#!/usr/bin/env python3
"""
build_trt_engine.py — Build TensorRT engines directly with the TRT Python API.

Uses the exact same `tensorrt` package that bench_rigorous._worker_trt uses,
so the serialised engine is guaranteed to deserialise correctly at benchmark
time (no Ultralytics wrapper, no version mismatch).

Supports FP32, FP16, and INT8 (entropy calibration from COCO images).
Uses torch CUDA tensors for calibration device memory — no pycuda required.

Compatible with TRT 8.x and TRT 10.x.

Usage (run from project root)
------------------------------
  # FP32
  python src/optimizer/quantization/build_trt_engine.py \\
      --onnx  models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \\
      --output models/tensorrt_exports/yolov8n_fp32.engine \\
      --precision fp32

  # FP16
  python src/optimizer/quantization/build_trt_engine.py \\
      --onnx  models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \\
      --output models/tensorrt_exports/yolov8n_fp16.engine \\
      --precision fp16

  # INT8 with COCO calibration
  python src/optimizer/quantization/build_trt_engine.py \\
      --onnx  models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \\
      --output models/tensorrt_exports/yolov8n_int8.engine \\
      --precision int8 \\
      --calib-dir  datasets/coco_subset/train_1percent/images \\
      --calib-cache models/tensorrt_exports/int8_calib.cache \\
      --n-calib 9999
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorrt as trt


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TRT_MAJOR = int(trt.__version__.split(".")[0])


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _letterbox(img: np.ndarray, target: int = 640) -> np.ndarray:
    """Resize + letterbox pad to target×target (matches YOLOv8 preprocessing)."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad = np.full((target, target, 3), 114, dtype=np.uint8)
    dh, dw = (target - nh) // 2, (target - nw) // 2
    pad[dh:dh + nh, dw:dw + nw] = img
    return pad


def _collect_image_paths(image_dir: str, n: int) -> list[str]:
    """Return sorted list of up to n image paths — no loading, no RAM cost."""
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
    )[:n]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    _log(f"Found {len(paths)} calibration images in {image_dir}")
    return paths


def _load_one(path: str, imgsz: int) -> np.ndarray:
    """Load, letterbox, and normalise a single image → float32 NCHW (1,3,H,W)."""
    img = cv2.imread(path)
    if img is None:
        raise OSError(f"cv2 could not read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _letterbox(img, target=imgsz)
    x = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[np.newaxis])  # (1,3,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# INT8 calibrator — IInt8EntropyCalibrator2, torch device memory, no pycuda
# Lazy loading: images are read from disk one at a time in get_batch(),
# so RAM usage stays flat (~50 MB) regardless of dataset size.
# ─────────────────────────────────────────────────────────────────────────────

class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 entropy calibrator backed by torch CUDA tensors (no pycuda).

    Images are loaded lazily — only one image lives in RAM at a time.
    """

    def __init__(
        self,
        image_dir: str,
        cache_file: str,
        n_images: int = 9999,
        imgsz: int = 640,
        batch_size: int = 1,
    ):
        super().__init__()
        self._cache_file = cache_file
        self._batch_size = batch_size
        self._imgsz = imgsz
        # Store only file paths — no images loaded yet
        self._paths = _collect_image_paths(image_dir, n_images)
        self._index = 0
        # Persistent device buffer — one image slot, reused every get_batch call
        self._buf = torch.zeros(
            batch_size, 3, imgsz, imgsz, dtype=torch.float32, device="cuda"
        )

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names: list[str]):            # noqa: ARG002  (names unused)
        if self._index >= len(self._paths):
            return None                               # signals end of calibration
        if self._index % 100 == 0:
            _log(f"Calibrating image {self._index + 1}/{len(self._paths)} …")
        arr = _load_one(self._paths[self._index], self._imgsz)
        self._index += 1
        self._buf.copy_(torch.from_numpy(arr))        # arr already on CPU, buf on CUDA
        return [self._buf.data_ptr()]

    def read_calibration_cache(self):
        if os.path.exists(self._cache_file):
            _log(f"Using existing calibration cache: {self._cache_file}")
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        Path(self._cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_file, "wb") as f:
            f.write(cache)
        _log(f"Calibration cache written: {self._cache_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Engine builder
# ─────────────────────────────────────────────────────────────────────────────

def build_engine(
    onnx_path: str,
    output_path: str,
    precision: str,           # "fp32" | "fp16" | "int8"
    workspace_gb: float,
    imgsz: int,
    batch: int,
    calib_dir: str | None,
    calib_cache: str,
    n_calib: int,
) -> None:
    _log(f"TensorRT {trt.__version__}  precision={precision}  onnx={onnx_path}")

    logger  = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)

    # Network flags — EXPLICIT_BATCH is required for ONNX dynamic shapes
    try:
        net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        net_flags = 0          # TRT 10+ dropped the flag (it's always explicit)

    network = builder.create_network(net_flags)
    parser  = trt.OnnxParser(network, logger)

    _log(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
    _log(f"ONNX parsed — {network.num_layers} layers")

    config = builder.create_builder_config()

    # Workspace
    workspace_bytes = int(workspace_gb * (1 << 30))
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    except AttributeError:
        config.max_workspace_size = workspace_bytes     # TRT 8.x fallback

    # Precision flags
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            _log("WARNING: platform does not report fast FP16 — building anyway")
        config.set_flag(trt.BuilderFlag.FP16)
        _log("FP16 flag set")

    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            _log("WARNING: platform does not report fast INT8 — building anyway")
        config.set_flag(trt.BuilderFlag.INT8)

        if calib_dir is None:
            raise ValueError("--calib-dir is required for INT8 precision")

        calibrator = TorchEntropyCalibrator(
            image_dir=calib_dir,
            cache_file=calib_cache,
            n_images=n_calib,
            imgsz=imgsz,
            batch_size=batch,
        )
        config.int8_calibrator = calibrator
        _log(f"INT8 calibrator set  n_images={n_calib}  cache={calib_cache}")

    # Optimisation profile for dynamic batch/shape axes (if any)
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name   = input_tensor.name
    shape = (batch, 3, imgsz, imgsz)
    profile.set_shape(input_name, min=shape, opt=shape, max=shape)
    config.add_optimization_profile(profile)
    _log(f"Optimisation profile: {input_name} = {shape}")

    # Build
    _log("Building serialised engine (this may take several minutes for INT8) …")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None — check VERBOSE log above")
    elapsed = time.perf_counter() - t0
    # IHostMemory exposes .nbytes; convert to memoryview for file I/O
    _log(f"Build complete in {elapsed:.1f}s  ({serialized.nbytes / 1e6:.1f} MB)")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(memoryview(serialized))
    _log(f"Engine saved → {out}")

    # Quick deserialisation check — same as _worker_trt does at benchmark time
    _log("Verifying engine can be deserialised …")
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(out.read_bytes())
    if engine is None:
        raise RuntimeError("Verification failed: engine deserialised as None")
    _log("Verification OK")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a TensorRT engine from FP32 ONNX using the TRT Python API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--onnx",    required=True, help="Path to FP32 ONNX model")
    ap.add_argument("--output",  required=True, help="Output .engine path")
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    ap.add_argument("--imgsz",   type=int,   default=640)
    ap.add_argument("--batch",   type=int,   default=1)
    ap.add_argument("--workspace-gb", type=float, default=1.0,
                    help="Builder workspace in GB (default: 1)")
    ap.add_argument("--calib-dir",   default=None,
                    help="Directory of COCO calibration images (required for int8)")
    ap.add_argument("--calib-cache", default="models/tensorrt_exports/int8_calib.cache",
                    help="Path to read/write calibration cache")
    ap.add_argument("--n-calib", type=int, default=200,
                    help="Number of calibration images (default: 200)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.onnx):
        sys.exit(f"ONNX file not found: {args.onnx}")

    build_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        precision=args.precision,
        workspace_gb=args.workspace_gb,
        imgsz=args.imgsz,
        batch=args.batch,
        calib_dir=args.calib_dir,
        calib_cache=args.calib_cache,
        n_calib=args.n_calib,
    )


if __name__ == "__main__":
    main()
