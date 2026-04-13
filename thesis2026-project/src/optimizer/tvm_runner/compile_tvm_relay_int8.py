#!/usr/bin/env python3
"""
compile_tvm_relay_int8.py  —  INT8 quantization and compilation via TVM Relay.

WHY THIS EXISTS
---------------
The QDQ ONNX approach (compile_tvm_int8.py) injects explicit QuantizeLinear /
DequantizeLinear nodes.  TVM's Relax ONNX importer keeps these as separate
relax.quantize / relax.dequantize ops that MetaSchedule schedules
independently — so you get FP32 arithmetic plus extra cast and memory-copy
kernels, i.e. SLOWER than FP32.

This script uses TVM Relay's own quantization pipeline:
  FP32 ONNX  →  Relay IR  →  relay.quantize (kl_divergence calibration)
                            →  qnn.conv2d ops (native INT8)  →  MetaSchedule
                            →  dp4a / INT8 tensor-core kernels on sm87  →  .so

The output .so is loaded with tvm.contrib.graph_executor (not VirtualMachine).
The bench_rigorous _worker_tvm_relay_int8 and TVMRelayBackend in
yolo_accuracy_comparison handle this automatically.

PIPELINE
--------
1. Load FP32 ONNX → relay.frontend.from_onnx
2. relay.quantize: annotate → calibrate (KL-divergence on COCO subset) → realize
3. relay.qnn.transform.CanonicalizeOps  →  relay.transform.FoldConstant
4. meta_schedule.tune_relay (MetaSchedule, builds dp4a-aware schedules)
5. relay.build with MetaSchedule DB  →  GraphExecutorFactoryModule
6. export_library  →  .so

USAGE (run from src/optimizer/tvm_runner/)
------------------------------------------
  # Full pipeline (tune + calibrate + export):
  python compile_tvm_relay_int8.py all \\
      --onnx ../../../models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \\
      --calib-dir ../../../datasets/coco_subset/train_1percent/images \\
      --output ../../../models/compiled_tvm_models/yolov8n_orin_relay_int8_1pct.so \\
      --db-dir out/tvm_tuning_logs/yolov8n_relay_int8_v1

  # Tune only (resume later):
  python compile_tvm_relay_int8.py tune ...

  # Export only (use existing DB):
  python compile_tvm_relay_int8.py export ...

  # Quick smoke-test (global_scale calibration, no dataset needed):
  python compile_tvm_relay_int8.py all --no-dataset-calib \\
      --onnx ../../../models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \\
      --output ../../../models/compiled_tvm_models/yolov8n_orin_relay_int8_globalscale.so \\
      --db-dir out/tvm_tuning_logs/yolov8n_relay_int8_v1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import onnx
import torch

import tvm
import tvm.relay as relay
import tvm.relay.quantize as qtz
from tvm import meta_schedule as ms

# Relay meta_schedule integration: tune_relay / ApplyHistoryBest
try:
    from tvm.meta_schedule import tune_relay
    from tvm.meta_schedule.database import JSONDatabase
except ImportError:
    # Some TVM builds expose this differently
    from tvm.meta_schedule.integration import tune_relay  # type: ignore[no-redef]
    from tvm.meta_schedule.database import JSONDatabase   # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Log:
    def __init__(self, path: Optional[Path] = None):
        self._path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def _ts(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def info(self, msg: str) -> None:
        line = f"{self._ts()} [INFO]  {msg}"
        print(line, flush=True)
        if self._path:
            self._path.open("a").write(line + "\n")

    def warn(self, msg: str) -> None:
        line = f"{self._ts()} [WARN]  {msg}"
        print(line, file=sys.stderr, flush=True)
        if self._path:
            self._path.open("a").write(line + "\n")


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------

def _letterbox(img: np.ndarray, target: int = 640) -> np.ndarray:
    """Resize + pad to target × target (same as YOLOv8 preprocessing)."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    import cv2  # lazy import: only needed during calibration
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad = np.full((target, target, 3), 114, dtype=np.uint8)
    dh, dw = (target - nh) // 2, (target - nw) // 2
    pad[dh : dh + nh, dw : dw + nw] = img
    return pad


def calibration_images(
    image_dir: str,
    n_images: int = 200,
    imgsz: int = 640,
    input_name: str = "images",
) -> Generator[Dict[str, np.ndarray], None, None]:
    """Yield preprocessed COCO images as Relay calibration records."""
    import cv2

    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg"))
                   + glob.glob(os.path.join(image_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    paths = paths[:n_images]

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = _letterbox(img, target=imgsz)
        x = img.astype(np.float32) / 255.0          # [0, 1]
        x = np.transpose(x, (2, 0, 1))[np.newaxis]  # NCHW
        yield {input_name: x}


# ---------------------------------------------------------------------------
# TVM target
# ---------------------------------------------------------------------------

def make_target(device: str, arch: Optional[str], log: Log) -> tvm.target.Target:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        auto_arch = f"sm_{cc_major}{cc_minor}"
        chosen_arch = arch or auto_arch
        log.info(f"CUDA target  arch={chosen_arch}")
        return tvm.target.Target(
            f"cuda -arch={chosen_arch}",
            host="llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a78",
        )
    elif device == "cpu":
        log.info("CPU target  (aarch64 cortex-a78)")
        return tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a78 -num-cores=6")
    else:
        raise ValueError(f"Unknown device {device!r}")


# ---------------------------------------------------------------------------
# MetaSchedule DB sanitiser  (mirrors run.py)
# ---------------------------------------------------------------------------

def _strip_feature_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_feature_keys(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("feature."))}
    if isinstance(obj, list):
        return [_strip_feature_keys(x) for x in obj]
    return obj


def _sanitize_json_file(path: Path) -> None:
    if not path.exists():
        return
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return
    dec = json.JSONDecoder()
    try:
        data = dec.decode(txt)
        path.write_text(json.dumps(_strip_feature_keys(data), separators=(",", ":")), encoding="utf-8")
        return
    except json.JSONDecodeError:
        pass
    objs, i, n = [], 0, len(txt)
    while i < n:
        while i < n and txt[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(txt, i)
        objs.append(_strip_feature_keys(obj))
        i = j
    path.write_text("".join(json.dumps(o, separators=(",", ":")) + "\n" for o in objs), encoding="utf-8")


def sanitize_ms_db(work_dir: str) -> None:
    wd = Path(work_dir)
    _sanitize_json_file(wd / "database_workload.json")
    _sanitize_json_file(wd / "database_tuning_record.json")


# ---------------------------------------------------------------------------
# Step 1 — Load FP32 ONNX → Relay
# ---------------------------------------------------------------------------

def load_relay(onnx_path: str, imgsz: int, batch: int, log: Log):
    """Returns (mod, params) in Relay IR."""
    log.info(f"Loading ONNX → Relay  path={onnx_path}")
    onnx_model = onnx.load(onnx_path)
    input_shape = {"images": (batch, 3, imgsz, imgsz)}
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        shape=input_shape,
        freeze_params=True,
    )
    mod = relay.transform.InferType()(mod)
    log.info("Relay IR loaded")
    return mod, params


# ---------------------------------------------------------------------------
# Step 2 — Relay quantization
# ---------------------------------------------------------------------------

def quantize_relay(
    mod,
    params: dict,
    calib_dir: Optional[str],
    n_calib_images: int,
    imgsz: int,
    log: Log,
    global_scale_only: bool = False,
):
    """Quantize mod+params using Relay's own quantization pipeline.

    Two calibration modes:
      global_scale_only=True  — fast, no dataset, uses a fixed global scale (8.0).
                                Good for a smoke-test or when you have no COCO images.
      global_scale_only=False — KL-divergence per-layer calibration with real images.
                                More accurate; requires calib_dir with JPEG/PNG images.
    """
    if global_scale_only:
        log.info("Quantizing with global_scale=8.0 (no calibration dataset)")
        qconfig = qtz.qconfig(
            calibrate_mode="global_scale",
            global_scale=8.0,
            weight_scale="power2",
            dtype_input="int8",
            dtype_weight="int8",
            dtype_activation="int8",
            skip_conv_layers=[],
            skip_dense_layer=False,
        )
        with qconfig:
            mod = qtz.quantize(mod, params)
    else:
        if not calib_dir:
            raise ValueError("--calib-dir is required unless --no-dataset-calib is set")
        log.info(f"Quantizing with KL-divergence  calib_dir={calib_dir}  n={n_calib_images}")

        qconfig = qtz.qconfig(
            calibrate_mode="kl_divergence",
            weight_scale="power2",
            dtype_input="int8",
            dtype_weight="int8",
            dtype_activation="int8",
            skip_conv_layers=[],
            skip_dense_layer=False,
        )
        dataset = list(calibration_images(calib_dir, n_images=n_calib_images, imgsz=imgsz))
        log.info(f"Loaded {len(dataset)} calibration images")

        with qconfig:
            mod = qtz.annotate()(mod)
            mod = qtz.calibrate(dataset=dataset)(mod)
            mod = qtz.realize()(mod)

    # Canonicalize QNN ops into primitives TVM can schedule with dp4a
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    mod = relay.transform.FoldConstant()(mod)
    log.info("Relay quantization done")
    return mod


# ---------------------------------------------------------------------------
# Step 3 — MetaSchedule tuning for Relay
# ---------------------------------------------------------------------------

def run_tune_relay(
    mod,
    params: dict,
    target: tvm.target.Target,
    db_dir: str,
    max_trials: int,
    max_trials_per_task: int,
    num_trials_per_iter: int,
    timeout_builder_s: int,
    timeout_runner_s: int,
    log: Log,
) -> None:
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    log.info(
        f"MetaSchedule tuning  max_trials={max_trials}  "
        f"per_task={max_trials_per_task}  db_dir={db_dir}"
    )
    try:
        tune_relay(
            mod=mod,
            params=params,
            target=target,
            work_dir=db_dir,
            max_trials_global=max_trials,
            max_trials_per_task=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            builder=ms.builder.LocalBuilder(timeout_sec=timeout_builder_s),
            runner=ms.runner.LocalRunner(timeout_sec=timeout_runner_s),
        )
    except RuntimeError as exc:
        if "feature.has_asimd" in str(exc):
            log.warn("MetaSchedule DB parse error — sanitizing and continuing")
            sanitize_ms_db(db_dir)
        else:
            raise
    sanitize_ms_db(db_dir)
    log.info("Tuning complete")


# ---------------------------------------------------------------------------
# Step 4 — Compile with DB and export .so
# ---------------------------------------------------------------------------

def compile_and_export(
    mod,
    params: dict,
    target: tvm.target.Target,
    db_dir: str,
    output_path: str,
    log: Log,
) -> None:
    db_workload = Path(db_dir) / "database_workload.json"
    if not db_workload.exists():
        raise FileNotFoundError(
            f"MetaSchedule DB not found: {db_workload}\n"
            "Run 'tune' first, or use 'all' command."
        )
    log.info(f"Compiling with MetaSchedule DB  db_dir={db_dir}")
    sanitize_ms_db(db_dir)

    database = JSONDatabase(work_dir=db_dir)
    with ms.ApplyHistoryBest(database):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_meta_schedule": True}):
            lib = relay.build(mod, target=target, params=params)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Exporting .so  path={out}")
    lib.export_library(str(out))
    log.info(f"Done  size={out.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="TVM Relay INT8 quantization + MetaSchedule compile for Orin Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "cmd", choices=["tune", "export", "all"],
        help="tune: MetaSchedule tuning only; export: compile+export with existing DB; all: both",
    )
    ap.add_argument("--onnx", required=True, help="FP32 ONNX model path")
    ap.add_argument("--output", required=True, help="Output .so path")
    ap.add_argument(
        "--db-dir", default="out/tvm_tuning_logs/yolov8n_relay_int8_v1",
        help="MetaSchedule DB directory",
    )
    ap.add_argument("--calib-dir", default=None,
                    help="Directory of COCO calibration images (JPEG/PNG)")
    ap.add_argument("--n-calib", type=int, default=200,
                    help="Number of calibration images to use (default: 200)")
    ap.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    ap.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--arch", default=None, help="CUDA arch, e.g. sm87 (auto-detected)")
    ap.add_argument("--max-trials", type=int, default=500,
                    help="Max global tuning trials (default: 500)")
    ap.add_argument("--max-trials-per-task", type=int, default=20)
    ap.add_argument("--num-trials-per-iter", type=int, default=8)
    ap.add_argument("--timeout-builder", type=int, default=300)
    ap.add_argument("--timeout-runner", type=int, default=120)
    ap.add_argument(
        "--no-dataset-calib", action="store_true",
        help="Use global_scale=8.0 calibration (no dataset required — fast smoke test)",
    )
    ap.add_argument("--log-file", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()

    out_path = Path(args.output)
    log_path = (
        Path(args.log_file) if args.log_file
        else out_path.parent / f"{out_path.stem}_relay_compile.log"
    )
    log = Log(log_path)
    log.info(f"compile_tvm_relay_int8.py  cmd={args.cmd}")
    log.info(f"onnx={args.onnx}  output={args.output}  db={args.db_dir}")

    os.environ.setdefault("TVM_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    target = make_target(args.device, args.arch, log)
    mod, params = load_relay(args.onnx, args.imgsz, args.batch, log)
    mod = quantize_relay(
        mod, params,
        calib_dir=args.calib_dir,
        n_calib_images=args.n_calib,
        imgsz=args.imgsz,
        log=log,
        global_scale_only=args.no_dataset_calib,
    )

    if args.cmd in ("tune", "all"):
        run_tune_relay(
            mod=mod, params=params, target=target,
            db_dir=args.db_dir,
            max_trials=args.max_trials,
            max_trials_per_task=args.max_trials_per_task,
            num_trials_per_iter=args.num_trials_per_iter,
            timeout_builder_s=args.timeout_builder,
            timeout_runner_s=args.timeout_runner,
            log=log,
        )

    if args.cmd in ("export", "all"):
        compile_and_export(
            mod=mod, params=params, target=target,
            db_dir=args.db_dir,
            output_path=args.output,
            log=log,
        )

    log.info(f"Finished in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
