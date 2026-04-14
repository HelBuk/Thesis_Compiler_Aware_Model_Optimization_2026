#!/usr/bin/env python3
"""
quantize_onnx_trt.py — Generate a TensorRT-compatible INT8 QDQ ONNX.

Key differences from quantize_onnx.py:
  - ActivationSymmetric=True  → zero point always 0 (TRT requirement)
  - WeightSymmetric=True      → consistent with activation symmetry
  - QDQ format                → QuantizeLinear/DequantizeLinear pairs that
                                 TRT 10.x fuses into native INT8 kernels
  - Entropy calibration       → better than MinMax for detection models

Usage (run from project root):
  python src/optimizer/quantization/quantize_onnx_trt.py \
      --onnx   models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \
      --output models/quantized_models/onnx/yolov8n_int8_QDQ_sym_1pct.onnx \
      --calib-dir datasets/coco_subset/train_1percent/images

Then build TRT engine:
  python src/optimizer/quantization/build_trt_engine.py \
      --onnx   models/quantized_models/onnx/yolov8n_int8_QDQ_sym_1pct.onnx \
      --output models/tensorrt_exports/yolov8n_int8_qdq.engine \
      --precision int8
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing (letterbox, same as YOLOv8 / build_trt_engine.py)
# ─────────────────────────────────────────────────────────────────────────────

_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR  # Pillow ≥9.1 / <9.1


def _letterbox(img: np.ndarray, target: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = np.array(Image.fromarray(img).resize((nw, nh), _BILINEAR))
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    dh, dw = (target - nh) // 2, (target - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas


def _preprocess(path: str, imgsz: int = 640) -> np.ndarray:
    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"))
    img = _letterbox(img, target=imgsz)
    arr = img.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis]   # (1, 3, H, W)
    return np.ascontiguousarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration data reader — lazy, one image at a time
# ─────────────────────────────────────────────────────────────────────────────

class LazyImageReader(CalibrationDataReader):
    def __init__(self, image_dir: str, input_name: str, n: int = 9999, imgsz: int = 640):
        self._input_name = input_name
        self._imgsz = imgsz
        self._paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.png"))
        )[:n]
        if not self._paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        print(f"[calib] {len(self._paths)} images from {image_dir}")
        self._index = 0

    def get_next(self):
        if self._index >= len(self._paths):
            return None
        if self._index % 100 == 0:
            print(f"[calib] {self._index + 1}/{len(self._paths)} ...")
        arr = _preprocess(self._paths[self._index], self._imgsz)
        self._index += 1
        return {self._input_name: arr}

    def rewind(self):
        self._index = 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Quantize FP32 ONNX to TRT-compatible symmetric INT8 QDQ ONNX"
    )
    ap.add_argument("--onnx",       required=True, help="Input FP32 ONNX")
    ap.add_argument("--output",     required=True, help="Output QDQ ONNX path")
    ap.add_argument("--calib-dir",  required=True, help="Calibration image directory")
    ap.add_argument("--input-name", default="images", help="ONNX input tensor name")
    ap.add_argument("--n-calib",    type=int, default=9999,
                    help="Max calibration images (default: all)")
    ap.add_argument("--imgsz",      type=int, default=640)
    ap.add_argument("--calibration-method", default="Entropy",
                    choices=["Entropy", "MinMax", "Percentile"],
                    help="Calibration algorithm (default: Entropy)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(f"ONNX not found: {args.onnx}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    calib_method = {
        "Entropy":    CalibrationMethod.Entropy,
        "MinMax":     CalibrationMethod.MinMax,
        "Percentile": CalibrationMethod.Percentile,
    }[args.calibration_method]

    reader = LazyImageReader(
        image_dir=args.calib_dir,
        input_name=args.input_name,
        n=args.n_calib,
        imgsz=args.imgsz,
    )

    print(f"[quantize] {args.onnx} → {args.output}")
    print(f"[quantize] format=QDQ  symmetric=True  method={args.calibration_method}")

    quantize_static(
        model_input=args.onnx,
        model_output=args.output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=calib_method,
        extra_options={
            "ActivationSymmetric": True,   # zero point = 0  (TRT requirement)
            "WeightSymmetric": True,       # consistent
            "EnableSubgraph": False,       # don't quantize inside subgraphs
        },
    )

    print(f"[quantize] Done → {args.output}")


if __name__ == "__main__":
    main()
