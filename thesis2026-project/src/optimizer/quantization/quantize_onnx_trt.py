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
import onnx
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
# Post-processing: strip QDQ nodes from bias tensors (TRT does not support
# quantized biases — only INT8 activations and weights are allowed).
#
# ORT inserts QuantizeLinear/DequantizeLinear pairs around every initializer
# including 1-D bias tensors.  TRT rejects these with:
#   "IDequantizeLayer can only run in DataType::kINT8/kFP8/kINT4 precision"
# Solution: find every Conv/Gemm bias input that is routed through a QDQ pair
# and reconnect it directly to the original FP32 initializer.
# ─────────────────────────────────────────────────────────────────────────────

def _remove_bias_qdq(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove DequantizeLinear (and optional preceding QuantizeLinear) nodes
    that wrap bias tensors (1-D initializers).

    ORT uses two patterns for bias quantization:
      Pattern A — FP32 bias: fp32_init → QuantizeLinear → DequantizeLinear → Conv
      Pattern B — INT32 bias: int32_init → DequantizeLinear → Conv
                  (ORT pre-multiplies scale so bias is already in INT32; no Q node)

    TRT rejects both because it does not support quantized biases at all.
    Pattern A: rewire Conv directly to the original FP32 initializer.
    Pattern B: dequantize the INT32 values back to FP32 and insert a new
               FP32 initializer.
    """
    import onnx.numpy_helper as nph

    graph = model.graph
    init_map  = {init.name: init for init in graph.initializer}
    init_rank = {init.name: len(init.dims) for init in graph.initializer}

    output_to_node: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    nodes_to_remove: set[int] = set()
    remap: dict[str, str] = {}       # dq_output → replacement_init_name
    new_inits: list = []

    for node in graph.node:
        if node.op_type != "DequantizeLinear":
            continue

        dq_in    = node.input[0]
        dq_scale = node.input[1] if len(node.input) > 1 else None
        dq_zp    = node.input[2] if len(node.input) > 2 else None
        dq_out   = node.output[0]

        # ── Pattern A: FP32_init → QuantizeLinear → (dq_in) → DequantizeLinear ──
        q_node = output_to_node.get(dq_in)
        if q_node is not None and q_node.op_type == "QuantizeLinear":
            orig = q_node.input[0]
            if init_rank.get(orig, -1) == 1:          # 1-D → bias
                nodes_to_remove.add(id(node))
                nodes_to_remove.add(id(q_node))
                remap[dq_out] = orig
                continue

        # ── Pattern B: INT32/INT8_init → DequantizeLinear directly ──────────────
        if init_rank.get(dq_in, -1) != 1:
            continue                                   # not a 1-D bias initializer

        scale_init = init_map.get(dq_scale) if dq_scale else None
        if scale_init is None:
            continue

        q_arr  = nph.to_array(init_map[dq_in]).astype(np.float64)
        s_arr  = nph.to_array(scale_init).astype(np.float64)
        zp_arr = (nph.to_array(init_map[dq_zp]).astype(np.float64)
                  if dq_zp and dq_zp in init_map else np.zeros(1, np.float64))

        fp32_bias = ((q_arr - zp_arr) * s_arr).astype(np.float32)
        fp32_name = dq_in + "_fp32_bias"
        new_inits.append(onnx.numpy_helper.from_array(fp32_bias, name=fp32_name))

        nodes_to_remove.add(id(node))
        remap[dq_out] = fp32_name

    if not remap:
        print("[strip_bias_qdq] no bias DQ nodes found — check graph structure")
        return model

    # Rewire consumers
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in remap:
                node.input[i] = remap[inp]

    # Drop removed nodes
    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Add dequantized FP32 bias initializers
    graph.initializer.extend(new_inits)

    print(f"[strip_bias_qdq] rewired {len(remap)} bias tensors "
          f"({len(nodes_to_remove)} Q/DQ nodes removed)")
    return model


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
    ap.add_argument("--calibration-method", default="MinMax",
                    choices=["Entropy", "MinMax", "Percentile"],
                    help="Calibration algorithm (default: MinMax — low RAM; "
                         "Entropy is higher quality but OOMs on Orin Nano)")
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
    print(f"[quantize] ORT quantization done")

    # TRT does not support quantized biases — strip QDQ pairs from bias tensors.
    print(f"[quantize] Stripping bias QDQ nodes for TRT compatibility ...")
    model = onnx.load(args.output)
    model = _remove_bias_qdq(model)
    onnx.save(model, args.output)

    print(f"[quantize] Done → {args.output}")


if __name__ == "__main__":
    main()
