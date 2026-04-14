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
    graph = model.graph

    # Build lookup: tensor_name → initializer
    init_map = {init.name: init for init in graph.initializer}

    # Build lookup: output_name → node  (for tracing Q/DQ chains)
    output_to_node: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Collect names of all 1-D initializers (biases are always rank-1)
    bias_init_names: set[str] = {
        init.name
        for init in graph.initializer
        if len(init.dims) == 1
    }

    # For Conv/Gemm/ConvTranspose nodes, bias is input index 2.
    # Walk: bias_input → DequantizeLinear → QuantizeLinear → original_init
    # We want to replace the DQ output consumed by the op with the original init.
    nodes_to_remove: set[str] = set()   # node names/ids to drop
    remap: dict[str, str] = {}          # old_tensor → original_init_name

    for node in graph.node:
        if node.op_type not in ("Conv", "ConvTranspose", "Gemm"):
            continue
        if len(node.input) < 3 or not node.input[2]:
            continue

        bias_input = node.input[2]

        # Check if bias_input comes from a DequantizeLinear node
        dq_node = output_to_node.get(bias_input)
        if dq_node is None or dq_node.op_type != "DequantizeLinear":
            continue

        # The DQ node's first input should come from a QuantizeLinear node
        q_input = dq_node.input[0]
        q_node = output_to_node.get(q_input)
        if q_node is None or q_node.op_type != "QuantizeLinear":
            continue

        # The Q node's first input should be the original bias initializer
        orig_bias = q_node.input[0]
        if orig_bias not in bias_init_names:
            continue

        # Mark this QDQ pair for removal and record the remap
        nodes_to_remove.add(id(dq_node))
        nodes_to_remove.add(id(q_node))
        remap[bias_input] = orig_bias

    if not remap:
        return model   # nothing to do

    # Apply remap to all nodes that consume the DQ output
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in remap:
                node.input[i] = remap[inp]

    # Remove the QDQ nodes
    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Remove the Q/DQ scale+zeropoint initializers that are now orphaned
    # (keep the original bias initializers)
    still_used: set[str] = set()
    for node in graph.node:
        still_used.update(node.input)
    new_inits = [init for init in graph.initializer if init.name in still_used
                 or init.name in bias_init_names]
    del graph.initializer[:]
    graph.initializer.extend(new_inits)

    removed_pairs = len(nodes_to_remove) // 2
    print(f"[strip_bias_qdq] removed {removed_pairs} QDQ pairs from bias tensors")
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
