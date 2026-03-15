#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto, shape_inference


def _tensorproto_to_np_dtype(elem_type: int):
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT8: np.int8,
        TensorProto.INT16: np.int16,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.UINT8: np.uint8,
    }
    return mapping.get(elem_type)


def _build_type_map(model: onnx.ModelProto) -> dict[str, int]:
    type_map: dict[str, int] = {}

    def add_vi(vi):
        if vi.type.HasField("tensor_type"):
            elem_type = vi.type.tensor_type.elem_type
            if vi.name:
                type_map[vi.name] = elem_type

    for vi in model.graph.input:
        add_vi(vi)
    for vi in model.graph.output:
        add_vi(vi)
    for vi in model.graph.value_info:
        add_vi(vi)

    for init in model.graph.initializer:
        type_map[init.name] = init.data_type

    return type_map


def _build_initializer_map(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    return {init.name: init for init in model.graph.initializer}


def _is_all_zero_initializer(init: onnx.TensorProto) -> bool:
    arr = numpy_helper.to_array(init)
    return np.all(arr == 0)


def rewrite_int32_dequant_zero_points(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    # Try to infer types first so value_info is populated.
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    type_map = _build_type_map(model)
    init_map = _build_initializer_map(model)

    new_nodes = []
    changed = 0

    for node in model.graph.node:
        if node.op_type != "DequantizeLinear":
            new_nodes.append(node)
            continue

        # Need 3-input form to rewrite
        if len(node.input) != 3:
            new_nodes.append(node)
            continue

        x_name, scale_name, zp_name = node.input

        x_dtype = type_map.get(x_name)
        zp_init = init_map.get(zp_name)

        # Rewrite only if:
        # 1) x is int32
        # 2) zero_point is an initializer
        # 3) zero_point tensor is all zeros
        if x_dtype == TensorProto.INT32 and zp_init is not None and _is_all_zero_initializer(zp_init):
            new_node = helper.make_node(
                "DequantizeLinear",
                inputs=[x_name, scale_name],   # drop zero_point
                outputs=list(node.output),
                name=node.name,
            )

            # preserve attributes like axis / block_size / output_dtype if present
            for attr in node.attribute:
                new_node.attribute.append(attr)

            new_nodes.append(new_node)
            changed += 1
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, changed


def main():
    src = Path("../models/quantized_models/onnx/yolov8n_opset17_int8_qdq.onnx")
    dst = Path("../models/quantized_models/onnx/yolov8n_opset17_int8_qdq_tvmfix.onnx")

    model = onnx.load(str(src))
    model, changed = rewrite_int32_dequant_zero_points(model)
    onnx.save(model, str(dst))
    print(f"Saved {dst} (rewrote {changed} DequantizeLinear nodes)")


if __name__ == "__main__":
    main()