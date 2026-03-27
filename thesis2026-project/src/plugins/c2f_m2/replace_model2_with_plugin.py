"""
replace_model2_with_plugin.py

Replaces the model.2 C2f sub-graph in the ONNX graph with a single
YoloC2fM2_TRT plugin node.

Boundary:
  input  — tensor feeding /model.2/cv1/conv/Conv
  output — tensor produced by /model.2/cv2/act/Mul
"""

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs

IN_ONNX      = "./models/quantized_models/onnx/yolov8n_opset17_fp32.onnx"
OUT_ONNX     = "./models/quantized_models/onnx/yolov8n_model2_plugin.onnx"
WEIGHTS_PATH = WEIGHTS_PATH = str(Path("./models/plugin_weights/model2_c2f_folded.bin").resolve())
NPZ_PATH     = "./models/plugin_weights/model2_c2f_folded.npz"  # for shape inspection

# ---------------------------------------------------------------------------
# Derive Cin / Cout from the exported weights so we can store them in the
# ONNX node attrs for documentation / future use.
# ---------------------------------------------------------------------------
npz = np.load(NPZ_PATH)
Cin   = int(npz["cv1_w"].shape[1])   # input  channels (32 for model.2)
Cout  = int(npz["cv2_w"].shape[0])   # output channels (64 for model.2)
halfC = Cout // 2

print(f"Cin={Cin}  Cout={Cout}  halfC={halfC}")

# ---------------------------------------------------------------------------
# Locate the C2f sub-graph boundaries
# ---------------------------------------------------------------------------
model = onnx.load(IN_ONNX)
graph = gs.import_onnx(model)
nodes = graph.nodes

input_tensor  = None
output_tensor = None

for n in nodes:
    if n.name == "/model.2/cv1/conv/Conv":
        input_tensor = n.inputs[0]
    if n.name == "/model.2/cv2/act/Mul":
        output_tensor = n.outputs[0]

assert input_tensor  is not None, "Could not find model.2 input tensor"
assert output_tensor is not None, "Could not find model.2 output tensor"

print("INPUT :", input_tensor.name,  input_tensor.shape)
print("OUTPUT:", output_tensor.name, output_tensor.shape)

# ---------------------------------------------------------------------------
# Remove all model.2 nodes and insert plugin node
# ---------------------------------------------------------------------------
selected     = [n for n in nodes if n.name.startswith("/model.2/")]
selected_ids = {id(n) for n in selected}
print(f"Removing {len(selected)} nodes from model.2")

output_tensor.inputs.clear()

plugin_node = gs.Node(
    op="YoloC2fM2_TRT",
    name="YoloC2fM2_TRT_0",
    inputs=[input_tensor],
    outputs=[output_tensor],
    attrs={
        "weights_path": str(Path(WEIGHTS_PATH).resolve()),
        "plugin_version": "1",
        "plugin_namespace": "",
        "cin": Cin,
        "cout": Cout,
        "halfc": halfC,
    }
)

graph.nodes = [n for n in graph.nodes if id(n) not in selected_ids]
graph.nodes.append(plugin_node)

remaining = [n.name for n in graph.nodes if n.name.startswith("/model.2/")]
print("Remaining model.2 nodes:", remaining)

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), OUT_ONNX)
print(f"\nSaved: {OUT_ONNX}")
