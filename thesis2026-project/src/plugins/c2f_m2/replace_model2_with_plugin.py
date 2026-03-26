from pathlib import Path
import onnx
import onnx_graphsurgeon as gs

IN_ONNX = "../models/quantized_models/onnx/yolov8n_opset17_fp32.onnx"
OUT_ONNX = "../models/quantized_models/onnx/yolov8n_model2_plugin.onnx"
WEIGHTS_PATH = "../models/plugin_weights/model2_c2f_folded.npz"

model = onnx.load(IN_ONNX)
graph = gs.import_onnx(model)

nodes = graph.nodes

input_tensor = None
output_tensor = None

for n in nodes:
    if n.name == "/model.2/cv1/conv/Conv":
        input_tensor = n.inputs[0]
    if n.name == "/model.2/cv2/act/Mul":
        output_tensor = n.outputs[0]

assert input_tensor is not None, "Could not find model.2 input tensor"
assert output_tensor is not None, "Could not find model.2 output tensor"

print("INPUT:", input_tensor.name, input_tensor.shape)
print("OUTPUT:", output_tensor.name, output_tensor.shape)

selected = [n for n in nodes if n.name.startswith("/model.2/")]
selected_ids = {id(n) for n in selected}

print(f"Removing {len(selected)} nodes from model.2")

# Detach output tensor from its old producer
output_tensor.inputs.clear()

plugin_node = gs.Node(
    op="YoloC2fM2_TRT",
    name="YoloC2fM2_TRT_0",
    inputs=[input_tensor],
    outputs=[output_tensor],
    attrs={
        "weights_path": WEIGHTS_PATH,
        "plugin_version": "1",
        "plugin_namespace": ""
    }
)

graph.nodes = [n for n in graph.nodes if id(n) not in selected_ids]
graph.nodes.append(plugin_node)

left = [n.name for n in graph.nodes if n.name.startswith("/model.2/")]
print("Remaining model.2 nodes:", left)

graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), OUT_ONNX)
print(f"saved: {OUT_ONNX}")