import time
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_executor

"""
TVM YOLOv8n (ONNX) benchmark script for Raspberry Pi 5 CPU.

Usage (on Pi):
  1) Export ONNX on your dev machine:
     yolo export model=yolov8n.pt format=onnx imgsz=640 dynamic=False
     Default nms = false
  2) Copy the ONNX to the Pi, e.g. .../models/yolov8n.onnx
  3) Run:
     python3 tvm_optimizations.py

What it does:
  - Imports the ONNX graph into TVM Relay
  - Compiles it for Pi 5 (ARM64 Cortex-A76) using LLVM
  - Runs warmup + timed inference loops
  - Reports FPS and prints output tensor shape

Notes:
  - This script benchmarks raw model inference only (no NMS/post-processing).
  - For best performance, ensure your TVM build has LLVM enabled.
"""

MODEL_PATH_ONNX = "../models/yolov8n.onnx"
MODEL_PATH_PT = "../models/yolov8n.pt"

IMG_SIZE = 640
DTYPE = "float32"
NUM_RUNS = 100
NUM_WARMUP = 10

# Pi 5 CPU target (ARM64 Cortex-A76)
TARGET = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+fp16 -mcpu=cortex-a76"

def load_onnx_model(path, img_size):
    """
    Parse the ONNX graph into TVM's Relay IR
    :param path:
    :param img_size:
    :return:
        mod: the Relay module (the graph)
        params: model weight as TVM constant
    """
    import onnx
    onnx_model = onnx.load(path)
    input_name = onnx_model.graph.input[0].name
    shape_dict = {input_name: (1, 3, img_size, img_size)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    return mod, params, input_name

def main():
    mod, params, input_name = load_onnx_model(MODEL_PATH_ONNX, IMG_SIZE)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=TARGET, params=params)

    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))

    # dummy input
    x = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(DTYPE)
    module.set_input(input_name, x)

    # warmup
    for _ in range(NUM_WARMUP):
        module.run()

    # timed runs
    start = time.time()
    for _ in range(NUM_RUNS):
        module.run()
    end = time.time()

    elapsed = end - start
    fps = NUM_RUNS / elapsed
    print(f"Runs: {NUM_RUNS}, Total time: {elapsed:.3f}s, FPS: {fps:.2f}")

    # read output (just to confirm it works)
    out = module.get_output(0).asnumpy()
    print("Output shape:", out.shape)

if __name__ == "__main__":
    # main()

    load_onnx_model(MODEL_PATH_PT, IMG_SIZE)