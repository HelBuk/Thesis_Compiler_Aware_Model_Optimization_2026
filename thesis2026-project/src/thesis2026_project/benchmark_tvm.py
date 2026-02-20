import argparse
import os
import time
import numpy as np
from PIL import Image
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

def letterbox(img, new_shape):
    w, h = img.size
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    nw, nh = new_shape

    scale = min(nw / w, nh / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (nw, nh), (114, 114, 114))
    pad_w = (nw - new_w) // 2
    pad_h = (nh - new_h) // 2
    canvas.paste(img_resized, (pad_w, pad_h))
    return canvas

def preprocess(img_path, imgsz):
    img = Image.open(img_path).convert("RGB")
    img = letterbox(img, imgsz)
    x = np.asarray(img, dtype="float32") / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x

def get_input_name(onnx_model):
    init_names = {i.name for i in onnx_model.graph.initializer}
    for inp in onnx_model.graph.input:
        if inp.name not in init_names:
            return inp.name
    return onnx_model.graph.input[0].name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="yolov8n.onnx")
    ap.add_argument("--img", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--target", choices=["llvm", "metal"], default="llvm")
    ap.add_argument("--mcpu", default=None, help="LLVM -mcpu value (e.g., apple-m2)")
    ap.add_argument("--exec-mode", choices=["bytecode", "compiled"], default="bytecode")
    ap.add_argument("--threads", type=int, default=None, help="TVM_NUM_THREADS / OMP_NUM_THREADS")
    ap.add_argument(
        "--relax-pipeline",
        choices=["zero", "default", "default_build"],
        default="default_build",
        help="Relax pipeline to use (default_build is VM-friendly)",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    if args.threads is not None:
        os.environ["TVM_NUM_THREADS"] = str(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    x = preprocess(args.img, args.imgsz)

    onnx_model = onnx.load(args.onnx)
    input_name = get_input_name(onnx_model)

    mod = from_onnx(onnx_model, shape_dict={input_name: list(x.shape)}, dtype_dict="float32") #IR Module
    mod = relax.get_pipeline("default_build")(mod)
    if args.target == "llvm" and args.mcpu:
        target = tvm.target.Target({"kind": "llvm", "mcpu": args.mcpu})
    else:
        target = args.target
    ex = tvm.compile(mod, target=target)
    dev = tvm.cpu() if args.target == "llvm" else tvm.metal()
    vm = relax.VirtualMachine(ex, dev)

    x_nd = tvm.runtime.tensor(x, device=dev)

    # Warmup
    for _ in range(args.warmup):
        vm["main"](x_nd)
    if hasattr(dev, "sync"):
        dev.sync()

    # Timed runs
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        vm["main"](x_nd)
        if hasattr(dev, "sync"):
            dev.sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    print(f"TVM ({args.target}) avg: {np.mean(times):.3f} ms")
    print(f"TVM ({args.target}) p50: {np.percentile(times,50):.3f} ms")
    print(f"TVM ({args.target}) p90: {np.percentile(times,90):.3f} ms")

if __name__ == "__main__":
    main()
