import argparse
import os
import time
import numpy as np
from PIL import Image
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
import IPython

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
    ap.add_argument("--exec-mode", choices=["bytecode", "compiled"], default="bytecode")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--save-lib", default=None, help="Path to save compiled TVM VM library (.so/.dylib)")
    ap.add_argument("--load-lib", default=None, help="Path to previously compiled TVM VM library")
    ap.add_argument("--compile-only", action="store_true", help="Compile (or load-check) and exit")
    args = ap.parse_args()

    # if args.threads is not None:
    #     os.environ["TVM_NUM_THREADS"] = str(args.threads)
    #     os.environ["OMP_NUM_THREADS"] = str(args.threads)

    dev = tvm.cpu() if args.target == "llvm" else tvm.metal()
    x = preprocess(args.img, args.imgsz)

    if args.load_lib:
        t_load0 = time.perf_counter()
        loaded = tvm.runtime.load_module(args.load_lib)
        vm = relax.VirtualMachine(loaded, dev)
        t_load1 = time.perf_counter()
        print(f"Loaded VM library in {(t_load1 - t_load0):.3f} s from: {args.load_lib}")
    else:
        t_compile0 = time.perf_counter()
        onnx_model = onnx.load(args.onnx)
        input_name = get_input_name(onnx_model)
        mod = from_onnx(onnx_model, shape_dict={input_name: list(x.shape)}, dtype_dict="float32")
        mod = relax.get_pipeline("default_build")(mod)
        print(IPython.display.Code(mod.script(), language="python"))
        ex = tvm.compile(mod, target=args.target)
        vm = relax.VirtualMachine(ex, dev)
        t_compile1 = time.perf_counter()
        print(f"Compile time: {(t_compile1 - t_compile0):.3f} s")

        if args.save_lib:
            ex.export_library(args.save_lib)
            print(f"Saved VM library to: {args.save_lib}")

    if args.compile_only:
        print("Compile-only mode complete; skipping inference benchmark.")
        return



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
