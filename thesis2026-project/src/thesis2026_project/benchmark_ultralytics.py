import argparse
import time

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--img", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    model = YOLO(args.model).model
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    x = preprocess(args.img, args.imgsz)
    x_t = torch.from_numpy(x).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x_t)
        if device.type == "mps":
            torch.mps.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(args.runs):
            t0 = time.perf_counter()
            _ = model(x_t)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    print(f"Ultralytics ({args.device}) avg: {np.mean(times):.3f} ms")
    print(f"Ultralytics ({args.device}) p50: {np.percentile(times,50):.3f} ms")
    print(f"Ultralytics ({args.device}) p90: {np.percentile(times,90):.3f} ms")

if __name__ == "__main__":
    main()
