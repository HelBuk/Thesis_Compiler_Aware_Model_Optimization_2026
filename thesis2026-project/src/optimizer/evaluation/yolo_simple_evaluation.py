#!/usr/bin/env python3
from __future__ import annotations

import argparse

from optimizer.evaluation.yolo_metrics import (
    build_backend,
    detect_best_device,
    eval_backend,
)


def parse_args():
    ap = argparse.ArgumentParser("Evaluate model outputs using Ultralytics COCO metrics")

    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--max_batches", type=int, default=0)

    ap.add_argument("--project", type=str, default="runs/val_custom")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--save_json", action="store_true")

    ap.add_argument("--model_a", required=True)
    ap.add_argument("--model_b", required=True)

    ap.add_argument("--backend_a", choices=["tflite", "ort", "torch"], default=None)
    ap.add_argument("--backend_b", choices=["tflite", "ort", "torch"], default=None)
    ap.add_argument("--backend", choices=["tflite", "ort", "torch"], default=None)

    ap.add_argument("--device_a", default=None)
    ap.add_argument("--device_b", default=None)

    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tflite_delegate", choices=["cpu", "gpu"], default="cpu")

    return ap.parse_args()


def main():
    args = parse_args()

    backend_a_kind = args.backend_a or args.backend
    backend_b_kind = args.backend_b or args.backend
    if backend_a_kind is None or backend_b_kind is None:
        raise SystemExit("Specify either --backend (both) or both --backend_a and --backend_b.")

    print(f"[A] {args.model_a} (backend={backend_a_kind}, device={args.device_a or detect_best_device()})")
    backend_a = build_backend(backend_a_kind, args.model_a, args, device_override=args.device_a)
    stats_a = eval_backend(
        backend=backend_a,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name="A_eval",
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
        max_batches=args.max_batches,
    )

    print("\n===== A stats =====")
    for k, v in stats_a.items():
        print(f"{k}: {v}")

    print(f"\n[B] {args.model_b} (backend={backend_b_kind}, device={args.device_b or detect_best_device()})")
    backend_b = build_backend(backend_b_kind, args.model_b, args, device_override=args.device_b)
    stats_b = eval_backend(
        backend=backend_b,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name="B_eval",
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
        max_batches=args.max_batches,
    )

    print("\n===== B stats =====")
    for k, v in stats_b.items():
        print(f"{k}: {v}")

    print("\n===== Delta (B - A) =====")
    for key in sorted(set(stats_a.keys()) & set(stats_b.keys())):
        try:
            print(f"{key}: {float(stats_b[key]) - float(stats_a[key]):+.6f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()