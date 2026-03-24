# yolov8_retrain.py

# Usage:
# python yolov8_retrain.py \
#   --weights ../models/yolov8n_pruned.pth \
#   --data /path/to/data.yaml \
#   --epochs 100 \
#   --imgsz 640 \
#   --batch 16 \
#   --device cuda:0
#   --fraction 0.01


from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


def load_pruned_yolo(weights: str, fallback_cfg: str = "yolov8n.yaml") -> YOLO:
    """
    Supports:
    - Ultralytics checkpoint (.pt)
    - Raw pruned module saved via torch.save(model, "...pth")
    - Dict checkpoint containing {"model": nn.Module}
    """
    w = Path(weights)

    if w.suffix == ".pt":
        return YOLO(str(w))

    obj = torch.load(w, map_location="cpu")

    if isinstance(obj, nn.Module):
        pruned_model = obj
    elif isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        pruned_model = obj["model"]
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(obj)}")

    yolo = YOLO(fallback_cfg)  # builds task/trainer scaffolding
    yolo.model = pruned_model
    yolo.ckpt = {"model": pruned_model}
    return yolo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Pruned .pt/.pth path")
    parser.add_argument("--data", type=str, required=True, help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001, help="Start lower after pruning")
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--project", type=str, default="runs/prune_finetune")
    parser.add_argument("--name", type=str, default="yolov8n_pruned_ft")
    parser.add_argument("--fallback-cfg", type=str, default="yolov8n.yaml",
                        help="Used when loading raw .pth module")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training dataset to use (0 < fraction <= 1.0)",
    )

    args = parser.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")


    model = load_pruned_yolo(args.weights, fallback_cfg=args.fallback_cfg)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        lrf=args.lrf,
        project=args.project,
        name=args.name,
        pretrained=False,
        resume=False,
        fraction=args.fraction,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(metrics)


if __name__ == "__main__":
    main()
