#!/usr/bin/env python3
"""
benchmark_pruned_edge.py
------------------------
Benchmark pruned YOLOv8n models on edge hardware (Orin Nano GPU, RPi5 CPU)
without any retraining.

Measures latency, throughput, and mAP50-95 for up to three .pt checkpoints:
  - stock YOLOv8n baseline
  - fine-tune base (unpruned, same 50-ep schedule)
  - pruned + fine-tuned model

Supports CUDA (Orin Nano), CPU (RPi5), and MPS (Mac) backends.
Saves results to a JSON file for thesis comparison tables.

Usage — Orin Nano (GPU):
    python -m src.optimizer.pruning.benchmark_pruned_edge \
        --stock   models/yolov8n.pt \
        --base    models/pruned_models/combo_17_layers_e50.pt \
        --pruned  models/pruned_models/combo_17_layers_e50_v2.pt \
        --data    coco.yaml \
        --device  cuda:0 \
        --out     src/optimizer/pruning/runs/edge_pruning_orin.json

Usage — Raspberry Pi 5 (CPU):
    python -m src.optimizer.pruning.benchmark_pruned_edge \
        --stock   models/yolov8n.pt \
        --base    models/pruned_models/combo_17_layers_e50.pt \
        --pruned  models/pruned_models/combo_17_layers_e50_v2.pt \
        --data    coco.yaml \
        --device  cpu \
        --threads 4 \
        --out     src/optimizer/pruning/runs/edge_pruning_rpi5.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def sync(device: str) -> None:
    """Block until all pending ops on device finish."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def make_dummy_input(batch: int, imgsz: int, device: str) -> torch.Tensor:
    rng = np.random.default_rng(42)
    x_np = rng.random((batch, 3, imgsz, imgsz), dtype=np.float32)
    x = torch.from_numpy(x_np)
    if device not in ("cpu",):
        x = x.to(device, non_blocking=True)
    return x


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(
    model: torch.nn.Module,
    device: str,
    imgsz: int = 640,
    batch: int = 1,
    warmup: int = 50,
    runs: int = 200,
) -> Dict[str, float]:
    model.eval()
    x = make_dummy_input(batch, imgsz, device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        sync(device)

        latencies_ms: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            sync(device)
            latencies_ms.append((time.perf_counter() - t0) * 1_000)

    lat = np.array(latencies_ms)
    mean_ms = float(lat.mean())
    return {
        "warmup": warmup,
        "runs": runs,
        "batch": batch,
        "mean_ms": mean_ms,
        "median_ms": float(np.median(lat)),
        "std_ms": float(lat.std(ddof=0)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "fps": round(1_000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Accuracy evaluation via Ultralytics val()
# ---------------------------------------------------------------------------

def eval_accuracy(
    model_path: str,
    data: str,
    device: str,
    imgsz: int = 640,
    batch: int = 1,
    workers: int = 0,
) -> Dict[str, float]:
    y = YOLO(model_path)
    m = y.val(
        data=data,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        split="val",
        verbose=False,
        plots=False,
    )
    out: Dict[str, float] = {
        "mAP50-95": float(m.box.map),
        "mAP50": float(m.box.map50),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    if hasattr(m, "speed") and isinstance(m.speed, dict):
        out["ultralytics_inference_ms"] = float(m.speed.get("inference", 0))
    return out


# ---------------------------------------------------------------------------
# Single model runner
# ---------------------------------------------------------------------------

def load_torch_model(path: str, device: str) -> torch.nn.Module:
    """Load a .pt file as a raw PyTorch model on the requested device."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Ultralytics checkpoints wrap the model under 'model' key
    if isinstance(ckpt, dict):
        model = ckpt.get("model") or ckpt.get("ema") or next(iter(ckpt.values()))
    else:
        model = ckpt
    model = model.float().eval().to(device)
    return model


def run_one(
    label: str,
    model_path: str,
    data: str,
    device: str,
    imgsz: int,
    batch: int,
    warmup: int,
    runs: int,
    workers: int,
    threads: int,
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  path   : {model_path}")
    print(f"  device : {device}")
    print(f"{'='*60}")

    if device == "cpu" and threads > 0:
        torch.set_num_threads(threads)

    # --- latency ---
    print("  [1/2] Latency benchmark ...")
    model = load_torch_model(model_path, device)
    lat = benchmark_latency(model, device, imgsz=imgsz, batch=batch,
                            warmup=warmup, runs=runs)
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"        mean {lat['mean_ms']:.2f} ms | "
          f"median {lat['median_ms']:.2f} ms | "
          f"p95 {lat['p95_ms']:.2f} ms | "
          f"{lat['fps']:.1f} FPS")

    # --- accuracy ---
    print("  [2/2] Accuracy evaluation (COCO val) ...")
    acc = eval_accuracy(model_path, data, device, imgsz=imgsz,
                        batch=batch, workers=workers)
    print(f"        mAP50-95 {acc['mAP50-95']:.4f} | mAP50 {acc['mAP50']:.4f}")

    return {
        "label": label,
        "path": model_path,
        "device": device,
        "latency": lat,
        "accuracy": acc,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark pruned YOLOv8n on edge hardware (no training required)"
    )
    ap.add_argument("--stock",  required=True,
                    help="Stock YOLOv8n checkpoint (e.g. models/yolov8n.pt)")
    ap.add_argument("--base",   default="",
                    help="Fine-tune base — unpruned, same training schedule")
    ap.add_argument("--pruned", default="",
                    help="Pruned + fine-tuned checkpoint")
    ap.add_argument("--data",   required=True,
                    help="Ultralytics dataset YAML (e.g. coco.yaml)")
    ap.add_argument("--device", default="",
                    help="cuda:0 | cpu | mps  (auto-detected if omitted)")
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--batch",   type=int, default=1)
    ap.add_argument("--warmup",  type=int, default=50,
                    help="Warm-up iterations before timing")
    ap.add_argument("--runs",    type=int, default=200,
                    help="Timed inference iterations")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers for accuracy eval")
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads (only used when --device cpu)")
    ap.add_argument("--out",     default="",
                    help="Path to write results JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # auto-detect device
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
    print(f"\nDevice: {device}")

    common = dict(
        data=args.data,
        device=device,
        imgsz=args.imgsz,
        batch=args.batch,
        warmup=args.warmup,
        runs=args.runs,
        workers=args.workers,
        threads=args.threads,
    )

    results: List[Dict] = []

    results.append(run_one("stock_yolov8n", args.stock, **common))

    if args.base:
        results.append(run_one("fine_tune_base_unpruned", args.base, **common))

    if args.pruned:
        results.append(run_one("pruned_fine_tuned", args.pruned, **common))

    # --- summary ---
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    header = f"  {'Model':<30} {'mAP50-95':>10} {'mAP50':>8} {'Mean ms':>9} {'FPS':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['label']:<30} "
              f"{r['accuracy']['mAP50-95']:>10.4f} "
              f"{r['accuracy']['mAP50']:>8.4f} "
              f"{r['latency']['mean_ms']:>9.2f} "
              f"{r['latency']['fps']:>7.1f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
