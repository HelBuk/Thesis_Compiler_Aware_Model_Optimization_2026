#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from optimizer.evaluation.yolo_metrics import (
    Backend,
    build_backend,
    detect_best_device,
    eval_backend,
)


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_backend_latency(
    backend: Backend,
    imgsz: int,
    batch: int,
    runs: int,
    warmup: int,
    device: str,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    x_np = rng.random((batch, 3, imgsz, imgsz), dtype=np.float32)
    x = torch.from_numpy(x_np)

    for _ in range(warmup):
        _ = backend.infer_batch(x)
    sync_device(device)

    latencies_ms = []
    t0 = time.perf_counter()
    for _ in range(runs):
        t_start = time.perf_counter()
        _ = backend.infer_batch(x)
        sync_device(device)
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000.0)
    t1 = time.perf_counter()

    lat = np.array(latencies_ms, dtype=np.float64)

    total_ms = (t1 - t0) * 1000.0
    mean_ms = float(lat.mean())
    median_ms = float(np.median(lat))
    std_ms = float(lat.std(ddof=0))
    p90_ms = float(np.percentile(lat, 90))
    p95_ms = float(np.percentile(lat, 95))
    p99_ms = float(np.percentile(lat, 99))

    fps_batch = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    fps_images = fps_batch * batch

    throughput_images_per_s = (runs * batch) / ((t1 - t0) if (t1 - t0) > 0 else 1e-12)

    return {
        "batch": int(batch),
        "runs": int(runs),
        "warmup": int(warmup),
        "mean_ms_per_batch": mean_ms,
        "median_ms_per_batch": median_ms,
        "std_ms_per_batch": std_ms,
        "p90_ms_per_batch": p90_ms,
        "p95_ms_per_batch": p95_ms,
        "p99_ms_per_batch": p99_ms,
        "fps_batches": float(fps_batch),
        "fps_images": float(fps_images),
        "throughput_images_per_s": float(throughput_images_per_s),
        "total_wall_ms": float(total_ms),
    }


def print_dict(title: str, d: Dict[str, Any]) -> None:
    print(f"\n===== {title} =====")
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


def free_backend(backend: Optional[Backend]) -> None:
    if backend is None:
        return
    try:
        backend.close()
    except Exception:
        pass
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare_stats(
    name_a: str,
    stats_a: Dict[str, Any],
    name_b: str,
    stats_b: Dict[str, Any],
) -> None:
    print(f"\n===== Delta ({name_b} - {name_a}) =====")
    common = sorted(set(stats_a.keys()) & set(stats_b.keys()))
    for key in common:
        va = stats_a[key]
        vb = stats_b[key]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = float(vb) - float(va)
            print(f"{key}: {delta:+.6f}")


def build_arg_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """
    Reuse build_backend(...) from yolo_metrics.py.
    It expects fields like imgsz/conf/iou/max_det/threads/etc.
    """
    return argparse.Namespace(
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        threads=args.threads,
        tflite_delegate="cpu",
        trt_fp16=False,
        trt_int8=False,
        trt_engine_cache=False,
        trt_engine_cache_path=args.trt_engine_cache_path,
        trt_workspace_size=args.trt_workspace_size,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark and evaluate two TensorRT .engine files"
    )

    ap.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    ap.add_argument("--engine_a", required=True, help="Path to engine A (.engine)")
    ap.add_argument("--engine_b", required=True, help="Path to engine B (.engine)")

    ap.add_argument("--name_a", default="fp32_engine", help="Display name for engine A")
    ap.add_argument("--name_b", default="int8_engine", help="Display name for engine B")

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)

    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)

    ap.add_argument("--device_a", default="cuda:0")
    ap.add_argument("--device_b", default="cuda:0")

    ap.add_argument("--bench_runs", type=int, default=200)
    ap.add_argument("--bench_warmup", type=int, default=30)

    ap.add_argument("--eval_max_batches", type=int, default=0, help="0 = full val set")
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--project", type=str, default="runs/engine_eval")
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--plots", action="store_true")

    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--trt_engine_cache_path", type=str, default="./trt_cache")
    ap.add_argument("--trt_workspace_size", type=int, default=2147483648)

    ap.add_argument("--out_json", type=str, default="", help="Optional path to save combined results JSON")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    backend_args = build_arg_namespace(args)

    backend_a: Optional[Backend] = None
    backend_b: Optional[Backend] = None

    try:
        print(f"[A] Loading {args.name_a}: {args.engine_a} on {args.device_a}")
        backend_a = build_backend(
            kind="trt_engine",
            model_path=args.engine_a,
            args=backend_args,
            device_override=args.device_a,
        )

        print(f"[B] Loading {args.name_b}: {args.engine_b} on {args.device_b}")
        backend_b = build_backend(
            kind="trt_engine",
            model_path=args.engine_b,
            args=backend_args,
            device_override=args.device_b,
        )

        print("\n### Benchmarking latency / FPS")
        bench_a = benchmark_backend_latency(
            backend=backend_a,
            imgsz=args.imgsz,
            batch=args.batch,
            runs=args.bench_runs,
            warmup=args.bench_warmup,
            device=args.device_a,
        )
        bench_b = benchmark_backend_latency(
            backend=backend_b,
            imgsz=args.imgsz,
            batch=args.batch,
            runs=args.bench_runs,
            warmup=args.bench_warmup,
            device=args.device_b,
        )

        print_dict(f"{args.name_a} benchmark", bench_a)
        print_dict(f"{args.name_b} benchmark", bench_b)
        compare_stats(args.name_a, bench_a, args.name_b, bench_b)

        # print("\n### Evaluating detection accuracy")
        # stats_a = eval_backend(
        #     backend=backend_a,
        #     data_yaml=args.data,
        #     imgsz=args.imgsz,
        #     batch=args.batch,
        #     conf=args.conf,
        #     iou=args.iou,
        #     max_det=args.max_det,
        #     project=args.project,
        #     name=f"{args.name_a}_eval",
        #     save_json=args.save_json,
        #     save_txt=args.save_txt,
        #     plots=args.plots,
        #     max_batches=args.eval_max_batches,
        #     workers=args.workers,
        # )

        # stats_b = eval_backend(
        #     backend=backend_b,
        #     data_yaml=args.data,
        #     imgsz=args.imgsz,
        #     batch=args.batch,
        #     conf=args.conf,
        #     iou=args.iou,
        #     max_det=args.max_det,
        #     project=args.project,
        #     name=f"{args.name_b}_eval",
        #     save_json=args.save_json,
        #     save_txt=args.save_txt,
        #     plots=args.plots,
        #     max_batches=args.eval_max_batches,
        #     workers=args.workers,
        # )

        # print_dict(f"{args.name_a} accuracy", stats_a)
        # print_dict(f"{args.name_b} accuracy", stats_b)
        # compare_stats(args.name_a, stats_a, args.name_b, stats_b)

        # results = {
        #     "engine_a": {
        #         "name": args.name_a,
        #         "path": args.engine_a,
        #         "device": args.device_a,
        #         "benchmark": bench_a,
        #         "accuracy": stats_a,
        #     },
        #     "engine_b": {
        #         "name": args.name_b,
        #         "path": args.engine_b,
        #         "device": args.device_b,
        #         "benchmark": bench_b,
        #         "accuracy": stats_b,
        #     },
        # }

        # if args.out_json:
        #     out_path = Path(args.out_json)
        #     out_path.parent.mkdir(parents=True, exist_ok=True)
        #     out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        #     print(f"\nSaved results JSON to: {out_path}")

    finally:
        free_backend(backend_a)
        free_backend(backend_b)


if __name__ == "__main__":
    main()