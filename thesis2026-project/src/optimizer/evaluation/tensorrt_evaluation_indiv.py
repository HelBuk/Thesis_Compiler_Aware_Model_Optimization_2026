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
import nvtx

from src.optimizer.evaluation.yolo_metrics import (
    Backend,
    build_backend,
    eval_backend,
)


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def free_backend(backend: Optional[Backend]) -> None:
    if backend is None:
        return
    try:
        backend.close()
    except Exception:
        pass
    del backend
    free_cuda()


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
        y = backend.infer_batch(x)
        del y
    sync_device(device)
    free_cuda()

    latencies_ms = []
    t0 = time.perf_counter()
    for _ in range(runs):
        t_start = time.perf_counter()
        y = backend.infer_batch(x)
        sync_device(device)
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000.0)
        del y
    t1 = time.perf_counter()

    del x
    del x_np
    free_cuda()

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
    throughput_images_per_s = (runs * batch) / max(t1 - t0, 1e-12)

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


def build_arg_namespace(args: argparse.Namespace) -> argparse.Namespace:
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
        trt_plugin_so=args.trt_plugin_so,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark and evaluate one TensorRT engine in an isolated process")

    ap.add_argument("--data", required=False, help="Ultralytics dataset YAML")
    ap.add_argument("--engine", required=True, help="Path to TensorRT engine")
    ap.add_argument("--name", required=True, help="Display name")
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)

    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)

    ap.add_argument("--bench_runs", type=int, default=200)
    ap.add_argument("--bench_warmup", type=int, default=30)

    ap.add_argument("--eval_max_batches", type=int, default=0, help="0 = full val set")
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--project", type=str, default="runs/trt_engine_eval")
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--plots", action="store_true")

    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--trt_engine_cache_path", type=str, default="./trt_cache")
    ap.add_argument("--trt_workspace_size", type=int, default=2147483648)

    ap.add_argument("--out_json", required=False, help="Output JSON path for this single engine result")
    ap.add_argument("--trt_plugin_so", type=str, default=None,
                help="Optional path to TensorRT plugin .so")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    backend_args = build_arg_namespace(args)

    backend: Optional[Backend] = None
    try:
        print(f"[LOAD] {args.name}: {args.engine} on {args.device}")
        backend = build_backend(
            kind="trt_engine",
            model_path=args.engine,
            args=backend_args,
            device_override=args.device,
        )

        print("\n### Benchmarking latency / FPS")
        bench = benchmark_backend_latency(
            backend=backend,
            imgsz=args.imgsz,
            batch=args.batch,
            runs=args.bench_runs,
            warmup=args.bench_warmup,
            device=args.device,
        )
        print_dict(f"{args.name} benchmark", bench)

        free_cuda()

    #     print("\n### Evaluating detection accuracy")
    #     stats = eval_backend(
    #         backend=backend,
    #         data_yaml=args.data,
    #         imgsz=args.imgsz,
    #         batch=args.batch,
    #         conf=args.conf,
    #         iou=args.iou,
    #         max_det=args.max_det,
    #         project=args.project,
    #         name=f"{args.name}_eval",
    #         save_json=args.save_json,
    #         save_txt=args.save_txt,
    #         plots=args.plots,
    #         max_batches=args.eval_max_batches,
    #         workers=args.workers,
    #     )
    #     print_dict(f"{args.name} accuracy", stats)

    #     result = {
    #         "name": args.name,
    #         "path": args.engine,
    #         "device": args.device,
    #         "benchmark": bench,
    #         "accuracy": stats,
    #     }

    #     out_path = Path(args.out_json)
    #     out_path.parent.mkdir(parents=True, exist_ok=True)
    #     out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    #     print(f"\nSaved single-engine result to: {out_path}")

    finally:
        free_backend(backend)


if __name__ == "__main__":
    main()