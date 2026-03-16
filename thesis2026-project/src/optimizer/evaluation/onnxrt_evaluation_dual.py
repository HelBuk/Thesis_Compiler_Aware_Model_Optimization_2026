#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch


ProviderSpec = Union[str, Tuple[str, Dict[str, Any]]]


def _to_ort_provider_name(name: str) -> str:
    n = name.strip().lower()
    if n == "cpu":
        return "CPUExecutionProvider"
    if n == "cuda":
        return "CUDAExecutionProvider"
    if n in ("trt", "tensorrt"):
        return "TensorrtExecutionProvider"
    raise ValueError(f"Unsupported provider: {name}")


def _ort_dtype_to_numpy(ort_type: str) -> np.dtype:
    m = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
    }
    if ort_type not in m:
        raise ValueError(f"Unsupported ONNX input type: {ort_type}")
    return m[ort_type]


def _provider_uses_cuda(provider_name: str) -> bool:
    p = provider_name.lower()
    return p in ("cudaexecutionprovider", "tensorrtexecutionprovider")


class ORTBackend:
    def __init__(
        self,
        model_path: str,
        provider: str,
        threads: int = 1,
        use_arena: bool = True,
        trt_engine_cache_enable: bool = True,
        trt_engine_cache_path: str = "./trt_cache",
        trt_fp16_enable: bool = False,
        trt_max_workspace_size: int = 2147483648,
        trt_int8_enable: bool = False,
        trt_int8_calibration_table_name: str = ""
    ) -> None:
        self.model_path = model_path
        self.provider = _to_ort_provider_name(provider)

        so = ort.SessionOptions()
        so.enable_profiling = True
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_cpu_mem_arena = bool(use_arena)

        providers: Sequence[ProviderSpec]

        if self.provider == "TensorrtExecutionProvider":
            trt_options = {
                "trt_engine_cache_enable": bool(trt_engine_cache_enable),
                "trt_engine_cache_path": trt_engine_cache_path,
                "trt_fp16_enable": bool(trt_fp16_enable),
                "trt_max_workspace_size": int(trt_max_workspace_size),
            }

            if trt_int8_enable:
                trt_options["trt_int8_enable"] = True
                if trt_int8_calibration_table_name:
                    trt_options["trt_int8_calibration_table_name"] = trt_int8_calibration_table_name

            providers = [
                ("TensorrtExecutionProvider", trt_options),
                ("CUDAExecutionProvider", {}),
                ("CPUExecutionProvider", {}),
            ]

        elif self.provider == "CUDAExecutionProvider":
            providers = [
                ("CUDAExecutionProvider", {}),
                ("CPUExecutionProvider", {}),
            ]

        else:
            providers = [
                ("CPUExecutionProvider", {}),
            ]

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=providers,
        )

        self.active_providers = self.session.get_providers()
        self.input_meta = self.session.get_inputs()[0]
        self.output_meta = self.session.get_outputs()
        self.input_name = self.input_meta.name
        self.input_dtype = _ort_dtype_to_numpy(self.input_meta.type)

    def infer_batch(self, x: torch.Tensor) -> List[np.ndarray]:
        x_np = x.detach().cpu().numpy()
        x_np = np.ascontiguousarray(x_np.astype(self.input_dtype, copy=False))
        return self.session.run(None, {self.input_name: x_np})

    def needs_cuda_sync(self) -> bool:
        return any(_provider_uses_cuda(p) for p in self.active_providers)

    def close(self) -> None:
        del self.session


def sync_backend(backend: ORTBackend) -> None:
    if backend.needs_cuda_sync() and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_backend_latency(
    backend: ORTBackend,
    imgsz: int,
    batch: int,
    runs: int,
    warmup: int,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    x_np = rng.random((batch, 3, imgsz, imgsz), dtype=np.float32)
    x = torch.from_numpy(x_np)

    for _ in range(warmup):
        _ = backend.infer_batch(x)
    sync_backend(backend)

    latencies_ms: List[float] = []
    t0 = time.perf_counter()
    for _ in range(runs):
        t_start = time.perf_counter()
        _ = backend.infer_batch(x)
        sync_backend(backend)
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


def free_backend(backend: Optional[ORTBackend]) -> None:
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark one ONNX Runtime backend (CPU/CUDA/TensorRT EP)."
    )

    ap.add_argument("--model", required=True, help="Path to model (.onnx)")
    ap.add_argument("--name", default="model", help="Display name")

    ap.add_argument("--provider", default="cuda", choices=["cpu", "cuda", "trt", "tensorrt"])

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)

    ap.add_argument("--bench_runs", type=int, default=200)
    ap.add_argument("--bench_warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--use_arena", action="store_true")
    ap.add_argument("--no_use_arena", dest="use_arena", action="store_false")
    ap.set_defaults(use_arena=True)

    ap.add_argument("--trt_engine_cache_enable", action="store_true")
    ap.add_argument("--no_trt_engine_cache_enable", dest="trt_engine_cache_enable", action="store_false")
    ap.set_defaults(trt_engine_cache_enable=True)

    ap.add_argument("--trt_engine_cache_path", type=str, default="../models/quantized_models/onnx/trt_cache")
    ap.add_argument("--trt_fp16_enable", action="store_true")
    ap.add_argument("--trt_max_workspace_size", type=int, default=2147483648)
    ap.add_argument("--trt_int8_enable", action="store_true")
    ap.add_argument("--trt_int8_calibration_table_name", type=str, default="")

    ap.add_argument("--out_json", type=str, default="", help="Optional output JSON path")
    return ap.parse_args()


def main() -> None:
    '''
    TensorRT:
     python -m optimizer.evaluation.onnxrt_evaluation_dual  \
     --model ../models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \
     --name trt   --provider trt   --bench_runs 1000   --bench_warmup 500 \
     --batch 1   --imgsz 640

    ONNXRT:

    python -m optimizer.evaluation.onnxrt_evaluation_dual \
     --model ../models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \
     --name cuda   --provider cuda   --bench_runs 1000   --bench_warmup 500 \
     --batch 1   --imgsz 640
    '''
    args = parse_args()
    backend: Optional[ORTBackend] = None

    try:
        print(f"Loading {args.name}: {args.model} with {args.provider}")
        backend = ORTBackend(
            model_path=args.model,
            provider=args.provider,
            threads=args.threads,
            use_arena=args.use_arena,
            trt_engine_cache_enable=args.trt_engine_cache_enable,
            trt_engine_cache_path=args.trt_engine_cache_path,
            trt_fp16_enable=args.trt_fp16_enable,
            trt_max_workspace_size=args.trt_max_workspace_size,
            trt_int8_enable=args.trt_int8_enable,
            trt_int8_calibration_table_name=args.trt_int8_calibration_table_name,
        )
        print(f"active providers: {backend.active_providers}")
        print(f"input tensor: {backend.input_name} {backend.input_meta.type}")

        print("\n### Benchmarking latency / FPS")
        bench = benchmark_backend_latency(
            backend=backend,
            imgsz=args.imgsz,
            batch=args.batch,
            runs=args.bench_runs,
            warmup=args.bench_warmup,
            seed=args.seed,
        )

        print_dict(f"{args.name} benchmark", bench)

        profile_path = backend.session.end_profiling()
        print(f"ONNX Runtime profile saved to: {profile_path}")

        if args.out_json:
            results = {
                "model": {
                    "name": args.name,
                    "path": args.model,
                    "provider_requested": args.provider,
                    "providers_active": backend.active_providers,
                    "benchmark": bench,
                }
            }
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nSaved results JSON to: {out_path}")

    finally:
        free_backend(backend)


if __name__ == "__main__":
    main()
