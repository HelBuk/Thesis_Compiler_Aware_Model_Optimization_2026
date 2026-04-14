#!/usr/bin/env python3
"""
yolo_accuracy_comparison.py — Multi-backend accuracy comparison for YOLOv8n.

Runs each backend sequentially against the COCO validation split, accumulates
mAP/precision/recall metrics, then prints a table of relative deltas vs the
designated baseline backend.

Memory design
-------------
* One backend is loaded at a time; it is fully freed (close() + del + gc.collect()
  + cuda cache clear) before the next backend is loaded.
* Inside each backend run, eval_backend() deletes imgs/preds/batch_data every
  batch and calls gc.collect() + cuda_empty_cache() every 200 batches.
* The Ultralytics DetectionValidator accumulates self.stats per batch. For the
  COCO 5k val set this is ~5000 small tuples ≈ 20–40 MB total — constant-rate
  growth that is acceptable. The stats list is freed when the validator goes out
  of scope after get_stats().
* Peak RAM per backend run ≈ model_weights + one batch of images + ~40 MB stats.

Usage examples
--------------
  # YAML config (recommended)
  python yolo_accuracy_comparison.py --config my_comparison.yaml

  # Ad-hoc CLI (two backends)
  python yolo_accuracy_comparison.py \\
      --data coco.yaml \\
      --backend torch  /models/yolov8n.pt        cpu \\
      --backend ort    /models/yolov8n.onnx       cpu \\
      --backend tflite /models/yolov8n.tflite     cpu \\
      --backend tvm    /models/yolov8n_tvm.so     cuda:0 \\
      --baseline torch

YAML config format
------------------
comparison:
  data: /path/to/coco.yaml
  imgsz: 640
  batch: 1
  conf: 0.001
  iou: 0.7
  max_det: 300
  max_batches: 0          # 0 = full val set
  workers: 0
  project: runs/accuracy_comparison
  save_json: false
  save_txt: false
  plots: false
  output_json: results.json

backends:
  - name: pytorch_fp32
    kind: torch            # torch | torch_compile | ort | tensorrt | trt_engine | tflite | tvm
    model: /path/to/yolov8n.pt
    device: cpu
    baseline: true         # optional; first entry is baseline if none marked

  - name: pytorch_compile
    kind: torch_compile
    model: /path/to/yolov8n.pt
    device: cpu
    compile_backend: inductor   # optional
    compile_mode: default        # optional

  - name: onnxrt_cpu
    kind: ort
    model: /path/to/yolov8n.onnx
    device: cpu

  - name: tflite_fp32
    kind: tflite
    model: /path/to/yolov8n.tflite
    device: cpu
    threads: 4
    tflite_delegate: cpu

  - name: trt_fp32
    kind: trt_engine
    model: /path/to/yolov8n_fp32.engine
    device: cuda:0

  - name: trt_fp16
    kind: trt_engine
    model: /path/to/yolov8n_fp16.engine
    device: cuda:0

  - name: tvm_gpu
    kind: tvm
    model: /path/to/yolov8n_tvm.so
    device: cuda:0
    tvm_device_type: cuda   # cuda | cpu
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import multiprocessing as mp
import sys
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Imports from the sibling yolo_metrics module
# ---------------------------------------------------------------------------
from optimizer.evaluation.yolo_metrics import (
    Backend,
    TorchBackend,
    build_backend,
    detect_best_device,
    eval_backend,
    yolo8_output_to_preds_ultra,
)


# ---------------------------------------------------------------------------
# TVM backend (accuracy evaluation)
# ---------------------------------------------------------------------------

class TVMBackend(Backend):
    """TVM Relax VirtualMachine backend for YOLOv8n accuracy evaluation.

    Loads a compiled TVM `.so` module (produced by bench_rigorous / TVM tuning)
    and runs inference through the Relax VirtualMachine API.
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        tvm_device_type: str = "cuda",
        input_dtype: str = "float32",
    ):
        super().__init__(imgsz, conf, iou, max_det, device)

        try:
            import tvm
            import tvm.relax
        except ImportError as e:
            raise ImportError(
                "TVM is not installed. Install with `pip install apache-tvm` "
                "or build from source."
            ) from e

        self._tvm = tvm
        self._device_type = tvm_device_type.lower()
        self._np_dtype = np.float16 if input_dtype.lower() == "float16" else np.float32

        if self._device_type == "cuda":
            self._dev = tvm.cuda(0)
        elif self._device_type == "cpu":
            self._dev = tvm.cpu(0)
        else:
            raise ValueError(f"Unsupported TVM device type: {tvm_device_type!r}")

        lib = tvm.runtime.load_module(model_path)
        self._vm = tvm.relax.VirtualMachine(lib, self._dev)
        self._torch_device = "cpu"  # outputs come back to CPU via numpy

        # Keep one persistent TVM input tensor and update it via copyfrom().
        # This matches run.py's TVM calling pattern (vm["main"](x_tvm)).
        self._x_tvm = tvm.runtime.tensor(
            np.zeros((1, 3, imgsz, imgsz), dtype=self._np_dtype),
            self._dev,
        )

        # Probe once to catch dtype mismatches early.
        try:
            self._x_tvm.copyfrom(np.zeros((1, 3, imgsz, imgsz), dtype=self._np_dtype))
            _y = self._vm["main"](self._x_tvm)
            if self._device_type == "cuda":
                self._dev.sync()
            del _y
            if self._device_type == "cuda":
                self._dev.sync()
            print(f"[TVM] probe OK  model={Path(model_path).name}  "
                  f"input_dtype={self._np_dtype.__name__}")
        except ValueError as exc:
            if "dtype" in str(exc).lower():
                self._np_dtype = (
                    np.float16 if self._np_dtype == np.float32 else np.float32
                )
                self._x_tvm = tvm.runtime.tensor(
                    np.zeros((1, 3, imgsz, imgsz), dtype=self._np_dtype),
                    self._dev,
                )
                print(
                    f"[TVM] auto-corrected input dtype → {self._np_dtype.__name__} "
                    f"(annotation: {exc})"
                )
            else:
                raise

    def infer_batch(self, imgs_bchw01: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        tvm = self._tvm
        x_np = imgs_bchw01.detach().cpu().numpy().astype(self._np_dtype)

        preds = []
        for i in range(x_np.shape[0]):
            self._x_tvm.copyfrom(np.ascontiguousarray(x_np[i : i + 1]))
            y_nd = self._vm["main"](self._x_tvm)

            # Sync before reading: ensures kernels finish before .numpy().
            if self._device_type == "cuda":
                self._dev.sync()

            if isinstance(y_nd, (tuple, list)):
                y = y_nd[0].numpy()
            else:
                y = y_nd.numpy()

            del y_nd
            if self._device_type == "cuda":
                self._dev.sync()

            preds.append(
                yolo8_output_to_preds_ultra(
                    out=y,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    torch_device="cpu",
                )
            )

        return preds

    def close(self) -> None:
        try:
            if self._device_type == "cuda":
                self._dev.sync()
            del self._vm
            del self._x_tvm
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TVM Relay backend (graph_executor, produced by compile_tvm_relay_int8.py)
# ---------------------------------------------------------------------------

class TVMRelayBackend(Backend):
    """TVM Relay GraphExecutor backend for YOLOv8n accuracy evaluation.

    Loads a compiled Relay .so (produced by compile_tvm_relay_int8.py) and
    runs inference through tvm.contrib.graph_executor.  This is the correct
    loader for relay.build() output; do NOT use relax.VirtualMachine here.
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        tvm_device_type: str = "cuda",
        input_name: str = "images",
    ):
        super().__init__(imgsz, conf, iou, max_det, device)

        try:
            import tvm
            from tvm.contrib import graph_executor
        except ImportError as exc:
            raise ImportError("TVM is not installed.") from exc

        self._tvm = tvm
        self._device_type = tvm_device_type.lower()
        self._input_name = input_name

        if self._device_type == "cuda":
            self._dev = tvm.cuda(0)
        elif self._device_type == "cpu":
            self._dev = tvm.cpu(0)
        else:
            raise ValueError(f"Unsupported TVM device type: {tvm_device_type!r}")

        lib = tvm.runtime.load_module(model_path)
        self._m = graph_executor.GraphModule(lib["default"](self._dev))
        self._torch_device = "cpu"

        # Probe once to verify the module loads correctly
        probe = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        self._m.set_input(self._input_name, probe)
        self._m.run()
        if self._device_type == "cuda":
            self._dev.sync()
        print(f"[TVMRelay] probe OK  model={Path(model_path).name}")

    def infer_batch(self, imgs_bchw01: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        x_np = imgs_bchw01.detach().cpu().numpy().astype(np.float32)
        preds = []
        for i in range(x_np.shape[0]):
            self._m.set_input(self._input_name, np.ascontiguousarray(x_np[i : i + 1]))
            self._m.run()
            if self._device_type == "cuda":
                self._dev.sync()
            y = self._m.get_output(0).numpy()
            preds.append(
                yolo8_output_to_preds_ultra(
                    out=y,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    torch_device="cpu",
                )
            )
        return preds

    def close(self) -> None:
        try:
            if self._device_type == "cuda":
                self._dev.sync()
            del self._m
        except Exception:
            pass


# ---------------------------------------------------------------------------
# torch.compile backend
# ---------------------------------------------------------------------------

class TorchCompileBackend(TorchBackend):
    """PyTorch eager + torch.compile() backend.

    Wraps TorchBackend and applies torch.compile() after model load.
    The first forward pass triggers compilation; subsequent calls are fast.
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        precision: str = "fp32",
        compile_backend: str = "inductor",
        compile_mode: str = "default",
    ):
        super().__init__(model_path, imgsz, conf, iou, max_det, device, precision=precision)
        self._compile_backend = compile_backend
        self._compile_mode = compile_mode

        print(
            f"[torch.compile] backend={compile_backend!r} mode={compile_mode!r} "
            f"device={self.torch_device!r}"
        )
        try:
            self.model = torch.compile(
                self.model,
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=False,
            )
        except Exception as e:
            print(f"[torch.compile] WARNING: compile failed ({e}), falling back to eager.")


# ---------------------------------------------------------------------------
# BackendSpec — declarative description of one backend entry
# ---------------------------------------------------------------------------

@dataclass
class BackendSpec:
    name: str
    kind: str                       # torch | torch_compile | ort | tensorrt | trt_engine | tflite | tvm
    model: str
    device: str = ""
    baseline: bool = False
    precision: str = "fp32"          # fp32 | fp16 (for torch/torch_compile)

    # torch.compile options
    compile_backend: str = "inductor"
    compile_mode: str = "default"

    # TFLite options
    threads: int = 4
    tflite_delegate: str = "cpu"

    # ORT / TensorRT-via-ORT options
    trt_fp16: bool = False
    trt_int8: bool = False
    trt_engine_cache: bool = False
    trt_engine_cache_path: str = "./trt_cache"
    trt_workspace_size: int = 2_147_483_648

    # TRT native engine options
    trt_plugin_so: Optional[str] = None

    # TVM options
    tvm_device_type: str = "cuda"
    tvm_input_dtype: str = "float32"   # float32 | float16

    # Extra raw kwargs for forward-compatibility
    extra: Dict = field(default_factory=dict)


def spec_from_dict(d: Dict) -> BackendSpec:
    known = {f.name for f in BackendSpec.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in d.items() if k in known}
    extra = {k: v for k, v in d.items() if k not in known}
    return BackendSpec(**kwargs, extra=extra)


# ---------------------------------------------------------------------------
# Build a backend from a BackendSpec
# ---------------------------------------------------------------------------

def _make_simple_args(spec: BackendSpec):
    """Return a simple namespace compatible with yolo_metrics.build_backend()."""
    import types
    ns = types.SimpleNamespace()
    ns.imgsz = 640          # overridden by caller
    ns.conf = 0.001         # overridden by caller
    ns.iou = 0.7            # overridden by caller
    ns.max_det = 300        # overridden by caller
    ns.threads = spec.threads
    ns.tflite_delegate = spec.tflite_delegate
    ns.trt_fp16 = spec.trt_fp16
    ns.trt_int8 = spec.trt_int8
    ns.trt_engine_cache = spec.trt_engine_cache
    ns.trt_engine_cache_path = spec.trt_engine_cache_path
    ns.trt_workspace_size = spec.trt_workspace_size
    ns.trt_plugin_so = spec.trt_plugin_so
    ns.precision = spec.precision
    return ns


def build_backend_from_spec(
    spec: BackendSpec,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> Backend:
    device = spec.device or detect_best_device()
    kind = spec.kind.lower()

    if kind == "torch_compile":
        return TorchCompileBackend(
            model_path=spec.model,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            precision=spec.precision,
            compile_backend=spec.compile_backend,
            compile_mode=spec.compile_mode,
        )

    if kind == "tvm_relay_int8":
        return TVMRelayBackend(
            model_path=spec.model,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            tvm_device_type=spec.tvm_device_type,
            input_name=spec.extra.get("input_name", "images"),
        )

    if kind in ("tvm", "tvm_int8"):
        # tvm_int8 uses the same TVMBackend: a QDQ-compiled .so accepts FLOAT32
        # input (the QuantizeLinear is the first op inside the graph boundary).
        return TVMBackend(
            model_path=spec.model,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            tvm_device_type=spec.tvm_device_type,
            input_dtype=spec.tvm_input_dtype,
        )

    # Delegate all other kinds to the existing factory
    args = _make_simple_args(spec)
    args.imgsz = imgsz
    args.conf = conf
    args.iou = iou
    args.max_det = max_det
    return build_backend(kind, spec.model, args, device_override=device)


# ---------------------------------------------------------------------------
# Memory-safe backend teardown
# ---------------------------------------------------------------------------

def free_backend(backend: Optional[Backend]) -> None:
    if backend is None:
        return
    try:
        backend.close()
    except Exception:
        pass
    if isinstance(backend, TorchBackend):
        try:
            backend.model.to("cpu")
        except Exception:
            pass
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Subprocess-isolated eval for backends that share a CUDA process unsafely
# (primarily TVM: its C++ abort() on CUDA errors would kill the whole script)
# ---------------------------------------------------------------------------

# Must be a top-level function so multiprocessing 'spawn' can pickle it.
def _isolated_eval_worker(
    spec_dict: dict,
    eval_kwargs: dict,
    queue: "mp.Queue[tuple]",
) -> None:
    """Runs inside a fresh spawned child process with a clean CUDA context."""
    import traceback as _tb
    try:
        # Fresh import in the child — no prior CUDA allocator state.
        from optimizer.evaluation.yolo_accuracy_comparison import (  # noqa: PLC0415
            BackendSpec,
            build_backend_from_spec,
            free_backend,
        )
        from optimizer.evaluation.yolo_metrics import eval_backend  # noqa: PLC0415

        spec = BackendSpec(**spec_dict)
        backend = build_backend_from_spec(
            spec,
            imgsz=eval_kwargs["imgsz"],
            conf=eval_kwargs["conf"],
            iou=eval_kwargs["iou"],
            max_det=eval_kwargs["max_det"],
        )
        stats = eval_backend(backend=backend, **eval_kwargs)
        free_backend(backend)
        queue.put(("ok", stats))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{exc}\n{_tb.format_exc()}"))


def _run_eval_isolated(
    spec: BackendSpec,
    eval_kwargs: dict,
    timeout_s: int = 7200,
) -> Dict[str, float]:
    """Launch _isolated_eval_worker in a fresh 'spawn' process.

    'spawn' guarantees the child has no inherited CUDA context from the parent,
    so TVM / pycuda cannot conflict with PyTorch's or ORT's allocators.
    If the child calls abort() (as TVM does on CUDA errors) only the child
    dies; the parent catches a non-zero exit code and raises RuntimeError.
    """
    ctx = mp.get_context("spawn")
    queue: "mp.Queue[tuple]" = ctx.Queue()

    # dataclasses.asdict() deep-copies; all BackendSpec fields are primitives.
    spec_dict = dataclasses.asdict(spec)

    proc = ctx.Process(
        target=_isolated_eval_worker,
        args=(spec_dict, eval_kwargs, queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.kill()
        proc.join()
        raise RuntimeError(
            f"Isolated eval subprocess timed out after {timeout_s}s"
        )

    if proc.exitcode != 0:
        raise RuntimeError(
            f"Isolated eval subprocess crashed (exitcode={proc.exitcode}). "
            "Hint: rerun with CUDA_LAUNCH_BLOCKING=1 for a pinpointed traceback."
        )

    try:
        kind, data = queue.get(timeout=30)
    except Exception:
        raise RuntimeError("Isolated eval subprocess exited cleanly but returned no result.")

    if kind == "error":
        raise RuntimeError(f"Isolated eval failed:\n{data}")

    return data  # type: ignore[return-value]


# Kinds that MUST run in a subprocess due to unrecoverable CUDA context conflicts
_ISOLATED_KINDS = frozenset({"tvm", "tvm_int8", "tvm_relay_int8"})


# ---------------------------------------------------------------------------
# Core: run one backend and return accuracy stats
# ---------------------------------------------------------------------------

def run_backend_eval(
    spec: BackendSpec,
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    max_det: int,
    project: str,
    max_batches: int,
    workers: int,
    save_json: bool,
    save_txt: bool,
    plots: bool,
) -> Dict[str, float]:
    """Load one backend, evaluate it on COCO val, free all resources, return stats."""
    print(
        f"\n{'=' * 70}\n"
        f"  Evaluating: {spec.name}  (kind={spec.kind}  device={spec.device or 'auto'})\n"
        f"  Model: {spec.model}\n"
        f"{'=' * 70}"
    )

    # TVM's C++ runtime calls abort() on CUDA errors, killing the whole process.
    # Run it in a subprocess so only the child dies if TVM crashes.
    if spec.kind.lower() in _ISOLATED_KINDS:
        print(f"  [subprocess-isolated — clean CUDA context per TVM requirement]")
        eval_kwargs = dict(
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            max_det=max_det,
            project=project,
            name=spec.name,
            save_json=save_json,
            save_txt=save_txt,
            plots=plots,
            max_batches=max_batches,
            workers=workers,
        )
        return _run_eval_isolated(spec, eval_kwargs)

    backend: Optional[Backend] = None
    try:
        backend = build_backend_from_spec(spec, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det)

        stats = eval_backend(
            backend=backend,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            max_det=max_det,
            project=project,
            name=spec.name,
            save_json=save_json,
            save_txt=save_txt,
            plots=plots,
            max_batches=max_batches,
            workers=workers,
        )
    finally:
        free_backend(backend)
        backend = None

    return stats


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

_DISPLAY_METRICS: List[Tuple[str, str]] = [
    ("metrics/mAP50(B)",    "mAP50"),
    ("metrics/mAP50-95(B)", "mAP50-95"),
    ("metrics/precision(B)","Prec"),
    ("metrics/recall(B)",   "Recall"),
]


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _fmt_delta(d: float, ref: float) -> Tuple[str, str]:
    sign = "+" if d >= 0 else ""
    pct = (d / ref * 100.0) if abs(ref) > 1e-9 else float("nan")
    pct_str = f"{sign}{pct:.2f}%" if not np.isnan(pct) else "  n/a"
    return f"{sign}{d:.4f}", pct_str


def print_comparison_table(
    baseline_name: str,
    all_stats: Dict[str, Dict[str, float]],
) -> None:
    # Split into successful (non-empty stats) and failed backends.
    ok_stats  = {n: s for n, s in all_stats.items() if s}
    failed    = [n for n, s in all_stats.items() if not s]

    if not ok_stats:
        print("[WARNING] All backends failed — no results to display.")
        return

    # Metrics present in every successful backend (baseline must be among them).
    avail_metrics = []
    for disp_key, label in _DISPLAY_METRICS:
        if all(disp_key in s for s in ok_stats.values()):
            avail_metrics.append((disp_key, label))

    if not avail_metrics:
        print("[WARNING] No common display metrics found in successful backends.")
        for name, stats in ok_stats.items():
            print(f"  {name}: {stats}")
        if failed:
            print(f"  FAILED (no stats): {', '.join(failed)}")
        return

    # If baseline itself failed, pick the first successful backend as reference.
    if baseline_name not in ok_stats:
        baseline_name = next(iter(ok_stats))
        print(f"[WARNING] Original baseline failed; using {baseline_name!r} as reference.")

    baseline_stats = ok_stats[baseline_name]

    # Column widths — account for all names including failed ones.
    all_names = list(ok_stats.keys()) + [f"{n} (FAILED)" for n in failed]
    name_w = max(len(n) for n in all_names) + 2
    col_w = 13

    header_parts = [f"{'Backend':<{name_w}}"]
    for _, label in avail_metrics:
        header_parts.append(f"{label:>{col_w}}")
        header_parts.append(f"{'Δabs':>{col_w}}")
        header_parts.append(f"{'Δ%':>{col_w}}")
    header = "  ".join(header_parts)
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(f"  ACCURACY COMPARISON  —  baseline: {baseline_name}")
    print("=" * len(header))
    print(header)
    print(sep)

    for name, stats in ok_stats.items():
        tag = " (REF)" if name == baseline_name else ""
        row_parts = [f"{name + tag:<{name_w}}"]
        for disp_key, _ in avail_metrics:
            val = float(stats.get(disp_key, float("nan")))
            ref_val = float(baseline_stats.get(disp_key, float("nan")))
            row_parts.append(f"{_fmt(val):>{col_w}}")
            if name == baseline_name:
                row_parts.append(f"{'—':>{col_w}}")
                row_parts.append(f"{'—':>{col_w}}")
            else:
                d_str, p_str = _fmt_delta(val - ref_val, ref_val)
                row_parts.append(f"{d_str:>{col_w}}")
                row_parts.append(f"{p_str:>{col_w}}")
        print("  ".join(row_parts))

    for name in failed:
        row_parts = [f"{name + ' (FAILED)':<{name_w}}"]
        for _ in avail_metrics:
            row_parts += [f"{'n/a':>{col_w}}"] * 3
        print("  ".join(row_parts))

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class ComparisonConfig:
    data: str
    backends: List[BackendSpec]
    baseline: str = ""
    imgsz: int = 640
    batch: int = 1
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    max_batches: int = 0
    workers: int = 0
    project: str = "runs/accuracy_comparison"
    save_json: bool = False
    save_txt: bool = False
    plots: bool = False
    output_json: str = ""


def load_yaml_config(path: str) -> ComparisonConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    cmp = raw.get("comparison", {})
    specs_raw = raw.get("backends", [])

    specs: List[BackendSpec] = []
    baseline_name = ""
    for d in specs_raw:
        spec = spec_from_dict(d)
        specs.append(spec)
        if spec.baseline and not baseline_name:
            baseline_name = spec.name

    if not baseline_name and specs:
        baseline_name = specs[0].name

    return ComparisonConfig(
        data=cmp["data"],
        backends=specs,
        baseline=baseline_name,
        imgsz=cmp.get("imgsz", 640),
        batch=cmp.get("batch", 1),
        conf=cmp.get("conf", 0.001),
        iou=cmp.get("iou", 0.7),
        max_det=cmp.get("max_det", 300),
        max_batches=cmp.get("max_batches", 0),
        workers=cmp.get("workers", 0),
        project=cmp.get("project", "runs/accuracy_comparison"),
        save_json=cmp.get("save_json", False),
        save_txt=cmp.get("save_txt", False),
        plots=cmp.get("plots", False),
        output_json=cmp.get("output_json", ""),
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Multi-backend YOLOv8n accuracy comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    ap.add_argument("--config", metavar="YAML",
                    help="Path to YAML comparison config (recommended)")
    ap.add_argument("--backends", nargs="+", metavar="NAME", default=None,
                    help="Run only these backend names (by 'name:' field in YAML). "
                         "Useful for re-running a single backend without editing the YAML.")

    # Dataset / eval settings (CLI override or standalone)
    ap.add_argument("--data", help="Path to Ultralytics dataset YAML (e.g. coco.yaml)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--max_batches", type=int, default=0,
                    help="Limit to N batches per backend (0 = full val)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--project", default="runs/accuracy_comparison")
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--output_json", default="",
                    help="Save all stats to this JSON file")

    # Ad-hoc backends: --backend KIND MODEL_PATH DEVICE
    ap.add_argument("--backend", nargs=3, action="append",
                    metavar=("KIND", "MODEL", "DEVICE"),
                    help="Add a backend (repeatable). KIND: torch|torch_compile|ort|"
                         "tensorrt|trt_engine|tflite|tvm. "
                         "Example: --backend torch /models/yolov8n.pt cpu")
    ap.add_argument("--baseline", default="",
                    help="Name (KIND) of the baseline backend (default: first)")

    return ap.parse_args()


def build_config_from_args(args: argparse.Namespace) -> ComparisonConfig:
    if not args.data:
        raise SystemExit("--data is required when not using --config")
    if not args.backend:
        raise SystemExit("Specify at least one --backend KIND MODEL DEVICE")

    specs: List[BackendSpec] = []
    for kind, model, device in args.backend:
        specs.append(BackendSpec(name=kind, kind=kind, model=model, device=device))

    baseline = args.baseline or specs[0].name

    return ComparisonConfig(
        data=args.data,
        backends=specs,
        baseline=baseline,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        max_batches=args.max_batches,
        workers=args.workers,
        project=args.project,
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
        output_json=args.output_json,
    )


# ---------------------------------------------------------------------------
# Incremental JSON save helpers
# ---------------------------------------------------------------------------

def _now_ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _timestamped_output_path(base_path: Path, ts_token: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{ts_token}{base_path.suffix}")


def _build_payload(
    baseline: str,
    all_stats: Dict[str, Dict[str, float]],
    backend_status: Dict[str, Dict[str, str]],
    run_started_at: str,
    status: str,
) -> Dict:
    payload: Dict = {
        "run_started_at": run_started_at,
        "updated_at": _now_ts(),
        "status": status,  # running | completed
        "baseline": baseline,
        "backends": {name: stats for name, stats in all_stats.items()},
        "backend_status": backend_status,
        "relative": {},
    }

    baseline_s = all_stats.get(baseline, {})
    for name, stats in all_stats.items():
        rel: Dict[str, Dict[str, float]] = {}
        for k, val in stats.items():
            ref_val = baseline_s.get(k)
            if ref_val is not None:
                try:
                    delta = float(val) - float(ref_val)
                    pct = delta / float(ref_val) * 100.0 if abs(float(ref_val)) > 1e-9 else float("nan")
                    rel[k] = {"val": float(val), "delta": delta, "pct_delta": pct}
                except Exception:
                    pass
        payload["relative"][name] = rel
    return payload


def _save_results_json(
    out_path: Path,
    baseline: str,
    all_stats: Dict[str, Dict[str, float]],
    backend_status: Dict[str, Dict[str, str]],
    run_started_at: str,
    status: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(
        baseline=baseline,
        all_stats=all_stats,
        backend_status=backend_status,
        run_started_at=run_started_at,
        status=status,
    )
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_yaml_config(args.config)
        # CLI flags override YAML where provided
        if args.data:
            cfg.data = args.data
        if args.output_json:
            cfg.output_json = args.output_json
        if args.max_batches:
            cfg.max_batches = args.max_batches
    else:
        cfg = build_config_from_args(args)

    if not cfg.backends:
        raise SystemExit("No backends defined. Check your config or --backend flags.")

    # --backends filter: run only the named subset
    if getattr(args, "backends", None):
        allowed = set(args.backends)
        unknown = allowed - {s.name for s in cfg.backends}
        if unknown:
            raise SystemExit(f"--backends: unknown backend name(s): {sorted(unknown)}\n"
                             f"Available: {[s.name for s in cfg.backends]}")
        cfg.backends = [s for s in cfg.backends if s.name in allowed]
        print(f"[INFO] --backends filter active: running {[s.name for s in cfg.backends]}")

    if cfg.baseline not in {s.name for s in cfg.backends}:
        cfg.baseline = cfg.backends[0].name
        print(f"[INFO] baseline set to first backend: {cfg.baseline!r}")

    print(f"\n[INFO] Dataset     : {cfg.data}")
    print(f"[INFO] imgsz       : {cfg.imgsz}")
    print(f"[INFO] batch       : {cfg.batch}")
    print(f"[INFO] conf/iou    : {cfg.conf} / {cfg.iou}")
    print(f"[INFO] max_batches : {cfg.max_batches or 'all'}")
    print(f"[INFO] Backends    : {[s.name for s in cfg.backends]}")
    print(f"[INFO] Baseline    : {cfg.baseline}")

    all_stats: Dict[str, Dict[str, float]] = {}
    backend_status: Dict[str, Dict[str, str]] = {}
    run_started_at = _now_ts()
    run_file_ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_path = _timestamped_output_path(Path(cfg.output_json), run_file_ts) if cfg.output_json else None
    if out_path is not None:
        print(f"[INFO] Output JSON  : {out_path}")

    for spec in cfg.backends:
        try:
            stats = run_backend_eval(
                spec=spec,
                data_yaml=cfg.data,
                imgsz=cfg.imgsz,
                batch=cfg.batch,
                conf=cfg.conf,
                iou=cfg.iou,
                max_det=cfg.max_det,
                project=cfg.project,
                max_batches=cfg.max_batches,
                workers=cfg.workers,
                save_json=cfg.save_json,
                save_txt=cfg.save_txt,
                plots=cfg.plots,
            )
            all_stats[spec.name] = stats
            backend_status[spec.name] = {
                "state": "ok",
                "completed_at": _now_ts(),
            }
            print(f"[OK] {spec.name}: mAP50={stats.get('metrics/mAP50(B)', 'n/a'):.4f}  "
                  f"mAP50-95={stats.get('metrics/mAP50-95(B)', 'n/a'):.4f}")
        except Exception as exc:
            print(f"[ERROR] {spec.name} failed: {exc}", file=sys.stderr)
            all_stats[spec.name] = {}
            backend_status[spec.name] = {
                "state": "failed",
                "completed_at": _now_ts(),
                "error": str(exc),
            }

        if out_path is not None:
            _save_results_json(
                out_path=out_path,
                baseline=cfg.baseline,
                all_stats=all_stats,
                backend_status=backend_status,
                run_started_at=run_started_at,
                status="running",
            )
            print(f"[INFO] Checkpoint saved to {out_path} after backend={spec.name}")

    # Print relative table
    if cfg.baseline in all_stats and all_stats[cfg.baseline]:
        # Re-order so baseline is first
        ordered: Dict[str, Dict[str, float]] = {}
        ordered[cfg.baseline] = all_stats[cfg.baseline]
        for name, stats in all_stats.items():
            if name != cfg.baseline:
                ordered[name] = stats
        print_comparison_table(cfg.baseline, ordered)
    else:
        print("[WARNING] Baseline stats missing; cannot print relative table.")
        for name, stats in all_stats.items():
            print(f"  {name}: {stats}")

    if out_path is not None:
        _save_results_json(
            out_path=out_path,
            baseline=cfg.baseline,
            all_stats=all_stats,
            backend_status=backend_status,
            run_started_at=run_started_at,
            status="completed",
        )
        print(f"[INFO] Results saved to {out_path}")


if __name__ == "__main__":
    main()
