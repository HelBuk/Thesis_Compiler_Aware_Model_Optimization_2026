#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
from onnx import TensorProto
import onnxruntime as ort
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.relax_integration import extract_tasks, tune_relax

try:
    import yaml  # pip install pyyaml
except Exception as e:
    raise RuntimeError("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e

# ----------------------------
# Logging (human log + JSONL)
# ----------------------------
class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.log_path = run_dir / "run.log"
        self.jsonl_path = run_dir / "run.jsonl"

    def _ts(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def info(self, msg: str, **fields: Any) -> None:
        line = f"{self._ts()} [INFO] {msg}"
        print(line, flush=True)
        self.log_path.open("a", encoding="utf-8").write(line + "\n")
        rec = {"ts": self._ts(), "level": "INFO", "msg": msg, **fields}
        self.jsonl_path.open("a", encoding="utf-8").write(json.dumps(rec) + "\n")

    def warn(self, msg: str, **fields: Any) -> None:
        line = f"{self._ts()} [WARN] {msg}"
        print(line, file=sys.stderr, flush=True)
        self.log_path.open("a", encoding="utf-8").write(line + "\n")
        rec = {"ts": self._ts(), "level": "WARN", "msg": msg, **fields}
        self.jsonl_path.open("a", encoding="utf-8").write(json.dumps(rec) + "\n")

# ----------------------------
# dtype helpers
# ----------------------------
def precision_to_np(prec: str) -> np.dtype:
    prec = prec.lower()
    if prec in ("fp32", "float32"):
        return np.float32
    if prec in ("fp16", "float16"):
        return np.float16
    if prec in ("int8",):
        return np.int8
    raise ValueError(f"Unsupported precision: {prec}")


def precision_to_torch(prec: str) -> torch.dtype:
    prec = prec.lower()
    if prec in ("fp32", "float32"):
        return torch.float32
    if prec in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unsupported precision: {prec}")

# ----------------------------
# MetaSchedule DB sanitize 
# ----------------------------
def _strip_feature_keys(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("feature."):
                continue
            out[k] = _strip_feature_keys(v)
        return out
    if isinstance(obj, list):
        return [_strip_feature_keys(x) for x in obj]
    return obj


def _sanitize_json_file(path: Path):
    if not path.exists():
        return
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return

    dec = json.JSONDecoder()

    # 1) Try normal JSON
    try:
        data = dec.decode(txt)
        data = _strip_feature_keys(data)
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        return
    except json.JSONDecodeError:
        pass

    # 2) Concatenated JSON objects / JSONL
    objs = []
    i, n = 0, len(txt)
    while i < n:
        while i < n and txt[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(txt, i)
        objs.append(_strip_feature_keys(obj))
        i = j

    path.write_text("".join(json.dumps(o, separators=(",", ":")) + "\n" for o in objs), encoding="utf-8")


def sanitize_ms_db(work_dir: str):
    wd = Path(work_dir)
    _sanitize_json_file(wd / "database_workload.json")
    _sanitize_json_file(wd / "database_tuning_record.json")

# ----------------------------
# Config
# ----------------------------

def load_config(path: str) -> Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")
    for k in ["run", "model", "device", "precision", "tvm", "benchmark"]:
        if k not in raw:
            raise ValueError(f"Missing top-level key: {k}")
    return Config(raw=raw)

@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    @property
    def run_name(self) -> str:
        return self.raw["run"]["name"]
    
    @property
    def out_dir(self) -> Path:
        return Path(self.raw["run"]["out_dir"])

    @property
    def seed(self) -> int:
        return int(self.raw["run"].get("seed", 0))

    @property
    def onnx_path(self) -> str:
        return self.raw["model"]["onnx_path"]

    @property
    def pt_path(self) -> str:
        return self.raw["model"]["pt_path"]

    @property
    def imgsz(self) -> int:
        return int(self.raw["model"].get("imgsz", 640))

    @property
    def batch(self) -> int:
        return int(self.raw["model"].get("batch", 1))

    @property
    def device_kind(self) -> str:
        return self.raw["device"]["kind"]  # "cuda" or "cpu"

    @property
    def def_target(self) -> Dict[str, Any]:
        device = self.raw.get("device") or {}
        host = device.get("host") or {}
        kind = device.get("kind")
        if not kind:
            raise ValueError("device.kind is required")

        target: Dict[str, Any] = {"kind": kind}

        if kind == "cuda":
            for key in (
                "arch",
                "max_num_threads",
                "max_threads_per_block",
                "thread_warp_size",
                "max_shared_memory_per_block",
                "registers_per_block",
            ):
                value = device.get(key)
                if value is not None:
                    target[key] = value

            host_target: Dict[str, Any] = {}
            for key in ("kind", "mtriple", "mcpu", "num-cores"):
                value = host.get(key)
                if value is not None:
                    host_target[key] = value

            if host_target:
                target["host"] = host_target
        
        elif kind == "cpu":
            for key in ("kind", "mtriple", "mcpu", "num-cores"):
                value = host.get(key)
                if value is not None:
                    target[key] = value
        else:
            raise ValueError(f"device.kind: {kind} is not suppported")

        return target

    @property
    def precision(self) -> str:
        return self.raw["precision"]["name"]  # fp32/fp16/bf16/int8

    @property
    def tvm_log_dir(self) -> str:
        return self.raw["tvm"]["log_dir"]

    @property
    def tvm_export_path(self) -> str:
        return self.raw["tvm"]["export_path"]

    @property
    def use_existing_db(self) -> bool:
        return bool(self.raw["tvm"].get("use_existing_db", True))

    @property
    def max_trials_global(self) -> int:
        return int(self.raw["tvm"].get("max_trials_global", 3000))

    @property
    def max_trials_per_task(self) -> int:
        return int(self.raw["tvm"].get("max_trials_per_task", 100))

    @property
    def num_trials_per_iter(self) -> int:
        return int(self.raw["tvm"].get("num_trials_per_iter", 64))

    @property
    def timeout_builder_s(self) -> int:
        return int(self.raw["tvm"].get("timeout_builder_s", 300))

    @property
    def timeout_runner_s(self) -> int:
        return int(self.raw["tvm"].get("timeout_runner_s", 30))
    
    @property
    def tvm_targeted_tuning(self) -> bool:
        return bool(self.raw.get("tvm_targeted_tuning", {}).get("enable", False))

    @property
    def tvm_targeted_tuning_ops(self) -> List[str]:
        cfg = self.raw.get("tvm_targeted_tuning", {}) or {}
        if not bool(cfg.get("enable", False)):
            return []
        ops = cfg.get("operations") or []
        if isinstance(ops, str):
            return [ops]
        if not isinstance(ops, list):
            raise ValueError("tvm_targeted_tuning.operations must be a string or list of strings")
        return [o for o in ops if isinstance(o, str) and o.strip()]

    @property
    def bench_runs(self) -> int:
        return int(self.raw["benchmark"].get("runs", 200))

    @property
    def bench_warmup(self) -> int:
        return int(self.raw["benchmark"].get("warmup", 20))

    @property
    def ort_provider(self) -> str:
        return self.raw.get("onnxruntime", {}).get("provider", "CPUExecutionProvider")

    @property
    def ort_threads(self) -> int:
        return int(self.raw.get("onnxruntime", {}).get("threads", 4))

    @property
    def ort_trt_fp16_enable(self) -> bool:
        return bool(self.raw.get("onnxruntime", {}).get("trt", {}).get("fp16_enable", False))

    @property
    def ort_trt_cache_path(self) -> str:
        return self.raw.get("onnxruntime", {}).get("trt", {}).get("engine_cache_path", "./trt_cache/trt_current")

    @property
    def ort_trt_cache_enable(self) -> bool:
        return bool(self.raw.get("onnxruntime", {}).get("trt", {}).get("enable_engine_cache", True))

    @property
    def ort_use_arena(self) -> bool:
        val = self.raw.get("onnxruntime", {}).get("use_arena")
        return True if val is None else bool(val)

def make_run_dir(cfg: Config) -> Path:
    cfg_bytes = json.dumps(cfg.raw, sort_keys=True).encode("utf-8")
    h = hashlib.sha1(cfg_bytes).hexdigest()[:10]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.out_dir / f"{cfg.run_name}__{stamp}__{h}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "run.yaml").write_text(yaml.safe_dump(cfg.raw, sort_keys=False), encoding="utf-8")
    return run_dir

# ----------------------------
# TVM pipeline: load -> preprocess -> tasks -> tune -> compile -> export
# ----------------------------
def make_tvm_target(cfg: Config, log: RunLogger) -> tvm.target.Target:
    if cfg.device_kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        log.info("Using CUDA target", arch=f"sm_{cc_major}{cc_minor}")
        target = cfg.def_target
        if target.get("arch") is None: 
            target["arch"] = f"sm_{cc_major}{cc_minor}"
        return tvm.target.Target(target)
    elif cfg.device_kind == "cpu":
        log.info("Using LLVM CPU target")
        target_dict = cfg.def_target
        return tvm.target.Target(target_dict)
    else:
        raise ValueError(f"Unknown device.kind: {cfg.device_kind}")

def get_tvm_device(cfg: Config):
    if cfg.device_kind == "cuda":
        return tvm.cuda(0)
    if cfg.device_kind == "cpu":
        return tvm.cpu(0)
    else:
        raise ValueError(f"Unknown device.kind: {cfg.device_kind}")

def load_relax_module_from_onnx(cfg: Config, log: RunLogger) -> relax.IRModule:
    log.info("Loading ONNX", path=cfg.onnx_path)
    onnx_model = onnx.load(cfg.onnx_path)
    mod = from_onnx(onnx_model, keep_params_in_input=False)
    return mod

def preprocess_for_inference(mod: relax.IRModule, cfg: Config, target: tvm.target.Target) -> relax.IRModule:
    with target:
        mod_tune = tvm.transform.Sequential(
            [
                relax.transform.DecomposeOpsForInference(),
                relax.transform.CanonicalizeBindings(),
                relax.get_pipeline("zero")
            ]
        )(mod)
    return mod_tune

def extract_relax_tasks(mod_tune: relax.IRModule, target: tvm.target.Target) -> List[Any]:
    with target:
        tasks = extract_tasks(mod_tune, target=target, params={})
    return tasks

def tune_all(cfg: Config, log: RunLogger, mod_tune: relax.IRModule, tasks: List[Any], target: tvm.target.Target) -> None:
    if not tasks:
        log.warn("No tasks extracted; skipping tuning")
        return

    Path(cfg.tvm_log_dir).mkdir(parents=True, exist_ok=True)
    try:
        log.info("Starting tuning", tasks=len(tasks), log_dir=cfg.tvm_log_dir)
        tune_relax(
            mod=mod_tune,
            params={},
            target=target,
            work_dir=cfg.tvm_log_dir,
            max_trials_global=cfg.max_trials_global,
            max_trials_per_task=cfg.max_trials_per_task,
            num_trials_per_iter=cfg.num_trials_per_iter,
            builder=ms.builder.LocalBuilder(timeout_sec=cfg.timeout_builder_s),
            runner=ms.runner.LocalRunner(timeout_sec=cfg.timeout_runner_s),
        )
    except RuntimeError as e:
        if "feature.has_asimd" not in str(e):
            raise
        log.warn("MetaSchedule DB parse issue; sanitizing DB and continuing")
        sanitize_ms_db(cfg.tvm_log_dir)

    sanitize_ms_db(cfg.tvm_log_dir)

def tune_selected(cfg: Config, log: RunLogger, mod_tune: relax.IRModule, tasks: List[Any], target: tvm.target.Target) -> None:
    op_names = cfg.tvm_targeted_tuning_ops
    if not op_names:
        raise ValueError("No op_names provided for tune_chosen_operations (tvm_targeted_tuning.operations).")
    available = {t.task_name for t in tasks}
    chosen = [n for n in op_names if n in available]
    missing = [n for n in op_names if n not in available]
    if missing:
        log.warn("Some requested ops were not found in extracted tasks", missing=missing)

    if not chosen:
        raise RuntimeError("None of the requested ops matched extracted tasks.")

    Path(cfg.tvm_log_dir).mkdir(parents=True, exist_ok=True)

    try:
        log.info("Starting targeted tuning", chosen=len(chosen))
        tune_relax(
            mod=mod_tune,
            params={},
            target=target,
            work_dir=cfg.tvm_log_dir,
            op_names=chosen,
            max_trials_global=cfg.max_trials_global,
            max_trials_per_task=cfg.max_trials_per_task,
            num_trials_per_iter=cfg.num_trials_per_iter,
            builder=ms.builder.LocalBuilder(timeout_sec=cfg.timeout_builder_s),
            runner=ms.runner.LocalRunner(timeout_sec=cfg.timeout_runner_s),
        )
    except RuntimeError as e:
        if "feature.has_asimd" not in str(e):
            raise
        log.warn("MetaSchedule DB parse issue; sanitizing DB and continuing")
        sanitize_ms_db(cfg.tvm_log_dir)

    sanitize_ms_db(cfg.tvm_log_dir)

def compile_with_db(cfg: Config, log: RunLogger, mod_tune: relax.IRModule, target: tvm.target.Target) -> tvm.runtime.Module:
    log.info("Compiling with MetaSchedule DB", db_dir=cfg.tvm_log_dir)
    sanitize_ms_db(cfg.tvm_log_dir)

    db_workload = Path(cfg.tvm_log_dir) / "database_workload.json"
    if not db_workload.exists():
        raise FileNotFoundError(f"No MetaSchedule DB found at {db_workload}. Run `tune` first or set correct tvm.log_dir.")

    with target:
        mod_opt = relax.transform.MetaScheduleApplyDatabase(cfg.tvm_log_dir)(mod_tune)
        ex = tvm.compile(mod_opt, target=target)
    return ex


def export_library(cfg: Config, log: RunLogger, ex: tvm.runtime.Module) -> Path:
    export_path = Path(cfg.tvm_export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Exporting library", path=str(export_path))
    ex.export_library(str(export_path))
    return export_path


def load_vm_from_so(cfg: Config, log: RunLogger) -> relax.VirtualMachine:
    dev = get_tvm_device(cfg)
    so_path = Path(cfg.tvm_export_path)
    if not so_path.exists():
        raise FileNotFoundError(f"Exported module not found: {so_path}")
    log.info("Loading exported TVM module", so=str(so_path))
    rt_mod = tvm.runtime.load_module(str(so_path))
    return relax.VirtualMachine(rt_mod, dev)

# ----------------------------
# Benchmarks
# ----------------------------
def sync_tvm(dev):
    try:
        dev.sync()
    except Exception:
        pass


def onnx_input_dtype(path: str) -> np.dtype:
    model = onnx.load(path)
    inp = model.graph.input[0]

    elem_type = inp.type.tensor_type.elem_type

    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
        TensorProto.INT8: np.int8,
    }

    if elem_type not in mapping:
        raise ValueError(f"Unsupported ONNX dtype: {elem_type}")

    return mapping[elem_type]


def bench_tvm_vm(vm: relax.VirtualMachine, dev, x_np: np.ndarray, runs: int, warmup: int, onnx_path: str) -> Dict[str, float]:
    expected = onnx_input_dtype(onnx_path)
    x_np = np.ascontiguousarray(x_np.astype(expected, copy=False))
    x_tvm = x_tvm = tvm.runtime.tensor(x_np, dev) 
    for _ in range(warmup):
        _ = vm["main"](x_tvm)
    sync_tvm(dev)

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = vm["main"](x_tvm)
    sync_tvm(dev)
    t1 = time.perf_counter()

    ms_ = (t1 - t0) * 1000.0 / runs
    return {"ms": ms_, "fps": 1000.0 / ms_}


def sync_torch(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def bench_pt_ultralytics(pt_path: str, x: torch.Tensor, runs: int, warmup: int, compiled: bool = False) -> Dict[str, float]:
    from ultralytics import YOLO
    model = YOLO(pt_path).model.eval().to(device=x.device, dtype=x.dtype)

    if compiled:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        sync_torch(x.device)

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        sync_torch(x.device)
        t1 = time.perf_counter()

    ms_ = (t1 - t0) * 1000.0 / runs
    return {"ms": ms_, "fps": 1000.0 / ms_}


def ort_session(cfg: Config) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = cfg.ort_threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL 
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = cfg.ort_use_arena

    provider = cfg.ort_provider
    if provider == "TensorrtExecutionProvider":
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_engine_cache_enable": cfg.ort_trt_cache_enable,
                "trt_engine_cache_path": cfg.ort_trt_cache_path,
                "trt_fp16_enable": cfg.ort_trt_fp16_enable,
            }),
            ("CUDAExecutionProvider", {}),
        ]
    else:
        providers = [provider]

    _unwanted = {"AzureExecutionProvider"}
    providers = [p for p in providers if (p if isinstance(p, str) else p[0]) not in _unwanted]


    return ort.InferenceSession(cfg.onnx_path, sess_options=so, providers=providers)

def ort_expected_np_dtype(sess: ort.InferenceSession) -> np.dtype:
    t = sess.get_inputs()[0].type
    m = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
    }
    if t not in m:
        raise ValueError(f"Unsupported ONNX input type: {t}")
    return m[t]


def bench_ort(cfg: Config, x_np: np.ndarray, runs: int, warmup: int) -> Dict[str, float]:
    sess = ort_session(cfg)
    inp = sess.get_inputs()[0]
    inp_name = inp.name

    expected = ort_expected_np_dtype(sess)
    x_np = np.ascontiguousarray(x_np.astype(expected, copy=False))
    feed = {inp_name: x_np}

    for _ in range(warmup):
        _ = sess.run(None, feed)

    # sync for CUDA/TRT
    if "CUDAExecutionProvider" in sess.get_providers() or "TensorrtExecutionProvider" in sess.get_providers():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(None, feed)
    if "CUDAExecutionProvider" in sess.get_providers() or "TensorrtExecutionProvider" in sess.get_providers():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms_ = (t1 - t0) * 1000.0 / runs
    return {"ms": ms_, "fps": 1000.0 / ms_, "providers": sess.get_providers()}

# ----------------------------
# CLI commands
# ----------------------------
def set_global_env(cfg: Config):
    # keep it configurable, but you can also put this in YAML if you want
    os.environ.setdefault("TVM_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.set_num_interop_threads(1)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


def cmd_tune(cfg: Config, log: RunLogger) -> None:
    set_global_env(cfg)
    target = make_tvm_target(cfg, log)
    mod = load_relax_module_from_onnx(cfg, log)
    mod_tune = preprocess_for_inference(mod, cfg, target)
    tasks = extract_relax_tasks(mod_tune, target)
    log.info("Extracted tasks", count=len(tasks))
    tune_all(cfg, log, mod_tune, tasks, target)


def cmd_tune_chosen(cfg: Config, log: RunLogger) -> None:
    set_global_env(cfg)
    target = make_tvm_target(cfg, log)
    mod = load_relax_module_from_onnx(cfg, log)
    mod_tune = preprocess_for_inference(mod, cfg, target)
    tasks = extract_relax_tasks(mod_tune, target)
    log.info("Extracted tasks", count=len(tasks))
    tune_selected(cfg, log, mod_tune, tasks, target)


def cmd_compile(cfg: Config, log: RunLogger) -> None:
    set_global_env(cfg)
    target = make_tvm_target(cfg, log)
    mod = load_relax_module_from_onnx(cfg, log)
    mod_tune = preprocess_for_inference(mod, cfg, target)

    ex = compile_with_db(cfg, log, mod_tune, target)
    run_artifact_path = Path(log.run_dir) / "artifacts" / "compiled.so"
    log.info("Exporting compiled artifact into run dir", path=str(run_artifact_path))
    ex.export_library(str(run_artifact_path))


def cmd_export(cfg: Config, log: RunLogger) -> None:
    """
    Export from DB + compile, then write to tvm.export_path
    """
    set_global_env(cfg)
    target = make_tvm_target(cfg, log)
    mod = load_relax_module_from_onnx(cfg, log)
    mod_tune = preprocess_for_inference(mod, cfg, target)
    ex = compile_with_db(cfg, log, mod_tune, target)
    export_library(cfg, log, ex)


def cmd_bench(cfg: Config, log: RunLogger) -> None:
    set_global_env(cfg)

    np_dtype = precision_to_np(cfg.precision)
    torch_dtype = precision_to_torch(cfg.precision)

    x_np = np.random.randn(cfg.batch, 3, cfg.imgsz, cfg.imgsz).astype(np_dtype, copy=False)
    x_np = np.ascontiguousarray(x_np)

    torch_device = torch.device("cuda" if cfg.device_kind == "cuda" else "cpu")
    x_torch = torch.from_numpy(x_np.astype(np.float32, copy=False)) 
    if cfg.precision in ("fp16", "float16") and torch_device.type == "cuda":
        x_torch = x_torch.to(device=torch_device, dtype=torch.float16)
    else:
        x_torch = x_torch.to(device=torch_device, dtype=torch.float32)

    # TVM VM from exported .so
    vm = load_vm_from_so(cfg, log)
    dev = get_tvm_device(cfg)

    tvm_res = bench_tvm_vm(vm, dev, x_np, runs=cfg.bench_runs, warmup=cfg.bench_warmup, onnx_path=cfg.onnx_path)
    pt_res = bench_pt_ultralytics(cfg.pt_path, x_torch, runs=cfg.bench_runs, warmup=cfg.bench_warmup, compiled=False)

    if cfg.device_kind == "cuda":
        # ORT CUDA EP
        cfg_cuda = dataclasses.replace(cfg, raw={**cfg.raw, "onnxruntime": {**cfg.raw.get("onnxruntime", {}), "provider": "CUDAExecutionProvider"}})
        ort_cuda = bench_ort(cfg_cuda, x_np, runs=cfg.bench_runs, warmup=cfg.bench_warmup)

        # ORT TensorRT EP
        cfg_trt = dataclasses.replace(cfg, raw={**cfg.raw, "onnxruntime": {**cfg.raw.get("onnxruntime", {}), "provider": "TensorrtExecutionProvider"}})
        ort_trt = bench_ort(cfg_trt, x_np, runs=cfg.bench_runs, warmup=cfg.bench_warmup)
        log.info(
            "Benchmark results",
            tvm_ms=tvm_res["ms"], tvm_fps=tvm_res["fps"],
            pt_ms=pt_res["ms"], pt_fps=pt_res["fps"],
            ort_cuda_ms=ort_cuda["ms"], ort_cuda_fps=ort_cuda["fps"],
            ort_trt_ms=ort_trt["ms"], ort_trt_fps=ort_trt["fps"],
        )
        print("ONNXRuntime CUDA:", ort_cuda)
        print("ONNXRuntime TensorRT:", ort_trt)

    elif cfg.device_kind == "cpu":
        cfg_cpu = dataclasses.replace(cfg, raw={**cfg.raw, "onnxruntime": {**cfg.raw.get("onnxruntime", {}), "provider": "CPUExecutionProvider"}})
        ort_cpu = bench_ort(cfg_cpu, x_np, runs=cfg.bench_runs, warmup=cfg.bench_warmup)
        log.info("Benchmark results",
                tvm_ms=tvm_res["ms"], tvm_fps=tvm_res["fps"],
                pt_ms=pt_res["ms"], pt_fps=pt_res["fps"],
                ort_provider=cfg.ort_provider, ort_ms=ort_cpu["ms"], ort_fps=ort_cpu["fps"])
        print("ONNXRuntime (CPU):", ort_cpu, "providers:", ort_cpu.get("providers", []))
    else: 
        raise ValueError(f"Unknown device.kind: {cfg.device_kind}")
    
    print("TVM:", tvm_res)
    print("PyTorch:", pt_res)



def main():
    ap = argparse.ArgumentParser(description="TVM/ONNXRUNTIME/PyTorch/TensorRT")
    ap.add_argument("--config", required=True, help="Path to YAML config")

    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tune", help="Tune all extracted tasks and write MetaSchedule DB")
    sub.add_parser("tune_chosen_operations", help="Tune only chosen ops from a YAML list")
    sub.add_parser("compile", help="Compile using existing MetaSchedule DB (no export to tvm.export_path)")
    sub.add_parser("export", help="Compile using existing DB and export to tvm.export_path")
    sub.add_parser("bench", help="Benchmark TVM (.so), PyTorch, and ORT")

    args = ap.parse_args()
    cfg = load_config(args.config)

    run_dir = make_run_dir(cfg)
    log = RunLogger(run_dir)
    log.info("Run started", cmd=args.cmd, run_dir=str(run_dir))
    log.info("Config summary", config=cfg.raw)

    try:
        if args.cmd == "tune":
            cmd_tune(cfg, log)
        elif args.cmd == "tune_chosen_operations":
            cmd_tune_chosen(cfg, log)
        elif args.cmd == "compile":
            cmd_compile(cfg, log)
        elif args.cmd == "export":
            cmd_export(cfg, log)
        elif args.cmd == "bench":
            cmd_bench(cfg, log)
        else:
            raise ValueError(f"Unknown cmd: {args.cmd}")
        log.info("Run completed successfully")
    except Exception as e:
        log.warn("Run failed", error=str(e))
        raise


if __name__ == "__main__":
    main()      