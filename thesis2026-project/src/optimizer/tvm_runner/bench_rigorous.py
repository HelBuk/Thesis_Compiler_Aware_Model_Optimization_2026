#!/usr/bin/env python3
"""
bench_rigorous.py — Statistically robust, isolated benchmarking.

Each backend runs in a fresh subprocess per repetition to eliminate
cache contamination between frameworks. Full statistics are reported
(mean, std, 95% CI, min, median, max).

Usage:
    python bench_rigorous.py --config pi5_fp32.yaml
    python bench_rigorous.py --config pi5_fp32.yaml --repeats 7 --runs 200 --warmup 30
    python bench_rigorous.py --config pi5_fp32.yaml --backends tvm ort
    python bench_rigorous.py --config pi5_fp32.yaml --set-performance
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# CPU / thermal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_file(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def get_cpu_freq_mhz() -> Optional[float]:
    raw = _read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return int(raw) / 1000.0 if raw else None


def get_cpu_max_freq_mhz() -> Optional[float]:
    raw = _read_file("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    return int(raw) / 1000.0 if raw else None


def get_cpu_governor() -> Optional[str]:
    return _read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


def try_set_performance_governor() -> bool:
    for cmd in (
        ["sudo", "cpufreq-set", "-g", "performance"],
        ["sudo", "cpupower", "frequency-set", "-g", "performance"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=5)
            if r.returncode == 0:
                return True
        except Exception:
            pass
    return False


def get_temperature_c() -> Optional[float]:
    """Return the highest relevant thermal zone temperature in °C.

    Strategy (in order):
    1. Jetson Orin: read all thermal zones, return the maximum (covers GPU, CPU,
       SoC, memory — the watchdog cares about the hottest spot).
    2. Raspberry Pi: vcgencmd measure_temp (ARM core).
    3. Generic sysfs fallback (zone0).
    """
    # Collect all readable sysfs zones
    zone_temps: List[float] = []
    for zone in range(0, 16):
        raw = _read_file(f"/sys/class/thermal/thermal_zone{zone}/temp")
        if raw:
            try:
                val = int(raw)
                if val > 1000:          # value is in milli-°C
                    zone_temps.append(val / 1000.0)
                elif 0 < val < 200:     # already in °C (some boards)
                    zone_temps.append(float(val))
            except ValueError:
                pass
    if zone_temps:
        return max(zone_temps)          # worst-case hotspot

    # Raspberry Pi vcgencmd fallback
    try:
        r = subprocess.run(["vcgencmd", "measure_temp"],
                           capture_output=True, text=True, timeout=3)
        if r.returncode == 0:
            return float(r.stdout.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        pass
    return None


def print_system_info() -> None:
    print("\n" + "=" * 62)
    print("  SYSTEM INFO")
    print("=" * 62)
    print(f"  Platform   : {platform.platform()}")
    print(f"  Python     : {platform.python_version()}")

    gov   = get_cpu_governor()
    freq  = get_cpu_freq_mhz()
    mfreq = get_cpu_max_freq_mhz()
    temp  = get_temperature_c()

    print(f"  Governor   : {gov or 'unknown'}")
    if freq and mfreq:
        pct = freq / mfreq * 100
        flag = "✓" if pct > 95 else "⚠  NOT AT MAX — lock with --set-performance"
        print(f"  CPU freq   : {freq:.0f} / {mfreq:.0f} MHz  ({pct:.1f}%)  {flag}")
    print(f"  Temperature: {f'{temp:.1f} °C' if temp is not None else 'unknown'}")
    print("=" * 62 + "\n")


def check_throttled(threshold_pct: float = 95.0) -> bool:
    freq, mfreq = get_cpu_freq_mhz(), get_cpu_max_freq_mhz()
    if freq and mfreq and (freq / mfreq * 100) < threshold_pct:
        print(f"  ⚠  CPU throttled: {freq:.0f}/{mfreq:.0f} MHz — results unreliable")
        return True
    return False


def cooldown(target_c: float = 55.0, max_wait_s: float = 90.0, verbose: bool = True) -> None:
    t0 = time.time()
    while True:
        temp = get_temperature_c()
        elapsed = time.time() - t0
        if temp is None or temp <= target_c or elapsed >= max_wait_s:
            break
        if verbose:
            print(f"  ⏳ Cooling … {temp:.1f} °C → {target_c:.1f} °C  ({elapsed:.0f}s elapsed)",
                  end="\r", flush=True)
        time.sleep(3)
    if verbose:
        temp = get_temperature_c()
        tag  = f"{temp:.1f} °C" if temp is not None else "?"
        print(f"  ✓  Ready — temperature: {tag}                              ")


# ─────────────────────────────────────────────────────────────────────────────
# Power measurement helpers
# ─────────────────────────────────────────────────────────────────────────────
# Module-level cache — each subprocess gets fresh globals, discovers once.
_POWER_SOURCE: str      = ""   # "ina3221" | "vcgencmd" | "none"
_POWER_PATHS:  List[str] = []  # sysfs paths for INA3221 rails


def _discover_power_source() -> tuple:
    """Find the first available power sensor on this platform."""
    import glob
    # Jetson INA3221 (Orin Nano, Xavier, AGX …) — values in µW
    for pattern in (
        "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input",
        "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/power*_input",
        "/sys/devices/platform/*/i2c-*/*/hwmon/hwmon*/power*_input",
    ):
        paths = sorted(glob.glob(pattern))
        if paths:
            return "ina3221", paths
    # Raspberry Pi 5 PMIC via vcgencmd pmic_read_adc
    try:
        r = subprocess.run(["vcgencmd", "pmic_read_adc"],
                           capture_output=True, text=True, timeout=2)
        if r.returncode == 0 and "EXT5V_V" in r.stdout:
            return "vcgencmd", []
    except Exception:
        pass
    return "none", []


def _sample_power_mw() -> Optional[float]:
    """Instantaneous board power in mW, or None if no sensor found."""
    global _POWER_SOURCE, _POWER_PATHS
    if not _POWER_SOURCE:
        _POWER_SOURCE, _POWER_PATHS = _discover_power_source()

    if _POWER_SOURCE == "ina3221":
        total = 0.0
        for p in _POWER_PATHS:
            raw = _read_file(p)
            if raw:
                try:
                    total += int(raw)       # µW
                except ValueError:
                    pass
        return total / 1000.0 if total > 0 else None   # µW → mW

    if _POWER_SOURCE == "vcgencmd":
        try:
            r = subprocess.run(["vcgencmd", "pmic_read_adc"],
                               capture_output=True, text=True, timeout=2)
            if r.returncode == 0:
                voltage = current = None
                for line in r.stdout.splitlines():
                    if "EXT5V_V" in line:
                        try:
                            voltage = float(line.split("=")[1].strip().rstrip("V").strip())
                        except Exception:
                            pass
                    elif "EXT5V_A" in line:
                        try:
                            current = float(line.split("=")[1].strip().rstrip("A").strip())
                        except Exception:
                            pass
                if voltage is not None and current is not None:
                    return voltage * current * 1000.0   # W → mW
        except Exception:
            pass

    return None


def _power_stats(samples: List[float]) -> Optional[Dict]:
    if not samples:
        return None
    return {
        "mean_mw": statistics.mean(samples),
        "max_mw" : max(samples),
        "min_mw" : min(samples),
        "n"      : len(samples),
    }


_PWR_EVERY = 10   # sample power every N inference runs (keeps overhead negligible)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

# Student-t critical values for 95% CI (two-tailed, df = n-1)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def compute_stats(values: List[float]) -> Dict:
    n    = len(values)
    mean = statistics.mean(values)
    std  = statistics.stdev(values) if n > 1 else 0.0
    t    = _T95.get(n - 1, 2.0)
    ci95 = t * std / (n ** 0.5) if n > 1 else 0.0
    return {
        "n"        : n,
        "mean_ms"  : mean,
        "std_ms"   : std,
        "ci95_ms"  : ci95,
        "min_ms"   : min(values),
        "median_ms": statistics.median(values),
        "max_ms"   : max(values),
        "fps_mean" : 1000.0 / mean,
        "fps_best" : 1000.0 / min(values),
        "fps_worst": 1000.0 / max(values),
        "rep_means": values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Workers  (each runs in a fresh subprocess — no shared state)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_tvm(cfg_raw: dict, runs: int, warmup: int) -> None:
    import numpy as np
    import tvm
    from tvm import relax
    import onnx

    device_kind = cfg_raw.get("device", {}).get("kind", "cpu")
    is_cuda     = device_kind == "cuda"

    if is_cuda:
        os.environ.pop("TVM_NUM_THREADS", None)
        os.environ.pop("OMP_NUM_THREADS", None)
    else:
        os.environ["TVM_NUM_THREADS"] = "4"
        os.environ["OMP_NUM_THREADS"] = "4"

    # Resolve input dtype from ONNX graph
    onnx_model = onnx.load(cfg_raw["model"]["onnx_path"])
    elem_type  = onnx_model.graph.input[0].type.tensor_type.elem_type
    np_dtype   = {1: np.float32, 10: np.float16}.get(elem_type, np.float32)

    batch = cfg_raw["model"].get("batch", 1)
    imgsz = cfg_raw["model"].get("imgsz", 640)
    x_np  = np.ascontiguousarray(
        np.random.randn(batch, 3, imgsz, imgsz).astype(np_dtype))

    rt_mod = tvm.runtime.load_module(cfg_raw["tvm"]["export_path"])
    dev    = tvm.cuda(0) if is_cuda else tvm.cpu(0)
    vm     = relax.VirtualMachine(rt_mod, dev)
    x_tvm  = tvm.nd.array(x_np, dev)

    for _ in range(warmup):
        vm["main"](x_tvm)

    if is_cuda:
        tvm.cuda(0).sync()  # ensure GPU kernels finish before timing

    pwr: List[float] = []
    times: List[float] = []
    if is_cuda:
        for i in range(runs):
            if i % _PWR_EVERY == 0:
                p = _sample_power_mw()
                if p is not None:
                    pwr.append(p)
            t0 = time.perf_counter()
            vm["main"](x_tvm)
            tvm.cuda(0).sync()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    else:
        for i in range(runs):
            if i % _PWR_EVERY == 0:
                p = _sample_power_mw()
                if p is not None:
                    pwr.append(p)
            t0 = time.perf_counter()
            vm["main"](x_tvm)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    print(json.dumps({"backend": "TVM", "device": device_kind,
                      "times_ms": times, "power": _power_stats(pwr)}))


def _worker_ort(cfg_raw: dict, runs: int, warmup: int) -> None:
    import numpy as np
    import onnxruntime as ort

    ort_cfg      = cfg_raw.get("onnxruntime", {})
    threads      = int(ort_cfg.get("threads", 4))
    device_kind  = cfg_raw.get("device", {}).get("kind", "cpu")
    is_cuda      = device_kind == "cuda"

    # Derive wanted provider from config; fall back to CPUExecutionProvider.
    # Never use ort.get_available_providers() as the session provider list —
    # that can include AzureExecutionProvider and other unwanted EPs.
    provider_cfg = ort_cfg.get("provider", None)
    if provider_cfg:
        # Config explicitly names a provider
        if provider_cfg == "TensorrtExecutionProvider":
            wanted = ["TensorrtExecutionProvider",
                      "CUDAExecutionProvider",
                      "CPUExecutionProvider"]
        elif provider_cfg == "CUDAExecutionProvider":
            wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            wanted = ["CPUExecutionProvider"]
    else:
        # No explicit provider: infer from device.kind
        wanted = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                  if is_cuda else ["CPUExecutionProvider"])

    available = set(ort.get_available_providers())
    providers = [p for p in wanted if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    so = ort.SessionOptions()
    so.intra_op_num_threads     = threads
    so.inter_op_num_threads     = 1
    so.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena     = True

    # TensorRT provider options (from config if present)
    provider_options = None
    if providers[0] == "TensorrtExecutionProvider":
        trt_cfg = ort_cfg.get("trt", {})
        trt_opts: dict = {}
        if trt_cfg.get("enable_engine_cache"):
            trt_opts["trt_engine_cache_enable"] = True
            trt_opts["trt_engine_cache_path"]   = trt_cfg.get("engine_cache_path", "./trt_cache")
        if trt_cfg.get("fp16_enable"):
            trt_opts["trt_fp16_enable"] = True
        provider_options = [trt_opts, {}, {}]  # one dict per provider in `providers`

    sess = ort.InferenceSession(
        cfg_raw["model"]["onnx_path"],
        sess_options=so,
        providers=providers,
        **({"provider_options": provider_options} if provider_options else {}),
    )

    inp_name = sess.get_inputs()[0].name
    inp_type = sess.get_inputs()[0].type
    np_dtype = np.float16 if "float16" in inp_type else np.float32

    batch = cfg_raw["model"].get("batch", 1)
    imgsz = cfg_raw["model"].get("imgsz", 640)
    x_np  = np.ascontiguousarray(
        np.random.randn(batch, 3, imgsz, imgsz).astype(np_dtype))
    feed  = {inp_name: x_np}

    for _ in range(warmup):
        sess.run(None, feed)

    pwr: List[float] = []
    times: List[float] = []
    for i in range(runs):
        if i % _PWR_EVERY == 0:
            p = _sample_power_mw()
            if p is not None:
                pwr.append(p)
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    actual_providers = sess.get_providers()
    print(json.dumps({"backend": "ONNXRuntime", "times_ms": times,
                      "providers": actual_providers, "power": _power_stats(pwr)}))


def _worker_pytorch(cfg_raw: dict, runs: int, warmup: int) -> None:
    import numpy as np
    import torch
    from ultralytics import YOLO

    device_kind = cfg_raw.get("device", {}).get("kind", "cpu")
    use_cuda    = device_kind == "cuda" and torch.cuda.is_available()
    device      = torch.device("cuda:0" if use_cuda else "cpu")

    if not use_cuda:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

    batch = cfg_raw["model"].get("batch", 1)
    imgsz = cfg_raw["model"].get("imgsz", 640)

    model = YOLO(cfg_raw["model"]["pt_path"]).model.eval().to(device)
    x     = torch.randn(batch, 3, imgsz, imgsz, dtype=torch.float32, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if use_cuda:
            torch.cuda.synchronize()

        pwr: List[float] = []
        times: List[float] = []
        if use_cuda:
            for i in range(runs):
                if i % _PWR_EVERY == 0:
                    p = _sample_power_mw()
                    if p is not None:
                        pwr.append(p)
                t0 = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
        else:
            for i in range(runs):
                if i % _PWR_EVERY == 0:
                    p = _sample_power_mw()
                    if p is not None:
                        pwr.append(p)
                t0 = time.perf_counter()
                model(x)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

    print(json.dumps({"backend": "PyTorch", "device": str(device),
                      "times_ms": times, "power": _power_stats(pwr)}))


def _worker_tflite(cfg_raw: dict, runs: int, warmup: int) -> None:
    import numpy as np

    tflite_cfg = cfg_raw.get("tflite", {})
    model_path = tflite_cfg.get("model_path")
    if not model_path:
        print(json.dumps({"error": "tflite.model_path not set in config"}),
              file=__import__("sys").stderr)
        raise SystemExit(1)

    threads = int(tflite_cfg.get("threads", 4))

    # Prefer the lightweight tflite_runtime package (available on Pi5 / Orin).
    # Fall back to the full tensorflow tf.lite.Interpreter if not installed.
    try:
        import tflite_runtime.interpreter as _tflite
        Interpreter = _tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=model_path, num_threads=threads)
    interp.allocate_tensors()

    inp_detail = interp.get_input_details()[0]
    in_dtype              = inp_detail["dtype"]
    in_scale, in_zero     = inp_detail.get("quantization", (0.0, 0))

    batch = cfg_raw["model"].get("batch", 1)
    imgsz = cfg_raw["model"].get("imgsz", 640)

    # TFLite uses NHWC layout (B, H, W, C)
    x_f32 = np.random.randn(batch, imgsz, imgsz, 3).astype(np.float32)

    if in_dtype in (np.uint8, np.int8):
        if in_scale == 0.0:
            raise RuntimeError("TFLite input scale is 0; model has no quantization params.")
        x_feed = np.round(x_f32 / in_scale + in_zero).astype(in_dtype)
    else:
        x_feed = x_f32.astype(in_dtype)

    for _ in range(warmup):
        interp.set_tensor(inp_detail["index"], x_feed)
        interp.invoke()

    pwr: List[float] = []
    times: List[float] = []
    for i in range(runs):
        if i % _PWR_EVERY == 0:
            p = _sample_power_mw()
            if p is not None:
                pwr.append(p)
        t0 = time.perf_counter()
        interp.set_tensor(inp_detail["index"], x_feed)
        interp.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    print(json.dumps({
        "backend"  : "TFLite",
        "model"    : model_path,
        "in_dtype" : str(in_dtype),
        "threads"  : threads,
        "times_ms" : times,
        "power"    : _power_stats(pwr),
    }))


def _worker_trt(cfg_raw: dict, runs: int, warmup: int) -> None:
    """TensorRT native inference — no ORT/PyTorch overhead.

    Works with both TRT 8.x (binding-index API) and TRT 10.x (tensor-name API).
    Uses torch CUDA tensors as I/O buffers to avoid a pycuda dependency.
    """
    import numpy as np
    import torch
    import tensorrt as trt

    trt_cfg     = cfg_raw.get("tensorrt", {})
    engine_path = trt_cfg.get("engine_path")
    if not engine_path:
        print(json.dumps({"error": "tensorrt.engine_path not set in config"}),
              file=__import__("sys").stderr)
        raise SystemExit(1)

    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    trt_major = int(trt.__version__.split(".")[0])
    batch     = cfg_raw["model"].get("batch", 1)
    imgsz     = cfg_raw["model"].get("imgsz", 640)
    x_np      = np.random.randn(batch, 3, imgsz, imgsz).astype(np.float32)

    # ── Helper: map TRT dtype → torch dtype ──────────────────────────────────
    def _torch_dtype(trt_dtype) -> torch.dtype:
        return torch.float16 if trt_dtype == trt.float16 else torch.float32

    # ── Allocate I/O buffers and build the callable that runs one inference ──

    if trt_major >= 10:
        # TRT 10.x: tensor-name API (num_io_tensors, set_tensor_address)
        bufs: dict = {}
        for i in range(engine.num_io_tensors):
            name  = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            shape = tuple(batch if d < 0 else d for d in shape)
            bufs[name] = torch.zeros(shape,
                                     dtype=_torch_dtype(engine.get_tensor_dtype(name)),
                                     device="cuda")
            context.set_tensor_address(name, int(bufs[name].data_ptr()))

        input_name = engine.get_tensor_name(0)
        bufs[input_name].copy_(torch.from_numpy(x_np).cuda())
        stream = torch.cuda.current_stream().cuda_stream

        def _run() -> None:
            context.execute_async_v3(stream)

    else:
        # TRT 8.x: binding-index API (num_bindings, execute_v2)
        buf_list: list = []
        for i in range(engine.num_bindings):
            shape = tuple(context.get_binding_shape(i))
            shape = tuple(batch if d < 0 else d for d in shape)
            buf_list.append(torch.zeros(shape,
                                        dtype=_torch_dtype(engine.get_binding_dtype(i)),
                                        device="cuda"))
        buf_list[0].copy_(torch.from_numpy(x_np).cuda())
        bindings = [int(t.data_ptr()) for t in buf_list]

        def _run() -> None:
            context.execute_v2(bindings)

    # ── Warmup ────────────────────────────────────────────────────────────────
    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()

    # ── Timed runs ────────────────────────────────────────────────────────────
    pwr: List[float] = []
    times: List[float] = []
    for i in range(runs):
        if i % _PWR_EVERY == 0:
            p = _sample_power_mw()
            if p is not None:
                pwr.append(p)
        t0 = time.perf_counter()
        _run()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    print(json.dumps({
        "backend"    : "TensorRT",
        "trt_version": trt.__version__,
        "engine_path": engine_path,
        "times_ms"   : times,
        "power"      : _power_stats(pwr),
    }))


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess launcher
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_FN = {
    "tvm"    : "_worker_tvm",
    "ort"    : "_worker_ort",
    "pytorch": "_worker_pytorch",
    "tflite" : "_worker_tflite",
    "trt"    : "_worker_trt",
}


def run_isolated(backend: str, cfg_raw: dict, runs: int, warmup: int) -> Optional[dict]:
    """
    Spawn a fresh Python process for one backend × one repetition.
    Passes config as JSON via env var to avoid shell-quoting issues.
    """
    env = os.environ.copy()
    env["_BENCH_CFG"]    = json.dumps(cfg_raw)
    env["_BENCH_RUNS"]   = str(runs)
    env["_BENCH_WARMUP"] = str(warmup)

    worker_code = (
        f"import json, os, sys, time\n"
        f"from typing import List\n"
        f"sys.path.insert(0, {repr(str(Path(__file__).parent))})\n"
        f"from {Path(__file__).stem} import {_WORKER_FN[backend]}\n"
        f"cfg = json.loads(os.environ['_BENCH_CFG'])\n"
        f"runs = int(os.environ['_BENCH_RUNS'])\n"
        f"warmup = int(os.environ['_BENCH_WARMUP'])\n"
        f"{_WORKER_FN[backend]}(cfg, runs, warmup)\n"
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", worker_code],
            capture_output=True, text=True,
            env=env, timeout=900,
        )
    except subprocess.TimeoutExpired:
        print("  ✗  Subprocess timed out", file=sys.stderr)
        return None

    if result.returncode != 0:
        # Print last 60 lines of stderr for diagnosis
        err_lines = result.stderr.strip().splitlines()
        print(f"  ✗  Worker failed (exit {result.returncode}):", file=sys.stderr)
        for line in err_lines[-60:]:
            print(f"     {line}", file=sys.stderr)
        return None

    # Parse last valid JSON line (ignore LLVM/ORT warnings printed to stdout)
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            data = json.loads(line)
            if "times_ms" in data:
                return data
        except json.JSONDecodeError:
            continue

    print("  ✗  No valid JSON found in worker output", file=sys.stderr)
    print(f"     stdout tail: {result.stdout[-500:]}", file=sys.stderr)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_rep(rep: int, total: int, mean_ms: float, std_ms: float,
              temp: Optional[float], freq: Optional[float],
              power_mw: Optional[float] = None) -> None:
    temp_s  = f"{temp:.1f}°C"    if temp     is not None else "?"
    freq_s  = f"{freq:.0f}MHz"   if freq     is not None else "?"
    power_s = f"{power_mw:.0f}mW" if power_mw is not None else "n/a"
    print(f"    rep {rep:>2}/{total}  "
          f"mean={mean_ms:>8.2f}ms  ±{std_ms:>6.2f}ms  "
          f"temp={temp_s}  freq={freq_s}  pwr={power_s}")


def print_stats_block(label: str, s: Dict) -> None:
    print(f"\n  ┌─ {label} ─────────────────────────────────")
    print(f"  │  mean   : {s['mean_ms']:.2f} ms")
    print(f"  │  std    : ±{s['std_ms']:.2f} ms")
    print(f"  │  95% CI : ±{s['ci95_ms']:.2f} ms")
    print(f"  │  median : {s['median_ms']:.2f} ms")
    print(f"  │  min    : {s['min_ms']:.2f} ms")
    print(f"  │  max    : {s['max_ms']:.2f} ms")
    print(f"  │  FPS    : {s['fps_mean']:.2f}  "
          f"(best: {s['fps_best']:.2f}, worst: {s['fps_worst']:.2f})")
    if s.get("power_mean_mw") is not None:
        efficiency = s["fps_mean"] / (s["power_mean_mw"] / 1000.0)  # FPS/W
        print(f"  │  Power  : {s['power_mean_mw']:.0f} mW  "
              f"(max: {s.get('power_max_mw', 0):.0f} mW)  "
              f"efficiency: {efficiency:.2f} FPS/W")
    else:
        print(f"  │  Power  : n/a (no sensor detected)")
    print(f"  │  n reps : {s['n']}")
    print(f"  └────────────────────────────────────────────")


def print_summary(all_results: Dict[str, Dict]) -> None:
    has_power = any(s.get("power_mean_mw") is not None for s in all_results.values())
    width = 80 if has_power else 62
    print(f"\n{'='*width}")
    print("  FINAL SUMMARY  (thesis-ready numbers)")
    print(f"{'='*width}")
    if has_power:
        hdr = (f"  {'Backend':<18} {'Mean ± Std':>18}  {'95% CI':>10}"
               f"  {'FPS':>7}  {'Power':>8}  {'FPS/W':>6}")
    else:
        hdr = f"  {'Backend':<18} {'Mean ± Std':>18}  {'95% CI':>10}  {'FPS':>8}"
    print(hdr)
    print(f"  {'-'*(width-2)}")
    for name, s in all_results.items():
        ms_str = f"{s['mean_ms']:.1f} ± {s['std_ms']:.1f} ms"
        ci_str = f"±{s['ci95_ms']:.1f} ms"
        if has_power:
            pwr    = s.get("power_mean_mw")
            pwr_s  = f"{pwr:.0f} mW" if pwr else "  n/a   "
            eff_s  = f"{s['fps_mean']/(pwr/1000):.1f}" if pwr else "  n/a"
            print(f"  {name:<18} {ms_str:>18}  {ci_str:>10}"
                  f"  {s['fps_mean']:>7.2f}  {pwr_s:>8}  {eff_s:>6}")
        else:
            print(f"  {name:<18} {ms_str:>18}  {ci_str:>10}  {s['fps_mean']:>8.2f}")
    print(f"{'='*width}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_LABELS = {
    "tvm"    : "TVM",
    "ort"    : "ONNXRuntime",
    "pytorch": "PyTorch",
    "tflite" : "TFLite",
    "trt"    : "TensorRT",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Rigorous isolated benchmark")
    ap.add_argument("--config",          required=True,
                    help="Path to YAML config (same format as run.py)")
    ap.add_argument("--repeats",         type=int, default=5,
                    help="Independent outer repetitions per backend (default: 5)")
    ap.add_argument("--runs",            type=int, default=200,
                    help="Inner timing runs per repetition (default: 200)")
    ap.add_argument("--warmup",          type=int, default=30,
                    help="Warmup runs before each repetition (default: 30)")
    ap.add_argument("--cooldown-temp",   type=float, default=55.0,
                    help="Wait for this °C between reps (default: 55)")
    ap.add_argument("--no-cooldown",     action="store_true",
                    help="Skip thermal cooldown between reps")
    ap.add_argument("--backends",        nargs="+",
                    default=["tvm", "ort", "pytorch"],
                    choices=["tvm", "ort", "pytorch", "tflite", "trt"])
    ap.add_argument("--set-performance", action="store_true",
                    help="Try sudo cpufreq-set -g performance before benchmarking")
    ap.add_argument("--out",             default=None,
                    help="JSON output path (default: next to config file)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg_raw  = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # ── Optionally set performance governor ──────────────────────────────────
    if args.set_performance:
        ok = try_set_performance_governor()
        print(f"  CPU governor → performance: {'✓' if ok else '✗ (run with sudo or install cpufrequtils)'}")

    print_system_info()

    all_results: Dict[str, Dict] = {}

    for backend in args.backends:
        label = BACKEND_LABELS[backend]
        print(f"\n{'─'*62}")
        print(f"  Backend: {label}   ({args.repeats} reps × {args.runs} runs, warmup={args.warmup})")
        print(f"{'─'*62}")

        rep_means:      List[float] = []
        rep_power_means: List[float] = []
        rep_power_maxes: List[float] = []
        extra_info: dict            = {}

        for rep in range(1, args.repeats + 1):
            # Cooldown before every rep (including first to ensure stable start)
            if not args.no_cooldown:
                cooldown(target_c=args.cooldown_temp)

            check_throttled()

            temp = get_temperature_c()
            freq = get_cpu_freq_mhz()

            data = run_isolated(backend, cfg_raw, args.runs, args.warmup)

            if data is None:
                print(f"    rep {rep:>2}/{args.repeats}  ✗  failed — skipping")
                continue

            times   = data["times_ms"]
            mean_ms = statistics.mean(times)
            std_ms  = statistics.stdev(times) if len(times) > 1 else 0.0
            rep_means.append(mean_ms)

            pwr_rep = data.get("power")  # dict with mean_mw/max_mw/min_mw or None
            if pwr_rep:
                rep_power_means.append(pwr_rep["mean_mw"])
                rep_power_maxes.append(pwr_rep["max_mw"])

            if "providers" in data:
                extra_info["providers"] = data["providers"]

            print_rep(rep, args.repeats, mean_ms, std_ms, temp, freq,
                      power_mw=pwr_rep["mean_mw"] if pwr_rep else None)

        if not rep_means:
            print(f"  ✗  All repetitions failed for {label}")
            continue

        stats = compute_stats(rep_means)
        if rep_power_means:
            stats["power_mean_mw"] = statistics.mean(rep_power_means)
            stats["power_max_mw"]  = max(rep_power_maxes)
            stats["power_std_mw"]  = statistics.stdev(rep_power_means) if len(rep_power_means) > 1 else 0.0
        else:
            stats["power_mean_mw"] = None
            stats["power_max_mw"]  = None
        if extra_info:
            stats.update(extra_info)

        all_results[label] = stats
        print_stats_block(label, stats)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_summary(all_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else cfg_path.parent / "bench_rigorous_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform"    : platform.platform(),
        "config"      : str(cfg_path),
        "repeats"     : args.repeats,
        "runs_per_rep": args.runs,
        "warmup"      : args.warmup,
        "cpu_governor": get_cpu_governor(),
        "cpu_freq_mhz": get_cpu_freq_mhz(),
        "results"     : all_results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
