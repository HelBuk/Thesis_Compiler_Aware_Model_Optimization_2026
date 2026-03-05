import os
os.environ["TVM_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4" 

import json
import time
from pathlib import Path
# import logging, warnings
import sys


import numpy as np
import onnx
import onnxruntime as ort
import torch

import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.relax_integration import extract_tasks, tune_relax

from ultralytics import YOLO

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def info(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [INFO] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [WARN] {msg}", file=sys.stderr, flush=True)


def get_next_versioned_path(base_path: str) -> Path:
    """
    If base_path exists, append _v1, _v2, ... until free name is found.
    """
    path = Path(base_path)

    if not path.exists():
        return path

    stem = path.stem          # filename without extension
    suffix = path.suffix      # .so / .tar
    parent = path.parent

    i = 1
    while True:
        new_name = f"{stem}_v{i}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        i += 1


def sync_tvm(dev):
    try:
        dev.sync()
    except Exception:
        pass


def bench_tvm(vm, dev, x_np, runs=10, warmup=20):
    x_tvm = tvm.runtime.tensor(x_np, dev)

    for _ in range(warmup):
        _ = vm["main"](x_tvm)
    sync_tvm(dev)

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = vm["main"](x_tvm)
    sync_tvm(dev)
    t1 = time.perf_counter()

    ms_ = (t1 - t0) * 1000.0 / runs
    fps = 1000.0 / ms_
    return ms_, fps


def sync_torch(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def bench_pt(pt_path="../models/yolov8n.pt", imgsz=640, device="cpu",
             runs=10, warmup=20, compiled=False):
    y = YOLO(pt_path)
    model = y.model.eval().to(device)
    x = torch.randn(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)

    if compiled:
        # PyTorch 2.x Inductor
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

    for _ in range(warmup):
        _ = model(x)
    sync_torch(torch.device(device))

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    sync_torch(torch.device(device))
    t1 = time.perf_counter()

    ms_ = (t1 - t0) * 1000.0 / runs
    fps = 1000.0 / ms_
    return ms_, fps

def _resolve_torch_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    mapping = {
        "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "f16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[key]


@torch.inference_mode()
def bench_pt_tensor(
    pt_path,
    x,
    runs=10,
    warmup=20,
    compiled=False,
    device=None,   # e.g. "cuda", "cpu", torch.device("cuda:0")
    dtype=None,    # e.g. torch.float16, "fp16", "float32"
):
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    dev = torch.device(device) if device is not None else x.device
    dt = _resolve_torch_dtype(dtype) or x.dtype

    if dev.type == "cpu" and dt == torch.float16:
        raise ValueError("FP16 on CPU is usually unsupported/slow. Use float32 or bfloat16.")

    y = YOLO(pt_path)
    model = y.model.eval().to(device=dev, dtype=dt)
    x = x.to(device=dev, dtype=dt, non_blocking=True).contiguous()

    if compiled:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)

    for _ in range(warmup):
        _ = model(x)
    sync_torch(dev)

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    sync_torch(dev)
    t1 = time.perf_counter()

    ms_ = (t1 - t0) * 1000.0 / runs
    return ms_, 1000.0 / ms_

def _ort_expected_np_dtype(sess):
    t = sess.get_inputs()[0].type  # e.g. "tensor(float)", "tensor(float16)"
    m = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if t not in m:
        raise ValueError(f"Unsupported ONNX input type: {t}")
    return m[t]

def bench_ort(
    onnx_path: str,
    x_np: np.ndarray,
    runs: int = 10,
    warmup: int = 20,
    threads: int = 4,
    use_arena: bool = True
):
    """
    Benchmark ONNX Runtime on CPU with controlled threading.
    Assumes x_np is NCHW float32 with the right shape for the model.
    """
    # Thread control (ORT uses its own pools; this is the closest analog to TVM_NUM_THREADS)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Optional: memory arena can improve perf; keep it on unless you’re debugging memory behavior.
    if not use_arena:
        so.enable_cpu_mem_arena = False

    # CPU EP explicitly
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=["CUDAExecutionProvider"]
        # providers=["CPUExecutionProvider"],
    )

    # Resolve input name (YOLO ONNX input is often "images", but don’t assume)
    inp = sess.get_inputs()[0]
    inp_name = inp.name

    expected = _ort_expected_np_dtype(sess)
    x_np = np.ascontiguousarray(x_np.astype(expected, copy=False))

    # Warmup
    feed = {inp_name: x_np}
    for _ in range(warmup):
        _ = sess.run(None, feed)

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(None, feed)
    t1 = time.perf_counter()

    ms_ = (t1 - t0) * 1000.0 / runs
    fps = 1000.0 / ms_
    return ms_, fps


def _cast_to_onnx_input_dtype(sess, x_np: np.ndarray) -> np.ndarray:
    inp = sess.get_inputs()[0]
    typ = inp.type  # e.g. "tensor(float)", "tensor(float16)"
    type_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    if typ not in type_map:
        raise ValueError(f"Unsupported ONNX input type: {typ}")
    return np.ascontiguousarray(x_np.astype(type_map[typ], copy=False))


def bench_ort_trt(
    onnx_path: str,
    x_np: np.ndarray,
    runs: int = 10,
    warmup: int = 20,
    cache_dir: str = "./trt_cache",
    dtype=None
):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    TRT_FP16 = (dtype == "float16")
    providers = [
        ("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_fp16_enable": TRT_FP16,   # set False if you want pure FP32
        }),
        ("CUDAExecutionProvider", {}),
    ]

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    inp_name = sess.get_inputs()[0].name
    x_np_cast = _cast_to_onnx_input_dtype(sess, x_np)
    feed = {inp_name: x_np_cast}

    for _ in range(warmup):
        _ = sess.run(None, feed)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(None, feed)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms_ = (t1 - t0) * 1000.0 / runs
    return ms_, 1000.0 / ms_


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
    txt = path.read_text().strip()
    if not txt:
        return

    dec = json.JSONDecoder()

    # 1) Try normal full JSON (object/array)
    try:
        data = dec.decode(txt)
        data = _strip_feature_keys(data)
        path.write_text(json.dumps(data, separators=(",", ":")))
        return
    except json.JSONDecodeError:
        pass

    # 2) Parse concatenated JSON objects / JSONL safely
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

    path.write_text("".join(json.dumps(o, separators=(",", ":")) + "\n" for o in objs))



def sanitize_ms_db(work_dir: str):
    wd = Path(work_dir)
    _sanitize_json_file(wd / "database_workload.json")
    _sanitize_json_file(wd / "database_tuning_record.json")


def tune_model_tvm(mod, tasks, target, work_dir, max_trials_global=3000, max_trials_per_task=100, num_trials_per_iter=64, timeout_sec_builder=300, timeout_sec_runner=30):
    # 1) Tune (resume-capable if same work_dir)
    if tasks:
        try:
            _db = tune_relax(
                mod=mod,
                params={},
                target=target,
                work_dir=work_dir,
                max_trials_global=max_trials_global,
                max_trials_per_task=max_trials_per_task,
                num_trials_per_iter=num_trials_per_iter,
                builder=ms.builder.LocalBuilder(timeout_sec=timeout_sec_builder),
                runner=ms.runner.LocalRunner(timeout_sec=timeout_sec_runner),
            )
        except RuntimeError as e:
            # In your TVM build, this is thrown when DB contains feature.* keys
            if "feature.has_asimd" not in str(e):
                raise
            print("Known feature.* DB parse issue during tuning. Sanitizing DB and continuing...")
            sanitize_ms_db(work_dir)
    else:
        print("No tasks to tune; reuse existing DB only.")

def run_targeted_tuning(mod, target, work_dir, op_name, max_trials_global=200, max_trials_per_task=100, num_trials_per_iter=64, timeout_sec_builder=300, timeout_sec_runner=30):
    try:
        _db = tune_relax(
            mod=mod,
            params={},
            target=target,
            work_dir=work_dir,             # same DB dir, appends records
            op_names=op_name,              # tune only this task(s)
            max_trials_global=max_trials_global,            
            max_trials_per_task=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,             
            builder=ms.builder.LocalBuilder(timeout_sec=timeout_sec_builder),
            runner=ms.runner.LocalRunner(timeout_sec=timeout_sec_runner),
        )
    except RuntimeError as e:
        if "feature.has_asimd" not in str(e):
            raise
        print("Known feature.* DB parse issue during tuning. Sanitizing DB and continuing...")
        sanitize_ms_db(work_dir)


if __name__ == "__main__":

    print("script started")
    TYPE = "float16"
    NP_DTYPE = np.float16 if TYPE == "float16" else np.float32
    TORCH_DTYPE = torch.float16 if TYPE == "float16" else torch.float32
    TVM_INPUT_DTYPE = np.float32
    if TYPE == "float16":
        ONNX_MODEL_PATH = "../../models/yolov8n_fp16_fixed.onnx"
    else: 
        ONNX_MODEL_PATH = "../../models/yolov8n_fp32.onnx"
    LOG_DIR = f"../../tvm_tuning_logs/tuning_yolov8n_orin_gpu_{TYPE}_04"
    RUN_TUNING = False  # set False after this op gets non-N/A
    RUN_TARGETED_TUNING = False  # set False after this op gets non-N/A
    MISSING_OP = None
    RUN_COMPILATION = False  # set False after this op gets non-N/A # Has to be set for RUN_DEFAULT_INFERENCE
    RUN_DEFAULT_INFERENCE = False
    EXPORT_MODEL = False
    try_device = 'gpu' # 'cpu'
    # export_path = get_next_versioned_path(
    #     "../../models/compiled_tvm_models/tvm_yolov8n.so"
    # )
    export_path = f"../../models/compiled_tvm_models/tvm_yolov8n_gpu_one_round_66_tasks_{TYPE}_v7.so"
    max_trials_global=10000
    max_trials_per_task=150
    num_trials_per_iter=64
    timeout_sec_builder=300
    timeout_sec_runner=300

    PATH_TO_COMPILED_MODEL = export_path
    BENCH_RUNS = 200
    ex = None

    if try_device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available in this PyTorch env.")
        device = "cuda"
        dev = tvm.cuda(0)
    else:
        device = "cpu"
        dev = tvm.cpu(0)


    if RUN_TUNING or RUN_TARGETED_TUNING or RUN_COMPILATION:
        onnx_model = onnx.load(ONNX_MODEL_PATH)
        mod = from_onnx(onnx_model, keep_params_in_input=False)
        if device == 'cuda':
            info("Setting up cuda target")
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            target = tvm.target.Target({
                "kind": "cuda",
                "arch": f"sm_{cc_major}{cc_minor}",
                "max_num_threads": 1024,
                "max_threads_per_block": 1024,
                "thread_warp_size": 32,
                "max_shared_memory_per_block": 49152,
                "registers_per_block": 65536,
                "host": {
                    "kind": "llvm",
                    "mtriple": "aarch64-linux-gnu",
                    "mcpu": "cortex-a78",
                },
            })

        elif device == 'cpu':
            info("Setting up cpu target")
            target = tvm.target.Target({
                "kind": "llvm",
                "mtriple": "aarch64-linux-gnu",
                "mcpu": "cortex-a78",   
                "num-cores": 6
            })
        else:
            raise RuntimeError(f"Unknown device type: not cpu, neither gpu")
        if not RUN_DEFAULT_INFERENCE:
            with target:
                mod_tune = tvm.transform.Sequential(
                    [
                        relax.transform.DecomposeOpsForInference(),
                        relax.transform.CanonicalizeBindings(),
                        relax.get_pipeline("zero"),  # legalize/fuse/fuse_tir
                    ]
                )(mod)

                tasks = extract_tasks(mod_tune, target=target, params={})
                print("num extracted tasks:", len(tasks))


            if RUN_TUNING:
                tune_model_tvm(mod_tune, tasks, target, work_dir=LOG_DIR, max_trials_global=max_trials_global, max_trials_per_task=max_trials_per_task, num_trials_per_iter=num_trials_per_iter, timeout_sec_builder=timeout_sec_builder, timeout_sec_runner=timeout_sec_runner)
                sanitize_ms_db(LOG_DIR)

            if RUN_TARGETED_TUNING:
                if MISSING_OP is None:
                    raise RuntimeError("RUN_TARGETED_TUNING=True but MISSING_OP is None")

                # Accept either a single pattern string or a list/tuple/set of patterns.
                if isinstance(MISSING_OP, str):
                    missing_ops = [MISSING_OP]
                elif isinstance(MISSING_OP, (list, tuple, set)):
                    missing_ops = [op for op in MISSING_OP if isinstance(op, str) and op.strip()]
                else:
                    raise TypeError("MISSING_OP must be a string, list, tuple, or set of strings")

                if not missing_ops:
                    raise RuntimeError("MISSING_OP contains no valid operation patterns")

                matched = []
                for pattern in missing_ops:
                    hits = [t.task_name for t in tasks if pattern == t.task_name]
                    info(f"Pattern '{pattern}' matched: {hits}")
                    matched.extend(hits)

                matched = list(dict.fromkeys(matched))

                if not matched:
                    raise RuntimeError(f"No task matched: {MISSING_OP}")
                run_targeted_tuning(mod_tune, target, work_dir=LOG_DIR, op_name=matched, max_trials_global=max_trials_global, max_trials_per_task=max_trials_per_task, num_trials_per_iter=num_trials_per_iter, timeout_sec_builder=timeout_sec_builder, timeout_sec_runner=timeout_sec_runner)
                sanitize_ms_db(LOG_DIR)
            
    
    if RUN_COMPILATION:
        # 3) Apply DB + compile
        info(f"Start compilation")
        with target:
            if RUN_DEFAULT_INFERENCE:
                info("Running default inference")
                if device == 'cuda':
                    raise RuntimeError(f"Cannot run default inference on cuda")
                else:
                    mod_run = relax.get_pipeline("default")(mod)
                    ex = tvm.compile(mod_run, target=target)
            else:
                if not Path(LOG_DIR, "database_workload.json").exists():
                    raise RuntimeError(f"No tuning DB found in {LOG_DIR}")
                info(f"Applying MetaSchedule from {LOG_DIR}")
                sanitize_ms_db(LOG_DIR)
                mod_opt = relax.transform.MetaScheduleApplyDatabase(LOG_DIR)(mod_tune)
                ex = tvm.compile(mod_opt, target=target)

        # 3.1) Export model
        if EXPORT_MODEL:
            ex.export_library(str(export_path))

    # 4) Benchmark
    if RUN_COMPILATION and Path(export_path).exists():
        info(f"Using {export_path} to run inference")
        rt_mod = tvm.runtime.load_module(export_path)
        vm = relax.VirtualMachine(rt_mod, dev)
    elif RUN_COMPILATION and ex is not None:
        warn("Compiled .so file does not exist; using in-memory compiled module.")
        vm = relax.VirtualMachine(ex, dev)
    elif RUN_COMPILATION == False and (PATH_TO_COMPILED_MODEL is not None):
        info(f"Using default compiled model:{PATH_TO_COMPILED_MODEL} to run inference")
        rt_mod = tvm.runtime.load_module(PATH_TO_COMPILED_MODEL)
        vm = relax.VirtualMachine(rt_mod, dev)
    else:
        raise FileNotFoundError(
            f"No compiled module available: missing '{export_path}' and no in-memory module `ex`."
        )
    
    info(f"Starting to benchmark")
    x_np = np.ascontiguousarray(np.random.randn(1, 3, 640, 640).astype(NP_DTYPE))
    x = torch.from_numpy(x_np).to(device=device, dtype=TORCH_DTYPE)
    x_np_tvm = np.ascontiguousarray(x_np.astype(TVM_INPUT_DTYPE, copy=False))
    ms_tvm_cpu, fps_tvm_cpu = bench_tvm(vm, dev, x_np_tvm, runs=BENCH_RUNS)
    print("TVM", ms_tvm_cpu, fps_tvm_cpu)
    ms_pt_cpu, fps_pt_cpu = bench_pt_tensor("../../models/yolov8n.pt", x, runs=BENCH_RUNS, dtype=TYPE, device=device)
    print("PyTorch", ms_pt_cpu, fps_pt_cpu)
    # ms_pt_comp,  fps_pt_comp  = bench_pt_tensor("../../models/yolov8n.pt", x, compiled=True, runs=BENCH_RUNS)
    # print("PyTorch (Compiled)", ms_pt_comp,  fps_pt_comp)
    ms_ort_gpu, fps_ort_gpu = bench_ort(ONNX_MODEL_PATH, x_np, runs=BENCH_RUNS, threads=4)
    print("ONNX Runtime", ms_ort_gpu, fps_ort_gpu)
    print("ORT providers:", ort.get_available_providers())
    print("TVM CUDA:", tvm.cuda(0).exist)
    ms_trt, fps_trt = bench_ort_trt(ONNX_MODEL_PATH, x_np, runs=BENCH_RUNS, dtype=TYPE)
    print("TensorRT (ORT EP)", ms_trt, fps_trt)

    
    


