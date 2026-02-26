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

@torch.inference_mode()
def bench_pt_tensor(pt_path, x, runs=10, warmup=20, compiled=False):
    y = YOLO(pt_path)
    model = y.model.eval().to(x.device)
    if compiled:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    for _ in range(warmup):
        _ = model(x)
    sync_torch(x.device)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    sync_torch(x.device)
    t1 = time.perf_counter()
    ms_ = (t1 - t0) * 1000.0 / runs
    return ms_, 1000.0 / ms_

def bench_ort(
    onnx_path: str,
    x_np: np.ndarray,
    runs: int = 10,
    warmup: int = 20,
    threads: int = 4,
    use_arena: bool = True,
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
        providers=["CPUExecutionProvider"],
    )

    # Resolve input name (YOLO ONNX input is often "images", but don’t assume)
    inp = sess.get_inputs()[0]
    inp_name = inp.name

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

    # Try full JSON first
    try:
        data = json.loads(txt)
        data = _strip_feature_keys(data)
        path.write_text(json.dumps(data, separators=(",", ":")))
        return
    except json.JSONDecodeError:
        pass

    # Fallback: JSON lines
    out_lines = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj = _strip_feature_keys(obj)
        out_lines.append(json.dumps(obj, separators=(",", ":")))
    path.write_text("\n".join(out_lines) + "\n")


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
            work_dir=work_dir,                  # same DB dir, appends records
            op_names=[op_name],              # tune only this task
            max_trials_global=max_trials_global,               # 64~128 is fine
            max_trials_per_task=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,              # avoid chunky allocation
            builder=ms.builder.LocalBuilder(timeout_sec=timeout_sec_builder),
            runner=ms.runner.LocalRunner(timeout_sec=timeout_sec_runner),
        )
    except RuntimeError as e:
        # In your TVM build, this is thrown when DB contains feature.* keys
        if "feature.has_asimd" not in str(e):
            raise
        print("Known feature.* DB parse issue during tuning. Sanitizing DB and continuing...")
        sanitize_ms_db(work_dir)


if __name__ == "__main__":

    print("script started")
    ONNX_MODEL_PATH = "../models/yolov8n.onnx"
    LOG_DIR = "tuning_yolov8n_pi5_11"
    RUN_TUNING = False  # set False after this op gets non-N/A
    RUN_TARGETED_TUNING = True  # set False after this op gets non-N/A
    MISSING_OP = "fused_conv2d14_add10_tir_sigmoid7_multiply7" #None #"fused_conv2d34_add15_tir_sigmoid11_multiply11"
    RUN_COMPILATION = True  # set False after this op gets non-N/A
    RUN_DEFAULT_INFERENCE = False
    EXPORT_MODEL = False
    device = 'cpu'
    export_path = get_next_versioned_path(
        "./compiled_models/tvm_yolov8n_relax.so"
    )
    PATH_TO_COMPILED_MODEL = None #'./tvm_yolov8n_relax.so'
    BENCH_RUNS = 10
    ex = None
    if RUN_TUNING or RUN_TARGETED_TUNING or RUN_COMPILATION:
        onnx_model = onnx.load(ONNX_MODEL_PATH)
        mod = from_onnx(onnx_model, keep_params_in_input=False)
        target = tvm.target.Target(
            {
                "kind": "llvm",
                "mtriple": "aarch64-linux-gnu",
                "mcpu": "cortex-a76",
                "num-cores": 4,
            }
        )

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
        tune_model_tvm(mod_tune, tasks, target, work_dir=LOG_DIR, max_trials_global=30, max_trials_per_task=10, num_trials_per_iter=1, timeout_sec_builder=300, timeout_sec_runner=30)
        sanitize_ms_db(LOG_DIR)

    if RUN_TARGETED_TUNING:
        matched = None
        if MISSING_OP is not None:
            matched = [t.task_name for t in tasks if MISSING_OP in t.task_name]
            info(f"Matched missing-op tasks: {matched}")
        if not matched:
            raise RuntimeError(f"No task matched: {MISSING_OP}")
        run_targeted_tuning(mod_tune, target, work_dir=LOG_DIR, op_name=MISSING_OP, max_trials_global=10, max_trials_per_task=10, num_trials_per_iter=10, timeout_sec_builder=300, timeout_sec_runner=30)
        sanitize_ms_db(LOG_DIR)
    
    
    if RUN_COMPILATION:
        # 3) Apply DB + compile
        info(f"Start compilation")
        with target:
            if RUN_DEFAULT_INFERENCE:
                info("Running default inference")
                #mod_opt = relax.get_pipeline("default")(mod_tune)
                ex = tvm.compile(mod_tune, target=target)
            else:
                if not Path(LOG_DIR, "database_workload.json").exists():
                    raise RuntimeError(f"No tuning DB found in {LOG_DIR}")
                info(f"Applying MetaSchedule from {LOG_DIR}")
                mod_opt = relax.transform.MetaScheduleApplyDatabase(LOG_DIR)(mod_tune)
                ex = tvm.compile(mod_opt, target=target)

        # 3.1) Export model
        if EXPORT_MODEL:
            ex.export_library(str(export_path))

    # 4) Benchmark
    dev = tvm.cpu(0)
    if RUN_COMPILATION and export_path.exists():
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
    x_np = np.random.randn(1,3,640,640).astype("float32")
    x = torch.from_numpy(x_np).to(device)
    ms_tvm_cpu, fps_tvm_cpu = bench_tvm(vm, dev, x_np, runs=BENCH_RUNS)
    print("TVM", ms_tvm_cpu, fps_tvm_cpu)
    ms_pt_cpu, fps_pt_cpu = bench_pt_tensor("../models/yolov8n.pt", x, runs=BENCH_RUNS)
    print("PyTorch", ms_pt_cpu, fps_pt_cpu)
    ms_pt_comp,  fps_pt_comp  = bench_pt_tensor("../models/yolov8n.pt", x, compiled=True, runs=BENCH_RUNS)
    print("PyTorch (Compiled)", ms_pt_comp,  fps_pt_comp)
    ms_ort_cpu, fps_ort_cpu = bench_ort("../models/yolov8n.onnx", x_np, runs=BENCH_RUNS, threads=4)
    print("ONNX Runtime", ms_ort_cpu, fps_ort_cpu)

    
    


