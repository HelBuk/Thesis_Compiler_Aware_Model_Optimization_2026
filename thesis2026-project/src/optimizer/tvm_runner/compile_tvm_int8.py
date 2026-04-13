#!/usr/bin/env python3
"""
compile_tvm_int8.py  —  Compile a pre-quantized INT8 QDQ ONNX model to a
TVM Relax shared library (.so) for use with bench_rigorous and
yolo_accuracy_comparison.

The DequantizeLinear TVM-compatibility fix is applied inline (no need to run
quantize_onnx_fixed.py first).  The graph input stays FLOAT32; TVM maps the
QDQ pattern through its own integer-arithmetic lowering inside the Relax
compilation pipeline.

Note on TVM INT8 lowering (TVM 0.24.dev0)
------------------------------------------
When a QDQ ONNX is imported via from_onnx, TVM Relax represents
QuantizeLinear/DequantizeLinear as relax.op.quantize / dequantize.
After DecomposeOpsForInference these fuse into the surrounding Conv kernels
and MetaSchedule schedules them as int8 WMMA / dp4a tiles on sm87.
The resulting .so accepts FLOAT32 input (matching the ONNX graph boundary)
and outputs FLOAT32 detections, with INT8 arithmetic in the Conv layers.

Usage
-----
  # (run from src/optimizer/tvm_runner/)

  # Step 1 — tune MetaSchedule DB (skip if DB already exists):
  python compile_tvm_int8.py tune \\
      --onnx ../../../models/quantized_models/onnx/yolov8n_opset17_int8_QDQ_1percent_coco.onnx \\
      --output ../../../models/compiled_tvm_models/yolov8n_orin_gpu_int8_1pct.so \\
      --db-dir out/tvm_tuning_logs/yolov8n_orin_gpu_int8_v1 \\
      --max-trials 500 --max-trials-per-task 20

  # Step 2 — compile with existing DB and export:
  python compile_tvm_int8.py export \\
      --onnx ../../../models/quantized_models/onnx/yolov8n_opset17_int8_QDQ_1percent_coco.onnx \\
      --output ../../../models/compiled_tvm_models/yolov8n_orin_gpu_int8_1pct.so \\
      --db-dir out/tvm_tuning_logs/yolov8n_orin_gpu_int8_v1

  # Tune + export in one shot:
  python compile_tvm_int8.py all \\
      --onnx ../../../models/quantized_models/onnx/yolov8n_opset17_int8_QDQ_1percent_coco.onnx \\
      --output ../../../models/compiled_tvm_models/yolov8n_orin_gpu_int8_1pct.so \\
      --db-dir out/tvm_tuning_logs/yolov8n_orin_gpu_int8_v1

  # CPU target (e.g. for cross-compilation checks):
  python compile_tvm_int8.py export --device cpu \\
      --onnx ... --output ... --db-dir ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
import torch

import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

# TVM meta_schedule: try standard path first, fall back to the s_tir alias
# used in run.py on some custom builds.
try:
    from tvm import meta_schedule as ms
    from tvm.meta_schedule.relax_integration import extract_tasks, tune_relax
except ImportError:
    try:
        from tvm.s_tir import meta_schedule as ms  # type: ignore[no-redef]
        from tvm.s_tir.meta_schedule.relax_integration import (  # type: ignore[no-redef]
            extract_tasks,
            tune_relax,
        )
    except ImportError as exc:
        raise ImportError(
            "Cannot import tvm.meta_schedule.  "
            "Make sure TVM 0.24.dev0 is installed and on PYTHONPATH."
        ) from exc


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Log:
    def __init__(self, log_path: Optional[Path] = None):
        self._path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def _ts(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def info(self, msg: str) -> None:
        line = f"{self._ts()} [INFO] {msg}"
        print(line, flush=True)
        if self._path:
            self._path.open("a").write(line + "\n")

    def warn(self, msg: str) -> None:
        line = f"{self._ts()} [WARN] {msg}"
        print(line, file=sys.stderr, flush=True)
        if self._path:
            self._path.open("a").write(line + "\n")


# ---------------------------------------------------------------------------
# DequantizeLinear TVM fix  (same logic as quantize_onnx_fixed.py)
#
# TVM 0.24 chokes on DequantizeLinear nodes whose zero_point input is an
# INT32 initializer.  When zero_point is all-zeros we can safely drop it
# (two-input form is equivalent to three-input with zp=0).
# ---------------------------------------------------------------------------

def _fix_dequant_zero_points(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Remove all-zero INT32 zero_point inputs from DequantizeLinear nodes.

    Returns (fixed_model, n_fixed) where n_fixed is the number of nodes patched.
    """
    # Build zero-initialiser set: name → True if INT32 and all zeros
    zero_init: set[str] = set()
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT32:
            arr = numpy_helper.to_array(init)
            if (arr == 0).all():
                zero_init.add(init.name)

    # Find DequantizeLinear nodes with zero_point in that set
    patched = 0
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        if len(node.input) < 3:
            continue
        zp_name = node.input[2]
        if zp_name in zero_init:
            del node.input[2]
            patched += 1

    return model, patched


def fix_onnx_for_tvm(onnx_path: str, log: Log) -> onnx.ModelProto:
    """Load INT8 QDQ ONNX, apply the DequantizeLinear fix, run shape inference."""
    log.info(f"Loading ONNX: {onnx_path}")
    model = onnx.load(onnx_path)

    # Quick sanity check: verify it looks like a QDQ model
    qdq_ops = {n.op_type for n in model.graph.node}
    has_qdq = "QuantizeLinear" in qdq_ops or "DequantizeLinear" in qdq_ops
    if not has_qdq:
        log.warn(
            "No QuantizeLinear/DequantizeLinear nodes found — "
            "this may not be a QDQ model.  Proceeding anyway."
        )

    model, n_fixed = _fix_dequant_zero_points(model)
    log.info(f"DequantizeLinear fix applied: {n_fixed} nodes patched")

    # Shape inference so TVM sees tensor shapes
    try:
        model = onnx.shape_inference.infer_shapes(model)
        log.info("ONNX shape inference OK")
    except Exception as exc:
        log.warn(f"Shape inference failed ({exc}); continuing without it")

    return model


# ---------------------------------------------------------------------------
# MetaSchedule DB sanitiser  (copied from run.py)
# ---------------------------------------------------------------------------

def _strip_feature_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_feature_keys(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("feature."))}
    if isinstance(obj, list):
        return [_strip_feature_keys(x) for x in obj]
    return obj


def _sanitize_json_file(path: Path) -> None:
    if not path.exists():
        return
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return
    dec = json.JSONDecoder()
    try:
        data = dec.decode(txt)
        path.write_text(json.dumps(_strip_feature_keys(data), separators=(",", ":")),
                        encoding="utf-8")
        return
    except json.JSONDecodeError:
        pass
    objs, i, n = [], 0, len(txt)
    while i < n:
        while i < n and txt[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(txt, i)
        objs.append(_strip_feature_keys(obj))
        i = j
    path.write_text(
        "".join(json.dumps(o, separators=(",", ":")) + "\n" for o in objs),
        encoding="utf-8",
    )


def sanitize_ms_db(work_dir: str) -> None:
    wd = Path(work_dir)
    _sanitize_json_file(wd / "database_workload.json")
    _sanitize_json_file(wd / "database_tuning_record.json")


# ---------------------------------------------------------------------------
# TVM pipeline
# ---------------------------------------------------------------------------

def make_target(device: str, arch: Optional[str], log: Log) -> tvm.target.Target:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        cc_major, cc_minor = torch.cuda.get_device_capability(0)
        auto_arch = f"sm_{cc_major}{cc_minor}"
        chosen_arch = arch or auto_arch
        log.info(f"CUDA target: arch={chosen_arch}  (auto-detected: {auto_arch})")
        return tvm.target.Target({
            "kind": "cuda",
            "arch": chosen_arch,
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
    elif device == "cpu":
        log.info("CPU (LLVM aarch64) target")
        return tvm.target.Target({
            "kind": "llvm",
            "mtriple": "aarch64-linux-gnu",
            "mcpu": "cortex-a78",
            "num-cores": 6,
        })
    else:
        raise ValueError(f"Unknown device: {device!r}. Use 'cuda' or 'cpu'.")


def load_relax_from_onnx(onnx_model: onnx.ModelProto, log: Log) -> relax.IRModule:
    log.info("Converting ONNX → TVM Relax IRModule")
    mod = from_onnx(onnx_model, keep_params_in_input=False)
    log.info("Relax IRModule loaded")
    return mod


def preprocess(mod: relax.IRModule, target: tvm.target.Target, log: Log) -> relax.IRModule:
    log.info("Preprocessing IRModule for inference")
    with target:
        mod = tvm.transform.Sequential([
            relax.transform.DecomposeOpsForInference(),
            relax.transform.CanonicalizeBindings(),
            relax.get_pipeline("zero"),
        ])(mod)
    log.info("Preprocessing done")
    return mod


def run_tune(
    mod: relax.IRModule,
    target: tvm.target.Target,
    db_dir: str,
    max_trials: int,
    max_trials_per_task: int,
    num_trials_per_iter: int,
    timeout_builder_s: int,
    timeout_runner_s: int,
    log: Log,
) -> None:
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    tasks = extract_tasks(mod, target=target, params={})
    log.info(f"Extracted {len(tasks)} MetaSchedule tasks")
    if not tasks:
        log.warn("No tasks to tune — model may be all-elementwise (no conv/matmul).")
        return

    log.info(
        f"Starting tuning: max_trials={max_trials}, "
        f"max_per_task={max_trials_per_task}, "
        f"num_per_iter={num_trials_per_iter}, "
        f"db_dir={db_dir}"
    )
    try:
        tune_relax(
            mod=mod,
            params={},
            target=target,
            work_dir=db_dir,
            max_trials_global=max_trials,
            max_trials_per_task=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            builder=ms.builder.LocalBuilder(timeout_sec=timeout_builder_s),
            runner=ms.runner.LocalRunner(timeout_sec=timeout_runner_s),
        )
    except RuntimeError as exc:
        if "feature.has_asimd" in str(exc):
            log.warn("MetaSchedule DB parse issue; sanitizing DB and continuing")
            sanitize_ms_db(db_dir)
        else:
            raise
    sanitize_ms_db(db_dir)
    log.info("Tuning complete")


def compile_and_export(
    mod: relax.IRModule,
    target: tvm.target.Target,
    db_dir: str,
    output_path: str,
    log: Log,
) -> None:
    db_workload = Path(db_dir) / "database_workload.json"
    if not db_workload.exists():
        raise FileNotFoundError(
            f"MetaSchedule DB not found at {db_workload}.  "
            "Run 'tune' first or check --db-dir."
        )

    log.info(f"Compiling with MetaSchedule DB: {db_dir}")
    sanitize_ms_db(db_dir)

    with target:
        mod_opt = relax.transform.MetaScheduleApplyDatabase(db_dir)(mod)
        ex = tvm.compile(mod_opt, target=target)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Exporting .so → {out}")
    ex.export_library(str(out))
    log.info(f"Export done: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compile INT8 QDQ ONNX to TVM Relax .so (TVM 0.24.dev0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "cmd",
        choices=["tune", "export", "all"],
        help=(
            "tune: run MetaSchedule tuning and save DB; "
            "export: compile with existing DB and write .so; "
            "all: tune then export"
        ),
    )
    ap.add_argument(
        "--onnx", required=True,
        help="Path to INT8 QDQ ONNX model (DequantizeLinear fix applied inline)"
    )
    ap.add_argument(
        "--output", required=True,
        help="Output path for the compiled TVM .so"
    )
    ap.add_argument(
        "--db-dir", default="out/tvm_tuning_logs/yolov8n_int8_v1",
        help="MetaSchedule DB directory (default: out/tvm_tuning_logs/yolov8n_int8_v1)"
    )
    ap.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="Target device (default: cuda)"
    )
    ap.add_argument(
        "--arch", default=None,
        help="CUDA arch, e.g. sm87 for Orin Nano (auto-detected if omitted)"
    )
    ap.add_argument(
        "--max-trials", type=int, default=500,
        help="Max global MetaSchedule trials (default: 500)"
    )
    ap.add_argument(
        "--max-trials-per-task", type=int, default=20,
        help="Max trials per extracted task (default: 20)"
    )
    ap.add_argument(
        "--num-trials-per-iter", type=int, default=8,
        help="Trials per MetaSchedule iteration (default: 8)"
    )
    ap.add_argument(
        "--timeout-builder", type=int, default=300,
        help="Builder timeout in seconds (default: 300)"
    )
    ap.add_argument(
        "--timeout-runner", type=int, default=120,
        help="Runner timeout in seconds (default: 120)"
    )
    ap.add_argument(
        "--skip-fix", action="store_true",
        help="Skip the DequantizeLinear fix (use if already applied)"
    )
    ap.add_argument(
        "--log-file", default=None,
        help="Optional log file path (default: next to --output)"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    # Set up log file next to the .so output if not specified
    out_path = Path(args.output)
    log_path = (
        Path(args.log_file) if args.log_file
        else out_path.parent / f"{out_path.stem}_compile.log"
    )
    log = Log(log_path)
    log.info(f"compile_tvm_int8.py  cmd={args.cmd}  onnx={args.onnx}")
    log.info(f"output={args.output}  db_dir={args.db_dir}  device={args.device}")

    # Thread config
    os.environ.setdefault("TVM_NUM_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    # ── Load and fix ONNX ────────────────────────────────────────────────────
    if args.skip_fix:
        log.info("--skip-fix: loading ONNX without DequantizeLinear patch")
        onnx_model = onnx.load(args.onnx)
        try:
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
        except Exception as exc:
            log.warn(f"Shape inference skipped: {exc}")
    else:
        onnx_model = fix_onnx_for_tvm(args.onnx, log)

    # ── TVM target ───────────────────────────────────────────────────────────
    target = make_target(args.device, args.arch, log)

    # ── ONNX → Relax → preprocess ────────────────────────────────────────────
    mod = load_relax_from_onnx(onnx_model, log)
    mod = preprocess(mod, target, log)

    # ── Tune ─────────────────────────────────────────────────────────────────
    if args.cmd in ("tune", "all"):
        run_tune(
            mod=mod,
            target=target,
            db_dir=args.db_dir,
            max_trials=args.max_trials,
            max_trials_per_task=args.max_trials_per_task,
            num_trials_per_iter=args.num_trials_per_iter,
            timeout_builder_s=args.timeout_builder,
            timeout_runner_s=args.timeout_runner,
            log=log,
        )

    # ── Compile + export ─────────────────────────────────────────────────────
    if args.cmd in ("export", "all"):
        compile_and_export(
            mod=mod,
            target=target,
            db_dir=args.db_dir,
            output_path=args.output,
            log=log,
        )

    elapsed = time.perf_counter() - t_start
    log.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
