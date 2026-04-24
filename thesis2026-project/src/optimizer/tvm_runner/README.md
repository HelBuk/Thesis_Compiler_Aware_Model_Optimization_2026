# tvm_runner

A small, repeatable runner to **tune, compile, export, and benchmark** ONNX models across:

- **TVM Relax + MetaSchedule** (compiled `.so` + Relax VM)
- **PyTorch (Ultralytics .pt)** baseline
- **ONNX Runtime** with:
  - CPUExecutionProvider
  - CUDAExecutionProvider
  - TensorrtExecutionProvider (TRT EP, with engine cache)

Everything is driven by a YAML config; each run gets its own timestamped output directory
with logs and a snapshot of the exact config used.

---

## Repo layout

```
tvm_runner/
├── run.py
├── config.schema.yaml
├── configs/
│   └── yolov8n_orin_gpu_fp32.yaml
├── out/
│   ├── experiments/          # Experiment logs
│   ├── tvm_compiled_models/  # Compiled .so files
│   └── tvm_tuning_logs/      # MetaSchedule tuning logs
└── trt_cache/                # TensorRT engine cache
```

---

## Requirements

### Python environment

```bash
source ~/venvs/tvm/bin/activate
```

Dependencies:

```
numpy                  1.26.4
onnx                   1.20.1
onnxruntime-gpu        1.23.0
onnxslim               0.1.86
torch                  2.8.0
torchvision            0.23.0
ultralytics            8.4.17
```

Install TVM from source:
https://tvm.apache.org/docs/install/from_source.html

Install ONNX Runtime (Jetson-specific wheel):

```bash
python -m pip install --no-cache-dir "numpy==1.26.4"
python -m pip install --no-cache-dir --no-deps \
  --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  "onnxruntime-gpu==1.23.0"
```

---

## Quick start

### 1. Tune (MetaSchedule)

```bash
python run.py --config configs/CONF_NAME.yaml tune
```

### 2. Export a compiled TVM module

```bash
python run.py --config configs/CONF_NAME.yaml export
```

### 3. Benchmark

```bash
python run.py --config configs/CONF_NAME.yaml bench
```

### 4. Tune only chosen operations

```bash
python run.py --config configs/CONF_NAME.yaml tune_chosen_operations
```

---

## Outputs & logging

Each invocation creates a unique run directory:

```
{run.out_dir}/{run.name}__{timestamp}__{hash}/
  run.log          # human-readable log
  run.jsonl        # JSON lines log (easy to parse)
  run.yaml         # snapshot of config used
  artifacts/
    compiled.so    # only for the `export` command
```
