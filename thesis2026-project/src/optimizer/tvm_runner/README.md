# tvm_runner

A small, repeatable runner to **tune, compile, export, and benchmark** ONNX models across:

- **TVM Relax + MetaSchedule** (compiled `.so` + Relax VM)
- **PyTorch (Ultralytics .pt)** baseline
- **ONNX Runtime** with:
  - CPUExecutionProvider
  - CUDAExecutionProvider
  - TensorrtExecutionProvider (TRT EP, with engine cache)

It is designed for experiments: everything is driven by a YAML config and each run gets its own timestamped output directory with logs + a copy of the exact config.

---

## Repo layout

Layout (yours may vary):

tvm_runner/
run.py
config.schema.yaml
configs/
yolov8n_orin_gpu_fp32.yaml
out/                           
    /experiments            # Experiment logs
    /tvm_compiled_models    # Compiled by TVM .so Files
    /tvm_tuning_logs        # TVM default logs
trt_cache/                  # TensorRT engine cache 

---

## Requirements

### Python environment

```bash
source ~/venvs/tvm/bin/activate

Needs: 
    - Installed TVM: 
    https://tvm.apache.org/docs/install/from_source.html
    - Installation for ONNX Runtime: 
    python -m pip install --no-cache-dir "numpy==1.26.4"
    python -m pip install --no-cache-dir --no-deps \
  --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126  "onnxruntime-gpu==1.23.0"
    - Main Requirements:
    numpy                  1.26.4
    onnx                   1.20.1
    onnxruntime-gpu        1.23.0
    onnxslim               0.1.86
    torch                  2.8.0
    torchvision            0.23.0
    ultralytics            8.4.17

## Quick start

1.  Tune (MetaSchedule)
```bash
python run.py --config configs/CONF_NAME.yaml tune

2. Export a compiled TVM module

```bash
python run.py --config configs/CONF_NAME.yaml export

3. Benchmark

```bash
python run.py --config configs/CONF_NAME.yaml bench

4. Tune only chosen operatinos

```bash
python run.py --config configs/CONF_NAME.yaml tune_chosen_operations

Outputs & logging

Each invocation creates a unique run directory:

{run.out_dir}/{run.name}__{timestamp}__{hash}/
  run.log          # human log
  run.jsonl        # JSON lines log (easy to parse)
  run.yaml         # snapshot of config used
  artifacts/
    compiled.so    # only for the `export` command