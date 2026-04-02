# Compiler-Aware Model Optimization — NTNU Master's Thesis 2026

Performance optimization of YOLOv8n inference on NVIDIA Orin Nano (sm87) using
TensorRT, custom CUDA plugins, quantization, pruning, and roofline-driven
hardware analysis.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware Target](#hardware-target)
- [Software Stack](#software-stack)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Workflow](#workflow)
  - [1. Hardware Characterization](#1-hardware-characterization)
  - [2. Quantization](#2-quantization)
  - [3. Pruning](#3-pruning)
  - [4. Custom TensorRT Plugin (C2f Fusion)](#4-custom-tensorrt-plugin-c2f-fusion)
  - [5. Evaluation & Comparison](#5-evaluation--comparison)
- [Datasets](#datasets)
- [Output & Results](#output--results)

---

## Project Overview

This thesis investigates compiler-aware optimisation of deep neural network
inference on NVIDIA Jetson (Orin family) embedded GPUs. The target model is
**YOLOv8n** (object detection). Three orthogonal axes are explored:

| Axis | Technique | Key files |
|---|---|---|
| **Precision** | INT8 / FP16 / FP32 quantization | `src/optimizer/quantization/` |
| **Architecture** | Manual structured pruning | `src/optimizer/pruning/` |
| **Kernel fusion** | Custom TensorRT plugin for model.2 C2f block | `src/plugins/c2f_m2/` |

All variants are benchmarked for latency, throughput, and mAP on COCO, with
statistical significance tested via bootstrap confidence intervals.

---

## Hardware Target

| Property | Value |
|---|---|
| Board | NVIDIA Orin Nano / NX / AGX |
| GPU | Ampere GA10B — **sm87** |
| RAM | 8 GB LPDDR5 unified (CPU + GPU) |
| JetPack | 5.x (CUDA 11.4) / 6.x (CUDA 12.x) |
| Measured BW | ~25 GB/s (DRAM), ~810 GFLOP/s (FP32 peak) |

Secondary targets: Raspberry Pi 5 (TVM), A100 / H100 (baseline reference).

---

## Software Stack

| Component | Version | Role |
|---|---|---|
| PyTorch | 2.x | Baseline inference, pruning, training |
| TensorRT | 8.6 / 10.x | Primary inference compiler |
| ONNX | 1.20+ | Model serialization & graph surgery |
| ONNX Runtime | 1.23-gpu | Alternative backend, INT8 calibration |
| TensorFlow / TFLite | 2.20 | Quantization pipeline, TFLite backend |
| Ultralytics | 8.4+ | YOLOv8 model loading, training, export |
| cuDNN | 8.6 / 9.x | Convolution primitives (plugin) |
| Apache TVM | 1.0+ | Experimental compiler backend |
| thop | — | FLOPs estimation |
| OpenCV | 4.13+ | Image I/O |

---

## Repository Structure

```
thesis2026-project/
├── src/
│   ├── optimizer/
│   │   ├── roofline.py                  # Roofline model: BW, peak FLOP/s, per-layer AI
│   │   ├── yolo_layer_analysis.py       # Per-layer param/activation/FLOPs inventory → CSV
│   │   ├── quantization/
│   │   │   ├── tensorrt_compile_yolo.py # Export YOLOv8 → TRT engine (INT8/FP16/FP32)
│   │   │   └── quantize_onnx.py         # Static INT8 via ONNX Runtime calibration
│   │   ├── evaluation/
│   │   │   ├── tensorrt_evaluation_indiv.py  # Benchmark single TRT engine (latency, FPS)
│   │   │   ├── tensorrt_evaluation_dual.py   # Side-by-side TRT engine comparison
│   │   │   ├── onnxrt_evaluation_dual.py     # ONNX Runtime evaluation
│   │   │   ├── pytorch_evaluation_dual.py    # PyTorch baseline comparison
│   │   │   ├── yolo_metrics.py               # Backend abstraction + NMS + mAP evaluation
│   │   │   └── bootstrap_comparison.py       # Bootstrap CI & p-value significance testing
│   │   ├── pruning/
│   │   │   ├── manual_yolov8_pruner.py  # CSV-driven structured pruning
│   │   │   ├── yolov8_pruning.py        # Magnitude/gradient-based pruning
│   │   │   └── yolov8_retrain.py        # Fine-tuning after pruning
│   │   ├── tvm_runner/
│   │   │   └── run.py                   # Apache TVM compilation (experimental)
│   │   └── utils/
│   │       ├── download_from_roboflow.py
│   │       ├── make_coco_subset.py       # Generate 0.1% / 1% / 10% COCO splits
│   │       └── upload_to_roboflow.py
│   └── plugins/
│       └── c2f_m2/                      # Custom TensorRT plugin — model.2 C2f fusion
│           ├── CMakeLists.txt
│           ├── c2f_m2_plugin.cpp        # TRT IPluginV2DynamicExt interface
│           ├── c2f_m2_fused.cu          # Fused CUDA kernel (tiled, shared-mem)
│           ├── c2f_m2_kernels.cu        # SiLU, slice, element-wise helpers
│           ├── c2f_m2_plugin_runtime.cu # Enqueue dispatch
│           ├── c2f_m2_winograd.cu       # Experimental Winograd path
│           ├── export_model2_weights.py # BN folding → .bin + .npz weight export
│           └── replace_model2_with_plugin.py  # ONNX graph surgery (onnx-graphsurgeon)
├── models/
│   ├── yolov8n.pt                       # Baseline PyTorch model
│   ├── yolov8n.onnx                     # ONNX export (opset 17)
│   ├── pruned_models/
│   ├── quantized_models/
│   │   ├── onnx/                        # INT8 / FP16 / FP32 ONNX
│   │   └── tensorrt/                    # TRT engines
│   ├── tensorrt_exports/
│   └── plugin_weights/                  # BN-folded weights for C2f plugin
├── datasets/
│   ├── coco/                            # Full COCO validation
│   ├── coco_subset/                     # 0.1% / 1% / 10% splits
│   └── EmbeddedAIProject-Skrews/        # Custom industrial screw dataset
├── notebooks/
│   ├── research_notebook.ipynb
│   ├── optimizations.ipynb
│   ├── tvm.ipynb
│   └── yolov8_recreation.ipynb
├── output/
│   └── pdfs/                            # Roofline plots, analysis figures
├── tests/
│   └── test_manual_yolov8_pruner.py
├── pyproject.toml
└── README.md
```

---

## Setup

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Plugin build (on Orin)

```bash
cd src/plugins/c2f_m2
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# → libc2f_m2_plugin.so
```

Requires: CMake ≥ 3.18, CUDA toolkit, TensorRT headers, cuDNN headers.

---

## Workflow

### 1. Hardware Characterization

Profile peak bandwidth, compute, and per-layer arithmetic intensity:

```bash
python src/optimizer/roofline.py
# → output/pdfs/roofline_<timestamp>.pdf
```

Inventory per-layer parameter and activation memory:

```bash
python src/optimizer/yolo_layer_analysis.py
# → CSV with param bytes, activation bytes, FLOPs per layer
```

### 2. Quantization

**ONNX static INT8** (calibrated on COCO subset):

```bash
python src/optimizer/quantization/quantize_onnx.py
# → models/quantized_models/onnx/yolov8n_opset17_int8.onnx
```

**TensorRT INT8 / FP16 / FP32 engine**:

```bash
python src/optimizer/quantization/tensorrt_compile_yolo.py
# → models/quantized_models/tensorrt/yolov8n_int8.engine
```

### 3. Pruning

Define targets in a CSV, then prune and retrain:

```bash
python -m src.optimizer.pruning.manual_yolov8_pruner \
  --targets "model.1,model.3.m.0" \
  --output models/pruned_models/pruned_v1.pt

python src/optimizer/pruning/yolov8_retrain.py \
  --model models/pruned_models/pruned_v1.pt
```

### 4. Custom TensorRT Plugin (C2f Fusion)

Fuses the **model.2 C2f block** (Cin=32 → Cout=64, 160×160) into a single
cuDNN-backed kernel, eliminating the NCHW↔NHWC copy overhead at the Split.

```bash
# Step 1 — export BN-folded weights
cd src/plugins/c2f_m2
python export_model2_weights.py
# → ../../../models/plugin_weights/model2_c2f_folded.bin

# Step 2 — patch the ONNX graph
python replace_model2_with_plugin.py
# → models/yolov8n_model2_plugin.onnx

# Step 3 — build the shared library (see Setup above)

# Step 4 — compile TensorRT engine with plugin
trtexec \
  --onnx=models/yolov8n_model2_plugin.onnx \
  --plugins=src/plugins/c2f_m2/build/libc2f_m2_plugin.so \
  --saveEngine=models/plugin_weights/yolov8_model2_plugin.engine \
  --fp32
```

### 5. Evaluation & Comparison

**Benchmark a single engine** (latency, FPS, percentiles):

```bash
python -m src.optimizer.evaluation.tensorrt_evaluation_indiv \
  --engine models/quantized_models/tensorrt/yolov8n_fp32.engine \
  --name fp32_baseline \
  --bench_runs 200 \
  --bench_warmup 30 \
  --out_json output/fp32_results.json
```

**With plugin**:

```bash
python -m src.optimizer.evaluation.tensorrt_evaluation_indiv \
  --engine models/plugin_weights/yolov8_model2_plugin.engine \
  --trt_plugin_so src/plugins/c2f_m2/build/libc2f_m2_plugin.so \
  --name c2f_plugin
```

**Side-by-side accuracy + latency comparison**:

```bash
python -m src.optimizer.evaluation.tensorrt_evaluation_dual \
  --engine_a models/quantized_models/tensorrt/yolov8n_fp32.engine \
  --engine_b models/quantized_models/tensorrt/yolov8n_int8.engine \
  --data datasets/coco/data.yaml
```

**Bootstrap statistical significance**:

```bash
python -m src.optimizer.evaluation.bootstrap_comparison \
  --model_a models/yolov8n.pt \
  --model_b models/quantized_models/onnx/yolov8n_opset17_int8.onnx \
  --backend_a torch \
  --backend_b onnxrt \
  --data datasets/coco/data.yaml \
  --bootstrap_iters 1000
```

---

## Datasets

| Dataset | Location | Purpose |
|---|---|---|
| COCO val2017 | `datasets/coco/` | Final accuracy evaluation |
| COCO 0.1% subset | `datasets/coco_subset/train_0_1percent/` | INT8 calibration |
| COCO 1% subset | `datasets/coco_subset/train_1percent/` | Fast pruning experiments |
| COCO 10% subset | `datasets/coco_subset/train_10percent/` | Full quantization trials |
| Skrews (Roboflow) | `datasets/EmbeddedAIProject-Skrews/` | Custom domain fine-tuning |

Generate COCO subsets:

```bash
python src/optimizer/utils/make_coco_subset.py
```

---

## Output & Results

All results land in `output/`:

| File | Description |
|---|---|
| `output/pdfs/roofline_*.pdf` | Roofline plots (vector PDF) |
| `output/*.json` | Benchmark results (latency percentiles, FPS) |
| `output/raspberry_pi_tvm_logs/` | TVM auto-tuning logs from Pi 5 |

Metrics captured per engine:

- **Latency**: mean, median, std, P90, P95, P99 (ms)
- **Throughput**: images/sec, batches/sec
- **Accuracy**: mAP50, mAP50-95, precision, recall (COCO protocol)
- **Statistical significance**: 95% bootstrap CI, two-sided p-value
