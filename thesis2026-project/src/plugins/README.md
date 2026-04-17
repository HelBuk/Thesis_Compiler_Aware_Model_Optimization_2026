# YoloC2fM2_TRT — Custom TensorRT Plugin

Custom TensorRT plugin that replaces the **model.2 C2f block** of YOLOv8n with a
single fused CUDA/cuDNN node, eliminating the Split copy overhead and intermediate
reformatting tensors that appear in the baseline profiler trace.

## What it does

The C2f block (model.2) in YOLOv8n performs:

```
input [N, 32, 160, 160]
  └─► cv1  (1×1 conv+BN+SiLU, 32→64)
        ├── x1 = first  32 channels
        └── x2 = second 32 channels
              └─► m0.cv1 (3×3, 32→32) + SiLU
                    └─► m0.cv2 (3×3, 32→32) + SiLU
                          └─► m0out = m0.cv2_out + x2  (shortcut)
  concat [x1 | x2 | m0out]  → [N, 96, 160, 160]
  └─► cv2  (1×1 conv+BN+SiLU, 96→64)
output [N, 64, 160, 160]
```

In the default TRT engine, TRT inserts **Split copy nodes** and NCHW↔NHWC
reformatting kernels around this block. The plugin fuses all 7 ops into one node,
keeping intermediate buffers in a single workspace allocation.

**Key optimisations over vanilla TRT:**
- BN folded into conv weights at export time (no BN ops at inference)
- cuDNN descriptors cached once per engine load (not re-created per inference)
- TF32 tensor-op math enabled — Ampere TensorCores handle FP32 convolutions
- Heuristic cuDNN algorithm selection at engine-build time
- Vectorised float4 SiLU with `__expf` fast-math

---

## Requirements

| Dependency        | JetPack 6.x (TRT 10.x) |
|-------------------|------------------------|
| CUDA              | 12.x                   |
| TensorRT          | 10.x                  |
| cuDNN             | 9.x                   |
| CMake             | ≥ 3.18                |
| GCC               | 12                    |
| Python            | 3.10+                 |
| ultralytics       | 8.x                   |
| onnx-graphsurgeon | latest                |

Target GPU: **NVIDIA Jetson Orin Nano**. The CMakeLists sets
`CUDA_ARCHITECTURES 87`. 

### TRT version compatibility

The plugin uses `IPluginV2DynamicExt`. Two API differences exist between TRT versions:

- **`attachToContext(cudnnContext*, cublasContext*, IGpuAllocator*)`** — present in
  TRT 8.x, **removed** in TRT 10.x. Guarded with `#if NV_TENSORRT_MAJOR < 10`.
  On TRT 10.x the plugin creates its own cuDNN handle in `initialize()` instead.

- **`--explicitBatch`** flag in `trtexec` — removed in TRT 10. Omit it if on JetPack 6.x.

- **`cudnnGetConvolutionForwardAlgorithm_v7`** — deprecated in cuDNN 9.x but still
  present. A `#pragma GCC diagnostic` suppresses the warning.

If you are unsure which version is installed on the Orin, run:
```bash
dpkg -l | grep tensorrt
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

---

## File Layout

```
src/plugins/
├── README.md                        ← this file
└── c2f_m2/
    ├── c2f_m2_plugin.hpp            plugin class + ConvDescSet struct
    ├── c2f_m2_plugin.cpp            weight loading, descriptors, TRT interface
    ├── c2f_m2_plugin_runtime.cu     enqueue() — C2f forward pass
    ├── c2f_m2_kernels.cu            SiLU, slice, add_inplace kernels
    ├── export_model2_weights.py     fold BN → write .npz + .bin
    ├── replace_model2_with_plugin.py ONNX graph surgery
    └── CMakeLists.txt

models/plugin_weights/               (generated — not in repo)
    model2_c2f_folded.npz            NumPy archive (inspection / backup)
    model2_c2f_folded.bin            flat binary consumed by C++ plugin
    yolov8_model2_plugin.engine      serialised TRT engine

models/quantized_models/onnx/        (generated)
    yolov8n_opset17_fp32.onnx        baseline ONNX
    yolov8n_model2_plugin.onnx       ONNX with plugin node substituted
```

### Weight binary format (`.bin`)

```
magic        : 4 bytes   'C2FW'
num_tensors  : uint32
[ name_len(u32)  name(char[])  ndim(u32)  dims(u32[ndim])
  data_len(u32)  data(float32[]) ] × num_tensors
```

Tensors stored: `cv1_w`, `cv1_b`, `m0_cv1_w`, `m0_cv1_b`, `m0_cv2_w`,
`m0_cv2_b`, `cv2_w`, `cv2_b`, `shortcut`, `meta_cin`, `meta_cout`, `meta_halfc`.

---

## Step-by-step Workflow

### 1. Export BN-folded weights

Run from `src/plugins/c2f_m2/`:

```bash
python export_model2_weights.py
```

Produces `model2_c2f_folded.bin` (consumed by plugin) and `model2_c2f_folded.npz`
(human-readable backup). Expected output:

```
Cin=32  Cout=64  halfC=32
  cv1_w                shape=(64, 32, 1, 1)   dtype=float32
  cv2_w                shape=(64, 96, 1, 1)   dtype=float32
  ...
```

### 2. Patch the ONNX graph

```bash
python replace_model2_with_plugin.py
```

Removes all `/model.2/*` nodes and inserts a single `YoloC2fM2_TRT` node.

### 3. Build the shared library

```bash
cd src/plugins/c2f_m2
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
# Output: libc2f_m2_plugin.so
```

### 4. Build the TensorRT engine

> **Note:** `--explicitBatch` was removed in TRT 10. Omit it if using TRT ≥ 10.

```bash
trtexec \
  --onnx=./models/quantized_models/onnx/yolov8n_model2_plugin.onnx \
  --plugins=src/plugins/c2f_m2/build/libc2f_m2_plugin.so \
  --saveEngine=./models/plugin_weights/yolov8_model2_plugin_v4.engine \
  --profilingVerbosity=detailed \
  --dumpLayerInfo 
```

#### Verify plugin insertion

In the `--dumpLayerInfo` output, look for:

```
Layer: YoloC2fM2_TRT_0  Type: PLUGIN_V2  ...
```

If you see `/model.2/cv1/conv/Conv` as a separate layer, the plugin was **not**
inserted — check the `.so` path and that `weights_path` in the ONNX node is correct.

### 5. Run inference benchmark

```bash
python -m src.optimizer.evaluation.tensorrt_evaluation_indiv \
  --engine  ./models/plugin_weights/yolov8_model2_plugin_v4.engine \
  --name    model2_plugin \
  --device  cuda:0 \
  --imgsz   640 \
  --batch   1 \
  --bench_runs    2 \
  --bench_warmup   1 \
  --trt_plugin_so src/plugins/c2f_m2/build/libc2f_m2_plugin.so
```

---

## Profiling with Nsight Systems

### 1. Collect trace

```bash
nsys profile \
  --output src/profiling/trt_fp32_profile_plugin_v4 \
  --trace  cuda,nvtx,osrt \
  python -m src.optimizer.evaluation.tensorrt_evaluation_indiv \
    --engine  ./models/plugin_weights/yolov8_model2_plugin_v4.engine \
    --name    model2_plugin \
    --device  cuda:0 \
    --imgsz   640 \
    --batch   1 \
    --bench_runs    2 \
    --bench_warmup   1 \
    --trt_plugin_so src/plugins/c2f_m2/build/libc2f_m2_plugin.so
```

Output: `profiling/trt_fp32_profile_plugin.nsys-rep`

### 2. Convert to SQLite

```bash
nsys export \
  --type sqlite \
  src/profiling/trt_fp32_profile_plugin.nsys-rep
```

### 3. Export statistics

```bash
nsys stats \
  --report nvtx_sum \
  --report cuda_api_sum \
  --report cuda_gpu_kern_sum \
  --report cuda_gpu_mem_time_sum \
  --report cuda_gpu_mem_size_sum \
  src/profiling/trt_fp32_profile_plugin.sqlite \
  > src/profiling/trt_fp32_profile_plugin.txt 2>&1
```

### What to look for vs baseline

Compare against `src/profiling/trt_fp32_profile_16032026_v1.txt`:

| Metric | Baseline | Plugin (expected) |
|--------|----------|-------------------|
| `/model.2/Split_output_0 copy` | ~109 µs | absent |
| `/model.2/Split_output_1 copy` | ~91 µs  | absent |
| `Reformatting CopyNode … /model.2/…` | present | reduced |
| `YoloC2fM2_TRT_0` enqueue | — | single kernel group |

