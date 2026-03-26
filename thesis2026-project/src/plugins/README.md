This project implements a custom TensorRT plugin:
	•	Plugin name: YoloC2fM2_TRT
	•	Purpose: Replace YOLOv8 C2f block with a fused CUDA implementation
	•	Backend: TensorRT + CUDA
	•	Profiling: Nsight Systems (nsys)

Requirements
	•	CUDA (≥ 12.x)
	•	TensorRT
	•	CMake ≥ 3.18
	•	GCC (tested with 11)


# Export .pt Weights

```bash
export_model2_weights.py
```

# Replace layer model.2 with custom plugin in ONNX Graph

```bash
replace_model2_with_plugin.py
```

# Build Steps

```bash
cd src/plugins/c2f_m2

rm -rf build
mkdir build
cd build

cmake ..
make -j
```

# Output

```bash
libc2f_m2_plugin.so
```

# Build TensorRT Engine

```bash
trtexec \
  --onnx=../models/quantized_models/onnx/yolov8n_model2_plugin.onnx \
  --plugins=src/plugins/c2f_m2/build/libc2f_m2_plugin.so \
  --saveEngine=../models/plugin_weights/yolov8_model2_plugin.engine \
  --explicitBatch \
  --profilingVerbosity=detailed \
  --dumpLayerInfo
```

# Notes
	•	--plugins is required → loads custom plugin
	•	--dumpLayerInfo verifies plugin insertion

# Run Inference

```bash
python -m optimizer.evaluation.tensorrt_evaluation_indiv \
  --engine ../models/plugin_weights/yolov8_model2_plugin_v3.engine \
  --name model2_plugin_custom_v3 \
  --device cuda:0 \
  --imgsz 640 \
  --batch 1 \
  --bench_runs 20 \
  --bench_warmup 5 \
  --trt_plugin_so src/plugins/c2f_m2/build/libc2f_m2_plugin.so
```
# Profiling with Nsight Systems

1. Collect profile

```bash
nsys profile \
  -o profiling/trt_fp32_profile_plugin_v3 \
  --trace=cuda,nvtx,osrt \
  python -m optimizer.evaluation.tensorrt_evaluation_indiv \
    --engine ../models/plugin_weights/yolov8_model2_plugin_v3.engine \
    --name model2_plugin_custom_v3 \
    --device cuda:0 \
    --imgsz 640 \
    --batch 1 \
    --bench_runs 2 \
    --bench_warmup 1 \
    --trt_plugin_so ./plugins/c2f_m2/build/libc2f_m2_plugin.so
```

Output: .nsys-rep

2. Export to SQLite

```bash
nsys export \
  --type sqlite \
  profiling/trt_fp32_profile_plugin_v3.nsys-rep

```

3. Export stats
```bash
nsys stats   --report nvtx_sum   --report cuda_api_sum   --report cuda_gpu_kern_sum   --report cuda_gpu_mem_time_sum   --report cuda_gpu_mem_size_sum   profiling/trt_fp32_profile_plugin_v3.sqlite   > profiling/trt_fp32_profile_plugin_v3.txt 2>&1
```