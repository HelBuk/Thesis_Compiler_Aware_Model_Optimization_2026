# Orin Nano & Raspberry Pi 5 (Useful Commands):

1. Enable maximum Jetson performance

```bash 
sudo nvpmodel -m 0
```

2. To force maximum Jetson frequency:

```bash 
sudo jetson_clocks
```

3. Observe RAM usage:

```bash 
watch free -h
```

4. Monitor real frequencies:

```bash
sudo tegrastats
```

5. Before profiling:

```bash
    sudo sync
    sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
    sudo swapoff -a && sudo swapon -a
```

6. Kill leftover processes (Python):

```bash
    ps aux | grep python
    kill -9 PID
```

7. NVIDIA Nsight Compute NCU (timeline and GPU usega): 

```bash
sudo HOME=HOME_DIR /usr/local/cuda-12.6/bin/ncu \ # Check with: readlink -f "$(which ncu)"
  --set basic \         # Basic/full
  --launch-skip 50 \    # Ignore the first 50 kernel launches.
  --launch-count 20 \   # Only profile the next 20 launches after the skip.
  -o trt_fp32_basic \   # Name of the file
  PYTHONPATH \          # Check with "which python" in chosen env
  -m optimizer.evaluation.onnxrt_evaluation_dual \
    --model ../models/quantized_models/onnx/yolov8n_opset17_fp32.onnx \
    --provider trt \
    --bench_runs 2 \
    --bench_warmup 1 \
    --seed 42
```

7.1 Print NCU text summaries:
```bash
/usr/local/cuda-12.6/bin/ncu \ # Check with: readlink -f "$(which ncu)"
--import FILE_NAME.ncu-rep \
--page details
```

8. NSYS Profiling: 

```bash
nsys profile --trace=cuda,nvtx,osrt -o FILE_NAME python -m SCRIPT_NAME
```

8.1 NSYS stats:

```bash
nsys stats trt_fp32_prof.nsys-rep
```

Or with Nsight Systems GUI:
```bash
nsys-ui
```