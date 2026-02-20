import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from thop import profile


# ----------------------------
# 0) Device helpers
# ----------------------------
def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def dtype_nbytes(t: torch.Tensor) -> int:
    # bytes per element
    return t.element_size()


# ----------------------------
# 1) FLOPs + Bytes estimators (Conv2d, Linear)
# ----------------------------
def conv2d_flops(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> int:
    # FLOPs = 2 * Cout * Hout * Wout * (Cin/groups) * Kh * Kw
    # Batch multiplies the whole thing
    b = x.shape[0]
    cout = y.shape[1]
    hout, wout = y.shape[2], y.shape[3]
    cin = x.shape[1]
    kh, kw = m.kernel_size
    g = m.groups
    return int(2 * b * cout * hout * wout * (cin // g) * kh * kw)


def linear_flops(m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> int:
    # FLOPs = 2 * batch * out_features * in_features
    # x could be (B, in) or (B, *, in). Flatten batch dims:
    in_features = m.in_features
    out_features = m.out_features
    batch = int(x.numel() // in_features)
    return int(2 * batch * out_features * in_features)


def min_bytes_moved(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    # Lower bound: read input + read weights (+ bias) + write output
    # (Doesn't account for caches, im2col, extra reads, etc.)
    nb = x.element_size()
    bytes_in = x.numel() * nb
    bytes_out = y.numel() * nb
    bytes_w = 0
    for p in m.parameters(recurse=False):
        bytes_w += p.numel() * p.element_size()
    return int(bytes_in + bytes_w + bytes_out)


# ----------------------------
# 2) Per-layer measurement via hooks
# ----------------------------
@dataclass
class LayerStat:
    name: str
    type: str
    flops: int
    bytes_lb: int
    time_s: float  # measured wall time
    ai: float      # arithmetic intensity = flops/bytes
    perf_gflops: float  # achieved = flops/time


def profile_layers(model: nn.Module, example_input: torch.Tensor, device: str) -> List[LayerStat]:
    model = model.to(device)
    model.eval()

    stats: List[LayerStat] = []
    handles = []
    name_of: Dict[nn.Module, str] = {m: n for n, m in model.named_modules()}
    supported = (nn.Conv2d, nn.Linear)

    # we measure time inside forward hook by syncing around the module call
    start_times: Dict[int, float] = {}

    def pre_hook(m: nn.Module, inputs: Tuple[Any, ...]):
        if not isinstance(m, supported):
            return
        sync(device)
        start_times[id(m)] = time.perf_counter()

    def post_hook(m: nn.Module, inputs: Tuple[Any, ...], output: Any):
        if not isinstance(m, supported):
            return
        sync(device)
        t1 = time.perf_counter()
        t0 = start_times.pop(id(m), None)
        if t0 is None:
            return

        x = inputs[0]
        y = output
        if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            return

        if isinstance(m, nn.Conv2d):
            fl = conv2d_flops(m, x, y)
        elif isinstance(m, nn.Linear):
            fl = linear_flops(m, x, y)
        else:
            fl = 0

        b = min_bytes_moved(m, x, y)
        ai = fl / b if b > 0 else 0.0
        dt = max(t1 - t0, 1e-12)
        perf = (fl / dt) / 1e9  # GFLOP/s

        stats.append(
            LayerStat(
                name=name_of.get(m, m.__class__.__name__),
                type=m.__class__.__name__,
                flops=fl,
                bytes_lb=b,
                time_s=dt,
                ai=ai,
                perf_gflops=perf
            )
        )

    for m in model.modules():
        handles.append(m.register_forward_pre_hook(pre_hook))
        handles.append(m.register_forward_hook(post_hook))

    with torch.no_grad():
        _ = model(example_input.to(device))

    for h in handles:
        h.remove()

    return stats


# ----------------------------
# 3) Hardware calibration: peak BW + peak compute (quick, portable)
# ----------------------------
def measure_bandwidth_gbs(device: str, size_mb: int = 256, iters: int = 50, dtype=torch.float32) -> float:
    # measure device-to-device copy bandwidth: y.copy_(x)
    nbytes = size_mb * 1024 * 1024
    n = nbytes // torch.tensor([], dtype=dtype).element_size()

    x = torch.empty(n, device=device, dtype=dtype)
    y = torch.empty(n, device=device, dtype=dtype)

    # warmup
    for _ in range(5):
        y.copy_(x)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    sync(device)
    t1 = time.perf_counter()

    total_bytes = iters * nbytes
    bw = (total_bytes / (t1 - t0)) / 1e9  # GB/s
    return bw


def measure_peak_gflops(device: str, m: int = 2048, k: int = 2048, n: int = 2048, iters: int = 50,
                        dtype=torch.float16) -> float:
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)

    # warmup
    for _ in range(5):
        _ = a @ b
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    sync(device)
    t1 = time.perf_counter()

    flops_per = 2 * m * k * n
    gflops = (iters * flops_per / (t1 - t0)) / 1e9
    return gflops

def profile_model(model: nn.Module, example_input: torch.Tensor, device: str, iter: int) -> LayerStat:
    def tensor_nbytes(obj: Any) -> int:
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (tuple, list)):
            return sum(tensor_nbytes(o) for o in obj)
        if isinstance(obj, dict):
            return sum(tensor_nbytes(v) for v in obj.values())
        return 0

    model = model.to(device).eval()
    example_input = example_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(30):
            _ = model(example_input)

    sync(device)

    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(iter):
            out = model(example_input)

    sync(device)

    t1 = time.perf_counter()

    elapsed_s = (t1 - t0) / iter
    print(f"Latency: {elapsed_s*1e3:.3f} ms")

    # THOP FLOPs for reference (use the actual input)
    flops, params = profile(model, inputs=(example_input,), verbose=False)

    # Full-model lower-bound bytes: input + ALL weights + output
    bytes_in = example_input.numel() * example_input.element_size()
    bytes_w = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_out = tensor_nbytes(out)
    b = int(bytes_in + bytes_w + bytes_out)

    ai = flops / b if b > 0 else 0.0
    perf = (flops / max(elapsed_s, 1e-12)) / 1e9

    return LayerStat(
        name="full_model",
        type=model.__class__.__name__,
        flops=int(flops),
        bytes_lb=b,
        time_s=elapsed_s,
        ai=ai,
        perf_gflops=perf,
    )


# ----------------------------
# 4) Roofline plot
# ----------------------------
def plot_roofline(stats: List[LayerStat], stats_model: LayerStat, peak_gflops: float, peak_bw_gbs: float, title: str = "Roofline"):
    # Convert to arrays
    ai = [s.ai for s in stats if s.flops > 0 and s.bytes_lb > 0]
    perf = [s.perf_gflops for s in stats if s.flops > 0 and s.bytes_lb > 0]

    # Build roofline curves
    ai_min = max(min(ai) / 2, 1e-6)
    ai_max = max(ai) * 2
    xs = torch.logspace(math.log10(ai_min), math.log10(ai_max), steps=200).tolist()
    mem_roof = [peak_bw_gbs * x for x in xs]
    comp_roof = [peak_gflops for _ in xs]
    roof = [min(mem_roof[i], comp_roof[i]) for i in range(len(xs))]

    plt.figure(figsize=(9, 6))
    plt.loglog(xs, mem_roof, label=f"Memory roof: BW={peak_bw_gbs:.1f} GB/s")
    plt.loglog(xs, comp_roof, label=f"Compute roof: Peak={peak_gflops:.1f} GFLOP/s")
    plt.loglog(xs, roof, label="Roofline", linewidth=2)

    plt.scatter(ai,perf,color='blue',label='Layer Performance')

    plt.scatter(stats_model.ai, stats_model.perf_gflops, color="green", label='Full Model (THOP) Performance')

    total_flops = sum(s.flops for s in stats)
    total_bytes = sum(s.bytes_lb for s in stats)
    elapsed_s = stats_model.time_s  # from profile_model

    ai_model = total_flops / total_bytes
    perf_model = (total_flops / elapsed_s) / 1e9

    plt.scatter(ai_model, perf_model, color="red", label='Conv2d+Linear Layers Full Model')


    plt.xlabel("Arithmetic Intensity (FLOPs / byte)  [lower-bound bytes]")
    plt.ylabel("Achieved Performance (GFLOP/s)")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from ultralytics import YOLO
    # PATH_YOLO = 'yolov8n.pt'
    PATH_YOLO = '/Users/helbuk/Library/Mobile Documents/com~apple~CloudDocs/NTNU/NTNU_Thesis_2026/Compiler-Aware_Model_Optimization/Thesis_Compiler_Aware_Model_Optimization_2026/thesis2026-project/models/yolov8n.pt'

    y = YOLO(PATH_YOLO)
    net = y.model

    device = "mps"  # "cpu", "cuda:0", or "mps"
    x = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    if device == 'cpu':
        torch.set_num_threads(os.cpu_count())

    print("Threads:", torch.get_num_threads())
    print("Interop threads:", torch.get_num_interop_threads())


    bw = measure_bandwidth_gbs(device=device, size_mb=1024, iters=300, dtype=torch.float32)
    print("Measure Bandwidth:", bw, "GB/s")

    peak = measure_peak_gflops(device=device, m=2048, k=2048, n=2048, iters=300,
                               dtype=torch.float32)
    print("Measure Peak Compute performance:", peak, "GFlops")

    stats = profile_layers(net, x, device=device)

    stats_model = profile_model(net, x, device=device, iter=50)
    plot_roofline(stats, stats_model, peak_gflops=peak, peak_bw_gbs=bw, title=f"Roofline ({device}) - YOLOv8")
    print(f"Full model: {stats_model}")

    for s in stats:
        print(f"{s.name:40s} {s.type:8s} time={s.time_s*1e3:7.2f} ms  AI={s.ai:7.3f}  perf={s.perf_gflops:8.1f} GF/s")

