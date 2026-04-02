import math
import time
import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import statistics
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from thop import profile
from ultralytics import YOLO


# ----------------------------
# 0) Device helpers
# ----------------------------
def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    # CPU: ops are synchronous — no-op is correct


# ----------------------------
# 1) FLOPs + Bytes estimators
# ----------------------------
def conv2d_flops(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> int:
    b = x.shape[0]
    cout = y.shape[1]
    hout, wout = y.shape[2], y.shape[3]
    cin = x.shape[1]
    kh, kw = m.kernel_size
    g = m.groups
    return int(2 * b * cout * hout * wout * (cin // g) * kh * kw)


def linear_flops(m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> int:
    in_features = m.in_features
    out_features = m.out_features
    batch = int(x.numel() // in_features)
    return int(2 * batch * out_features * in_features)


def min_bytes_moved(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    nb = x.element_size()
    bytes_in = x.numel() * nb
    bytes_out = y.numel() * nb
    bytes_w = sum(p.numel() * p.element_size() for p in m.parameters(recurse=False))
    return int(bytes_in + bytes_w + bytes_out)


# ----------------------------
# 2) Stat record
# ----------------------------
@dataclass
class LayerStat:
    name: str
    type: str
    flops: int
    bytes_lb: int
    time_s: float
    ai: float
    perf_gflops: float


# ----------------------------
# 3) Per-layer profiling
#    FIX: run hooked forward N times, take median per layer
# ----------------------------
def profile_layers(
    model: nn.Module,
    example_input: torch.Tensor,
    device: str,
    warmup: int = 30,
    timed_runs: int = 20,       # <-- NEW: median over this many runs
) -> List[LayerStat]:
    model = model.to(device).eval()
    x = example_input.to(device)

    # Warmup — forces CPU caches + kernel scheduling to settle
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    sync(device)

    # Collect raw timings: name -> list of (flops, bytes, dt)
    raw: Dict[str, list] = {}
    name_of: Dict[nn.Module, str] = {m: n for n, m in model.named_modules()}
    supported = (nn.Conv2d, nn.Linear)
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

        x_in = inputs[0]
        y_out = output
        if not (isinstance(x_in, torch.Tensor) and isinstance(y_out, torch.Tensor)):
            return

        fl = conv2d_flops(m, x_in, y_out) if isinstance(m, nn.Conv2d) else linear_flops(m, x_in, y_out)
        b  = min_bytes_moved(m, x_in, y_out)
        dt = max(t1 - t0, 1e-12)

        name = name_of.get(m, m.__class__.__name__)
        if name not in raw:
            raw[name] = {"type": m.__class__.__name__, "flops": fl, "bytes": b, "times": []}
        raw[name]["times"].append(dt)

    handles = []
    for m in model.modules():
        handles.append(m.register_forward_pre_hook(pre_hook))
        handles.append(m.register_forward_hook(post_hook))

    with torch.no_grad():
        for _ in range(timed_runs):
            _ = model(x)

    for h in handles:
        h.remove()

    stats = []
    for name, d in raw.items():
        dt = statistics.median(d["times"])     # median across runs
        fl = d["flops"]
        b  = d["bytes"]
        ai   = fl / b if b > 0 else 0.0
        perf = (fl / dt) / 1e9
        stats.append(LayerStat(name=name, type=d["type"],
                               flops=fl, bytes_lb=b,
                               time_s=dt, ai=ai, perf_gflops=perf))
    return stats


# ----------------------------
# 4) Hardware calibration
#    FIX: use matrix-multiply for bandwidth too so thread count matches
# ----------------------------
def measure_bandwidth_gbs(device: str, size_mb: int = 512, iters: int = 100,
                           dtype=torch.float32) -> float:
    """
    Estimates effective memory bandwidth using a DAXPY-style op (c = a + b).
    This uses the same thread pool as the compute benchmark, giving a
    consistent ridge point for the roofline model.
    Reads 2 × size_mb, writes 1 × size_mb  →  3 × size_mb total per iter.
    """
    nbytes = size_mb * 1024 * 1024
    n = nbytes // torch.tensor([], dtype=dtype).element_size()
    a = torch.randn(n, device=device, dtype=dtype)
    b = torch.randn(n, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        c = a + b
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        c = a + b          # 2 reads + 1 write = 3 × nbytes
    sync(device)
    t1 = time.perf_counter()

    total_bytes = iters * 3 * nbytes
    return (total_bytes / (t1 - t0)) / 1e9


def measure_peak_gflops(device: str, m: int = 2048, k: int = 2048, n: int = 2048,
                        iters: int = 100, dtype=torch.float32) -> float:
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)

    for _ in range(5):
        _ = a @ b
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    sync(device)
    t1 = time.perf_counter()

    return (iters * 2 * m * k * n / (t1 - t0)) / 1e9


# ----------------------------
# 5) Full-model profiling
# ----------------------------
def profile_model(model: nn.Module, example_input: torch.Tensor,
                  device: str, iters: int = 50) -> LayerStat:
    def tensor_nbytes(obj: Any) -> int:
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (tuple, list)):
            return sum(tensor_nbytes(o) for o in obj)
        if isinstance(obj, dict):
            return sum(tensor_nbytes(v) for v in obj.values())
        return 0

    model = model.to(device).eval()
    x = example_input.to(device)

    with torch.no_grad():
        for _ in range(30):
            _ = model(x)
    sync(device)

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            out = model(x)
            sync(device)
            times.append(time.perf_counter() - t0)

    elapsed_s = statistics.median(times)
    print(f"Latency (median): {elapsed_s * 1e3:.3f} ms")

    flops, _ = profile(model, inputs=(x,), verbose=False)
    bytes_in  = x.numel() * x.element_size()
    bytes_w   = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_out = tensor_nbytes(out)
    b = int(bytes_in + bytes_w + bytes_out)

    ai   = flops / b if b > 0 else 0.0
    perf = (flops / max(elapsed_s, 1e-12)) / 1e9

    return LayerStat(name="full_model", type=model.__class__.__name__,
                     flops=int(flops), bytes_lb=b,
                     time_s=elapsed_s, ai=ai, perf_gflops=perf)


# ----------------------------
# 6) Roofline plot
# ----------------------------
def plot_roofline(stats: List[LayerStat], stats_model: LayerStat,
                  peak_gflops: float, peak_bw_gbs: float,
                  title: str = "Roofline", save_path: str = None):
    ai   = [s.ai          for s in stats if s.flops > 0 and s.bytes_lb > 0]
    perf = [s.perf_gflops for s in stats if s.flops > 0 and s.bytes_lb > 0]

    ai_min = max(min(ai) / 2, 1e-6)
    ai_max = max(ai) * 2
    xs = torch.logspace(math.log10(ai_min), math.log10(ai_max), steps=200).tolist()
    mem_roof  = [peak_bw_gbs * x for x in xs]
    comp_roof = [peak_gflops     for _ in xs]
    roof      = [min(mem_roof[i], comp_roof[i]) for i in range(len(xs))]

    ridge = peak_gflops / peak_bw_gbs

    plt.figure(figsize=(12, 7))
    plt.loglog(xs, mem_roof,  linestyle="--", linewidth=2, label=f"Memory roof  {peak_bw_gbs:.1f} GB/s")
    plt.loglog(xs, comp_roof, linestyle="--", linewidth=2, label=f"Compute roof {peak_gflops:.1f} GFLOP/s")
    plt.loglog(xs, roof, "k-", linewidth=2.5, label="Roofline")
    plt.axvline(ridge, color="gray", linestyle=":", linewidth=1.5, label=f"Ridge point {ridge:.1f}")

    plt.scatter(ai, perf, color="blue", zorder=5, s=60, label="Layers")
    plt.scatter(stats_model.ai, stats_model.perf_gflops, color="green",
                zorder=6, s=200, marker="*", label="Full model (THOP)")

    total_flops = sum(s.flops for s in stats)
    total_bytes = sum(s.bytes_lb for s in stats)
    ai_agg   = total_flops / total_bytes
    perf_agg = (total_flops / stats_model.time_s) / 1e9
    plt.scatter(ai_agg, perf_agg, color="red", zorder=6, s=200, marker="D",
                label="Conv+Linear aggregate")

    plt.xlabel("Arithmetic Intensity (FLOP/byte)  [lower-bound bytes]", fontsize=16)
    plt.ylabel("Achieved Performance (GFLOP/s)", fontsize=16)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.tick_params(axis="both", labelsize=13)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(fontsize=13, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


# ----------------------------
# 7) Entry point
# ----------------------------
if __name__ == "__main__":
    # Force PyTorch + oneDNN to use all physical cores
    # so matmul benchmark and conv2d kernels use the same thread count
    num_cores = os.cpu_count() or 4
    torch.set_num_threads(num_cores)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_cores)

    print(f"PyTorch threads: {torch.get_num_threads()}")  # should now print 4

    PATH_YOLO = "./models/yolov8n.pt"
    device    = "cpu"

    y   = YOLO(PATH_YOLO)
    net = y.model

    x = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    print(f"PyTorch threads: {torch.get_num_threads()}")

    print("Measuring bandwidth...")
    bw = measure_bandwidth_gbs(device=device, size_mb=512, iters=500, dtype=torch.float32)
    print(f"Bandwidth:    {bw:.2f} GB/s")

    print("Measuring peak compute...")
    peak = measure_peak_gflops(device=device, m=2048, k=2048, n=2048, iters=100, dtype=torch.float32)
    print(f"Peak compute: {peak:.2f} GFLOP/s")

    print("Profiling full model...")
    stats_model = profile_model(net, x, device=device, iters=50)

    print("Profiling layers (with warmup + median over 20 runs)...")
    stats = profile_layers(net, x, device=device, warmup=30, timed_runs=20)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_roofline(stats, stats_model, peak_gflops=peak, peak_bw_gbs=bw,
                  title=f"Roofline (cpu) — YOLOv8n",
                  save_path=f"./output/roofline_pi5_cpu_{ts}.pdf")

    print(f"\nFull model: {stats_model}\n")
    print(f"{'Layer':<40} {'Type':<8} {'time(ms)':>9} {'AI':>8} {'perf(GF/s)':>12}")
    print("-" * 82)
    for s in stats:
        print(f"{s.name:<40} {s.type:<8} {s.time_s*1e3:9.2f} {s.ai:8.3f} {s.perf_gflops:12.1f}")
