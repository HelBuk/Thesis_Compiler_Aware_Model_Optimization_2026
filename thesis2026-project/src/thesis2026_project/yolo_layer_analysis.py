import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn

# ----------------------------
# Helpers
# ----------------------------
def is_leaf_module(m: nn.Module) -> bool:
    # Leaf = no children modules (so we don't double-count big containers)
    return len(list(m.children())) == 0

def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def nested_tensor_bytes(obj: Any) -> int:
    """Sum bytes of all tensors contained in obj (tensor / tuple / list / dict)."""
    if torch.is_tensor(obj):
        return tensor_bytes(obj)
    if isinstance(obj, (tuple, list)):
        return sum(nested_tensor_bytes(o) for o in obj)
    if isinstance(obj, dict):
        return sum(nested_tensor_bytes(v) for v in obj.values())
    return 0

def nested_tensor_shapes(obj: Any, max_items: int = 4) -> List[Tuple[Tuple[int, ...], str]]:
    """Return a few (shape, dtype) pairs for debug/inspection."""
    out = []
    def walk(x: Any):
        nonlocal out
        if len(out) >= max_items:
            return
        if torch.is_tensor(x):
            out.append((tuple(x.shape), str(x.dtype).replace("torch.", "")))
            return
        if isinstance(x, (tuple, list)):
            for xx in x:
                walk(xx)
                if len(out) >= max_items:
                    return
        if isinstance(x, dict):
            for vv in x.values():
                walk(vv)
                if len(out) >= max_items:
                    return
    walk(obj)
    return out

def param_bytes(m: nn.Module) -> Tuple[int, int, int]:
    """(total, weights, bias) bytes for parameters owned by this module (not children)."""
    total = 0
    w = 0
    b = 0
    for name, p in m.named_parameters(recurse=False):
        pb = p.numel() * p.element_size()
        total += pb
        if "bias" in name:
            b += pb
        else:
            w += pb
    return total, w, b

def conv2d_flops(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> int:
    # FLOPs = 2 * B * Cout * Hout * Wout * (Cin/groups) * Kh * Kw
    b = x.shape[0]
    cout = y.shape[1]
    hout, wout = y.shape[2], y.shape[3]
    cin = x.shape[1]
    kh, kw = m.kernel_size
    g = m.groups
    return int(2 * b * cout * hout * wout * (cin // g) * kh * kw)

def linear_flops(m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> int:
    # FLOPs = 2 * batch * out_features * in_features
    in_features = m.in_features
    out_features = m.out_features
    batch = int(x.numel() // in_features)
    return int(2 * batch * out_features * in_features)

def format_bytes(n: int) -> str:
    # human-friendly
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    for u in units:
        if f < 1024.0 or u == units[-1]:
            return f"{f:.2f} {u}"
        f /= 1024.0
    return f"{f:.2f} GB"

# ----------------------------
# Stat record
# ----------------------------
@dataclass
class ModuleStat:
    name: str
    type: str
    params_total_b: int
    params_weight_b: int
    params_bias_b: int
    in_act_b: int
    out_act_b: int
    # FLOPs only for Conv/Linear; 0 otherwise
    flops: int
    # small summary of output shapes/dtypes (first few tensors)
    out_shapes: str

# ----------------------------
# Main profiler
# ----------------------------
def profile_yolo_modules(
    model: nn.Module,
    example_input: torch.Tensor,
    device: str = "cpu",
    leaf_only: bool = True,
) -> List[ModuleStat]:
    model = model.to(device).eval()

    name_of: Dict[nn.Module, str] = {m: n for n, m in model.named_modules()}

    stats: List[ModuleStat] = []
    handles = []

    def hook(m: nn.Module, inputs: Tuple[Any, ...], output: Any):
        # Optionally restrict to leaf modules to avoid double counting
        if leaf_only and not is_leaf_module(m):
            return

        mname = name_of.get(m, m.__class__.__name__)
        mtype = m.__class__.__name__

        in_b = sum(nested_tensor_bytes(x) for x in inputs)

        out_b = nested_tensor_bytes(output)

        p_total, p_w, p_b = param_bytes(m)

        # FLOPs (Conv2d/Linear only) based on first tensor input/output
        fl = 0
        x0 = inputs[0] if len(inputs) > 0 else None

        y0: Optional[torch.Tensor] = None
        if torch.is_tensor(output):
            y0 = output
        elif isinstance(output, (tuple, list)):
            for o in output:
                if torch.is_tensor(o):
                    y0 = o
                    break
        elif isinstance(output, dict):
            for o in output.values():
                if torch.is_tensor(o):
                    y0 = o
                    break

        if torch.is_tensor(x0) and torch.is_tensor(y0):
            if isinstance(m, nn.Conv2d):
                fl = conv2d_flops(m, x0, y0)
            elif isinstance(m, nn.Linear):
                fl = linear_flops(m, x0, y0)

        shapes = nested_tensor_shapes(output, max_items=4)
        shapes_str = "; ".join([f"{s} {dt}" for s, dt in shapes]) if shapes else "-"

        stats.append(
            ModuleStat(
                name=mname,
                type=mtype,
                params_total_b=p_total,
                params_weight_b=p_w,
                params_bias_b=p_b,
                in_act_b=in_b,
                out_act_b=out_b,
                flops=fl,
                out_shapes=shapes_str,
            )
        )

    # register hooks
    for m in model.modules():
        handles.append(m.register_forward_hook(hook))

    # run once
    with torch.no_grad():
        _ = model(example_input.to(device))

    for h in handles:
        h.remove()

    return stats

# ----------------------------
# Convenience: print summary
# ----------------------------
def summarize(stats: List[ModuleStat], topk: int = 15):
    total_params = sum(s.params_total_b for s in stats)
    total_out = sum(s.out_act_b for s in stats)  # not "peak", just sum of module outputs (double-counts lifetimes)
    total_flops = sum(s.flops for s in stats)

    peak_out = max((s.out_act_b for s in stats), default=0)
    peak_layer = max(stats, key=lambda s: s.out_act_b) if stats else None

    print(f"Modules profiled: {len(stats)}")
    print(f"Total params (bytes): {total_params} ({format_bytes(total_params)})")
    print(f"Sum of module outputs (bytes): {total_out} ({format_bytes(total_out)})  [double-counts lifetimes]")
    print(f"Peak single-module output: {peak_out} ({format_bytes(peak_out)})")
    if peak_layer:
        print(f"  Peak output layer: {peak_layer.name} ({peak_layer.type}) -> {peak_layer.out_shapes}")
    print(f"Total FLOPs (Conv+Linear only): {total_flops / 1e9:.3f} GFLOPs")

    print("\nTop layers by output activation bytes:")
    for s in sorted(stats, key=lambda x: x.out_act_b, reverse=True)[:topk]:
        print(
            f"{s.name:35s} {s.type:10s} "
            f"out={format_bytes(s.out_act_b):>10s}  "
            f"params={format_bytes(s.params_total_b):>10s}  "
            f"flops={s.flops/1e9:>7.3f}G  "
            f"shapes={s.out_shapes}"
        )

if __name__ == "__main__":
    from ultralytics import YOLO

    # Load model
    PATH_YOLO = '/Users/helbuk/Library/Mobile Documents/com~apple~CloudDocs/NTNU/NTNU_Thesis_2026/Compiler-Aware_Model_Optimization/Thesis_Compiler_Aware_Model_Optimization_2026/thesis2026-project/models/yolov8n.pt'
    y = YOLO(PATH_YOLO)
    net = y.model

    device = "cpu"  # "cpu", "cuda:0", or "mps"
    x = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    stats = profile_yolo_modules(net, x, device=device, leaf_only=True)
    summarize(stats, topk=20)

    # Optional: export to CSV
    import csv
    PATH_to_save = '/Users/helbuk/Library/Mobile Documents/com~apple~CloudDocs/NTNU/NTNU_Thesis_2026/Compiler-Aware_Model_Optimization/Thesis_Compiler_Aware_Model_Optimization_2026/thesis2026-project/tmp'
    with open(PATH_to_save + "yolov8n_module_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(stats[0]).keys()))
        w.writeheader()
        for s in stats:
            w.writerow(asdict(s))
    print("\nWrote yolov8n_module_stats.csv")
