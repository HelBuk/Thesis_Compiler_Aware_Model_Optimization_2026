# yolov8_pruning.py
# Research-backed base:
#   DepGraph / Torch-Pruning (CVPR 2023)
#
# Safer defaults for YOLOv8-specific graph quirks:
# - Patches C2f -> forward_split for stable dependency tracing
# - Ignores task heads + grouped/depthwise convs
# - Optionally freezes SPPF in first pass
# - Uses robust output_transform for Ultralytics outputs
#
# Usage:
# python -m optimizer.pruning.yolov8_pruning \
#   --weights ../models/yolov8n.pt \
#   --out ../models/yolov8n_pruned.pth \
#   --imgsz 640 \
#   --device cpu \
#   --pruning-ratio 0.15 \
#   --iterative-steps 8 \
#   --round-to 1

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
# add imports
from ultralytics.nn.modules import C2f, Conv, Bottleneck


HEAD_LIKE_NAMES = {
    "detect",
    "segment",
    "pose",
    "obb",
    "classify",
    "rtdetrdecoder",
    "worlddetect",
    "dfl",
}

def _copy_conv_bn(dst: Conv, src: Conv, out_slice: slice | None = None) -> None:
    if out_slice is None:
        dst.load_state_dict(src.state_dict(), strict=True)
        return

    s = out_slice
    dst.conv.weight.data.copy_(src.conv.weight.data[s].clone())
    if src.conv.bias is not None and dst.conv.bias is not None:
        dst.conv.bias.data.copy_(src.conv.bias.data[s].clone())

    if isinstance(dst.bn, nn.BatchNorm2d) and isinstance(src.bn, nn.BatchNorm2d):
        dst.bn.weight.data.copy_(src.bn.weight.data[s].clone())
        dst.bn.bias.data.copy_(src.bn.bias.data[s].clone())
        dst.bn.running_mean.data.copy_(src.bn.running_mean.data[s].clone())
        dst.bn.running_var.data.copy_(src.bn.running_var.data[s].clone())
        dst.bn.num_batches_tracked.data.copy_(src.bn.num_batches_tracked.data)


class C2fPrunable(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1a = Conv(c1, self.c, 1, 1)
        self.cv1b = Conv(c1, self.c, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1a(x), self.cv1b(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    @classmethod
    def from_c2f(cls, old: C2f) -> "C2fPrunable":
        c1 = old.cv1.conv.in_channels
        c2 = old.cv2.conv.out_channels
        n = len(old.m)
        e = float(old.c) / float(c2)
        shortcut = bool(getattr(old.m[0], "add", False)) if n else False
        g = int(getattr(old.m[0].cv2.conv, "groups", 1)) if n else 1

        new = cls(c1=c1, c2=c2, n=n, shortcut=shortcut, g=g, e=e)
        _copy_conv_bn(new.cv1a, old.cv1, slice(0, old.c))
        _copy_conv_bn(new.cv1b, old.cv1, slice(old.c, 2 * old.c))
        new.m.load_state_dict(old.m.state_dict(), strict=True)
        new.cv2.load_state_dict(old.cv2.state_dict(), strict=True)
        return new


def replace_c2f(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, C2f):
            new_child = C2fPrunable.from_c2f(child).to(child.cv1.conv.weight.device)
            new_child = new_child.to(dtype=child.cv1.conv.weight.dtype)
            setattr(module, name, new_child)
            replaced += 1
        else:
            replaced += replace_c2f(child)
    return replaced


def get_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but not available: {device_str}")
    return torch.device(device_str)


def build_example_inputs(imgsz: int, device: torch.device) -> torch.Tensor:
    return torch.randn(1, 3, imgsz, imgsz, device=device)


def patch_c2f_for_depgraph(model: nn.Module) -> int:
    patched = 0
    for m in model.modules():
        if m.__class__.__name__.lower() == "c2f" and hasattr(m, "forward_split"):
            m.forward = m.forward_split
            patched += 1
    return patched


def collect_ignored_layers(model: nn.Module, freeze_sppf: bool = True) -> list[nn.Module]:
    ignored: set[nn.Module] = set()

    for m in model.modules():
        name = m.__class__.__name__.lower()

        if name in HEAD_LIKE_NAMES:
            ignored.update(m.modules())
            continue

        if freeze_sppf and name == "sppf":
            ignored.update(m.modules())
            continue

        # Depthwise/grouped convs are fragile for first pruning pass in YOLOv8.
        if isinstance(m, nn.Conv2d) and m.groups != 1:
            ignored.add(m)

    return list(ignored)


def _first_tensor(x: Any) -> torch.Tensor | None:
    if torch.is_tensor(x):
        return x
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    return None


def output_transform(out: Any) -> torch.Tensor:
    t = _first_tensor(out)
    if t is None:
        raise TypeError(f"Unsupported model output type for pruning: {type(out)}")
    return t


def count_model_stats(model: nn.Module, example_inputs: torch.Tensor) -> tuple[float, float]:
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return float(macs), float(params)


def make_pruner(
    model: nn.Module,
    example_inputs: torch.Tensor,
    pruning_ratio: float,
    iterative_steps: int,
    global_pruning: bool,
    round_to: int,
    freeze_sppf: bool,
) -> tp.pruner.BasePruner:
    ignored_layers = collect_ignored_layers(model, freeze_sppf=freeze_sppf)
    importance = tp.importance.GroupMagnitudeImportance(p=2)

    return tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_ratio,
        global_pruning=global_pruning,
        ignored_layers=ignored_layers,
        round_to=round_to,
        root_module_types=[nn.Conv2d],
        output_transform=output_transform,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 .pt weights")
    parser.add_argument("--out", type=str, required=True, help="Output path for pruned model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pruning-ratio", type=float, default=0.15, help="0 < ratio < 1")
    parser.add_argument("--iterative-steps", type=int, default=8, help="More steps = safer pruning")
    parser.add_argument("--round-to", type=int, default=1, help="Start with 1 for stability")
    parser.add_argument("--global-pruning", action="store_true", help="Usually less stable on YOLOv8")
    parser.add_argument(
        "--prune-sppf",
        action="store_true",
        help="If set, allows pruning SPPF; default keeps SPPF frozen for stability",
    )
    args = parser.parse_args()

    if not (0.0 < args.pruning_ratio < 1.0):
        raise ValueError(f"--pruning-ratio must be in (0,1), got {args.pruning_ratio}")
    if args.iterative_steps < 1:
        raise ValueError(f"--iterative-steps must be >= 1, got {args.iterative_steps}")
    if args.round_to < 1:
        raise ValueError(f"--round-to must be >= 1, got {args.round_to}")

    device = get_device(args.device)

    yolo = YOLO(args.weights)
    model = yolo.model.eval().to(device)

    num_replaced = replace_c2f(model)
    print(f"[Info] Replaced C2f -> C2fPrunable: {num_replaced}")

    # patched_c2f = patch_c2f_for_depgraph(model)
    # if patched_c2f:
    #     print(f"[Info] Patched C2f forward -> forward_split on {patched_c2f} module(s)")

    for p in model.parameters():
        p.requires_grad_(True)

    example_inputs = build_example_inputs(args.imgsz, device)

    # Sanity forward before pruning.
    with torch.no_grad():
        _ = model(example_inputs)

    base_macs, base_params = count_model_stats(model, example_inputs)
    print(f"[Before] MACs={base_macs/1e9:.3f}G  Params={base_params/1e6:.3f}M")

    pruner = make_pruner(
        model=model,
        example_inputs=example_inputs,
        pruning_ratio=args.pruning_ratio,
        iterative_steps=args.iterative_steps,
        global_pruning=args.global_pruning,
        round_to=args.round_to,
        freeze_sppf=not args.prune_sppf,
    )

    for step in range(args.iterative_steps):
        pruner.step()

        # Fail early if graph got broken in this step.
        with torch.no_grad():
            _ = model(example_inputs)

        macs, params = count_model_stats(model, example_inputs)
        print(
            f"[Step {step+1}/{args.iterative_steps}] "
            f"MACs={macs/1e9:.3f}G  Params={params/1e6:.3f}M"
        )

    model.zero_grad(set_to_none=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu().eval()

    raw_out = out_path.with_suffix(".pth")
    torch.save(model_cpu, raw_out)

    ckpt_out = out_path.with_suffix(".pt")
    ckpt = {
        "model": model_cpu,
        "epoch": -1,
        "train_args": getattr(yolo, "overrides", {}),
    }
    torch.save(ckpt, ckpt_out)

    print(f"[Saved raw module] {raw_out}")
    print(f"[Saved checkpoint] {ckpt_out}")


if __name__ == "__main__":
    main()
