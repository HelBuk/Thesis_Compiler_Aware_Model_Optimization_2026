# manual_yolov8_pruner.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
from dataclasses import dataclass

import yaml
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.models.yolo.detect import DetectionTrainer

RX_C2F_CV1 = re.compile(r"^model\.(\d+)\.cv1$")
RX_SPPF_CV1 = re.compile(r"^model\.(\d+)\.cv1$")

@dataclass
class ResolvedTargets:
    block_targets: list[int]
    c2f_hidden_targets: list[int]
    sppf_hidden_targets: list[int]
    unknown: list[str]


@dataclass
class Change:
    path: str
    old_out: int
    new_out: int
    old_params: int
    new_params: int


def params_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def fp32_param_mb(m: nn.Module) -> float:
    return params_count(m) * 4 / (1024**2)

def _detect_layer_idx(model: nn.Module) -> int | None:
    for i, m in enumerate(model.model):
        if m.__class__.__name__.lower() == "detect":
            return i
    return None

### --------------- ###

# Determine output conv in Conv, C2f, SPPF blocks, add them to a table

### --------------- ###

def _first_wrapper_with_conv(mod: nn.Module) -> nn.Module | None:
    if hasattr(mod, "conv") and isinstance(mod.conv, nn.Conv2d):
        return mod
    for ch in mod.children():
        w = _first_wrapper_with_conv(ch)
        if w is not None:
            return w
    return None

def _block_output_wrapper(block: nn.Module) -> nn.Module | None:
    cls = block.__class__.__name__.lower()
    if cls in {"c2f", "sppf"} and hasattr(block, "cv2"):
        return _first_wrapper_with_conv(block.cv2)
    return _first_wrapper_with_conv(block)


def build_output_target_table(model: nn.Module) -> dict[str, int]:
    name_of = {id(m): n for n, m in model.named_modules()}
    table: dict[str, int] = {}
    det_i = _detect_layer_idx(model)

    for i, block in enumerate(model.model):
        if det_i is not None and i == det_i:
            continue
        w = _block_output_wrapper(block)
        if w is None or not hasattr(w, "conv") or not isinstance(w.conv, nn.Conv2d):
            continue
        if w.conv.groups != 1:
            continue

        p = name_of.get(id(w), f"model.{i}")
        if det_i is not None and p.startswith(f"model.{det_i}."):
            continue
        table[p] = i
    return table

### --------------- ###

# Prepare structure ResolvedTargets with block_targets, hidden_targets and unknown

### --------------- ###

def _norm_target(t: str) -> str:
    t = t.strip().replace("model.model[", "model.").replace("]", "")
    if t.endswith(".conv"):
        t = t[:-5]
    return t


def _layer_idx_from_target(t: str) -> int | None:
    p = t.split(".")
    if len(p) < 2 or p[0] != "model" or not p[1].isdigit():
        return None
    return int(p[1])


def _is_prunable_wrapper(m: nn.Module) -> bool:
    return hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d) and m.conv.groups == 1


def resolve_targets(model: nn.Module, targets_csv: str) -> ResolvedTargets:
    named = dict(model.named_modules())
    out_table = build_output_target_table(model)  # output-wrapper path -> top-level block idx
    det_i = _detect_layer_idx(model)

    block_targets: set[int] = set()
    c2f_hidden_targets: set[int] = set()
    sppf_hidden_targets: set[int] = set()
    unknown: list[str] = []

    for raw in [x.strip() for x in targets_csv.split(",") if x.strip()]:
        t = _norm_target(raw)

        li = _layer_idx_from_target(t)
        if li is None or li >= len(model.model):
            unknown.append(t)
            continue

        if det_i is not None and t.startswith(f"model.{det_i}."):
            unknown.append(f"{t} (Detect targets disabled)")
            continue

        # direct output-conv target (Conv block, C2f.cv2, SPPF.cv2, etc.)
        if t in out_table:
            bi = out_table[t]
            if det_i is not None and bi == det_i:
                unknown.append(f"{t} (Detect targets disabled)")
            else:
                block_targets.add(bi)
            continue

        m = named.get(t)
        if m is None:
            unknown.append(t)
            continue

        # if raw leaf Conv2d is passed, try parent wrapper
        if isinstance(m, nn.Conv2d):
            parent = t.rsplit(".", 1)[0]
            pm = named.get(parent)
            if pm is not None and _is_prunable_wrapper(pm):
                t = parent
                m = pm
            else:
                unknown.append(f"{t} (raw Conv2d without supported wrapper)")
                continue

        if not _is_prunable_wrapper(m):
            unknown.append(f"{t} (not a prunable Conv wrapper)")
            continue

        cls = model.model[li].__class__.__name__.lower()

        if cls == "c2f":
            # internal C2f targets map to hidden prune (safe + shape-consistent)
            if re.match(rf"^model\.{li}\.cv1$", t) or re.match(rf"^model\.{li}\.m\.\d+\.cv[12]$", t):
                c2f_hidden_targets.add(li)
            elif re.match(rf"^model\.{li}\.cv2$", t):
                block_targets.add(li)  # output prune
            else:
                unknown.append(f"{t} (unsupported C2f path)")
            continue

        if cls == "sppf":
            if t == f"model.{li}.cv1":
                sppf_hidden_targets.add(li)
            elif t == f"model.{li}.cv2":
                block_targets.add(li)
            else:
                unknown.append(f"{t} (unsupported SPPF path)")
            continue

        # plain top-level Conv block
        if t == f"model.{li}":
            block_targets.add(li)
        else:
            unknown.append(f"{t} (unsupported non-C2f/SPPF local path)")

    return ResolvedTargets(
        block_targets=sorted(block_targets),
        c2f_hidden_targets=sorted(c2f_hidden_targets),
        sppf_hidden_targets=sorted(sppf_hidden_targets),
        unknown=unknown,
    )

### --------------- ###

# Structural Pruning

### --------------- ###

def _sources(i: int, layer: nn.Module) -> list[int]:
    f = getattr(layer, "f", -1)
    refs = [f] if isinstance(f, int) else list(f)
    return [i - 1 if r == -1 else int(r) for r in refs]


def _build_old_out_channels(layers: nn.Sequential, model_in_ch: int = 3) -> dict[int, int]:
    old_out: dict[int, int] = {}
    for i, m in enumerate(layers):
        cls = m.__class__.__name__.lower()
        src = _sources(i, m)

        if cls == "concat":
            old_out[i] = int(sum(old_out[s] for s in src))
            continue
        if cls in {"upsample", "split"}:
            old_out[i] = int(old_out[src[0]])
            continue
        if cls == "detect":
            old_out[i] = int(sum(old_out[s] for s in src))
            continue

        w = _block_output_wrapper(m)
        if w is not None and hasattr(w, "conv") and isinstance(w.conv, nn.Conv2d):
            old_out[i] = int(w.conv.out_channels)
        else:
            old_out[i] = int(model_in_ch if not src else sum(old_out[s] for s in src))
    return old_out


def _choose_keep_idx(conv: nn.Conv2d, prune_ratio: float, round_to: int) -> torch.Tensor:
    # 
    # Computes which output channels to keep. Ranking is based on L2 norm
    # 
    cout = int(conv.out_channels)
    keep = max(1, int(round(cout * (1.0 - prune_ratio))))
    if round_to > 1:
        keep = max(round_to, (keep // round_to) * round_to)
        keep = min(keep, cout)
        if keep < 1:
            keep = 1
    if keep >= cout:
        return torch.arange(cout, dtype=torch.long)

    score = conv.weight.detach().flatten(1).norm(p=2, dim=1)  # L2 per output channel
    return score.topk(keep).indices.sort().values.cpu()


def _rebuild_conv_out_select(old: nn.Conv2d, keep_idx: torch.Tensor) -> nn.Conv2d:
    idx = keep_idx.tolist()
    new = nn.Conv2d(
        in_channels=old.in_channels,
        out_channels=len(idx),
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight[idx].clone())
        if old.bias is not None:
            new.bias.copy_(old.bias[idx].clone())
    return new


def _rebuild_bn_out_select(old_bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
    idx = keep_idx.tolist()
    new_bn = nn.BatchNorm2d(
        num_features=len(idx),
        eps=old_bn.eps,
        momentum=old_bn.momentum,
        affine=old_bn.affine,
        track_running_stats=old_bn.track_running_stats,
    ).to(device=old_bn.weight.device, dtype=old_bn.weight.dtype)
    with torch.no_grad():
        new_bn.weight.copy_(old_bn.weight[idx].clone())
        new_bn.bias.copy_(old_bn.bias[idx].clone())
        new_bn.running_mean.copy_(old_bn.running_mean[idx].clone())
        new_bn.running_var.copy_(old_bn.running_var[idx].clone())
        new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
    return new_bn


def _prune_wrapper_output_channels(wrapper: nn.Module, keep_idx: torch.Tensor) -> bool:
    if not (hasattr(wrapper, "conv") and isinstance(wrapper.conv, nn.Conv2d)):
        return False
    old = wrapper.conv
    if old.groups != 1:
        return False
    if len(keep_idx) == old.out_channels:
        return True

    wrapper.conv = _rebuild_conv_out_select(old, keep_idx)
    if hasattr(wrapper, "bn") and isinstance(wrapper.bn, nn.BatchNorm2d):
        wrapper.bn = _rebuild_bn_out_select(wrapper.bn, keep_idx)
    return True


def _concat_input_keep_idx(
    src: list[int],
    keep_map: dict[int, torch.Tensor],
    old_out: dict[int, int],
    model_in_ch: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    offset = 0
    for s in src:
        if s < 0:
            idx = torch.arange(model_in_ch, dtype=torch.long) + offset
            offset += model_in_ch
        else:
            idx = keep_map[s] + offset
            offset += old_out[s]
        parts.append(idx)
    return torch.cat(parts, dim=0)


def _rebuild_conv_in_select(old: nn.Conv2d, in_keep_idx: torch.Tensor) -> nn.Conv2d:
    idx = in_keep_idx.tolist()
    if old.groups != 1:
        return old
    new = nn.Conv2d(
        in_channels=len(idx),
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.copy_(old.weight[:, idx].clone())
        if old.bias is not None:
            new.bias.copy_(old.bias.clone())
    return new

def _set_wrapper_input_channels(wrapper: nn.Module, in_keep_idx: torch.Tensor) -> bool:
    if not (hasattr(wrapper, "conv") and isinstance(wrapper.conv, nn.Conv2d)):
        return False
    old = wrapper.conv
    if old.groups != 1:
        return False
    if len(in_keep_idx) == old.in_channels and torch.equal(
        in_keep_idx.cpu(), torch.arange(old.in_channels, dtype=torch.long)
    ):
        return True
    wrapper.conv = _rebuild_conv_in_select(old, in_keep_idx)
    return True

def _adjust_detect_inputs(detect: nn.Module, src: list[int], keep_map: dict[int, torch.Tensor]) -> list[str]:
    changed: list[str] = []
    for b, s in enumerate(src):
        idx = keep_map[s]

        if hasattr(detect, "cv2") and b < len(detect.cv2):
            seq = detect.cv2[b]
            if len(seq) > 0 and _set_wrapper_input_channels(seq[0], idx):
                changed.append(f"detect.cv2[{b}].0.in={len(idx)}")

        if hasattr(detect, "cv3") and b < len(detect.cv3):
            seq = detect.cv3[b]
            if len(seq) > 0 and _set_wrapper_input_channels(seq[0], idx):
                changed.append(f"detect.cv3[{b}].0.in={len(idx)}")
    return changed


def _block_input_wrapper(block: nn.Module) -> nn.Module | None:
    cls = block.__class__.__name__.lower()
    if cls in {"c2f", "sppf"} and hasattr(block, "cv1"):
        return _first_wrapper_with_conv(block.cv1)
    return _first_wrapper_with_conv(block)


def apply_structural_pruning_and_realign(
    model: nn.Module,
    blocks: Iterable[int],
    prune_ratio: float,
    round_to: int,
    model_in_ch: int = 3,
) -> tuple[list[Change], list[str], list[str], dict[int, torch.Tensor]]:
    layers = model.model
    old_out = _build_old_out_channels(layers, model_in_ch=model_in_ch)

    selected_keep: dict[int, torch.Tensor] = {}
    changes: list[Change] = []
    skipped: list[str] = []

    # 1) Prune selected block outputs by L2 channel ranking (true structural pruning).
    for i in blocks:
        m = layers[i]
        cls = m.__class__.__name__.lower()
        if cls in {"concat", "upsample", "split", "detect"}:
            skipped.append(f"model.{i} ({m.__class__.__name__})")
            continue

        out_w = _block_output_wrapper(m)
        if out_w is None or not hasattr(out_w, "conv") or not isinstance(out_w.conv, nn.Conv2d):
            skipped.append(f"model.{i} (no prunable output conv)")
            continue
        if out_w.conv.groups != 1:
            skipped.append(f"model.{i} (groups={out_w.conv.groups})")
            continue

        old_params = params_count(m)
        old_cout = int(out_w.conv.out_channels)

        keep_idx = _choose_keep_idx(out_w.conv, prune_ratio=prune_ratio, round_to=round_to)
        if len(keep_idx) == old_cout:
            skipped.append(f"model.{i} (ratio/round keeps all channels)")
            continue

        _prune_wrapper_output_channels(out_w, keep_idx)
        new_params = params_count(m)
        new_cout = int(out_w.conv.out_channels)

        selected_keep[i] = keep_idx
        changes.append(
            Change(
                path=f"model.{i}",
                old_out=old_cout,
                new_out=new_cout,
                old_params=old_params,
                new_params=new_params,
            )
        )

    # 2) Realign all consumers (Concat/Upsample/Split bookkeeping + actual conv input surgery).
    keep_map: dict[int, torch.Tensor] = {}
    for i in range(len(layers)):
        if i in selected_keep:
            keep_map[i] = selected_keep[i]
        else:
            keep_map[i] = torch.arange(old_out[i], dtype=torch.long)

    realign_changes: list[str] = []

    for i, m in enumerate(layers):
        cls = m.__class__.__name__.lower()
        src = _sources(i, m)

        if cls == "concat":
            keep_map[i] = _concat_input_keep_idx(src, keep_map, old_out, model_in_ch)
            continue


        if cls in {"upsample", "split"}:
            keep_map[i] = keep_map[src[0]].clone()
            continue

        if cls == "detect":
            realign_changes.extend([f"model.{i} {s}" for s in _adjust_detect_inputs(m, src, keep_map)])
            keep_map[i] = torch.arange(sum(old_out[s] for s in src), dtype=torch.long)
            continue

        in_keep = (
            _concat_input_keep_idx(src, keep_map, old_out, model_in_ch)
            if src
            else torch.arange(model_in_ch, dtype=torch.long)
        )

        in_w = _block_input_wrapper(m)
        if in_w is not None and _set_wrapper_input_channels(in_w, in_keep):
            realign_changes.append(f"model.{i} input={len(in_keep)}")

    return changes, skipped, realign_changes, keep_map


### --------------- ###

# Structural Pruning of C2F

### --------------- ###


def _topk_keep_from_score(score: torch.Tensor, prune_ratio: float, round_to: int) -> torch.Tensor:
    n = int(score.numel())
    keep = max(1, int(round(n * (1.0 - prune_ratio))))
    if round_to > 1:
        keep = max(round_to, (keep // round_to) * round_to)
        keep = min(keep, n)
    return score.topk(keep).indices.sort().values.cpu()


def prune_c2f_cv1(model: nn.Module, li: int, prune_ratio: float, round_to: int) -> Change | None:
    c2f = model.model[li]
    if c2f.__class__.__name__.lower() != "c2f":
        return None

    old_params = params_count(c2f)
    old_hidden = int(c2f.c)  # hidden channels in C2f
    old_out = int(c2f.cv1.conv.out_channels)  # should be 2*hidden

    if old_out != 2 * old_hidden:
        return None

    # pairwise score for split halves
    s = c2f.cv1.conv.weight.detach().flatten(1).norm(p=2, dim=1)
    keep_h = _topk_keep_from_score(s[:old_hidden] + s[old_hidden:2 * old_hidden], prune_ratio, round_to)
    if len(keep_h) == old_hidden:
        return None

    keep_cv1 = torch.cat([keep_h, keep_h + old_hidden]).sort().values
    _prune_wrapper_output_channels(c2f.cv1, keep_cv1)

    # update hidden size used by forward
    c2f.c = int(len(keep_h))

    # prune all bottleneck internals to new hidden width
    for b in c2f.m:
        _set_wrapper_input_channels(b.cv1, keep_h)
        _prune_wrapper_output_channels(b.cv1, keep_h)
        _set_wrapper_input_channels(b.cv2, keep_h)
        _prune_wrapper_output_channels(b.cv2, keep_h)

    # cv2 input is concat([y0,y1,m0,...]) with chunks of old_hidden each
    parts = [keep_h + j * old_hidden for j in range(2 + len(c2f.m))]
    _set_wrapper_input_channels(c2f.cv2, torch.cat(parts))

    return Change(
        path=f"model.{li}.cv1",
        old_out=old_out,
        new_out=int(c2f.cv1.conv.out_channels),
        old_params=old_params,
        new_params=params_count(c2f),
    )



### --------------- ###

# Structural Pruning of SPPF

### --------------- ###


def prune_sppf_cv1(model: nn.Module, li: int, prune_ratio: float, round_to: int) -> Change | None:
    sppf = model.model[li]
    if sppf.__class__.__name__.lower() != "sppf":
        return None

    old_params = params_count(sppf)
    old_h = int(sppf.cv1.conv.out_channels)

    keep_h = _choose_keep_idx(sppf.cv1.conv, prune_ratio, round_to)
    if len(keep_h) == old_h:
        return None

    _prune_wrapper_output_channels(sppf.cv1, keep_h)
    # cv2 input is concat of 4 pooled branches
    in_keep = torch.cat([keep_h + j * old_h for j in range(4)])
    _set_wrapper_input_channels(sppf.cv2, in_keep)

    return Change(
        path=f"model.{li}.cv1",
        old_out=old_h,
        new_out=int(sppf.cv1.conv.out_channels),
        old_params=old_params,
        new_params=params_count(sppf),
    )

########################################

def make_pruned_trainer(pruned_nn: nn.Module):
    # prevents Ultralytics trainer from rebuilding a fresh model from YAML/config.
    class PrunedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return pruned_nn
    return PrunedTrainer


def eval_model(
    y: YOLO,
    data: str,
    imgsz: int,
    device: str,
    batch: int,
    *,
    split: str = "val",
    project: str = "./optimizer/pruning/runs/manual_prune",
) -> dict[str, float]:
    with open(data, "r") as f:
        d = yaml.safe_load(f) or {}
    if split == "val" and d.get("val") is None and d.get("valid") is not None:
        split = "valid"

    m = y.val(
        data=data,
        imgsz=imgsz,
        device=device,
        batch=batch,
        split=split,
        project=project,
        verbose=False,
    )
    return {
        "map50_95": float(m.box.map),
        "map50": float(m.box.map50),
        "lat_ms": float(m.speed.get("inference", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="../models/yolov8n.pt")
    ap.add_argument("--data", type=str, default="../datasets/coco/data.yaml")
    ap.add_argument("--prune-ratio", type=float, default=0.20, help="Per-selected-block output-channel prune ratio in (0,1)")
    ap.add_argument("--round-to", type=int, default=1, help="Round kept channels to a multiple (e.g. 8)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--project", type=str, default="./optimizer/pruning/runs/manual_prune")
    ap.add_argument("--name", type=str, default="yolov8_manual_structural_prune")
    ap.add_argument("--save-pruned", type=str, default="../models/pruned_models/yolov8_manual_pruned_v1.pt")
    ap.add_argument(
        "--targets",
        type=str,
        default="",
        help="Comma-separated conv targets, e.g. model.18.cv2,model.19.conv",
    )

    args = ap.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        raise ValueError("--fraction must be in (0,1].")
    if not (0.0 < args.prune_ratio < 1.0):
        raise ValueError("--prune-ratio must be in (0,1).")
    if args.round_to < 1:
        raise ValueError("--round-to must be >= 1.")

    # Baseline (unmodified model)
    base = YOLO(args.weights)
    base_stats = eval_model(base, args.data, args.imgsz, args.device, args.batch, project=args.project)
    base_params = params_count(base.model)
    base_mb = fp32_param_mb(base.model)

    # Pruned model
    y = YOLO(args.weights)
    max_idx = len(y.model.model) - 1

    if args.targets.strip():
        resolved = resolve_targets(y.model, args.targets)
        blocks = resolved.block_targets
        c2f_hidden_targets = resolved.c2f_hidden_targets
        sppf_hidden_targets = resolved.sppf_hidden_targets

        if resolved.unknown:
            print("[WARN] Unknown/unsupported targets:", ", ".join(resolved.unknown))
        if not blocks and not c2f_hidden_targets and not sppf_hidden_targets:
            raise ValueError("No valid targets resolved from --targets.")
    else:
        raise ValueError("Use --targets")


    # Prune Individual Conv Blocks
    changes, skipped, realign_changes, _ = apply_structural_pruning_and_realign(
        y.model,
        blocks=blocks,
        prune_ratio=args.prune_ratio,
        round_to=args.round_to,
        model_in_ch=3, #input image channels
    )

    # Prune C2F Conv Blocks
    for li in c2f_hidden_targets:
        ch = prune_c2f_cv1(y.model, li, args.prune_ratio, args.round_to)
        if ch is None:
            skipped.append(f"model.{li}.(C2f hidden) no-op")
        else:
            changes.append(ch)

    # Prune SPPF Conv Blocks
    for li in sppf_hidden_targets:
        ch = prune_sppf_cv1(y.model, li, args.prune_ratio, args.round_to)
        if ch is None:
            skipped.append(f"model.{li}.cv1 (SPPF hidden) no-op")
        else:
            changes.append(ch)


    if skipped:
        print("[INFO] Skipped:", ", ".join(skipped))
    if realign_changes:
        print("[INFO] Realigned:")
        for s in realign_changes:
            print(" -", s)

    # Sanity forward
    p0 = next(y.model.parameters())
    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=p0.device, dtype=p0.dtype)
    with torch.no_grad():
        _ = y.model(x)

    # Save pruned architecture
    pruned_ckpt = Path(args.save_pruned)
    pruned_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": y.model.eval()}, pruned_ckpt)

    # Eval before finetune
    # keep pruned model in memory
    pruned = YOLO(str(pruned_ckpt))
    preft_stats = eval_model(
        pruned, args.data, args.imgsz, args.device, args.batch, project=args.project
    )
    pruned_params = params_count(pruned.model)
    pruned_mb = fp32_param_mb(pruned.model)

    pruned.model = y.model  # y.model is already structurally pruned

    expected_params = params_count(pruned.model)

    def _check_train_model(trainer):
        got = sum(p.numel() for p in trainer.model.parameters())
        if got != expected_params:
            raise RuntimeError(f"Trainer rebuilt wrong architecture: got {got}, expected {expected_params}")

    pruned.add_callback("on_train_start", _check_train_model)

    for p in pruned.model.parameters():
        p.requires_grad = True

    pruned.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        project=args.project,
        name=args.name,
        pretrained=False,                 # no extra transfer from base
        resume=False,
        trainer=make_pruned_trainer(pruned.model),
    )


    save_dir = Path(pruned.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    final_pt = best_pt if best_pt.exists() else last_pt

    finetuned = YOLO(str(final_pt))
    postft_stats = eval_model(
        finetuned, args.data, args.imgsz, args.device, args.batch, project=args.project
    )
    postft_params = params_count(finetuned.model)
    postft_mb = fp32_param_mb(finetuned.model)

    # Report
    total_old = sum(c.old_params for c in changes)
    total_new = sum(c.new_params for c in changes)
    local_pruned_pct = 100.0 * (total_old - total_new) / max(1, total_old)

    global_pruned_pct = 100.0 * max(0, base_params - pruned_params) / max(1, base_params)
    param_delta_pct = 100.0 * (pruned_params - base_params) / max(1, base_params)

    print("\n=== LAYERS CHANGED ===")
    for c in changes:
        p = 100.0 * (c.old_params - c.new_params) / max(1, c.old_params)
        print(f"{c.path}: out {c.old_out}->{c.new_out}, params {c.old_params}->{c.new_params} ({p:.2f}%)")

    print("\n=== SIZE / PARAMS ===")
    print(f"Base params:   {base_params:,} ({base_mb:.2f} MB fp32)")
    print(f"Pruned params: {pruned_params:,} ({pruned_mb:.2f} MB fp32)")
    print(f"Global param pruning: {global_pruned_pct:.2f}%")
    print(f"Global param delta:   {param_delta_pct:+.2f}%")
    print(f"Saved param memory:   {base_mb - pruned_mb:.2f} MB (fp32)")
    print(f"Selected-layer pruning only: {local_pruned_pct:.2f}%")

    print("\n=== ACCURACY / LATENCY ===")
    print(f"Baseline       mAP50-95={base_stats['map50_95']:.4f}  mAP50={base_stats['map50']:.4f}  lat={base_stats['lat_ms']:.2f}ms")
    print(f"Pruned (pre-FT) mAP50-95={preft_stats['map50_95']:.4f}  mAP50={preft_stats['map50']:.4f}  lat={preft_stats['lat_ms']:.2f}ms")
    print(f"Pruned (post-FT) mAP50-95={postft_stats['map50_95']:.4f}  mAP50={postft_stats['map50']:.4f}  lat={postft_stats['lat_ms']:.2f}ms")
    print(f"Delta pre-FT vs base:  {preft_stats['map50_95'] - base_stats['map50_95']:+.4f}")
    print(f"Delta post-FT vs base: {postft_stats['map50_95'] - base_stats['map50_95']:+.4f}")


if __name__ == "__main__":
    main()
