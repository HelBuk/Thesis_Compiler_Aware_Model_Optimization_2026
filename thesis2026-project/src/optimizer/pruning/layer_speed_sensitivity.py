from __future__ import annotations

import argparse
import copy
import csv
import re
import statistics
import traceback
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from ultralytics import YOLO

try:
    # Works when run as: python -m src.optimizer.pruning.layer_speed_sensitivity
    from .manual_yolov8_pruner import (
        _block_input_wrapper,
        _block_output_wrapper,
        _set_wrapper_input_channels,
        _prune_wrapper_output_channels,
        _sync_bottleneck_add,
        _c2f_cv2_input_keep_for_bn_prune,
        _build_old_out_channels,
        _sources,
        _concat_input_keep_idx,
        fp32_param_mb,
        params_count,
    )
except ImportError:
    # Fallback: run from src/ as: python -m optimizer.pruning.layer_speed_sensitivity
    from optimizer.pruning.manual_yolov8_pruner import (  # type: ignore[no-redef]
        _block_input_wrapper,
        _block_output_wrapper,
        _set_wrapper_input_channels,
        _prune_wrapper_output_channels,
        _sync_bottleneck_add,
        _c2f_cv2_input_keep_for_bn_prune,
        _build_old_out_channels,
        _sources,
        _concat_input_keep_idx,
        fp32_param_mb,
        params_count,
    )


# -----------------------------
# Target discovery
# -----------------------------

def list_prunable_targets(weights: str) -> list[str]:
    """
    Enumerate targets compatible with the manual structural pruner logic.
    Includes:
      - top-level block outputs: model.i or model.i.cv2 / sppf cv2 equivalent
      - C2f hidden conv: model.i.cv1
      - C2f bottlenecks: model.i.m.j.cv1 / model.i.m.j.cv2
      - SPPF hidden conv: model.i.cv1
    Excludes Detect.
    """
    y = YOLO(weights)
    model = y.model
    out: list[str] = []

    for i, block in enumerate(model.model):
        cls = block.__class__.__name__.lower()
        if cls == "detect":
            continue

        if cls == "c2f":
            out.append(f"model.{i}.cv2")
            out.append(f"model.{i}.cv1")
            for j, _ in enumerate(block.m):
                out.append(f"model.{i}.m.{j}.cv1")
                out.append(f"model.{i}.m.{j}.cv2")
        elif cls == "sppf":
            out.append(f"model.{i}.cv2")
            out.append(f"model.{i}.cv1")
        else:
            w = _block_output_wrapper(block)
            if w is not None and hasattr(w, "conv") and isinstance(w.conv, nn.Conv2d):
                if w.conv.groups == 1:
                    out.append(f"model.{i}")

    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


# -----------------------------
# Exact keep-count pruning helpers
# -----------------------------

def choose_keep_idx_exact(conv: nn.Conv2d, keep: int) -> torch.Tensor:
    cout = int(conv.out_channels)
    keep = max(1, min(int(keep), cout))
    if keep >= cout:
        return torch.arange(cout, dtype=torch.long)

    score = conv.weight.detach().flatten(1).norm(p=2, dim=1)
    return score.topk(keep).indices.sort().values.cpu()


@dataclass
class KeepSpec:
    requested: int
    feasible: int
    parity_mode: str


def adjust_keep_count(requested: int, max_channels: int, parity_mode: str) -> KeepSpec:
    requested = max(1, min(int(requested), int(max_channels)))

    if parity_mode == "any":
        feasible = requested
    elif parity_mode == "even":
        feasible = requested if requested % 2 == 0 else requested - 1
        if feasible < 1:
            feasible = 2 if max_channels >= 2 else 1
    elif parity_mode == "odd":
        feasible = requested if requested % 2 == 1 else requested - 1
        if feasible < 1:
            feasible = 1
    else:
        raise ValueError(f"Unknown parity_mode={parity_mode}")

    feasible = max(1, min(feasible, max_channels))

    if parity_mode == "even" and feasible % 2 == 1 and max_channels >= 2:
        feasible = max(2, feasible - 1)
    if parity_mode == "odd" and feasible % 2 == 0:
        feasible = feasible - 1 if feasible > 1 else 1

    feasible = max(1, min(feasible, max_channels))
    return KeepSpec(requested=requested, feasible=feasible, parity_mode=parity_mode)


# -----------------------------
# Accuracy evaluation
# -----------------------------

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _count_val_images(data_yaml: str, split: str) -> int:
    """Count images in the validation split directory (best-effort)."""
    data_path = Path(data_yaml)
    with open(data_path, "r") as f:
        d = yaml.safe_load(f) or {}

    val_rel = d.get(split) or d.get("val") or d.get("valid") or ""
    if not val_rel:
        return 0

    for base in (data_path.parent, Path(".")):
        val_dir = (base / val_rel).resolve()
        if val_dir.is_dir():
            return sum(1 for p in val_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    return 0


def compute_val_fraction(data_yaml: str, split: str, batch: int, max_batches: int) -> float:
    """Return the dataset fraction that corresponds to at most max_batches batches."""
    if max_batches <= 0:
        return 1.0
    n_images = _count_val_images(data_yaml, split)
    if n_images <= 0:
        print("  [warn] could not count val images; ignoring --max-val-batches")
        return 1.0
    cap = max_batches * batch
    frac = min(1.0, cap / n_images)
    print(f"  [val] {n_images} val images, capping at {cap} (fraction={frac:.4f})")
    return frac


class PersistentCOCOEvaluator:
    """
    Builds the Ultralytics DetectionValidator + dataset subset ONCE and
    reuses them for every eval() call.

    On x86 with separate CPU/GPU RAM, batches can optionally be preloaded
    into CPU RAM to eliminate per-eval disk IO.  On Orin Nano (and other
    Jetson boards) CPU and GPU share the same 8 GB physical memory, so
    preloading 1000 images (~1.2 GB) leaves too little for CUDA — use
    preload=False on Jetson.  In that mode eval() wraps the dataset in a
    fresh pin_memory=False DataLoader each call; construction is ~instant
    because the dataset object (and its .cache index) already exists.
    """

    def __init__(
        self,
        weights: str,
        data: str,
        imgsz: int,
        device: str,
        batch: int,
        split: str = "val",
        val_fraction: float = 1.0,
        project: str = ".",
        preload: bool = True,
    ):
        import torch.utils.data as _tud
        from ultralytics.models.yolo.detect.val import DetectionValidator
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils.torch_utils import select_device
        from ultralytics.cfg import get_cfg, DEFAULT_CFG

        self._de_parallel = lambda m: m.module if hasattr(m, "module") else m
        self._device = select_device(device)
        self._batch = batch
        self._preload = preload

        overrides: dict = dict(
            model=weights,
            data=data,
            imgsz=imgsz,
            batch=batch,
            device=str(self._device),
            split=split,
            project=project,
            verbose=False,
            plots=False,
            save=False,
            workers=0,
        )

        cfg = get_cfg(DEFAULT_CFG, overrides)
        v = DetectionValidator(args=cfg)
        v.print_results = lambda: None  # silence per-eval "Class Images..." header
        v.data = check_det_dataset(data)
        v.device = self._device

        actual_split = split
        for candidate in (split, "valid", "val", "test"):
            if candidate in v.data:
                actual_split = candidate
                break

        print(f"  [PersistentEvaluator] building dataset "
              f"(split={actual_split}, fraction={val_fraction:.3f}) ...")
        dl_orig = v.get_dataloader(v.data[actual_split], batch)

        # Ultralytics 8.4 ignores 'fraction' for val — apply manually via Subset.
        full_len = len(dl_orig.dataset)
        n_keep = max(batch, round(full_len * val_fraction))
        if n_keep < full_len:
            dataset = _tud.Subset(dl_orig.dataset, list(range(n_keep)))
            print(f"  [PersistentEvaluator] subset: {full_len} → {n_keep} images")
        else:
            dataset = dl_orig.dataset

        collate_fn = getattr(dl_orig, "collate_fn", None)
        if collate_fn is None:
            collate_fn = getattr(type(dl_orig.dataset), "collate_fn", None)

        self._dataset = dataset
        self._collate_fn = collate_fn
        self._v = v

        if preload:
            # x86: load all batches into CPU RAM once — zero disk IO per eval.
            dl = _tud.DataLoader(dataset, batch_size=batch, shuffle=False,
                                 num_workers=0, pin_memory=False,
                                 collate_fn=collate_fn)
            self._batches: list[dict] = []
            for b in dl:
                self._batches.append(
                    {k: (t.cpu() if isinstance(t, torch.Tensor) else t)
                     for k, t in b.items()}
                )
            ram_mb = sum(b["img"].nbytes for b in self._batches) / 1e6
            n_img = sum(b["img"].shape[0] for b in self._batches)
            print(f"  [PersistentEvaluator] {n_img} images preloaded "
                  f"({ram_mb:.0f} MB) — zero disk IO per eval")
        else:
            # Jetson/unified-memory: skip preload; DataLoader wraps dataset
            # cheaply per eval using the already-cached index.
            n_img = len(dataset)
            print(f"  [PersistentEvaluator] {n_img} images ready "
                  f"(no preload — unified memory device)")

    def _make_dataloader(self):
        import torch.utils.data as _tud
        return _tud.DataLoader(
            self._dataset,
            batch_size=self._batch,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=self._collate_fn,
        )

    def eval(self, model: nn.Module) -> dict[str, float]:
        v = self._v
        m = model.float().eval().to(self._device)
        v.init_metrics(self._de_parallel(m))
        v.jdict = []

        source = self._batches if self._preload else self._make_dataloader()

        with torch.inference_mode():
            for raw in source:
                batch = v.preprocess(raw)
                preds = m(batch["img"])
                preds = v.postprocess(preds)
                v.update_metrics(preds, batch)

        stats = v.get_stats()
        try:
            v.finalize_metrics()
        except Exception:
            pass

        def _pick(d: dict, *keys: str) -> float:
            for k in keys:
                if k in d:
                    return float(d[k])
            return 0.0

        return {
            "map50_95": _pick(stats, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
            "map50":    _pick(stats, "metrics/mAP50(B)",    "metrics/mAP50"),
            "lat_ms_val": 0.0,
        }


def eval_model_no_train(
    y: YOLO,
    data: str,
    imgsz: int,
    device: str,
    batch: int,
    *,
    split: str = "val",
    project: str = "./optimizer/pruning/runs/layer_speed_sensitivity",
    val_fraction: float = 1.0,
) -> dict[str, float]:
    """
    Fallback: evaluate via YOLO.val() (rebuilds dataloader every call).
    Prefer PersistentCOCOEvaluator for sweeps.
    stdout is suppressed to avoid the per-call "Class Images Instances" spam.
    """
    import io
    import contextlib

    with open(data, "r") as f:
        d = yaml.safe_load(f) or {}

    if split == "val" and d.get("val") is None and d.get("valid") is not None:
        split = "valid"

    kwargs: dict = dict(
        data=data,
        imgsz=imgsz,
        device=device,
        batch=batch,
        split=split,
        project=project,
        verbose=False,
    )
    if val_fraction < 1.0:
        kwargs["fraction"] = val_fraction

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m = y.val(**kwargs)
    return {
        "map50_95": float(m.box.map),
        "map50": float(m.box.map50),
        "lat_ms_val": float(m.speed.get("inference", 0.0)),
    }


# -----------------------------
# Target application
# -----------------------------

def apply_exact_keep_to_target(model: nn.Module, target: str, keep_count: int) -> list[str]:
    """
    Prune exactly one target, structurally, using the same model-edit logic as the
    manual pruner, but with an exact keep-count instead of a global prune ratio.
    Returns a list of changed paths for bookkeeping.
    """
    changed: list[str] = []
    layers = model.model

    # top-level block output: model.i
    m = re.fullmatch(r"model\.(\d+)$", target)
    if m:
        li = int(m.group(1))
        block = layers[li]
        old_out = _build_old_out_channels(layers, model_in_ch=3)
        out_w = _block_output_wrapper(block)
        if out_w is None or not hasattr(out_w, "conv"):
            raise ValueError(f"Target {target}: no prunable output wrapper")

        keep_idx = choose_keep_idx_exact(out_w.conv, keep_count)
        _prune_wrapper_output_channels(out_w, keep_idx)

        selected_keep = {li: keep_idx}
        keep_map = {}
        for i in range(len(layers)):
            if i in selected_keep:
                keep_map[i] = selected_keep[i]
            else:
                keep_map[i] = torch.arange(old_out[i], dtype=torch.long)

        realign_changes: list[str] = []
        for i, layer in enumerate(layers):
            cls = layer.__class__.__name__.lower()
            src = _sources(i, layer)

            if cls == "concat":
                keep_map[i] = _concat_input_keep_idx(src, keep_map, old_out, 3)
                continue
            if cls in {"upsample", "split"}:
                keep_map[i] = keep_map[src[0]].clone()
                continue
            if cls == "detect":
                if hasattr(layer, "cv2"):
                    for b, s in enumerate(src):
                        idx = keep_map[s]
                        if b < len(layer.cv2) and len(layer.cv2[b]) > 0:
                            _set_wrapper_input_channels(layer.cv2[b][0], idx)
                if hasattr(layer, "cv3"):
                    for b, s in enumerate(src):
                        idx = keep_map[s]
                        if b < len(layer.cv3) and len(layer.cv3[b]) > 0:
                            _set_wrapper_input_channels(layer.cv3[b][0], idx)
                keep_map[i] = torch.arange(sum(old_out[s] for s in src), dtype=torch.long)
                continue

            in_keep = (
                _concat_input_keep_idx(src, keep_map, old_out, 3)
                if src else torch.arange(3, dtype=torch.long)
            )
            in_w = _block_input_wrapper(layer)
            if in_w is not None and _set_wrapper_input_channels(in_w, in_keep):
                realign_changes.append(f"model.{i} input={len(in_keep)}")

        changed.append(target)
        changed.extend(realign_changes)
        return changed

    # C2f / SPPF / bottleneck targets.
    m = re.fullmatch(r"model\.(\d+)\.cv1$", target)
    if m:
        li = int(m.group(1))
        block = layers[li]
        cls = block.__class__.__name__.lower()

        if cls == "c2f":
            old_hidden = int(block.c)
            ks = adjust_keep_count(keep_count, old_hidden, "any")

            s = block.cv1.conv.weight.detach().flatten(1).norm(p=2, dim=1)
            pair_score = s[:old_hidden] + s[old_hidden:2 * old_hidden]
            keep_hidden = pair_score.topk(ks.feasible).indices.sort().values.cpu()
            keep_cv1 = torch.cat([keep_hidden, keep_hidden + old_hidden]).sort().values

            _prune_wrapper_output_channels(block.cv1, keep_cv1)
            block.c = int(len(keep_hidden))

            for b in block.m:
                _set_wrapper_input_channels(b.cv1, keep_hidden)
                _prune_wrapper_output_channels(b.cv1, keep_hidden)
                _set_wrapper_input_channels(b.cv2, keep_hidden)
                _prune_wrapper_output_channels(b.cv2, keep_hidden)
                _sync_bottleneck_add(b)

            parts = [keep_hidden + j * old_hidden for j in range(2 + len(block.m))]
            _set_wrapper_input_channels(block.cv2, torch.cat(parts))

            changed.append(target)
            return changed

        if cls == "sppf":
            old_width = int(block.cv1.conv.out_channels)
            ks = adjust_keep_count(keep_count, old_width, "any")
            keep_h = choose_keep_idx_exact(block.cv1.conv, ks.feasible)

            _prune_wrapper_output_channels(block.cv1, keep_h)
            in_keep = torch.cat([keep_h + j * old_width for j in range(4)])
            _set_wrapper_input_channels(block.cv2, in_keep)

            changed.append(target)
            return changed

        raise ValueError(f"Unsupported target {target}")

    m = re.fullmatch(r"model\.(\d+)\.cv2$", target)
    if m:
        li = int(m.group(1))
        block = layers[li]
        out_w = _block_output_wrapper(block)
        if out_w is None or not hasattr(out_w, "conv"):
            raise ValueError(f"Target {target}: no prunable output conv")

        keep_idx = choose_keep_idx_exact(out_w.conv, keep_count)
        _prune_wrapper_output_channels(out_w, keep_idx)

        if block.__class__.__name__.lower() in {"c2f", "sppf"}:
            old_out = _build_old_out_channels(layers, model_in_ch=3)
            selected_keep = {li: keep_idx}
            keep_map = {}
            for i in range(len(layers)):
                keep_map[i] = selected_keep.get(i, torch.arange(old_out[i], dtype=torch.long))

            for i, layer in enumerate(layers):
                cls = layer.__class__.__name__.lower()
                src = _sources(i, layer)

                if cls == "concat":
                    keep_map[i] = _concat_input_keep_idx(src, keep_map, old_out, 3)
                    continue
                if cls in {"upsample", "split"}:
                    keep_map[i] = keep_map[src[0]].clone()
                    continue
                if cls == "detect":
                    if hasattr(layer, "cv2"):
                        for b, s in enumerate(src):
                            idx = keep_map[s]
                            if b < len(layer.cv2) and len(layer.cv2[b]) > 0:
                                _set_wrapper_input_channels(layer.cv2[b][0], idx)
                    if hasattr(layer, "cv3"):
                        for b, s in enumerate(src):
                            idx = keep_map[s]
                            if b < len(layer.cv3) and len(layer.cv3[b]) > 0:
                                _set_wrapper_input_channels(layer.cv3[b][0], idx)
                    continue

                in_keep = _concat_input_keep_idx(src, keep_map, old_out, 3) if src else torch.arange(3, dtype=torch.long)
                in_w = _block_input_wrapper(layer)
                if in_w is not None:
                    _set_wrapper_input_channels(in_w, in_keep)

        changed.append(target)
        return changed

    m = re.fullmatch(r"model\.(\d+)\.m\.(\d+)\.cv1$", target)
    if m:
        li, bi = int(m.group(1)), int(m.group(2))
        c2f = layers[li]
        b = c2f.m[bi]

        ks = adjust_keep_count(keep_count, int(b.cv1.conv.out_channels), "any")
        keep = choose_keep_idx_exact(b.cv1.conv, ks.feasible)

        _prune_wrapper_output_channels(b.cv1, keep)
        _set_wrapper_input_channels(b.cv2, keep)
        _sync_bottleneck_add(b)

        changed.append(target)
        return changed

    m = re.fullmatch(r"model\.(\d+)\.m\.(\d+)\.cv2$", target)
    if m:
        li, bi = int(m.group(1)), int(m.group(2))
        c2f = layers[li]
        b = c2f.m[bi]

        ks = adjust_keep_count(keep_count, int(b.cv2.conv.out_channels), "any")
        keep = choose_keep_idx_exact(b.cv2.conv, ks.feasible)
        in_keep_c2f_cv2 = _c2f_cv2_input_keep_for_bn_prune(c2f, bi, keep)

        _prune_wrapper_output_channels(b.cv2, keep)
        _sync_bottleneck_add(b)

        if bi + 1 < len(c2f.m):
            nb = c2f.m[bi + 1]
            _set_wrapper_input_channels(nb.cv1, keep)
            _sync_bottleneck_add(nb)

        _set_wrapper_input_channels(c2f.cv2, in_keep_c2f_cv2)

        changed.append(target)
        return changed

    raise ValueError(f"Unsupported target syntax: {target}")


# -----------------------------
# Benchmarking
# -----------------------------

def cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif str(device) == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark_forward_latency_ms(
    model: nn.Module,
    imgsz: int,
    batch: int,
    device: str,
    warmup: int,
    runs: int,
    use_half: bool,
) -> dict[str, float]:
    model = model.to(device).eval()
    dtype = torch.float16 if (use_half and str(device).startswith("cuda")) else torch.float32

    if dtype == torch.float16:
        model = model.half()
    else:
        model = model.float()

    x = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
    cuda_sync(device)

    times_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(runs):
            cuda_sync(device)
            if str(device).startswith("cuda"):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(x)
                end.record()
                torch.cuda.synchronize()
                times_ms.append(float(start.elapsed_time(end)))
            else:
                t0 = time.perf_counter()
                _ = model(x)
                times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(statistics.mean(times_ms))
    return {
        "lat_mean_ms": mean_ms,
        "lat_median_ms": float(statistics.median(times_ms)),
        "lat_std_ms": float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "fps_mean": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        "runs": float(runs),
        "warmup": float(warmup),
    }


# -----------------------------
# Planning keep counts
# -----------------------------

def candidate_keep_counts(channels: int, step_pct: int, include_full: bool = True) -> list[int]:
    """
    Generate exact keep counts corresponding to prune percentages 0, step, ..., 100.
    100% prune is not structurally feasible, so it is represented as keep=1.
    """
    out: list[int] = []
    if include_full:
        out.append(channels)

    for pct in range(step_pct, 101, step_pct):
        keep = int(round(channels * (1.0 - pct / 100.0)))
        keep = max(1, min(channels, keep))
        out.append(keep)

    uniq = []
    seen = set()
    for k in out:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def target_channel_count(model: nn.Module, target: str) -> int:
    layers = model.model

    m = re.fullmatch(r"model\.(\d+)$", target)
    if m:
        li = int(m.group(1))
        return int(_block_output_wrapper(layers[li]).conv.out_channels)

    m = re.fullmatch(r"model\.(\d+)\.cv1$", target)
    if m:
        li = int(m.group(1))
        block = layers[li]
        cls = block.__class__.__name__.lower()
        if cls == "c2f":
            return int(block.c)
        return int(block.cv1.conv.out_channels)

    m = re.fullmatch(r"model\.(\d+)\.cv2$", target)
    if m:
        li = int(m.group(1))
        return int(_block_output_wrapper(layers[li]).conv.out_channels)

    m = re.fullmatch(r"model\.(\d+)\.m\.(\d+)\.cv1$", target)
    if m:
        li, bi = int(m.group(1)), int(m.group(2))
        return int(layers[li].m[bi].cv1.conv.out_channels)

    m = re.fullmatch(r"model\.(\d+)\.m\.(\d+)\.cv2$", target)
    if m:
        li, bi = int(m.group(1)), int(m.group(2))
        return int(layers[li].m[bi].cv2.conv.out_channels)

    raise ValueError(f"Unsupported target {target}")


# -----------------------------
# Main sweep
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="../models/yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--step-pct", type=int, default=10)
    ap.add_argument("--project", type=str, default="./optimizer/pruning/runs/layer_speed_sensitivity")
    ap.add_argument("--csv-out", type=str, default="./optimizer/pruning/runs/layer_speed_sensitivity/results.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--targets", type=str, default="", help="Optional comma-separated subset of targets")
    ap.add_argument("--parity-modes", type=str, default="any,even,odd")
    ap.add_argument("--use-half", action="store_true", help="Benchmark in fp16 on CUDA")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads (torch.set_num_threads). Only used when --device cpu.")

    ap.add_argument("--data", type=str, default="../datasets/coco/data.yaml")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-split", type=str, default="val")
    ap.add_argument("--measure-accuracy", action="store_true")
    ap.add_argument("--max-val-batches", type=int, default=0,
                    help="Cap accuracy eval at this many batches (0 = full dataset). "
                         "E.g. 1000 with --eval-batch 1 evaluates ~1000 images.")
    ap.add_argument("--no-preload", action="store_true",
                    help="Do not preload val images into CPU RAM. Required on Jetson/"
                         "Orin where CPU and GPU share the same physical memory pool.")

    args = ap.parse_args()

    if args.device == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)

    # Pre-compute val fraction once (reads the val dir to count images)
    val_fraction = 1.0
    if args.measure_accuracy and args.max_val_batches > 0:
        val_fraction = compute_val_fraction(
            args.data, args.eval_split, args.eval_batch, args.max_val_batches
        )

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.project).mkdir(parents=True, exist_ok=True)

    baseline = YOLO(args.weights)
    base_params = params_count(baseline.model)
    base_mb = fp32_param_mb(baseline.model)
    base_bench = benchmark_forward_latency_ms(
        baseline.model,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        warmup=args.warmup,
        runs=args.runs,
        use_half=args.use_half,
    )

    # Build the persistent evaluator once (dataloader constructed here, reused for all configs).
    # Falls back gracefully to per-call YOLO.val() if Ultralytics internals break.
    evaluator: PersistentCOCOEvaluator | None = None
    base_acc = None
    if args.measure_accuracy:
        try:
            evaluator = PersistentCOCOEvaluator(
                weights=args.weights,
                data=args.data,
                imgsz=args.imgsz,
                device=args.device,
                batch=args.eval_batch,
                split=args.eval_split,
                val_fraction=val_fraction,
                project=args.project,
                preload=not args.no_preload,
            )
            base_acc = evaluator.eval(baseline.model)
        except Exception as e:
            import traceback as _tb
            print(f"\n  [ERROR] PersistentCOCOEvaluator failed — full traceback:")
            _tb.print_exc()
            print(f"  [warn] falling back to YOLO.val() per config\n")
            evaluator = None
            base_eval_runner = YOLO(args.weights)
            base_acc = eval_model_no_train(
                base_eval_runner,
                data=args.data,
                imgsz=args.imgsz,
                device=args.device,
                batch=args.eval_batch,
                split=args.eval_split,
                project=args.project,
                val_fraction=val_fraction,
            )

    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets.strip() else list_prunable_targets(args.weights)
    if args.limit > 0:
        targets = targets[:args.limit]

    parity_modes = [x.strip() for x in args.parity_modes.split(",") if x.strip()]

    total_iters = len(targets) * len(parity_modes)
    progress_i = 0

    fieldnames = [
        "target",
        "target_channels_before",
        "parity_mode",
        "requested_keep",
        "actual_keep",
        "prune_pct_requested",
        "prune_pct_actual",
        "params_after",
        "saved_params",
        "saved_params_pct",
        "saved_mb",
        "base_lat_mean_ms",
        "lat_mean_ms",
        "lat_median_ms",
        "lat_std_ms",
        "speedup_ratio",
        "lat_delta_ms",
        "base_fps_mean",
        "fps_mean",
        "base_map50_95",
        "map50_95",
        "acc_delta_abs",
        "acc_delta_pct",
        "base_map50",
        "map50",
        "status",
        "error",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for target in targets:
            try:
                fresh = YOLO(args.weights)
                channels = target_channel_count(fresh.model, target)
                keep_counts = candidate_keep_counts(channels, args.step_pct)

                for parity_mode in parity_modes:
                    progress_i += 1
                    print(
                        f"[{progress_i}/{total_iters}] target={target} parity={parity_mode} channels={channels}",
                        flush=True,
                    )

                    for requested_keep in keep_counts:
                        row = {
                            "target": target,
                            "target_channels_before": channels,
                            "parity_mode": parity_mode,
                            "requested_keep": requested_keep,
                            "actual_keep": "",
                            "prune_pct_requested": round(100.0 * (1.0 - requested_keep / channels), 4),
                            "prune_pct_actual": "",
                            "params_after": "",
                            "saved_params": "",
                            "saved_params_pct": "",
                            "saved_mb": "",
                            "base_lat_mean_ms": round(base_bench["lat_mean_ms"], 6),
                            "lat_mean_ms": "",
                            "lat_median_ms": "",
                            "lat_std_ms": "",
                            "speedup_ratio": "",
                            "lat_delta_ms": "",
                            "base_fps_mean": round(base_bench["fps_mean"], 6),
                            "fps_mean": "",
                            "base_map50_95": round(base_acc["map50_95"], 6) if base_acc is not None else "",
                            "map50_95": "",
                            "acc_delta_abs": "",
                            "acc_delta_pct": "",
                            "base_map50": round(base_acc["map50"], 6) if base_acc is not None else "",
                            "map50": "",
                            "status": "ok",
                            "error": "",
                        }

                        try:
                            spec = adjust_keep_count(requested_keep, channels, parity_mode)
                            row["actual_keep"] = spec.feasible
                            row["prune_pct_actual"] = round(100.0 * (1.0 - spec.feasible / channels), 4)

                            print(
                                f"  requested_keep={requested_keep} actual_keep={spec.feasible}",
                                flush=True,
                            )

                            y = YOLO(args.weights)
                            _ = apply_exact_keep_to_target(y.model, target, spec.feasible)

                            # Forward sanity
                            y.model = y.model.to(args.device).eval()
                            p0 = next(y.model.parameters())
                            dtype = torch.float16 if (args.use_half and str(args.device).startswith("cuda")) else p0.dtype
                            if dtype == torch.float16:
                                y.model = y.model.half()

                            x = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device, dtype=dtype)
                            with torch.inference_mode():
                                _ = y.model(x)

                            params_after = params_count(y.model)
                            saved_params = base_params - params_after
                            saved_mb = base_mb - fp32_param_mb(y.model.float())

                            row["params_after"] = params_after
                            row["saved_params"] = saved_params
                            row["saved_params_pct"] = round(100.0 * saved_params / base_params, 4)
                            row["saved_mb"] = round(saved_mb, 6)

                            # Accuracy eval in fp32
                            if args.measure_accuracy:
                                pruned_fp32 = copy.deepcopy(y.model).float().eval()
                                if evaluator is not None:
                                    # Fast path: reuse pre-built dataloader, no disk IO
                                    acc = evaluator.eval(pruned_fp32)
                                else:
                                    # Fallback: full YOLO.val() (rebuilds dataloader each time)
                                    eval_runner = YOLO(args.weights)
                                    eval_runner.model = pruned_fp32
                                    acc = eval_model_no_train(
                                        eval_runner,
                                        data=args.data,
                                        imgsz=args.imgsz,
                                        device=args.device,
                                        batch=args.eval_batch,
                                        split=args.eval_split,
                                        project=args.project,
                                        val_fraction=val_fraction,
                                    )

                                row["map50_95"] = round(acc["map50_95"], 6)
                                row["map50"] = round(acc["map50"], 6)

                                if base_acc is not None and base_acc["map50_95"] > 0:
                                    acc_delta_abs = acc["map50_95"] - base_acc["map50_95"]
                                    acc_delta_pct = (acc_delta_abs / base_acc["map50_95"]) * 100.0
                                    row["acc_delta_abs"] = round(acc_delta_abs, 6)
                                    row["acc_delta_pct"] = round(acc_delta_pct, 4)

                            # Speed benchmark
                            bench = benchmark_forward_latency_ms(
                                y.model,
                                imgsz=args.imgsz,
                                batch=args.batch,
                                device=args.device,
                                warmup=args.warmup,
                                runs=args.runs,
                                use_half=args.use_half,
                            )

                            row["lat_mean_ms"] = round(bench["lat_mean_ms"], 6)
                            row["lat_median_ms"] = round(bench["lat_median_ms"], 6)
                            row["lat_std_ms"] = round(bench["lat_std_ms"], 6)
                            row["fps_mean"] = round(bench["fps_mean"], 6)
                            row["speedup_ratio"] = (
                                round(base_bench["lat_mean_ms"] / bench["lat_mean_ms"], 6)
                                if bench["lat_mean_ms"] > 0
                                else ""
                            )
                            row["lat_delta_ms"] = round(base_bench["lat_mean_ms"] - bench["lat_mean_ms"], 6)

                        except Exception as e:
                            row["status"] = "error"
                            row["error"] = f"{type(e).__name__}: {e}"
                            traceback.print_exc()

                        writer.writerow(row)
                        f.flush()

            except Exception:
                traceback.print_exc()

    print(f"Done. CSV: {csv_path}")


if __name__ == "__main__":
    main()