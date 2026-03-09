#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from optimizer.evaluation.yolo_metrics import (
    Backend,
    build_backend,
    detect_best_device,
    make_validator,
)


def clone_value(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, np.ndarray):
        return np.array(x, copy=True)
    if isinstance(x, list):
        return [clone_value(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone_value(v) for v in x)
    if isinstance(x, dict):
        return {k: clone_value(v) for k, v in x.items()}
    return copy.deepcopy(x)


def clone_pred(pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in pred.items()}


class _NamesOnlyModel:
    def __init__(self, names):
        self.names = names


@dataclass
class ImageSample:
    batch_data: Dict[str, Any]
    pred: Dict[str, torch.Tensor]


def _slice_sequence_for_image(value: Any, i: int) -> Any:
    if isinstance(value, list):
        return [clone_value(value[i])]
    if isinstance(value, tuple):
        return (clone_value(value[i]),)
    return clone_value(value)


def split_batch_to_samples(
    batch_data: Dict[str, Any],
    preds: List[Dict[str, torch.Tensor]],
) -> List[ImageSample]:
    """
    Convert one preprocessed Ultralytics batch into a list of single-image samples.
    Each sample can later be fed back into DetectionValidator.update_metrics.
    """
    imgs = batch_data["img"]
    bsz = int(imgs.shape[0])

    batch_idx_all = batch_data.get("batch_idx", None)
    if batch_idx_all is not None:
        batch_idx_all = batch_idx_all.detach().cpu()

    samples: List[ImageSample] = []

    for i in range(bsz):
        sample: Dict[str, Any] = {}

        for k, v in batch_data.items():
            if k == "img":
                sample["img"] = v[i : i + 1].detach().cpu().clone()
            elif k in ("cls", "bboxes", "masks", "segments", "keypoints"):
                continue
            elif k == "batch_idx":
                continue
            elif isinstance(v, torch.Tensor):
                if v.ndim > 0 and v.shape[0] == bsz:
                    sample[k] = v[i : i + 1].detach().cpu().clone()
                else:
                    sample[k] = v.detach().cpu().clone()
            elif isinstance(v, (list, tuple)) and len(v) == bsz:
                sample[k] = _slice_sequence_for_image(v, i)
            else:
                sample[k] = clone_value(v)

        if batch_idx_all is not None:
            img_mask = (batch_idx_all.view(-1) == i)

            if "cls" in batch_data:
                cls = batch_data["cls"].detach().cpu()
                sample["cls"] = cls[img_mask].clone()

            if "bboxes" in batch_data:
                bboxes = batch_data["bboxes"].detach().cpu()
                sample["bboxes"] = bboxes[img_mask].clone()

            if "masks" in batch_data and isinstance(batch_data["masks"], torch.Tensor):
                sample["masks"] = batch_data["masks"].detach().cpu()[img_mask].clone()

            if "segments" in batch_data:
                segs = batch_data["segments"]
                if isinstance(segs, list):
                    seg_idx = torch.where(img_mask)[0].tolist()
                    sample["segments"] = [clone_value(segs[j]) for j in seg_idx]

            if "keypoints" in batch_data and isinstance(batch_data["keypoints"], torch.Tensor):
                sample["keypoints"] = batch_data["keypoints"].detach().cpu()[img_mask].clone()

            num_labels = int(img_mask.sum().item())
            sample["batch_idx"] = torch.zeros((num_labels,), dtype=torch.int64)
        else:
            sample["batch_idx"] = torch.zeros((0,), dtype=torch.int64)
            if "cls" not in sample:
                sample["cls"] = torch.zeros((0, 1), dtype=torch.float32)
            if "bboxes" not in sample:
                sample["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)

        samples.append(ImageSample(batch_data=sample, pred=clone_pred(preds[i])))

    return samples


def init_validator_for_metrics(
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    max_det: int,
    project: str,
    name: str,
    save_json: bool,
    save_txt: bool,
    plots: bool,
):
    v = make_validator(
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device="cpu",
        project=project,
        name=name,
        save_json=save_json,
        save_txt=save_txt,
        plots=plots,
    )

    names = v.data.get("names", None)
    if names is None:
        raise RuntimeError(
            "Dataset is missing 'names'. Add a 'names:' block to your data.yaml."
        )

    v.init_metrics(model=_NamesOnlyModel(names))
    return v


def stats_from_samples(
    samples: Sequence[ImageSample],
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    max_det: int,
    project: str,
    name: str,
    save_json: bool = False,
    save_txt: bool = False,
    plots: bool = False,
) -> Dict[str, float]:
    v = init_validator_for_metrics(
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        max_det=max_det,
        project=project,
        name=name,
        save_json=save_json,
        save_txt=save_txt,
        plots=plots,
    )

    for s in samples:
        v.update_metrics([clone_pred(s.pred)], clone_value(s.batch_data))

    return v.get_stats()


def collect_samples_for_backend(
    backend: Backend,
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    max_det: int,
    project: str,
    name: str,
    max_batches: int,
) -> List[ImageSample]:
    """
    Run inference once and cache per-image targets + predictions.
    """
    v = init_validator_for_metrics(
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        max_det=max_det,
        project=project,
        name=name,
        save_json=False,
        save_txt=False,
        plots=False,
    )

    v.dataloader = v.get_dataloader(v.data[v.args.split], batch)

    all_samples: List[ImageSample] = []

    for batch_i, batch_data in enumerate(v.dataloader):
        batch_data = v.preprocess(batch_data)
        imgs = batch_data["img"]
        preds = backend.infer_batch(imgs)

        if batch_i == 0:
            p0 = preds[0]
            print(
                "[DEBUG] first batch preds[0]:",
                p0["bboxes"].shape,
                p0["conf"].shape,
                p0["cls"].shape,
            )

        samples = split_batch_to_samples(batch_data, preds)
        all_samples.extend(samples)

        if max_batches and batch_i + 1 >= max_batches:
            break

    return all_samples


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_pvalue_from_deltas(deltas: np.ndarray) -> float:
    """
    Approximate two-sided p-value from bootstrap delta distribution.
    """
    p_le_0 = float(np.mean(deltas <= 0.0))
    p_ge_0 = float(np.mean(deltas >= 0.0))
    p = 2.0 * min(p_le_0, p_ge_0)
    return min(max(p, 0.0), 1.0)


def choose_metric_keys(
    stats_a: Dict[str, float],
    stats_b: Dict[str, float],
    requested: Optional[List[str]],
) -> List[str]:
    common = [k for k in stats_a.keys() if k in stats_b]
    if requested:
        missing = [k for k in requested if k not in common]
        if missing:
            raise ValueError(
                f"Requested metric(s) not found in stats: {missing}\nAvailable: {sorted(common)}"
            )
        return requested

    preferred = [
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(B)",
        "metrics/recall(B)",
    ]
    out = [k for k in preferred if k in common]
    return out if out else common


def run_paired_bootstrap(
    samples_a: Sequence[ImageSample],
    samples_b: Sequence[ImageSample],
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    max_det: int,
    project: str,
    metric_keys: List[str],
    iters: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    if len(samples_a) != len(samples_b):
        raise RuntimeError(f"sample count mismatch: A={len(samples_a)} B={len(samples_b)}")

    n = len(samples_a)
    if n == 0:
        raise RuntimeError("No samples collected; cannot bootstrap.")

    rng = np.random.default_rng(seed)
    deltas_by_metric = {k: np.zeros((iters,), dtype=np.float64) for k in metric_keys}

    for t in range(iters):
        idx = rng.integers(0, n, size=n, endpoint=False)
        boot_a = [samples_a[j] for j in idx]
        boot_b = [samples_b[j] for j in idx]

        stats_a_t = stats_from_samples(
            samples=boot_a,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=1,
            conf=conf,
            iou=iou,
            max_det=max_det,
            project=project,
            name=f"bootstrap_A_{t}",
            save_json=False,
            save_txt=False,
            plots=False,
        )
        stats_b_t = stats_from_samples(
            samples=boot_b,
            data_yaml=data_yaml,
            imgsz=imgsz,
            batch=1,
            conf=conf,
            iou=iou,
            max_det=max_det,
            project=project,
            name=f"bootstrap_B_{t}",
            save_json=False,
            save_txt=False,
            plots=False,
        )

        for k in metric_keys:
            deltas_by_metric[k][t] = float(stats_b_t[k]) - float(stats_a_t[k])

        if (t + 1) % 50 == 0 or (t + 1) == iters:
            print(f"[BOOTSTRAP] finished {t + 1}/{iters}")

    summary: Dict[str, Dict[str, float]] = {}
    for k, deltas in deltas_by_metric.items():
        ci_lo, ci_hi = percentile_ci(deltas, alpha=0.05)
        summary[k] = {
            "delta_mean_boot": float(np.mean(deltas)),
            "delta_median_boot": float(np.median(deltas)),
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "pvalue_two_sided": bootstrap_pvalue_from_deltas(deltas),
        }
    return summary


def print_stats(title: str, stats: Dict[str, float]):
    print(f"\n===== {title} =====")
    for k, v in stats.items():
        print(f"{k}: {v}")


def parse_args():
    ap = argparse.ArgumentParser(
        "Evaluate two detection models and test significance with paired bootstrap"
    )

    ap.add_argument("--data", required=True, help="Ultralytics dataset YAML")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = full val, else stop after N batches")

    ap.add_argument("--project", type=str, default="runs/val_custom")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--save_json", action="store_true")

    ap.add_argument("--model_a", required=True, help="Path to model A (.tflite/.onnx/.pt/.engine)")
    ap.add_argument("--model_b", required=True, help="Path to model B (.tflite/.onnx/.pt/.engine)")

    ap.add_argument("--backend_a", choices=["tflite", "ort", "torch", "tensorrt"], default=None, help="Backend for model A")
    ap.add_argument("--backend_b", choices=["tflite", "ort", "torch", "tensorrt"], default=None, help="Backend for model B")
    ap.add_argument(
        "--backend",
        choices=["tflite", "ort", "torch", "tensorrt"],
        default=None,
        help="(Deprecated) single backend for both A and B",
    )

    ap.add_argument("--trt_fp16", action="store_true", help="Enable TensorRT FP16")
    ap.add_argument("--trt_int8", action="store_true", help="Enable TensorRT INT8")
    ap.add_argument("--trt_engine_cache", action="store_true", help="Enable TensorRT engine cache")
    ap.add_argument("--trt_engine_cache_path", type=str, default="./trt_cache", help="TensorRT engine cache path")
    ap.add_argument("--trt_workspace_size", type=int, default=2147483648, help="TensorRT workspace size in bytes")

    ap.add_argument(
        "--device_a",
        default="cpu",
        help="Device for A (torch: cpu/mps/cuda:0; ort/tensorrt: cpu/cuda:0)",
    )
    ap.add_argument(
        "--device_b",
        default="cpu",
        help="Device for B (torch: cpu/mps/cuda:0; ort/tensorrt: cpu/cuda:0)",
    )
    ap.add_argument("--threads", type=int, default=4, help="Threads for TFLite")
    ap.add_argument("--tflite_delegate", choices=["cpu", "gpu"], default="cpu", help="TFLite delegate (gpu optional)")

    ap.add_argument("--bootstrap_iters", type=int, default=1000, help="Number of paired bootstrap resamples")
    ap.add_argument("--bootstrap_seed", type=int, default=42)
    ap.add_argument(
        "--bootstrap_metrics",
        nargs="*",
        default=None,
        help=(
            "Metric keys to test, e.g. "
            "'metrics/mAP50(B)' 'metrics/mAP50-95(B)'. "
            "If omitted, common detection metrics are used automatically."
        ),
    )
    ap.add_argument("--skip_bootstrap", action="store_true", help="Only compute point estimates")



    return ap.parse_args()


def main():
    args = parse_args()

    backend_a_kind = args.backend_a or args.backend
    backend_b_kind = args.backend_b or args.backend
    if backend_a_kind is None or backend_b_kind is None:
        raise SystemExit("Specify either --backend (both) or both --backend_a and --backend_b.")

    print(f"[A] {args.model_a} (backend={backend_a_kind}, device={args.device_a or detect_best_device()})")
    backend_a = build_backend(backend_a_kind, args.model_a, args, device_override=args.device_a)
    samples_a = collect_samples_for_backend(
        backend=backend_a,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name="A_collect",
        max_batches=args.max_batches,
    )
    print(f"[A] collected {len(samples_a)} images")

    print(f"\n[B] {args.model_b} (backend={backend_b_kind}, device={args.device_b or detect_best_device()})")
    backend_b = build_backend(backend_b_kind, args.model_b, args, device_override=args.device_b)
    samples_b = collect_samples_for_backend(
        backend=backend_b,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name="B_collect",
        max_batches=args.max_batches,
    )
    print(f"[B] collected {len(samples_b)} images")

    if len(samples_a) != len(samples_b):
        raise RuntimeError(f"Image count mismatch between A and B: {len(samples_a)} vs {len(samples_b)}")

    stats_a = stats_from_samples(
        samples=samples_a,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=1,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name="A_eval",
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
    )
    stats_b = stats_from_samples(
        samples=samples_b,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=1,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name="B_eval",
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
    )

    print_stats("A stats", stats_a)
    print_stats("B stats", stats_b)

    print("\n===== Delta (B - A) =====")
    common_keys = sorted(set(stats_a.keys()) & set(stats_b.keys()))
    for key in common_keys:
        try:
            da = float(stats_a[key])
            db = float(stats_b[key])
            print(f"{key}: {db - da:+.6f}")
        except Exception:
            pass

    if args.skip_bootstrap:
        return

    metric_keys = choose_metric_keys(stats_a, stats_b, args.bootstrap_metrics)

    print("\n===== Paired bootstrap significance test =====")
    print(f"images: {len(samples_a)}")
    print(f"bootstrap iterations: {args.bootstrap_iters}")
    print(f"seed: {args.bootstrap_seed}")
    print(f"metrics: {metric_keys}")

    boot = run_paired_bootstrap(
        samples_a=samples_a,
        samples_b=samples_b,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=1,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        metric_keys=metric_keys,
        iters=args.bootstrap_iters,
        seed=args.bootstrap_seed,
    )

    for k in metric_keys:
        a = float(stats_a[k])
        b = float(stats_b[k])
        d = b - a
        s = boot[k]
        significant = not (s["ci95_lo"] <= 0.0 <= s["ci95_hi"])

        print(f"\n--- {k} ---")
        print(f"A: {a:.6f}")
        print(f"B: {b:.6f}")
        print(f"delta (B - A): {d:+.6f}")
        print(f"bootstrap mean delta: {s['delta_mean_boot']:+.6f}")
        print(f"95% CI: [{s['ci95_lo']:+.6f}, {s['ci95_hi']:+.6f}]")
        print(f"approx. two-sided p-value: {s['pvalue_two_sided']:.6g}")
        print(f"significant at 95% level: {significant}")


if __name__ == "__main__":
    main()