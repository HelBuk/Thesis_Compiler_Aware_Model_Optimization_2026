#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from optimizer.evaluation.yolo_metrics import (
    Backend,
    TorchBackend,
    build_backend,
    detect_best_device,
    eval_backend,
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
class CompactImageSample:
    img_shape: Tuple[int, ...]
    meta: Dict[str, Any]
    pred: Dict[str, torch.Tensor]

    def to_validator_batch(self) -> Dict[str, Any]:
        batch = {k: clone_value(v) for k, v in self.meta.items()}
        batch["img"] = torch.empty(self.img_shape, dtype=torch.float32)
        return batch

    def pred_clone(self) -> Dict[str, torch.Tensor]:
        return clone_pred(self.pred)


def _slice_sequence_for_image(value: Any, i: int) -> Any:
    if isinstance(value, list):
        return [clone_value(value[i])]
    if isinstance(value, tuple):
        return (clone_value(value[i]),)
    return clone_value(value)


def split_batch_to_compact_samples(
    batch_data: Dict[str, Any],
    preds: List[Dict[str, torch.Tensor]],
) -> List[CompactImageSample]:
    imgs = batch_data["img"]
    bsz = int(imgs.shape[0])

    batch_idx_all = batch_data.get("batch_idx", None)
    if batch_idx_all is not None:
        batch_idx_all = batch_idx_all.detach().cpu()

    samples: List[CompactImageSample] = []

    for i in range(bsz):
        meta: Dict[str, Any] = {}
        img_shape = tuple(imgs[i : i + 1].shape)

        for k, v in batch_data.items():
            if k == "img":
                continue
            elif k in ("cls", "bboxes", "masks", "segments", "keypoints"):
                continue
            elif k == "batch_idx":
                continue
            elif isinstance(v, torch.Tensor):
                if v.ndim > 0 and v.shape[0] == bsz:
                    meta[k] = v[i : i + 1].detach().cpu().clone()
                else:
                    meta[k] = v.detach().cpu().clone()
            elif isinstance(v, (list, tuple)) and len(v) == bsz:
                meta[k] = _slice_sequence_for_image(v, i)
            else:
                meta[k] = clone_value(v)

        if batch_idx_all is not None:
            img_mask = batch_idx_all.view(-1) == i

            if "cls" in batch_data:
                meta["cls"] = batch_data["cls"].detach().cpu()[img_mask].clone()
            else:
                meta["cls"] = torch.zeros((0, 1), dtype=torch.float32)

            if "bboxes" in batch_data:
                meta["bboxes"] = batch_data["bboxes"].detach().cpu()[img_mask].clone()
            else:
                meta["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)

            if "masks" in batch_data and isinstance(batch_data["masks"], torch.Tensor):
                meta["masks"] = batch_data["masks"].detach().cpu()[img_mask].clone()

            if "segments" in batch_data and isinstance(batch_data["segments"], list):
                seg_idx = torch.where(img_mask)[0].tolist()
                meta["segments"] = [clone_value(batch_data["segments"][j]) for j in seg_idx]

            if "keypoints" in batch_data and isinstance(batch_data["keypoints"], torch.Tensor):
                meta["keypoints"] = batch_data["keypoints"].detach().cpu()[img_mask].clone()

            meta["batch_idx"] = torch.zeros((int(img_mask.sum().item()),), dtype=torch.int64)
        else:
            meta["batch_idx"] = torch.zeros((0,), dtype=torch.int64)
            meta["cls"] = torch.zeros((0, 1), dtype=torch.float32)
            meta["bboxes"] = torch.zeros((0, 4), dtype=torch.float32)

        samples.append(
            CompactImageSample(
                img_shape=img_shape,
                meta=meta,
                pred=clone_pred(preds[i]),
            )
        )

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
    workers: int = 0,
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
        workers=workers,
    )

    names = v.data.get("names", None)
    if names is None:
        raise RuntimeError("Dataset is missing 'names' in data.yaml.")

    v.init_metrics(model=_NamesOnlyModel(names))
    return v


class SampleStore:
    def append_many(self, samples: List[CompactImageSample]) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        pass

    def __len__(self) -> int:
        raise NotImplementedError

    def iter_all(self) -> Iterator[CompactImageSample]:
        raise NotImplementedError

    def iter_counts(self, counts: np.ndarray) -> Iterator[CompactImageSample]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RAMSampleStore(SampleStore):
    def __init__(self):
        self.samples: List[CompactImageSample] = []

    def append_many(self, samples: List[CompactImageSample]) -> None:
        self.samples.extend(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def iter_all(self) -> Iterator[CompactImageSample]:
        yield from self.samples

    def iter_counts(self, counts: np.ndarray) -> Iterator[CompactImageSample]:
        if len(counts) != len(self.samples):
            raise ValueError("counts length does not match RAMSampleStore size")
        for i, c in enumerate(counts.tolist()):
            if c <= 0:
                continue
            s = self.samples[i]
            for _ in range(c):
                yield s


class DiskSampleStore(SampleStore):
    def __init__(self, root_dir: str, prefix: str, shard_size: int = 512, cleanup: bool = False):
        self.root_dir = Path(root_dir) / prefix
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self.cleanup = cleanup

        self.buffer: List[CompactImageSample] = []
        self.shard_paths: List[Path] = []
        self.shard_lengths: List[int] = []
        self.total = 0
        self._shard_id = 0

    def _flush(self) -> None:
        if not self.buffer:
            return
        shard_path = self.root_dir / f"{self.prefix}_shard_{self._shard_id:06d}.pt"
        torch.save(self.buffer, shard_path)
        self.shard_paths.append(shard_path)
        self.shard_lengths.append(len(self.buffer))
        self.total += len(self.buffer)
        self._shard_id += 1
        self.buffer = []

    def append_many(self, samples: List[CompactImageSample]) -> None:
        self.buffer.extend(samples)
        while len(self.buffer) >= self.shard_size:
            chunk = self.buffer[: self.shard_size]
            self.buffer = self.buffer[self.shard_size :]
            shard_path = self.root_dir / f"{self.prefix}_shard_{self._shard_id:06d}.pt"
            torch.save(chunk, shard_path)
            self.shard_paths.append(shard_path)
            self.shard_lengths.append(len(chunk))
            self.total += len(chunk)
            self._shard_id += 1

    def finalize(self) -> None:
        self._flush()

    def __len__(self) -> int:
        return self.total + len(self.buffer)

    def iter_all(self) -> Iterator[CompactImageSample]:
        self.finalize()
        for shard_path in self.shard_paths:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            for s in shard:
                yield s

    def iter_counts(self, counts: np.ndarray) -> Iterator[CompactImageSample]:
        self.finalize()
        expected_len = sum(self.shard_lengths)
        if len(counts) != expected_len:
            raise ValueError(
                f"counts length ({len(counts)}) does not match DiskSampleStore size ({expected_len})"
            )

        pos = 0
        for shard_path, shard_len in zip(self.shard_paths, self.shard_lengths):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            for local_i in range(shard_len):
                c = int(counts[pos])
                if c > 0:
                    s = shard[local_i]
                    for _ in range(c):
                        yield s
                pos += 1

    def close(self) -> None:
        if self.cleanup and self.root_dir.exists():
            shutil.rmtree(self.root_dir, ignore_errors=True)


def build_sample_store(
    mode: str,
    cache_dir: str,
    prefix: str,
    shard_size: int,
    cleanup: bool,
) -> SampleStore:
    mode = mode.lower()
    if mode == "ram":
        return RAMSampleStore()
    if mode == "disk":
        return DiskSampleStore(
            root_dir=cache_dir,
            prefix=prefix,
            shard_size=shard_size,
            cleanup=cleanup,
        )
    raise ValueError(f"Unsupported sample store mode: {mode}")


def stats_from_store(
    store: SampleStore,
    data_yaml: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    project: str,
    name: str,
    counts: Optional[np.ndarray] = None,
    save_json: bool = False,
    save_txt: bool = False,
    plots: bool = False,
    workers: int = 0,
) -> Dict[str, float]:
    v = init_validator_for_metrics(
        data_yaml=data_yaml,
        imgsz=imgsz,
        batch=1,
        conf=conf,
        iou=iou,
        max_det=max_det,
        project=project,
        name=name,
        save_json=save_json,
        save_txt=save_txt,
        plots=plots,
        workers=workers,
    )

    iterator = store.iter_all() if counts is None else store.iter_counts(counts)
    for s in iterator:
        v.update_metrics([s.pred_clone()], s.to_validator_batch())

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
    store: SampleStore,
    workers: int = 0,
) -> SampleStore:
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
        workers=workers,
    )

    v.dataloader = v.get_dataloader(v.data[v.args.split], batch)
    print(f"[DEBUG dataloader] workers={getattr(v.dataloader, 'num_workers', 'N/A')}")

    seen = 0
    for batch_i, batch_data in enumerate(v.dataloader):
        batch_data = v.preprocess(batch_data)
        imgs = batch_data["img"]
        preds = backend.infer_batch(imgs)

        if batch_i == 0:
            p0 = preds[0]
            print("[DEBUG] first batch preds[0]:", p0["bboxes"].shape, p0["conf"].shape, p0["cls"].shape)

        compact_samples = split_batch_to_compact_samples(batch_data, preds)
        store.append_many(compact_samples)
        seen += len(compact_samples)

        del imgs, preds, batch_data, compact_samples

        if (batch_i + 1) % 200 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if (batch_i + 1) % 100 == 0:
            print(f"[COLLECT] {name}: processed {seen} images")

        if max_batches and batch_i + 1 >= max_batches:
            break

    store.finalize()
    return store


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.percentile(values, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(values, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def bootstrap_pvalue_from_deltas(deltas: np.ndarray) -> float:
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
    store_a: SampleStore,
    store_b: SampleStore,
    data_yaml: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    project: str,
    metric_keys: List[str],
    iters: int,
    seed: int,
    workers: int = 0,
) -> Dict[str, Dict[str, float]]:
    if len(store_a) != len(store_b):
        raise RuntimeError(f"sample count mismatch: A={len(store_a)} B={len(store_b)}")

    n = len(store_a)
    if n == 0:
        raise RuntimeError("No samples collected; cannot bootstrap.")

    rng = np.random.default_rng(seed)
    deltas_by_metric = {k: np.zeros((iters,), dtype=np.float64) for k in metric_keys}

    for t in range(iters):
        idx = rng.integers(0, n, size=n, endpoint=False)
        counts = np.bincount(idx, minlength=n)

        stats_a_t = stats_from_store(
            store=store_a,
            data_yaml=data_yaml,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            project=project,
            name=f"bootstrap_A_{t}",
            counts=counts,
            save_json=False,
            save_txt=False,
            plots=False,
            workers=workers,
        )
        stats_b_t = stats_from_store(
            store=store_b,
            data_yaml=data_yaml,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            project=project,
            name=f"bootstrap_B_{t}",
            counts=counts,
            save_json=False,
            save_txt=False,
            plots=False,
            workers=workers,
        )

        for k in metric_keys:
            deltas_by_metric[k][t] = float(stats_b_t[k]) - float(stats_a_t[k])

        if (t + 1) % 10 == 0 or (t + 1) == iters:
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
        "Evaluate two detection models and test significance with memory-efficient paired bootstrap"
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

    ap.add_argument("--backend_a", choices=["tflite", "ort", "torch", "tensorrt", "trt_engine"], default=None)
    ap.add_argument("--backend_b", choices=["tflite", "ort", "torch", "tensorrt", "trt_engine"], default=None)
    ap.add_argument("--backend", choices=["tflite", "ort", "torch", "tensorrt", "trt_engine"], default=None)

    ap.add_argument("--trt_fp16", action="store_true")
    ap.add_argument("--trt_int8", action="store_true")
    ap.add_argument("--trt_engine_cache", action="store_true")
    ap.add_argument("--trt_engine_cache_path", type=str, default="./trt_cache")
    ap.add_argument("--trt_workspace_size", type=int, default=2147483648)

    ap.add_argument("--device_a", default="cpu")
    ap.add_argument("--device_b", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tflite_delegate", choices=["cpu", "gpu"], default="cpu")

    ap.add_argument("--workers", type=int, default=0, help="Dataloader workers for Ultralytics validator")

    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--bootstrap_seed", type=int, default=42)
    ap.add_argument("--bootstrap_metrics", nargs="*", default=None)
    ap.add_argument("--skip_bootstrap", action="store_true")

    ap.add_argument("--sample_store", choices=["ram", "disk"], default="disk")
    ap.add_argument("--cache_dir", type=str, default="./bootstrap_cache")
    ap.add_argument("--cache_shard_size", type=int, default=512)
    ap.add_argument("--cleanup_cache", action="store_true")

    return ap.parse_args()


def free_backend(backend: Optional[Backend]) -> None:
    if backend is None:
        return
    try:
        backend.close()
    except Exception:
        pass
    if isinstance(backend, TorchBackend):
        try:
            backend.model.to("cpu")
        except Exception:
            pass
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    backend_a_kind = args.backend_a or args.backend
    backend_b_kind = args.backend_b or args.backend
    if backend_a_kind is None or backend_b_kind is None:
        raise SystemExit("Specify either --backend (both) or both --backend_a and --backend_b.")

    if args.skip_bootstrap:
        print(f"[A] {args.model_a} (backend={backend_a_kind}, device={args.device_a or detect_best_device()})")
        backend_a = build_backend(backend_a_kind, args.model_a, args, device_override=args.device_a)
        stats_a = eval_backend(
            backend=backend_a,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            project=args.project,
            name="A_eval",
            save_json=args.save_json,
            save_txt=args.save_txt,
            plots=args.plots,
            max_batches=args.max_batches,
            workers=args.workers,
        )
        free_backend(backend_a)

        print(f"\n[B] {args.model_b} (backend={backend_b_kind}, device={args.device_b or detect_best_device()})")
        backend_b = build_backend(backend_b_kind, args.model_b, args, device_override=args.device_b)
        stats_b = eval_backend(
            backend=backend_b,
            data_yaml=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            project=args.project,
            name="B_eval",
            save_json=args.save_json,
            save_txt=args.save_txt,
            plots=args.plots,
            max_batches=args.max_batches,
            workers=args.workers,
        )
        free_backend(backend_b)

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
        return

    store_a: Optional[SampleStore] = None
    store_b: Optional[SampleStore] = None
    backend_a: Optional[Backend] = None
    backend_b: Optional[Backend] = None

    try:
        print(f"[A] {args.model_a} (backend={backend_a_kind}, device={args.device_a or detect_best_device()})")
        backend_a = build_backend(backend_a_kind, args.model_a, args, device_override=args.device_a)
        store_a = build_sample_store(
            mode=args.sample_store,
            cache_dir=args.cache_dir,
            prefix="A",
            shard_size=args.cache_shard_size,
            cleanup=args.cleanup_cache,
        )
        collect_samples_for_backend(
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
            store=store_a,
            workers=args.workers,
        )
        print(f"[A] collected {len(store_a)} compact samples")
        free_backend(backend_a)
        backend_a = None

        print(f"\n[B] {args.model_b} (backend={backend_b_kind}, device={args.device_b or detect_best_device()})")
        backend_b = build_backend(backend_b_kind, args.model_b, args, device_override=args.device_b)
        store_b = build_sample_store(
            mode=args.sample_store,
            cache_dir=args.cache_dir,
            prefix="B",
            shard_size=args.cache_shard_size,
            cleanup=args.cleanup_cache,
        )
        collect_samples_for_backend(
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
            store=store_b,
            workers=args.workers,
        )
        print(f"[B] collected {len(store_b)} compact samples")
        free_backend(backend_b)
        backend_b = None

        if len(store_a) != len(store_b):
            raise RuntimeError(f"Image count mismatch between A and B: {len(store_a)} vs {len(store_b)}")

        stats_a = stats_from_store(
            store=store_a,
            data_yaml=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            project=args.project,
            name="A_eval",
            save_json=args.save_json,
            save_txt=args.save_txt,
            plots=args.plots,
            workers=args.workers,
        )
        stats_b = stats_from_store(
            store=store_b,
            data_yaml=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            project=args.project,
            name="B_eval",
            save_json=args.save_json,
            save_txt=args.save_txt,
            plots=args.plots,
            workers=args.workers,
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

        metric_keys = choose_metric_keys(stats_a, stats_b, args.bootstrap_metrics)

        print("\n===== Paired bootstrap significance test =====")
        print(f"images: {len(store_a)}")
        print(f"bootstrap iterations: {args.bootstrap_iters}")
        print(f"seed: {args.bootstrap_seed}")
        print(f"metrics: {metric_keys}")
        print(f"sample store: {args.sample_store}")

        boot = run_paired_bootstrap(
            store_a=store_a,
            store_b=store_b,
            data_yaml=args.data,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            project=args.project,
            metric_keys=metric_keys,
            iters=args.bootstrap_iters,
            seed=args.bootstrap_seed,
            workers=args.workers,
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
            print(f"%diff (B - A)/A: {(d/a):+.6f}")
            print(f"bootstrap mean delta: {s['delta_mean_boot']:+.6f}")
            print(f"95% CI: [{s['ci95_lo']:+.6f}, {s['ci95_hi']:+.6f}]")
            print(f"approx. two-sided p-value: {s['pvalue_two_sided']:.6g}")
            print(f"significant at 95% level: {significant}")

    finally:
        free_backend(backend_a)
        free_backend(backend_b)
        if store_a is not None:
            store_a.close()
        if store_b is not None:
            store_b.close()


if __name__ == "__main__":
    main()