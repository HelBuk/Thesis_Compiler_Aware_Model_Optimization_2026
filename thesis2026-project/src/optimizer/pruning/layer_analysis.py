# layer_analysis.py
from __future__ import annotations

import copy
import argparse
import csv
import re
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

from optimizer.pruning.manual_yolov8_pruner import (
    _norm_target,
    apply_structural_pruning_and_realign,
    eval_model,
    fp32_param_mb,
    make_pruned_trainer,
    params_count,
    prune_c2f_cv1,
    prune_sppf_cv1,
    prune_c2f_bottleneck_cv1,
    prune_c2f_bottleneck_cv2,
    resolve_targets,
)

def clone_trainable_model(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model).train()
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def safe_name(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    return s[:max_len].strip("_") or "target"


def list_conv2d_targets(weights: str) -> list[str]:
    y = YOLO(weights)
    out: list[str] = []
    for name, m in y.model.named_modules():
        if isinstance(m, nn.Conv2d) and not name.startswith("model.22"):
            out.append(name)
    return out


def read_last_losses(results_csv: Path) -> tuple[float | None, float | None, float | None]:
    if not results_csv.exists():
        return None, None, None
    rows: list[dict[str, str]] = []
    with results_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return None, None, None
    last = rows[-1]
    def getf(k: str) -> float | None:
        v = last.get(k, "")
        return float(v) if v not in ("", None) else None
    return getf("train/box_loss"), getf("train/cls_loss"), getf("train/dfl_loss")


def fps_from_lat_ms(lat_ms: float) -> float:
    return 1000.0 / lat_ms if lat_ms > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="../models/yolov8n.pt")
    ap.add_argument("--data", type=str, default="../datasets/coco/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prune-ratio", type=float, default=0.10)
    ap.add_argument("--round-to", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100) 
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--project", type=str, default="./optimizer/pruning/runs/manual_prune")
    ap.add_argument("--name-prefix", type=str, default="sweep")
    ap.add_argument("--csv-out", type=str, default="./optimizer/pruning/runs/manual_prune/sweep_results.csv")
    ap.add_argument("--limit", type=int, default=0, help="0=all targets")
    args = ap.parse_args()

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Baseline once
    base_for_params = YOLO(args.weights)
    base_params = params_count(base_for_params.model)
    base_mb = fp32_param_mb(base_for_params.model)

    base_for_eval = YOLO(args.weights)
    base_stats = eval_model(base_for_eval, args.data, args.imgsz, args.device, args.batch, project=args.project)

    targets = list_conv2d_targets(args.weights)
    if args.limit > 0:
        targets = targets[:args.limit]

    fieldnames = [
        "raw_target",
        "target_norm",
        "status",
        "error",
        "block_targets",
        "c2f_hidden_targets",
        "c2f_bn_cv1_targets",
        "c2f_bn_cv2_targets",
        "sppf_hidden_targets",
        "changed_paths",
        "base_params",
        "pruned_params_preft",
        "postft_params",
        "base_mb",
        "saved_mb_preft",
        "saved_mb_postft",
        "base_map50_95",
        "preft_map50_95",
        "postft_map50_95",
        "base_map50",
        "preft_map50",
        "postft_map50",
        "base_lat_ms",
        "preft_lat_ms",
        "postft_lat_ms",
        "base_fps",
        "preft_fps",
        "postft_fps",
        "train_box_loss_last",
        "train_cls_loss_last",
        "train_dfl_loss_last",
        "run_dir",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for raw_target in targets:
            row = {
                "raw_target": raw_target,
                "target_norm": _norm_target(raw_target),
                "status": "ok",
                "error": "",
                "block_targets": "",
                "c2f_hidden_targets": "",
                "c2f_bn_cv1_targets": "",
                "c2f_bn_cv2_targets": "",
                "sppf_hidden_targets": "",
                "changed_paths": "",
                "base_params": base_params,
                "pruned_params_preft": "",
                "postft_params": "",
                "base_mb": base_mb,
                "saved_mb_preft": "",
                "saved_mb_postft": "",
                "base_map50_95": base_stats["map50_95"],
                "preft_map50_95": "",
                "postft_map50_95": "",
                "base_map50": base_stats["map50"],
                "preft_map50": "",
                "postft_map50": "",
                "base_lat_ms": base_stats["lat_ms"],
                "preft_lat_ms": "",
                "postft_lat_ms": "",
                "base_fps": fps_from_lat_ms(base_stats["lat_ms"]),
                "preft_fps": "",
                "postft_fps": "",
                "train_box_loss_last": "",
                "train_cls_loss_last": "",
                "train_dfl_loss_last": "",
                "run_dir": "",
            }

            try:
                y = YOLO(args.weights)

                resolved = resolve_targets(y.model, raw_target)
                row["block_targets"] = ",".join(map(str, resolved.block_targets))
                row["c2f_hidden_targets"] = ",".join(map(str, resolved.c2f_hidden_targets))
                row["sppf_hidden_targets"] = ",".join(map(str, resolved.sppf_hidden_targets))
                row["c2f_bn_cv1_targets"] = ",".join(f"{li}.{bi}" for li, bi in resolved.c2f_bn_cv1_targets)
                row["c2f_bn_cv2_targets"] = ",".join(f"{li}.{bi}" for li, bi in resolved.c2f_bn_cv2_targets)

                if resolved.unknown:
                    row["status"] = "skip_unknown"
                    row["error"] = "; ".join(resolved.unknown)
                    writer.writerow(row)
                    continue

                if not (
                    resolved.block_targets
                    or resolved.c2f_hidden_targets
                    or resolved.sppf_hidden_targets
                    or resolved.c2f_bn_cv1_targets
                    or resolved.c2f_bn_cv2_targets
                ):
                    row["status"] = "skip_empty_resolution"
                    writer.writerow(row)
                    continue


                changes, skipped, _, _ = apply_structural_pruning_and_realign(
                    y.model,
                    blocks=resolved.block_targets,
                    prune_ratio=args.prune_ratio,
                    round_to=args.round_to,
                    model_in_ch=3,
                )

                for li in resolved.c2f_hidden_targets:
                    ch = prune_c2f_cv1(y.model, li, args.prune_ratio, args.round_to)
                    if ch is not None:
                        changes.append(ch)
                for li in resolved.sppf_hidden_targets:
                    ch = prune_sppf_cv1(y.model, li, args.prune_ratio, args.round_to)
                    if ch is not None:
                        changes.append(ch)

                for li, bi in resolved.c2f_bn_cv1_targets:
                    ch = prune_c2f_bottleneck_cv1(y.model, li, bi, args.prune_ratio, args.round_to)
                    if ch is not None:
                        changes.append(ch)

                for li, bi in resolved.c2f_bn_cv2_targets:
                    ch = prune_c2f_bottleneck_cv2(y.model, li, bi, args.prune_ratio, args.round_to)
                    if ch is not None:
                        changes.append(ch)

                if not changes:
                    row["status"] = "skip_no_change"
                    row["error"] = "; ".join(skipped) if skipped else "no structural change"
                    writer.writerow(row)
                    continue

                row["changed_paths"] = ",".join(c.path for c in changes)

                # Forward sanity
                p0 = next(y.model.parameters())
                x = torch.randn(1, 3, args.imgsz, args.imgsz, device=p0.device, dtype=p0.dtype)
                with torch.no_grad():
                    _ = y.model(x)

                # Pre-FT eval on dedicated eval copy
                preft_runner = YOLO(args.weights)
                preft_runner.model = copy.deepcopy(y.model).eval()
                preft = eval_model(preft_runner, args.data, args.imgsz, args.device, args.batch, project=args.project)

                # Training model on separate trainable copy
                pruned = YOLO(args.weights)
                pruned.model = clone_trainable_model(y.model)

                expected_params = params_count(pruned.model)
                pruned_mb = fp32_param_mb(pruned.model)

                row["pruned_params_preft"] = expected_params
                row["saved_mb_preft"] = base_mb - pruned_mb

                if row["pruned_params_preft"] > base_params:
                    row["status"] = "warn_param_increase"
                    row["error"] = "pruned_params_preft > base_params (possible state mismatch)"


                # row["pruned_params_preft"] = expected_params
                # row["saved_mb_preft"] = base_mb - pruned_mb
                row["preft_map50_95"] = preft["map50_95"]
                row["preft_map50"] = preft["map50"]
                row["preft_lat_ms"] = preft["lat_ms"]
                row["preft_fps"] = fps_from_lat_ms(preft["lat_ms"])

                if args.epochs <= 0:
                    if row["status"] == "ok":
                        row["status"] = "ok_preft_only"
                    else:
                        row["status"] = f"{row['status']}_preft_only"

                    row["postft_params"] = row["pruned_params_preft"]
                    row["saved_mb_postft"] = row["saved_mb_preft"]
                    row["postft_map50_95"] = row["preft_map50_95"]
                    row["postft_map50"] = row["preft_map50"]
                    row["postft_lat_ms"] = row["preft_lat_ms"]
                    row["postft_fps"] = row["preft_fps"]
                    writer.writerow(row)
                    continue


                def _check_train_model(trainer):
                    got = sum(p.numel() for p in trainer.model.parameters())
                    if got != expected_params:
                        raise RuntimeError(f"Trainer rebuilt wrong architecture: got {got}, expected {expected_params}")

                pruned.add_callback("on_train_start", _check_train_model)

                run_name = f"{args.name_prefix}_{safe_name(row['target_norm'])}"
                pruned.train(
                    data=args.data,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    workers=args.workers,
                    device=args.device,
                    fraction=args.fraction,
                    project=args.project,
                    name=run_name,
                    pretrained=False,
                    resume=False,
                    trainer=make_pruned_trainer(pruned.model),
                )

                save_dir = Path(pruned.trainer.save_dir)
                row["run_dir"] = str(save_dir)

                # last training losses
                box_l, cls_l, dfl_l = read_last_losses(save_dir / "results.csv")
                row["train_box_loss_last"] = box_l
                row["train_cls_loss_last"] = cls_l
                row["train_dfl_loss_last"] = dfl_l

                best_pt = save_dir / "weights" / "best.pt"
                last_pt = save_dir / "weights" / "last.pt"
                final_pt = best_pt if best_pt.exists() else last_pt

                finetuned = YOLO(str(final_pt))
                postft = eval_model(finetuned, args.data, args.imgsz, args.device, args.batch, project=args.project)
                postft_params = params_count(finetuned.model)
                postft_mb = fp32_param_mb(finetuned.model)

                row["postft_params"] = postft_params
                row["saved_mb_postft"] = base_mb - postft_mb
                row["postft_map50_95"] = postft["map50_95"]
                row["postft_map50"] = postft["map50"]
                row["postft_lat_ms"] = postft["lat_ms"]
                row["postft_fps"] = fps_from_lat_ms(postft["lat_ms"])

            except Exception as e:
                row["status"] = "error"
                row["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()

            writer.writerow(row)

    print(f"Done. CSV: {csv_path}")


if __name__ == "__main__":
    main()
