# OCSPrunner_test.py
import os
import sys
import copy
import math
import argparse
from dataclasses import dataclass, field

import torch
import numpy as np
from ultralytics import YOLO


def import_ocs_modules(ocspruner_root: str):
    root = os.path.abspath(os.path.expanduser(ocspruner_root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"OCS root not found: {root}")

    if root not in sys.path:
        sys.path.insert(0, root)

    try:
        import vainf_torch_pruning.torch_pruning as tp
        from ocspruner.pruner import OCSPruner
        from ocspruner.importance import GroupNormV2Importance
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Failed importing OCS modules from {root}. "
            f"Expected {root}/vainf_torch_pruning and {root}/ocspruner"
        ) from e

    return tp, OCSPruner, GroupNormV2Importance



def build_ignored_layers_for_yolo(det_model):
    ignored = []

    # Prefer explicit Detect-like head ignore
    head = None
    if hasattr(det_model, "model") and len(det_model.model) > 0:
        head = det_model.model[-1]

    if head is not None and "detect" in head.__class__.__name__.lower():
        for m in head.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):
                ignored.append(m)

    return ignored


def make_pruner(det_model, example_inputs, reg, group_cost, OCSPruner, GroupNormV2Importance):
    return OCSPruner(
        model=det_model,
        example_inputs=example_inputs,
        importance=GroupNormV2Importance(p=2),
        reg=reg,
        ignored_layers=build_ignored_layers_for_yolo(det_model),
        group_cost=group_cost,
    )


def compute_group_cost(det_model, example_inputs, reg, OCSPruner, GroupNormV2Importance, tp):
    base_ops, _ = tp.utils.count_ops_and_params(det_model, example_inputs=example_inputs)
    base_pruner = make_pruner(
        det_model, example_inputs, reg, None, OCSPruner, GroupNormV2Importance
    )

    flops_groups = {"base_model": (base_ops, -1, -1.0)}
    n_groups = len(base_pruner.groups_prune_candidate)

    for i in range(n_groups):
        m_tmp = copy.deepcopy(det_model)
        p_tmp = make_pruner(m_tmp, example_inputs, reg, None, OCSPruner, GroupNormV2Importance)

        g = p_tmp.groups_prune_candidate[i]
        module = g[0][0].target.module
        prune_fn = g[0][0].handler
        idxs = list(g[0][1])

        cand = idxs[:-1] if len(idxs) > 1 else idxs
        if len(cand) == 0:
            flops_groups[f"group{i+1}"] = (0.0, len(idxs), 0.0)
            continue

        pg = p_tmp.DG.get_pruning_group(module, prune_fn, cand)
        if p_tmp.DG.check_pruning_group(pg):
            pg.prune()

        pruned_ops, _ = tp.utils.count_ops_and_params(m_tmp, example_inputs=example_inputs)
        cost = float(base_ops - pruned_ops) / max(1, len(cand))
        flops_groups[f"group{i+1}"] = (cost, len(idxs), 0.0)

    vals = [v[0] for k, v in flops_groups.items() if k != "base_model"]
    if len(vals) == 0:
        return flops_groups

    exp_sum = sum(math.exp(v / 1e6) for v in vals)
    probs = [math.exp(v / 1e6) / exp_sum for v in vals]

    keys = [k for k in flops_groups.keys() if k != "base_model"]
    for i, key in enumerate(keys):
        c, n, _ = flops_groups[key]
        flops_groups[key] = (c, n, probs[i])

    return flops_groups


def search_groups_for_target_flops(
    pruner, target_flops_rr, layer_prune_limit, OCSPruner, GroupNormV2Importance
):
    base_ops, _ = pruner.get_model_info()
    all_groups = pruner.get_all_group_imp_scores(layer_max_prune_limit=layer_prune_limit)
    if len(all_groups) == 0:
        return None

    all_imp = torch.cat([g[-1] for g in all_groups], dim=0)
    sorted_vals, sorted_idx = torch.sort(all_imp)
    keep = torch.nonzero(sorted_vals != 100.0, as_tuple=False).flatten()
    if keep.numel() == 0:
        return None
    sorted_idx = sorted_idx[keep]

    left, right = 0, len(sorted_idx) - 1
    best = None

    while left <= right:
        mid = (left + right) // 2
        m_tmp = copy.deepcopy(pruner.model)
        p_tmp = make_pruner(
            m_tmp, pruner.example_inputs, pruner.reg, pruner.group_cost, OCSPruner, GroupNormV2Importance
        )

        gidx = pruner.get_group_pruning_indices(all_groups, all_imp, sorted_idx[mid])
        p_tmp.pruning(gidx)

        pruned_ops, _ = p_tmp.get_model_info()
        flop_rr = 1.0 - float(pruned_ops) / base_ops
        best = (gidx, flop_rr)

        if abs(flop_rr - target_flops_rr) <= 5e-4:
            break
        if flop_rr <= target_flops_rr:
            left = mid + 1
        else:
            right = mid - 1

    return best


def jaccard(a, b):
    inter = len(set(a).intersection(b))
    union = len(a) + len(b) - inter
    return float(inter) / max(1, union)


class StabilityTracker:
    def __init__(self, win=3):
        self.win = win
        self.net_hist = []
        self.raw = []
        self.avg = []

    def update(self, pruner):
        net = pruner.get_pruned_net_structure()
        self.net_hist.append(net)
        if len(self.net_hist) > self.win:
            self.net_hist.pop(0)

        if len(self.net_hist) < self.win:
            raw = 0.0
        else:
            first, last = self.net_hist[0], self.net_hist[-1]
            keys = [k for k in first.keys() if k in last]
            if len(keys) == 0:
                raw = 0.0
            else:
                sims = [jaccard(first[k], last[k]) for k in keys]
                raw = float(np.mean(sims))

        self.raw.append(raw)
        self.avg.append(float(np.mean(self.raw[-self.win:])))
        return raw, self.avg[-1]


def add_callback_safe(yolo_model, event, fn):
    try:
        yolo_model.add_callback(event, fn)
        return True
    except Exception:
        pass

    try:
        if hasattr(yolo_model, "callbacks") and event in yolo_model.callbacks:
            yolo_model.callbacks[event].append(fn)
            return True
    except Exception:
        pass

    return False


@dataclass
class OCSState:
    example: torch.Tensor = None
    pruner: object = None
    group_cost: dict = None
    tracker: StabilityTracker = None
    sl_started: bool = False
    sl_start_epoch: int = -1
    reg: float = 1e-4
    final_groups: dict = field(default_factory=dict)


def main(args):
    tp, OCSPruner, GroupNormV2Importance = import_ocs_modules(args.ocspruner_root)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    y = YOLO(args.weights)
    y.model.to(device)

    state = OCSState(
        example=torch.randn(1, 3, args.imgsz, args.imgsz, device=device),
        tracker=StabilityTracker(win=3),
        reg=args.reg,
    )

    def on_train_start(trainer):
        model = trainer.model
        state.pruner = make_pruner(
            model, state.example, state.reg, None, OCSPruner, GroupNormV2Importance
        )
        state.group_cost = compute_group_cost(
            model, state.example, state.reg, OCSPruner, GroupNormV2Importance, tp
        )
        state.pruner.group_cost = state.group_cost
        print(f"[OCS] initialized: groups={len(state.pruner.groups_prune_candidate)}")

    def on_before_zero_grad(trainer):
        # Per-step weight shrinking (Eq.9-style) after SL starts
        if not state.sl_started or state.pruner is None:
            return
        if not hasattr(trainer, "optimizer") or trainer.optimizer is None:
            return
        lr = trainer.optimizer.param_groups[0]["lr"]
        scale = max(0.0, 1.0 - state.reg * lr)
        state.pruner.scale_weights_towards_zero(weight_scale_fact=scale)

    def on_train_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0))

        state.pruner = make_pruner(
            trainer.model, state.example, state.reg, state.group_cost, OCSPruner, GroupNormV2Importance
        )

        if epoch < args.prune_monitor_start_epoch:
            print(f"[OCS][epoch {epoch}] monitor warmup")
            return

        result = search_groups_for_target_flops(
            state.pruner, args.target_flops_rr, args.layer_prune_limit, OCSPruner, GroupNormV2Importance
        )

        if result is None:
            print(f"[OCS][epoch {epoch}] no valid pruning candidates")
            return

        groups_idx, flop_rr = result
        if len(groups_idx) == 0:
            print(f"[OCS][epoch {epoch}] empty pruning indices")
            return

        state.pruner.update_pruning_groups(groups_idx, need_noprune_group_too=True)
        raw, avg = state.tracker.update(state.pruner)

        print(
            f"[OCS][epoch {epoch}] flop_rr={flop_rr:.4f} psi_raw={raw:.4f} "
            f"psi_avg={avg:.4f} reg={state.reg:.6f}"
        )

        if (not state.sl_started) and len(state.tracker.avg) >= 3:
            if abs(state.tracker.avg[-1] - state.tracker.avg[-3]) <= args.tau:
                state.sl_started = True
                state.sl_start_epoch = epoch
                print(f"[OCS] SL start at epoch {epoch}")

        if state.sl_started and (epoch % args.reg_add_interval == 0):
            state.reg += args.reg_delta

        if avg >= args.pruning_stability_thresh:
            state.final_groups = copy.deepcopy(groups_idx)
            trainer.stop = True
            print(f"[OCS] stable pruning epoch reached at {epoch}; stopping search")

    ok1 = add_callback_safe(y, "on_train_start", on_train_start)
    ok2 = add_callback_safe(y, "on_train_epoch_end", on_train_epoch_end)
    ok3 = add_callback_safe(y, "on_before_zero_grad", on_before_zero_grad)
    if not ok3:
        add_callback_safe(y, "on_train_batch_end", on_before_zero_grad)

    if not (ok1 and ok2):
        raise RuntimeError("Required callback hooks are unavailable in this Ultralytics version.")

    # Stage 1: search + sparsity learning
    y.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.search_epochs,
        fraction=args.search_fraction,
        device=str(device),
        project=args.project,
        name=args.name,
        exist_ok=True,
        save=True,
        val=True,
        amp=False
    )

    # Stage 2: final prune
    pruner = make_pruner(
        y.model, state.example, state.reg, state.group_cost, OCSPruner, GroupNormV2Importance
    )

    if not state.final_groups:
        result = search_groups_for_target_flops(
            pruner, args.target_flops_rr, args.layer_prune_limit, OCSPruner, GroupNormV2Importance
        )
        if result is None:
            raise RuntimeError("No valid pruning group found after search stage.")
        state.final_groups, _ = result

    pruner.update_pruning_groups(state.final_groups, need_noprune_group_too=True)
    speed_up = pruner.pruning(state.final_groups)
    y.model = pruner.model
    print(f"[OCS] final structural prune done. estimated speed-up={speed_up:.3f}x")

    os.makedirs(os.path.dirname(args.pruned_weights) or ".", exist_ok=True)
    y.save(args.pruned_weights)
    print(f"[OCS] pruned weights saved: {args.pruned_weights}")

    # Stage 3: finetune pruned model in a clean run
    y_ft = YOLO(args.pruned_weights)
    y_ft.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.finetune_epochs,
        fraction=args.finetune_fraction,
        device=str(device),
        project=args.project,
        name=f"{args.name}_finetune",
        exist_ok=True,
        save=True,
        val=True,
        amp=False
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocspruner-root", default="~/Documents/OCSPruner")
    ap.add_argument("--weights", default="../models/yolov8n.pt")
    ap.add_argument("--data", default="../datasets/coco/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--project", default="./optimizer/pruning/runs/ocs")
    ap.add_argument("--name", default="yolov8n_ocs")

    ap.add_argument("--target-flops-rr", type=float, default=0.35)
    ap.add_argument("--layer-prune-limit", type=float, default=0.75)
    ap.add_argument("--prune-monitor-start-epoch", type=int, default=2)

    ap.add_argument("--pruning-stability-thresh", type=float, default=0.98)
    ap.add_argument("--tau", type=float, default=1e-4)

    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--reg-delta", type=float, default=1e-4)
    ap.add_argument("--reg-add-interval", type=int, default=1)

    ap.add_argument("--search-epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=100)
    ap.add_argument("--pruned-weights", default="./optimizer/pruning/runs/ocs/yolov8n_ocs_pruned.pt")
    ap.add_argument("--search-fraction", type=float, default=0.01,
                    help="dataset fraction for OCS search stage (0,1].")
    ap.add_argument("--finetune-fraction", type=float, default=0.01,
                    help="dataset fraction for final finetune stage (0,1].")

    main(ap.parse_args())
