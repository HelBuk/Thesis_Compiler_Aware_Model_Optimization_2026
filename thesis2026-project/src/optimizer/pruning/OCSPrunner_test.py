# ocs_yolov8_flow.py

import sys
import torch

sys.path.append("/Users/helbuk/Documents/OCSPruner")

import copy
import math
import argparse
import numpy as np

from ultralytics import YOLO
import vainf_torch_pruning.torch_pruning as tp
from ocspruner.pruner import OCSPruner
from ocspruner.importance import GroupNormV2Importance


def build_ignored_layers_for_yolo(det_model):
    ignored = []
    # Ignore Detect head modules first (safer for YOLO shape constraints)
    detect = det_model.model[-1]
    for m in detect.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):
            ignored.append(m)
    return ignored

def make_pruner(det_model, example_inputs, reg, group_cost=None):
    pruner = OCSPruner(
        model=det_model,
        example_inputs=example_inputs,
        importance=GroupNormV2Importance(p=2),
        reg=reg,
        ignored_layers=build_ignored_layers_for_yolo(det_model),
        group_cost=group_cost,
    )
    return pruner

def compute_group_cost(det_model, example_inputs, reg):
    base_ops, _ = tp.utils.count_ops_and_params(det_model, example_inputs=example_inputs)
    base_pruner = make_pruner(det_model, example_inputs, reg)

    flops_groups = {"base_model": (base_ops, -1, -1.0)}
    for i in range(len(base_pruner.groups_prune_candidate)):
        m_tmp = copy.deepcopy(det_model)
        p_tmp = make_pruner(m_tmp, example_inputs, reg)

        g = p_tmp.groups_prune_candidate[i]
        module = g[0][0].target.module
        prune_fn = g[0][0].handler
        idxs = list(g[0][1])
        cand = idxs[:-1] if len(idxs) > 1 else idxs

        pg = p_tmp.DG.get_pruning_group(module, prune_fn, cand)
        if p_tmp.DG.check_pruning_group(pg):
            pg.prune()

        pruned_ops, _ = tp.utils.count_ops_and_params(m_tmp, example_inputs=example_inputs)
        cost = float(base_ops - pruned_ops) / max(1, len(cand))
        flops_groups[f"group{i+1}"] = (cost, len(idxs), 0.0)

    vals = [v[0] for k, v in flops_groups.items() if k != "base_model"]
    exp_sum = sum(math.exp(v / 1e6) for v in vals)
    probs = [math.exp(v / 1e6) / exp_sum for v in vals]

    for i, key in enumerate([k for k in flops_groups.keys() if k != "base_model"]):
        c, n, _ = flops_groups[key]
        flops_groups[key] = (c, n, probs[i])

    return flops_groups

def search_groups_for_target_flops(pruner, target_flops_rr, layer_prune_limit):
    base_ops, _ = pruner.get_model_info()
    all_groups = pruner.get_all_group_imp_scores(layer_max_prune_limit=layer_prune_limit)
    all_imp = torch.cat([g[-1] for g in all_groups], dim=0)
    sorted_vals, sorted_idx = torch.sort(all_imp)
    keep = torch.nonzero(sorted_vals != 100.0).squeeze()
    sorted_idx = sorted_idx[keep]

    left, right = 0, len(sorted_idx) - 1
    best = None

    while left <= right:
        mid = (left + right) // 2
        m_tmp = copy.deepcopy(pruner.model)
        p_tmp = make_pruner(m_tmp, pruner.example_inputs, reg=pruner.reg, group_cost=pruner.group_cost)

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

    return best  # (groups_pruning_indices, flop_reduction_ratio)

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
            sims = [jaccard(first[k], last[k]) for k in first.keys()]
            raw = float(np.mean(sims))

        self.raw.append(raw)
        self.avg.append(float(np.mean(self.raw[-self.win:])))
        return raw, self.avg[-1]

def main(args):
    device = torch.device(args.device)
    y = YOLO(args.weights)
    y.model.to(device)

    example = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    pruner = make_pruner(y.model, example, reg=args.reg)
    pruner.group_cost = compute_group_cost(y.model, example, reg=args.reg)

    tracker = StabilityTracker(win=3)
    sl_started = False

    for epoch in range(args.search_epochs):
        # One epoch YOLO train step (practical adapter)
        y.train(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            epochs=1,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            resume=(epoch > 0),
            save=False,
            val=True,
            fraction=args.fraction,
        )

        # rebuild pruner over updated model weights
        pruner = make_pruner(y.model, example, reg=args.reg, group_cost=pruner.group_cost)

        groups_idx, flop_rr = search_groups_for_target_flops(
            pruner, target_flops_rr=args.target_flops_rr, layer_prune_limit=args.layer_prune_limit
        )
        pruner.update_pruning_groups(groups_idx, need_noprune_group_too=True)

        raw, avg = tracker.update(pruner)
        print(f"[epoch {epoch}] flop_rr={flop_rr:.4f} psi_raw={raw:.4f} psi_avg={avg:.4f} reg={args.reg:.6f}")

        if len(tracker.avg) >= 3 and not sl_started:
            if abs(tracker.avg[-1] - tracker.avg[-3]) <= args.tau:
                sl_started = True
                print(f"SL start at epoch {epoch}")

        if sl_started:
            lr = y.trainer.optimizer.param_groups[0]["lr"]
            pruner.scale_weights_towards_zero(weight_scale_fact=max(0.0, 1.0 - args.reg * lr))
            args.reg += args.reg_delta

        if avg >= args.pruning_stability_thresh:
            print(f"Stable pruning epoch reached at {epoch}")
            break

    # Final prune
    pruner.pruning(pruner.groups_pruning_indices)
    y.model = pruner.model
    y.save(args.pruned_weights)

    # Fine-tune pruned model
    y.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.finetune_epochs,
        device=args.device,
        project=args.project,
        name=f"{args.name}_finetune",
        exist_ok=True,
        resume=False,
        fraction=args.fraction,
    )

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--data", required=True)  # e.g. coco128.yaml
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--project", default="runs/ocs")
    ap.add_argument("--name", default="yolov8n_ocs")
    ap.add_argument("--target-flops-rr", type=float, default=0.35)
    ap.add_argument("--layer-prune-limit", type=float, default=0.75)
    ap.add_argument("--pruning-stability-thresh", type=float, default=0.98)
    ap.add_argument("--tau", type=float, default=1e-4)      # tsl-start trigger
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--reg-delta", type=float, default=1e-4)
    ap.add_argument("--search-epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=100)
    ap.add_argument("--pruned-weights", default="yolov8n_ocs_pruned.pt")
    ap.add_argument("--fraction", type=float, default=0.01)
    main(ap.parse_args())
