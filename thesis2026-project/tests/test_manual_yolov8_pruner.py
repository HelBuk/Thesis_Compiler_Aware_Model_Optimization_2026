from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from ultralytics import YOLO




from optimizer.pruning.manual_yolov8_pruner import (
    apply_structural_pruning_and_realign,
    params_count,
    _block_output_wrapper,
)

def snapshot_conv_layers(model: nn.Module) -> dict[str, dict]:
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and not name.startswith("model.22"):
            snap[name] = {
                "id": id(m),
                "in_channels": m.in_channels,
                "out_channels": m.out_channels,
                "weight_shape": tuple(m.weight.shape),
                "params": sum(p.numel() for p in m.parameters()),
            }
    return snap


def test_block9_pruning_changes_structure_and_forward():
    y = YOLO("./models/yolov8n.pt")
    before = snapshot_conv_layers(y.model)
    base_params = params_count(y.model)

    changes, skipped, realign_changes, _ = apply_structural_pruning_and_realign(
        y.model,
        blocks=[9],
        prune_ratio=0.10,
        round_to=1,
        model_in_ch=3,
    )

    assert len(changes) > 0
    assert any(c.path == "model.9" for c in changes)

    after = snapshot_conv_layers(y.model)
    assert params_count(y.model) < base_params

    # verify at least one conv under block 9 changed
    changed_under_block9 = []
    for name, old in before.items():
        if not name.startswith("model.9"):
            continue
        if name not in after:
            changed_under_block9.append(name)
            continue
        new = after[name]
        if (
            old["id"] != new["id"] or
            old["in_channels"] != new["in_channels"] or
            old["out_channels"] != new["out_channels"] or
            old["weight_shape"] != new["weight_shape"] or
            old["params"] != new["params"]
        ):
            changed_under_block9.append(name)

    assert changed_under_block9, "Expected some Conv2d under model.9 to change"

    # check reported change matches actual block
    ch = next(c for c in changes if c.path == "model.9")
    block = y.model.model[9]
    out_w = _block_output_wrapper(block)

    assert out_w is not None
    assert isinstance(out_w.conv, nn.Conv2d)
    assert out_w.conv.out_channels == ch.new_out
    assert params_count(block) == ch.new_params
    assert ch.new_out < ch.old_out
    assert ch.new_params < ch.old_params

    # forward sanity
    p0 = next(y.model.parameters())
    x = torch.randn(1, 3, 640, 640, device=p0.device, dtype=p0.dtype)
    with torch.no_grad():
        out = y.model(x)
    assert out is not None


def test_all_wrapper_convs_remain_shape_consistent():
    y = YOLO("./models/yolov8n.pt")

    apply_structural_pruning_and_realign(
        y.model,
        blocks=[9],
        prune_ratio=0.10,
        round_to=1,
        model_in_ch=3,
    )

    for name, m in y.model.named_modules():
        if hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
            assert m.conv.weight.shape[0] == m.conv.out_channels, f"{name}: bad out_channels"
            assert m.conv.weight.shape[1] == m.conv.in_channels, f"{name}: bad in_channels"


from optimizer.pruning.manual_yolov8_pruner import (
    _norm_target,
    _concat_input_keep_idx,
    _detect_layer_idx,
    _block_output_wrapper,
    _choose_keep_idx,
    _prune_wrapper_output_channels,
    _set_wrapper_input_channels,
    apply_structural_pruning_and_realign,
    build_output_target_table,
    list_all_cv_targets,
    parse_blocks,
    params_count,
    prune_c2f_cv1,
    prune_sppf_cv1,
    resolve_targets,
)


class _Wrap(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)


@pytest.fixture(scope="session")
def weights_path() -> str:
    p = Path(__file__).resolve().parents[1] / "models" / "yolov8n.pt"
    if not p.exists():
        pytest.skip(f"weights not found: {p}")
    return str(p)


@pytest.fixture()
def det_model(weights_path: str):
    # returns DetectionModel (y.model), not YOLO wrapper
    return YOLO(weights_path).model


def test_parse_blocks_range_and_clamp():
    got = parse_blocks("1,3-5,9-7,100,-2,5", max_idx=9)
    assert got == [1, 3, 4, 5, 7, 8, 9]


def test_norm_target_alias():
    assert _norm_target(" model.model[4].m.1.cv2.conv ") == "model.4.m.1.cv2"
    assert _norm_target("model.9.cv1") == "model.9.cv1"


def test_concat_input_keep_idx_with_negative_source():
    keep_map = {
        1: torch.tensor([0, 2], dtype=torch.long),
        3: torch.tensor([1], dtype=torch.long),
    }
    old_out = {1: 4, 3: 2}
    idx = _concat_input_keep_idx(src=[-1, 1, 3], keep_map=keep_map, old_out=old_out, model_in_ch=3)
    # offsets: -1 uses [0,1,2], then src1 offset +3 => [3,5], then src3 offset +(3+4)=7 => [8]
    assert torch.equal(idx, torch.tensor([0, 1, 2, 3, 5, 8], dtype=torch.long))


def test_choose_keep_idx_round_to():
    conv = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
    keep = _choose_keep_idx(conv, prune_ratio=0.1, round_to=8)
    assert len(keep) % 8 == 0
    assert len(keep) < 128


def test_wrapper_output_and_input_rebuild_helpers():
    w = _Wrap(16, 32)
    old_out = w.conv.out_channels

    keep_out = torch.arange(0, old_out, 2, dtype=torch.long)  # keep half
    ok = _prune_wrapper_output_channels(w, keep_out)
    assert ok
    assert w.conv.out_channels == len(keep_out)
    assert w.bn.num_features == len(keep_out)

    keep_in = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.long)
    ok2 = _set_wrapper_input_channels(w, keep_in)
    assert ok2
    assert w.conv.in_channels == len(keep_in)


def test_target_listing_excludes_detect(det_model):
    targets = list_all_cv_targets(det_model)
    assert len(targets) > 0
    assert all(not t.startswith("model.22.") for t in targets)
    assert "model.4.cv1" in targets
    assert "model.4.cv2" in targets


def test_build_output_target_table_basic(det_model):
    table = build_output_target_table(det_model)
    assert "model.4.cv2" in table
    assert "model.9.cv2" in table
    assert all(not k.startswith("model.22.") for k in table.keys())


def test_resolve_targets_mixed_cases(det_model):
    r = resolve_targets(
        det_model,
        "model.4.cv2,model.4.m.1.cv2.conv,model.9.cv1,model.22.cv2.0.0.conv,bad.path",
    )
    # model.4.cv2 -> output prune block
    assert 4 in r.block_targets
    # model.4.m.1.cv2.conv -> C2f hidden prune on block 4
    assert 4 in r.c2f_hidden_targets
    # model.9.cv1 -> SPPF hidden prune
    assert 9 in r.sppf_hidden_targets
    # detect + bad path should be unknown
    assert any("Detect targets disabled" in u for u in r.unknown)
    assert any("bad.path" in u for u in r.unknown)


@pytest.mark.integration
def test_prune_c2f_hidden_updates_internal_shapes(det_model):
    li = 4
    c2f = det_model.model[li]
    old_c = int(c2f.c)
    old_cv1_out = int(c2f.cv1.conv.out_channels)

    ch = prune_c2f_cv1(det_model, li=li, prune_ratio=0.10, round_to=1)
    assert ch is not None

    new_c = int(c2f.c)
    assert new_c < old_c
    assert int(c2f.cv1.conv.out_channels) == 2 * new_c
    assert int(c2f.cv1.conv.out_channels) < old_cv1_out

    for b in c2f.m:
        assert b.cv1.conv.in_channels == new_c
        assert b.cv1.conv.out_channels == new_c
        assert b.cv2.conv.in_channels == new_c
        assert b.cv2.conv.out_channels == new_c

    assert c2f.cv2.conv.in_channels == new_c * (2 + len(c2f.m))


@pytest.mark.integration
def test_prune_sppf_hidden_updates_cv2_input(det_model):
    li = 9
    sppf = det_model.model[li]
    old_h = int(sppf.cv1.conv.out_channels)

    ch = prune_sppf_cv1(det_model, li=li, prune_ratio=0.10, round_to=1)
    assert ch is not None

    new_h = int(sppf.cv1.conv.out_channels)
    assert new_h < old_h
    assert sppf.cv2.conv.in_channels == 4 * new_h


@pytest.mark.integration
def test_block_output_prune_realign_and_forward(det_model):
    base_params = params_count(det_model)
    changes, skipped, realign, _ = apply_structural_pruning_and_realign(
        det_model,
        blocks=[18, 21],
        prune_ratio=0.10,
        round_to=1,
        model_in_ch=3,
    )

    assert len(changes) >= 1
    assert params_count(det_model) < base_params
    assert any("model.22 detect.cv2[1].0.in=" in s for s in realign)
    assert any("model.22 detect.cv3[2].0.in=" in s for s in realign)

    p0 = next(det_model.parameters())
    x = torch.randn(1, 3, 640, 640, device=p0.device, dtype=p0.dtype)
    with torch.no_grad():
        out = det_model(x)
    assert out is not None


@pytest.mark.integration
def test_block_output_wrapper_after_prune_matches_change(det_model):
    changes, _, _, _ = apply_structural_pruning_and_realign(
        det_model,
        blocks=[9],
        prune_ratio=0.10,
        round_to=1,
        model_in_ch=3,
    )
    ch = next(c for c in changes if c.path == "model.9")
    w = _block_output_wrapper(det_model.model[9])

    assert w is not None
    assert isinstance(w.conv, nn.Conv2d)
    assert w.conv.out_channels == ch.new_out
    assert ch.new_out < ch.old_out
    assert ch.new_params < ch.old_params

def test_output_prune_preserves_selected_filters_and_bn():
    w = _Wrap(4, 6)
    with torch.no_grad():
        w.conv.weight.copy_(torch.arange(w.conv.weight.numel(), dtype=w.conv.weight.dtype).view_as(w.conv.weight))
        w.bn.weight.copy_(torch.arange(6, dtype=w.bn.weight.dtype))
        w.bn.bias.copy_(torch.arange(6, dtype=w.bn.bias.dtype) + 10)
        w.bn.running_mean.copy_(torch.arange(6, dtype=w.bn.running_mean.dtype) + 20)
        w.bn.running_var.copy_(torch.arange(6, dtype=w.bn.running_var.dtype) + 30)

    old_w = w.conv.weight.detach().clone()
    old_bn_w = w.bn.weight.detach().clone()
    old_bn_b = w.bn.bias.detach().clone()
    old_rm = w.bn.running_mean.detach().clone()
    old_rv = w.bn.running_var.detach().clone()

    keep = torch.tensor([1, 3, 5], dtype=torch.long)
    assert _prune_wrapper_output_channels(w, keep)

    assert torch.equal(w.conv.weight.detach().cpu(), old_w[keep].cpu())
    assert torch.equal(w.bn.weight.detach().cpu(), old_bn_w[keep].cpu())
    assert torch.equal(w.bn.bias.detach().cpu(), old_bn_b[keep].cpu())
    assert torch.equal(w.bn.running_mean.detach().cpu(), old_rm[keep].cpu())
    assert torch.equal(w.bn.running_var.detach().cpu(), old_rv[keep].cpu())


def test_input_prune_preserves_selected_input_slices():
    w = _Wrap(6, 5)
    with torch.no_grad():
        w.conv.weight.copy_(torch.arange(w.conv.weight.numel(), dtype=w.conv.weight.dtype).view_as(w.conv.weight))

    old_w = w.conv.weight.detach().clone()
    keep_in = torch.tensor([0, 2, 5], dtype=torch.long)

    assert _set_wrapper_input_channels(w, keep_in)
    assert torch.equal(w.conv.weight.detach().cpu(), old_w[:, keep_in].cpu())


@pytest.mark.integration
def test_block_output_prune_keeps_old_weight_subset(det_model):
    li = 9  # SPPF block in YOLOv8n
    block = det_model.model[li]
    out_w = _block_output_wrapper(block)
    assert out_w is not None and hasattr(out_w, "conv")

    old_w = out_w.conv.weight.detach().clone()
    old_bn_w = out_w.bn.weight.detach().clone()
    old_bn_b = out_w.bn.bias.detach().clone()
    old_rm = out_w.bn.running_mean.detach().clone()
    old_rv = out_w.bn.running_var.detach().clone()

    keep = _choose_keep_idx(out_w.conv, prune_ratio=0.10, round_to=1)
    assert len(keep) < out_w.conv.out_channels

    apply_structural_pruning_and_realign(
        det_model,
        blocks=[li],
        prune_ratio=0.10,
        round_to=1,
        model_in_ch=3,
    )

    new_out_w = _block_output_wrapper(det_model.model[li])
    assert new_out_w is not None

    assert torch.equal(new_out_w.conv.weight.detach().cpu(), old_w[keep].cpu())
    assert torch.equal(new_out_w.bn.weight.detach().cpu(), old_bn_w[keep].cpu())
    assert torch.equal(new_out_w.bn.bias.detach().cpu(), old_bn_b[keep].cpu())
    assert torch.equal(new_out_w.bn.running_mean.detach().cpu(), old_rm[keep].cpu())
    assert torch.equal(new_out_w.bn.running_var.detach().cpu(), old_rv[keep].cpu())


def _forward_ok(det_model):
    p0 = next(det_model.parameters())
    x = torch.randn(1, 3, 640, 640, device=p0.device, dtype=p0.dtype)
    with torch.no_grad():
        out = det_model(x)
    assert out is not None


def _apply_targets_once(det_model, targets: str, prune_ratio: float = 0.10, round_to: int = 1):
    r = resolve_targets(det_model, targets)
    assert not r.unknown, f"Unexpected unknown targets: {r.unknown}"

    changes, skipped, realign, _ = apply_structural_pruning_and_realign(
        det_model,
        blocks=r.block_targets,
        prune_ratio=prune_ratio,
        round_to=round_to,
        model_in_ch=3,
    )

    for li in r.c2f_hidden_targets:
        ch = prune_c2f_cv1(det_model, li, prune_ratio, round_to)
        if ch is not None:
            changes.append(ch)

    for li in r.sppf_hidden_targets:
        ch = prune_sppf_cv1(det_model, li, prune_ratio, round_to)
        if ch is not None:
            changes.append(ch)

    return r, changes, skipped, realign


@pytest.mark.integration
def test_multi_target_conv_output_and_next_c2f_pipeline(det_model):
    # conv output + next C2f output in one command
    before_in = det_model.model[4].cv1.conv.in_channels  # input of C2f(4)
    r, changes, _, _ = _apply_targets_once(det_model, "model.3,model.4.cv2")

    assert set(r.block_targets) == {3, 4}
    assert any(c.path == "model.3" for c in changes)
    assert any(c.path == "model.4" for c in changes)

    after_in = det_model.model[4].cv1.conv.in_channels
    assert after_in < before_in
    assert after_in == det_model.model[3].conv.out_channels
    _forward_ok(det_model)


@pytest.mark.integration
def test_multi_target_bottleneck_out_and_next_bottleneck_in(det_model):
    # one bottleneck out (cv2) + next bottleneck in (cv1)
    r, changes, _, _ = _apply_targets_once(det_model, "model.4.m.0.cv2,model.4.m.1.cv1")

    # both map to one C2f hidden prune at layer 4
    assert r.c2f_hidden_targets == [4]
    assert any(c.path == "model.4.cv1" for c in changes)

    b0 = det_model.model[4].m[0]
    b1 = det_model.model[4].m[1]
    assert b0.cv2.conv.out_channels == b1.cv1.conv.in_channels
    _forward_ok(det_model)


@pytest.mark.integration
def test_multi_target_sppf_hidden_and_output(det_model):
    # interpreted "output and input" as SPPF cv2 + cv1 in one run
    r, changes, _, _ = _apply_targets_once(det_model, "model.9.cv1,model.9.cv2")

    assert 9 in r.sppf_hidden_targets
    assert 9 in r.block_targets
    assert len(changes) >= 1

    sppf = det_model.model[9]
    assert sppf.cv2.conv.in_channels == 4 * sppf.cv1.conv.out_channels
    _forward_ok(det_model)


@pytest.mark.integration
def test_multi_target_outputs_15_18_21(det_model):
    before = {}
    for i in [15, 18, 21]:
        w = _block_output_wrapper(det_model.model[i])
        assert w is not None
        before[i] = w.conv.out_channels

    r, changes, _, realign = _apply_targets_once(det_model, "model.15.cv2,model.18.cv2,model.21.cv2")

    assert set(r.block_targets) == {15, 18, 21}
    for i in [15, 18, 21]:
        w = _block_output_wrapper(det_model.model[i])
        assert w is not None
        assert w.conv.out_channels < before[i]

    assert any("detect.cv2[0].0.in=" in s for s in realign)
    assert any("detect.cv2[1].0.in=" in s for s in realign)
    assert any("detect.cv2[2].0.in=" in s for s in realign)
    _forward_ok(det_model)


@pytest.mark.integration
def test_multi_target_output_layer9_before_upsample_concat(det_model):
    before_12_in = det_model.model[12].cv1.conv.in_channels
    out6 = _block_output_wrapper(det_model.model[6]).conv.out_channels

    r, changes, _, _ = _apply_targets_once(det_model, "model.9.cv2")

    assert 9 in r.block_targets
    assert any(c.path == "model.9" for c in changes)

    out9 = _block_output_wrapper(det_model.model[9]).conv.out_channels
    after_12_in = det_model.model[12].cv1.conv.in_channels

    assert after_12_in < before_12_in
    assert after_12_in == out6 + out9
    _forward_ok(det_model)