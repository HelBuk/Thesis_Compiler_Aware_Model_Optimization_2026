# src/plugins/c2f_m2/export_model2_weights.py
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

OUT = Path("../models/plugin_weights/model2_c2f_folded.npz")
OUT.parent.mkdir(parents=True, exist_ok=True)

def fold_conv_bn(conv, bn):
    w = conv.weight.detach().cpu()
    if conv.bias is None:
        b = torch.zeros(w.shape[0], dtype=w.dtype)
    else:
        b = conv.bias.detach().cpu()

    gamma = bn.weight.detach().cpu()
    beta = bn.bias.detach().cpu()
    mean = bn.running_mean.detach().cpu()
    var = bn.running_var.detach().cpu()
    eps = bn.eps

    inv = gamma / torch.sqrt(var + eps)
    w_fold = w * inv.reshape(-1, 1, 1, 1)
    b_fold = beta + (b - mean) * inv
    return w_fold.numpy().astype(np.float32), b_fold.numpy().astype(np.float32)

y = YOLO("../models/yolov8n.pt")
m = y.model.model[2]  # model.2 C2f

arrays = {}

w, b = fold_conv_bn(m.cv1.conv, m.cv1.bn)
arrays["cv1_w"] = w
arrays["cv1_b"] = b

w, b = fold_conv_bn(m.m[0].cv1.conv, m.m[0].cv1.bn)
arrays["m0_cv1_w"] = w
arrays["m0_cv1_b"] = b

w, b = fold_conv_bn(m.m[0].cv2.conv, m.m[0].cv2.bn)
arrays["m0_cv2_w"] = w
arrays["m0_cv2_b"] = b

w, b = fold_conv_bn(m.cv2.conv, m.cv2.bn)
arrays["cv2_w"] = w
arrays["cv2_b"] = b

arrays["shortcut"] = np.array([1 if m.m[0].add else 0], dtype=np.int32)

np.savez(OUT, **arrays)

print(f"saved: {OUT}")
for k, v in arrays.items():
    print(k, v.shape, v.dtype)