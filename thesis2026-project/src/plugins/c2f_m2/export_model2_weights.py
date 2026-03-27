# src/plugins/c2f_m2/export_model2_weights.py
#
# Exports BN-folded weights for model.2 C2f block to:
#   - model2_c2f_folded.npz  (backup / Python inspection)
#   - model2_c2f_folded.bin  (flat binary consumed by the C++ plugin)
#
# Binary layout:
#   magic      : 4 bytes  'C2FW'
#   num_tensors: uint32
#   For each tensor:
#     name_len : uint32
#     name     : char[name_len]
#     ndim     : uint32
#     dims     : int32[ndim]
#     data_len : uint32   (bytes)
#     data     : float32[data_len/4]

import struct
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

NPZ_OUT = Path("../models/plugin_weights/model2_c2f_folded.npz")
BIN_OUT = Path("../models/plugin_weights/model2_c2f_folded.bin")
NPZ_OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# BN folding: merges conv + BN into a single conv+bias
# ---------------------------------------------------------------------------
def fold_conv_bn(conv, bn):
    w = conv.weight.detach().cpu()
    b = conv.bias.detach().cpu() if conv.bias is not None else torch.zeros(w.shape[0])

    gamma = bn.weight.detach().cpu()
    beta  = bn.bias.detach().cpu()
    mean  = bn.running_mean.detach().cpu()
    var   = bn.running_var.detach().cpu()
    eps   = bn.eps

    inv    = gamma / torch.sqrt(var + eps)
    w_fold = w * inv.reshape(-1, 1, 1, 1)
    b_fold = beta + (b - mean) * inv
    return w_fold.numpy().astype(np.float32), b_fold.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Binary writer
# ---------------------------------------------------------------------------
def write_bin(path: Path, arrays: dict):
    """Write weight dict to flat binary file parseable from C++."""
    with open(path, "wb") as f:
        f.write(b"C2FW")                                    # magic
        f.write(struct.pack("<I", len(arrays)))              # num_tensors

        for name, arr in arrays.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))      # name_len
            f.write(name_bytes)                              # name
            f.write(struct.pack("<I", arr.ndim))             # ndim
            for d in arr.shape:
                f.write(struct.pack("<I", int(d)))           # each dim
            data = arr.tobytes()
            f.write(struct.pack("<I", len(data)))            # data_len (bytes)
            f.write(data)                                    # raw float32 data


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------
y = YOLO("../models/yolov8n.pt")
m = y.model.model[2]   # model.2  C2f block

arrays = {}

w, b = fold_conv_bn(m.cv1.conv, m.cv1.bn)
arrays["cv1_w"] = w    # [Cout, Cin, 1, 1]  — encodes both Cin & Cout
arrays["cv1_b"] = b    # [Cout]

w, b = fold_conv_bn(m.m[0].cv1.conv, m.m[0].cv1.bn)
arrays["m0_cv1_w"] = w  # [halfC, halfC, 3, 3]
arrays["m0_cv1_b"] = b  # [halfC]

w, b = fold_conv_bn(m.m[0].cv2.conv, m.m[0].cv2.bn)
arrays["m0_cv2_w"] = w  # [halfC, halfC, 3, 3]
arrays["m0_cv2_b"] = b  # [halfC]

w, b = fold_conv_bn(m.cv2.conv, m.cv2.bn)
arrays["cv2_w"] = w    # [Cout, 3*halfC, 1, 1]
arrays["cv2_b"] = b    # [Cout]

# shortcut flag for the bottleneck residual add
arrays["shortcut"] = np.array([1 if m.m[0].add else 0], dtype=np.float32)

# Derived dims (convenient for the C++ loader)
Cin   = int(arrays["cv1_w"].shape[1])   # cv1 input channels  (e.g. 32)
Cout  = int(arrays["cv2_w"].shape[0])   # C2f output channels (e.g. 64)
halfC = Cout // 2

arrays["meta_cin"]   = np.array([Cin],   dtype=np.float32)
arrays["meta_cout"]  = np.array([Cout],  dtype=np.float32)
arrays["meta_halfc"] = np.array([halfC], dtype=np.float32)

# Save both formats
np.savez(NPZ_OUT, **arrays)
write_bin(BIN_OUT, arrays)

print(f"Saved NPZ : {NPZ_OUT}")
print(f"Saved BIN : {BIN_OUT}")
print(f"\nChannel layout: Cin={Cin}  Cout={Cout}  halfC={halfC}")
print("\nTensor shapes:")
for k, v in arrays.items():
    print(f"  {k:20s}  shape={v.shape}  dtype={v.dtype}")
