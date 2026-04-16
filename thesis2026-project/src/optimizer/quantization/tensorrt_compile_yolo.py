#!/usr/bin/env python3
import shutil
from pathlib import Path
from ultralytics import YOLO

MODEL_PT   = "../models/yolov8n.pt"
DATA_YAML  = "../datasets/coco_subset/train_1percent/coco_subset.yaml"
OUT_DIR    = Path("../models/tensorrt_exports")
IMGSZ      = 640
BATCH      = 1
DEVICE     = 0
WORKSPACE_GB = 4

OUT_DIR.mkdir(parents=True, exist_ok=True)

def export_engine(model_path: str, int8: bool, name: str) -> Path:
    model = YOLO(model_path)

    kwargs = dict(
        format="engine",
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workspace=WORKSPACE_GB,
        simplify=True,
        nms=False,
        verbose=True,
    )

    if int8:
        kwargs.update(int8=True, half=False, data=DATA_YAML, fraction=1.0, split="train")
    else:
        kwargs.update(int8=False, half=False)

    exported = Path(model.export(**kwargs))          # path Ultralytics chose

    dest = OUT_DIR / name
    shutil.move(str(exported), str(dest))            # move to your target name
    print(f"  → saved to {dest}")
    return dest

if __name__ == "__main__":
    # fp32_engine = export_engine(MODEL_PT, int8=False, name="yolov8n_fp32.engine")

    int8_engine = export_engine(MODEL_PT, int8=True, name="yolov8n_int8_tvm.engine")
    print(f"[OK] INT8 engine: {int8_engine}")
