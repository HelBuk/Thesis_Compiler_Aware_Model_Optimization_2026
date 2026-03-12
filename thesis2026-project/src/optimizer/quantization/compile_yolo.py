from ultralytics import YOLO

MODEL_PT = "../models/yolov8n.pt"
DATA_YAML = "../datasets/coco_subset/train_0_1percent/coco_subset.yaml"

model = YOLO(MODEL_PT)

engine_path = model.export(
    format="engine",
    imgsz=640,
    half=False,
    int8=True,
    data=DATA_YAML,
    device=0,
    workspace=4,
    fraction=1.0,
    batch=1,
    simplify=True,
    verbose=True,
)

print("Exported engine:", engine_path)

tflite_path = model.export(
    format="tflite",
    imgsz=640,
    int8=True,
    data=DATA_YAML,
    device="cpu",
    fraction=1.0,
    batch=1,
    verbose=True,
)

print("Exported tflite:", tflite_path)