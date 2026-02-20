from __future__ import annotations
from typing import Optional
import argparse

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate compiled YOLO models using Ultralytics validation metrics"
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLO weights (.pt)"
    )

    parser.add_argument(
        "--hef_path",
        type=str,
        required=True,
        help="Path to Hailo weights (.hef)"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda:0", "0"],
        help="Device to run on. Default: auto-detect (cuda > mps > cpu)"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Image size for validation"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for validation"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for validation (default: 0.001)"
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)"
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image"
    )

    return parser.parse_args()


def detect_best_device() -> str:
    """
    Returns the best available device string:
    'cuda:0' > 'mps' > 'cpu'
    """

    if torch.cuda.is_available():
        return "cuda:0"

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"

    return "cpu"


def evaluate_with_ultralytics_metrics(
    weights: str,
    hef_path=hef_path,
    data_yaml: str,
    imgsz: int = 640,
    batch: int = 1,
    conf: float = 0.001, # 0.001 - validation, 0.25 - inference/predict
    iou: float = 0.7, # Used inside non_max_suppression
    max_det: int = 300, # maximum number of detections kept per image after NMS
    device: Optional[str] = None, #"cuda:0", "mps"
):
    # 1) Load model
    yolo = YOLO(weights)
    if device is None:
        device = detect_best_device()
        print(f"[INFO] Using device: {device}")
    else:
        try:
            model = yolo.model.to(device)
        except RuntimeError as e:
            print(
                f"[WARN] Failed to use device '{device}': {e}\n"
                "[WARN] Falling back to CPU."
            )
            device = "cpu"
            model = yolo.model.to(device)

    model.eval()

    # 2) Create validator
    args = dict(
        model=weights,
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        split="val",
        task="detect",
        plots=False,
        save_json=False,
        save_txt=False,
        half=False,
        agnostic_nms=False,
        single_cls=False,
        verbose=False,
    )
    validator = DetectionValidator(args=args)

    # 3) Init metrics + dataloader
    validator.device = torch.device(device)
    validator.init_metrics(model)                 # sets up iouv, metrics, names, etc.
    validator.dataloader = validator.get_dataloader(validator.data[validator.args.split], batch)

    # 4) Hailo engine
    hailo_infer = HailoInfer(hef_path, batch_size=batch)
    H, W, C = hailo_infer.get_input_shape()

    # 4) Loop val set
    for batch_i, batch_data in enumerate(validator.dataloader):
        batch_data = validator.preprocess(batch_data)   # moves tensors to device, normalizes img/255

        imgs = batch_data["img"]

        # ---- YOUR inference path here ----
        # If you have raw outputs:
        with torch.no_grad():
            raw = model(imgs)  # raw preds tensor, as the model returns

        # 5) Convert raw -> NMS outputs in Ultralytics format
        preds = validator.postprocess(raw)  # list[dict{bboxes/conf/cls/extra}] per image

        # 6) Update Ultralytics metrics (same AP implementation as yolo val)
        validator.update_metrics(preds, batch_data)

    # 7) Final results dict (mAP50, mAP50-95, P, R, etc.)
    stats = validator.get_stats()
    return stats

def main():
    args = parse_args()

    stats = evaluate_with_ultralytics_metrics(
        weights=args.weights,  # "./models/rf_yolov8n_skrews_new/weights/best.pt",
        hef_path=args.hef_path,
        data_yaml=args.data,  # "./datasets/EmbeddedAIProject-7/data.yaml",
        imgsz=args.imgsz,  # 960,
        batch=args.batch,  # 1,
        conf=args.conf,
        iou=args.iou,
        device=args.device,  # or "cuda:0"
    )
    print("\n===== Evaluation results =====")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
