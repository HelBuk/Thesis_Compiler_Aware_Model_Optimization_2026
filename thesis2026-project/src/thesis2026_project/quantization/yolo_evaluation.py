#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import torch

from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import IterableSimpleNamespace


# -----------------------------
# Backend adapters
# -----------------------------
class Backend:
    """Return preds in Ultralytics validator format:
    list length B of dicts:
      {"bboxes": (N,4) xyxy in *imgsz* space, "conf": (N,), "cls": (N,), "extra": (N,0)}
    """

    def __init__(self, imgsz: int, conf: float, iou: float, max_det: int, device: str):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.device = device

    def infer_batch(self, imgs_bchw01: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        raise NotImplementedError


def detect_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def nms_numpy_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        order = order[1:][iou <= iou_thr]

    return np.array(keep, dtype=np.int64)


def yolo8_output_to_preds_ultra(
    out: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    torch_device: str,
) -> Dict[str, torch.Tensor]:
    """
    Works for common YOLOv8 exports where output is (1,84,8400) or (1,8400,84):
      [x,y,w,h, cls0..cls79] in the *imgsz* coordinate space.
    """
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    else:
        raise ValueError(f"Unexpected output shape {out.shape}, expected [1,*,*]")

    if out.shape[0] == 84:  # (84,8400)
        pred = out.T  # (8400,84)
    elif out.shape[1] == 84:  # (8400,84)
        pred = out
    else:
        raise ValueError(f"Can't interpret output shape {out.shape} as YOLOv8 (84 channels).")

    xywh = pred[:, :4]
    cls_scores = pred[:, 4:]

    cls_id = np.argmax(cls_scores, axis=1)
    score = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

    keep = score >= conf
    if not np.any(keep):
        b = torch.zeros((0, 4), dtype=torch.float32, device=torch_device)
        c = torch.zeros((0,), dtype=torch.float32, device=torch_device)
        k = torch.zeros((0,), dtype=torch.float32, device=torch_device)
        e = torch.zeros((0, 0), dtype=torch.float32, device=torch_device)
        return {"bboxes": b, "conf": c, "cls": k, "extra": e}

    xywh = xywh[keep]
    score = score[keep]
    cls_id = cls_id[keep]

    # xywh -> xyxy
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    # If boxes are normalized (0..1), scale to pixel coords
    if boxes.max() <= 2.0:
        boxes *= float(imgsz)

    # clip to imgsz
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, imgsz)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, imgsz)

    # class-wise NMS
    out_boxes, out_scores, out_cls = [], [], []
    for c in np.unique(cls_id):
        idx = np.where(cls_id == c)[0]
        b = boxes[idx]
        s = score[idx].astype(np.float32)
        kept = nms_numpy_xyxy(b, s, iou_thr=iou, max_det=max_det)
        if kept.size == 0:
            continue
        out_boxes.append(b[kept])
        out_scores.append(s[kept])
        out_cls.append(np.full((kept.size,), c, dtype=np.float32))

    if not out_boxes:
        b = torch.zeros((0, 4), dtype=torch.float32, device=torch_device)
        c = torch.zeros((0,), dtype=torch.float32, device=torch_device)
        k = torch.zeros((0,), dtype=torch.float32, device=torch_device)
        e = torch.zeros((0, 0), dtype=torch.float32, device=torch_device)
        return {"bboxes": b, "conf": c, "cls": k, "extra": e}

    b = np.concatenate(out_boxes, axis=0)
    s = np.concatenate(out_scores, axis=0)
    k = np.concatenate(out_cls, axis=0)

    # global top max_det
    if s.size > max_det:
        top = np.argsort(-s)[:max_det]
        b, s, k = b[top], s[top], k[top]

    bt = torch.from_numpy(b).to(torch_device)
    st = torch.from_numpy(s).to(torch_device)
    kt = torch.from_numpy(k).to(torch_device)
    et = torch.zeros((bt.shape[0], 0), dtype=torch.float32, device=torch_device)
    return {"bboxes": bt, "conf": st, "cls": kt, "extra": et}


class TFLiteBackend(Backend):
    def __init__(
        self,
        model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        threads: int,
        delegate: str,
    ):
        super().__init__(imgsz, conf, iou, max_det, device)
        import tensorflow as tf

        delegates = None
        if delegate == "gpu":
            # NOTE: This is Linux .so; on macOS it will likely fail.
            try:
                delegates = [tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")]
            except Exception as e:
                raise RuntimeError(f"Failed to load TFLite GPU delegate: {e}")

        self.interp = tf.lite.Interpreter(
            model_path=model_path, num_threads=threads, experimental_delegates=delegates
        )
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

        # print("[DEBUG] TFLite output shape:", self.out["shape"], "dtype:", self.out["dtype"], "quant:", self.out.get("quantization"))

    def infer_batch(self, imgs_bchw01: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        # Convert BCHW [0..1] -> NHWC
        imgs = (
            imgs_bchw01.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
        )  # B,H,W,C in [0..1]
        preds = []

        in_dtype = self.inp["dtype"]
        in_scale, in_zero = self.inp.get("quantization", (0.0, 0))

        out_dtype = self.out["dtype"]
        out_scale, out_zero = self.out.get("quantization", (0.0, 0))

        for i in range(imgs.shape[0]):
            x = imgs[i : i + 1]  # [1,H,W,C]

            # Quantize input if needed
            if in_dtype in (np.uint8, np.int8):
                if in_scale == 0:
                    raise RuntimeError("TFLite input scale is 0; cannot quantize input.")
                xq = np.round(x / in_scale + in_zero).astype(in_dtype)
                self.interp.set_tensor(self.inp["index"], xq)
            else:
                self.interp.set_tensor(self.inp["index"], x.astype(in_dtype))

            self.interp.invoke()
            y = self.interp.get_tensor(self.out["index"])

            # Dequantize output if needed
            if out_dtype in (np.uint8, np.int8):
                y = (y.astype(np.float32) - out_zero) * out_scale
            else:
                y = y.astype(np.float32)

            pred_i = yolo8_output_to_preds_ultra(
                out=y,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                torch_device="cpu",
            )
            preds.append(pred_i)

        return preds


class ORTBackend(Backend):
    def __init__(self, model_path: str, imgsz: int, conf: float, iou: float, max_det: int, device: str):
        super().__init__(imgsz, conf, iou, max_det, device)
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") or device == "0":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name

    def infer_batch(self, imgs_bchw01: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        x = imgs_bchw01.detach().cpu().numpy().astype(np.float32)
        y = self.sess.run(None, {self.in_name: x})[0]  # assume single output

        preds = []
        for i in range(x.shape[0]):
            pred_i = yolo8_output_to_preds_ultra(
                out=y[i : i + 1],
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                torch_device="cpu",
            )
            preds.append(pred_i)
        return preds


class TorchBackend(Backend):
    def __init__(self, model_path: str, imgsz: int, conf: float, iou: float, max_det: int, device: str):
        super().__init__(imgsz, conf, iou, max_det, device)
        from ultralytics import YOLO

        self.torch_device = device or detect_best_device()

        y = YOLO(model_path)
        self.model = y.model
        self.model.eval()
        self.model.to(self.torch_device)

    @torch.no_grad()
    def infer_batch(self, imgs_bchw01: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        x = imgs_bchw01.to(self.torch_device)
        if x.dtype != torch.float32:
            x = x.float()

        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        out_np = out.detach().cpu().numpy().astype(np.float32)

        preds = []
        for i in range(out_np.shape[0]):
            pred_i = yolo8_output_to_preds_ultra(
                out=out_np[i : i + 1],
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                torch_device="cpu",
            )
            preds.append(pred_i)
        return preds


# -----------------------------
# Evaluation with Ultralytics metrics
# -----------------------------
def make_validator(
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    project: str,
    name: str,
    save_json: bool,
    save_txt: bool,
    plots: bool,
):
    args = dict(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        split="val",
        task="detect",
        plots=plots,
        save_json=save_json,
        save_txt=save_txt,
        half=False,
        agnostic_nms=False,
        single_cls=False,
        verbose=False,
        project=project,
        name=name,
        exist_ok=True,
    )

    data_dict = check_det_dataset(data_yaml)
    v = DetectionValidator(args=IterableSimpleNamespace(**args))
    v.data = data_dict
    v.args.data = data_yaml
    v.device = torch.device("cpu")  # metrics on CPU; inference handled by backend
    return v

class _NamesOnlyModel:
    def __init__(self, names):
        self.names = names

def eval_backend(
    backend: Backend,
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
    max_batches: int
) -> Dict[str, float]:
    v = make_validator(
        data_yaml,
        imgsz,
        batch,
        conf,
        iou,
        max_det,
        device="cpu",
        project=project,
        name=name,
        save_json=save_json,
        save_txt=save_txt,
        plots=plots,
    )

    # v.init_metrics(model=None)
    # Ultralytics validator needs model.names for COCO mapping/metrics
    names = v.data.get("names", None)
    if names is None:
        raise RuntimeError(
            "Dataset is missing 'names'. Add a 'names:' block (e.g., 80 COCO classes) to your data.yaml."
        )

    v.init_metrics(model=_NamesOnlyModel(names))
    v.dataloader = v.get_dataloader(v.data[v.args.split], batch)

    for batch_i, batch_data in enumerate(v.dataloader):
        batch_data = v.preprocess(batch_data)
        imgs = batch_data["img"]
        preds = backend.infer_batch(imgs)
        v.update_metrics(preds, batch_data)

        if batch_i == 0:
            p0 = preds[0]
            print("[DEBUG] first batch preds[0]:", p0["bboxes"].shape, p0["conf"].shape, p0["cls"].shape)

            cls = preds[0]["cls"].cpu().numpy()
            conf = preds[0]["conf"].cpu().numpy()
            # print("[DEBUG] cls min/max:", cls.min() if cls.size else None, cls.max() if cls.size else None)
            # print("[DEBUG] conf min/max:", conf.min() if conf.size else None, conf.max() if conf.size else None)
            # print("[DEBUG] unique cls (first 20):", np.unique(cls)[:20])

            # b = preds[0]["bboxes"].cpu().numpy()
            # print("[DEBUG] box min/max:", b.min(), b.max())
            # print("[DEBUG] first 3 boxes:", b[:3])

        if max_batches and batch_i + 1 >= max_batches:
            break

    return v.get_stats()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser("Evaluate model outputs using Ultralytics COCO metrics")

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

    ap.add_argument("--model_a", required=True, help="Path to model A (.tflite/.onnx/.pt)")
    ap.add_argument("--model_b", required=True, help="Path to model B (.tflite/.onnx/.pt)")

    # New: per-model backends
    ap.add_argument("--backend_a", choices=["tflite", "ort", "torch"], default=None, help="Backend for model A")
    ap.add_argument("--backend_b", choices=["tflite", "ort", "torch"], default=None, help="Backend for model B")

    # Backward-compat: single backend for both
    ap.add_argument("--backend", choices=["tflite", "ort", "torch"], default=None,
                    help="(Deprecated) single backend for both A and B")

    # Optional per-model devices
    ap.add_argument("--device_a", default=None, help="Device for A (torch: cpu/mps/cuda:0; ort: cpu/cuda:0)")
    ap.add_argument("--device_b", default=None, help="Device for B (torch: cpu/mps/cuda:0; ort: cpu/cuda:0)")

    # Shared tflite knobs (used for whichever side uses tflite)
    ap.add_argument("--threads", type=int, default=4, help="Threads for TFLite")
    ap.add_argument("--tflite_delegate", choices=["cpu", "gpu"], default="cpu", help="TFLite delegate (gpu optional)")

    return ap.parse_args()


def build_backend(kind: str, model_path: str, args, device_override: Optional[str] = None) -> Backend:
    device = device_override or detect_best_device()

    if kind == "tflite":
        return TFLiteBackend(
            model_path=model_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            threads=args.threads,
            delegate=args.tflite_delegate,
        )
    if kind == "ort":
        return ORTBackend(
            model_path=model_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
        )
    if kind == "torch":
        return TorchBackend(
            model_path=model_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
        )
    raise ValueError(kind)


def main():
    args = parse_args()

    backend_a_kind = args.backend_a or args.backend
    backend_b_kind = args.backend_b or args.backend
    if backend_a_kind is None or backend_b_kind is None:
        raise SystemExit("Specify either --backend (both) or both --backend_a and --backend_b.")

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
        max_batches = args.max_batches
    )
    print("\n===== A stats =====")
    for k, v in stats_a.items():
        print(f"{k}: {v}")

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
        max_batches = args.max_batches
    )
    print("\n===== B stats =====")
    for k, v in stats_b.items():
        print(f"{k}: {v}")

    print("\n===== Delta (B - A) =====")
    for key in sorted(set(stats_a.keys()) & set(stats_b.keys())):
        try:
            da = float(stats_a[key])
            db = float(stats_b[key])
            print(f"{key}: {db - da:+.6f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()