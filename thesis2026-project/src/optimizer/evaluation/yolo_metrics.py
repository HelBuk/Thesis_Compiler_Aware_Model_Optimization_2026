from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import IterableSimpleNamespace


class Backend:
    """Return preds in Ultralytics validator format:
    list length B of dicts:
      {"bboxes": (N,4) xyxy in imgsz space, "conf": (N,), "cls": (N,), "extra": (N,0)}
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
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    else:
        raise ValueError(f"Unexpected output shape {out.shape}, expected [1,*,*]")

    if out.shape[0] == 84:
        pred = out.T
    elif out.shape[1] == 84:
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

    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    if boxes.max() <= 2.0:
        boxes *= float(imgsz)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, imgsz)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, imgsz)

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

    def infer_batch(self, imgs_bchw01: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        imgs = imgs_bchw01.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
        preds = []

        in_dtype = self.inp["dtype"]
        in_scale, in_zero = self.inp.get("quantization", (0.0, 0))
        out_dtype = self.out["dtype"]
        out_scale, out_zero = self.out.get("quantization", (0.0, 0))

        for i in range(imgs.shape[0]):
            x = imgs[i : i + 1]

            if in_dtype in (np.uint8, np.int8):
                if in_scale == 0:
                    raise RuntimeError("TFLite input scale is 0; cannot quantize input.")
                xq = np.round(x / in_scale + in_zero).astype(in_dtype)
                self.interp.set_tensor(self.inp["index"], xq)
            else:
                self.interp.set_tensor(self.inp["index"], x.astype(in_dtype))

            self.interp.invoke()
            y = self.interp.get_tensor(self.out["index"])

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
    def __init__(
        self,
        model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        provider_kind: str = "ort",
        trt_fp16: bool = False,
        trt_int8: bool = False,
        trt_engine_cache: bool = False,
        trt_engine_cache_path: str = "./trt_cache",
        trt_workspace_size: int = 2147483648,
    ):
        super().__init__(imgsz, conf, iou, max_det, device)
        import onnxruntime as ort

        self.provider_kind = provider_kind
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        provider_kind = provider_kind.lower()
        provider_options = None

        if provider_kind == "tensorrt":
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            provider_options = [
                {
                    "trt_fp16_enable": trt_fp16,
                    "trt_int8_enable": trt_int8,
                    "trt_engine_cache_enable": trt_engine_cache,
                    "trt_engine_cache_path": trt_engine_cache_path,
                    "trt_max_workspace_size": str(trt_workspace_size),
                },
                {},
                {},
            ]
        elif device.startswith("cuda") or device == "0":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        self.sess = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
        self.in_name = self.sess.get_inputs()[0].name

        print("[ORT] requested providers:", providers)
        print("[ORT] active providers:", self.sess.get_providers())

    def infer_batch(self, imgs_bchw01: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        x = imgs_bchw01.detach().cpu().numpy().astype(np.float32)
        y = self.sess.run(None, {self.in_name: x})[0]

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
        

class TensorRTBackend(ORTBackend):
    def __init__(
        self,
        model_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
        trt_fp16: bool = False,
        trt_int8: bool = False,
        trt_engine_cache: bool = False,
        trt_engine_cache_path: str = "./trt_cache",
        trt_workspace_size: int = 2147483648,
    ):
        super().__init__(
            model_path=model_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            provider_kind="tensorrt",
            trt_fp16=trt_fp16,
            trt_int8=trt_int8,
            trt_engine_cache=trt_engine_cache,
            trt_engine_cache_path=trt_engine_cache_path,
            trt_workspace_size=trt_workspace_size,
        )

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
    v.device = torch.device("cpu")
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
    max_batches: int,
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

        if max_batches and batch_i + 1 >= max_batches:
            break

    return v.get_stats()


def build_backend(kind: str, model_path: str, args, device_override: Optional[str] = None) -> Backend:
    device = device_override or detect_best_device()
    kind = kind.lower()

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
            provider_kind="ort",
            trt_fp16=getattr(args, "trt_fp16", False),
            trt_int8=getattr(args, "trt_int8", False),
            trt_engine_cache=getattr(args, "trt_engine_cache", False),
            trt_engine_cache_path=getattr(args, "trt_engine_cache_path", "./trt_cache"),
            trt_workspace_size=getattr(args, "trt_workspace_size", 2147483648),
        )

    if kind == "tensorrt":
        return TensorRTBackend(
            model_path=model_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            trt_fp16=getattr(args, "trt_fp16", False),
            trt_int8=getattr(args, "trt_int8", False),
            trt_engine_cache=getattr(args, "trt_engine_cache", False),
            trt_engine_cache_path=getattr(args, "trt_engine_cache_path", "./trt_cache"),
            trt_workspace_size=getattr(args, "trt_workspace_size", 2147483648),
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

    raise ValueError(f"Unsupported backend: {kind}")