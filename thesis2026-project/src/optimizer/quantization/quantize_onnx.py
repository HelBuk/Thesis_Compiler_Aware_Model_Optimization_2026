from pathlib import Path
import numpy as np
from PIL import Image

from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
    quantize_static,
)

# ---------------------------
# Config
# ---------------------------
MODEL_FP32 = "../models/quantized_models/onnx/yolov8n_opset17_fp32.onnx"
MODEL_INT8 = "../models/quantized_models/onnx/yolov8n_opset17_int8_qop.onnx"
IMAGE_DIR = "../datasets/coco_subset/train_0_1percent/images"
INPUT_NAME = "images"
IMG_SIZE = (640, 640)


def preprocess_image(image_path: str, img_size=(640, 640)) -> np.ndarray:
    target_h, target_w = img_size

    img = Image.open(image_path).convert("RGB")
    img = np.asarray(img)  # HWC uint8

    h, w = img.shape[:2]
    r = min(target_w / w, target_h / h)

    new_w = int(round(w * r))
    new_h = int(round(h * r))

    img_resized = np.array(
        Image.fromarray(img).resize((new_w, new_h), Image.Resampling.BILINEAR)
    )

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = img_resized

    arr = canvas.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))   # CHW
    arr = np.expand_dims(arr, 0)         # NCHW

    assert arr.shape == (1, 3, target_h, target_w), arr.shape
    return arr


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir: str, input_name: str, img_size=(640, 640), max_samples=None):
        self.input_name = input_name
        self.img_size = img_size

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = sorted(
            [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]
        )

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        self.data_list = [
            {self.input_name: preprocess_image(str(p), self.img_size)}
            for p in self.image_paths
        ]
        self.enum_data = None

        print(f"Loaded {len(self.data_list)} calibration images")

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


dr = ImageCalibrationDataReader(
    image_dir=IMAGE_DIR,
    input_name=INPUT_NAME,
    img_size=IMG_SIZE,
    max_samples=100,   # optional
)

quantize_static(
    model_input=MODEL_FP32,
    model_output=MODEL_INT8,
    calibration_data_reader=dr,
    quant_format=QuantFormat.QOperator, #QDQ
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
    op_types_to_quantize=["Conv", "MatMul"],
)

print(f"Saved INT8 model to: {MODEL_INT8}")