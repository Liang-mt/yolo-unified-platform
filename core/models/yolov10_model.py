"""
YOLOv10 Model Wrapper.
"""

from typing import Any, Dict, List, Union
from pathlib import Path
import torch

from ..base_model import BaseYOLOModel
from ..registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("yolov10")
class YOLOv10Model(BaseYOLOModel):
    """YOLOv10 via ultralytics library."""

    SIZE_MAP = {"n": "yolov10n", "s": "yolov10s", "m": "yolov10m", "l": "yolov10l", "x": "yolov10x", "b": "yolov10b"}

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self._ultra_model = None
        self._build_model()

    def _build_model(self):
        from ultralytics import YOLO
        size = self._cfg.get("size", "s")
        name = self.SIZE_MAP.get(size, f"yolov10{size}")
        self._ultra_model = YOLO(f"{name}.pt")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        return self._ultra_model(x, *args, **kwargs)

    def load_pretrained(self, weights: str) -> "YOLOv10Model":
        from ultralytics import YOLO
        self._ultra_model = YOLO(weights)
        return self

    def export_onnx(self, save_path, img_size=640, opset=17, simplify=True, dynamic=False) -> str:
        self._ultra_model.export(format="onnx", imgsz=img_size, opset=opset, simplify=simplify, dynamic=dynamic)
        return str(save_path)

    def predict(self, source, conf=0.25, iou=0.45, img_size=640, device="auto") -> list:
        return self._ultra_model.predict(source=source, conf=conf, iou=iou, imgsz=img_size, device=device, verbose=False)

    def val(self, data, img_size=640, batch_size=16, **kwargs) -> dict:
        results = self._ultra_model.val(data=data, imgsz=img_size, batch=batch_size, **kwargs)
        return {"mAP50": results.box.map50, "mAP50-95": results.box.map, "precision": results.box.mp, "recall": results.box.mr}

    def train(self, data, epochs=100, imgsz=640, batch=16, **kwargs):
        return self._ultra_model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, **kwargs)
