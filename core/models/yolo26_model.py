"""
YOLO26 Model Wrapper.
The latest YOLO variant, using ultralytics backend.
"""

from typing import Any, Dict
import torch

from ..base_model import BaseYOLOModel
from ..registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("yolo26")
class YOLO26Model(BaseYOLOModel):
    """YOLO26 model - latest generation YOLO."""

    SIZE_MAP = {"n": "yolo26n", "s": "yolo26s", "m": "yolo26m", "l": "yolo26l", "x": "yolo26x", "c": "yolo26c", "e": "yolo26e"}

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self._ultra_model = None
        self._build_model()

    def _build_model(self):
        from ultralytics import YOLO
        size = self._cfg.get("size", "s")
        name = self.SIZE_MAP.get(size, f"yolo26{size}")
        if self._cfg.get("pretrained", True):
            try:
                self._ultra_model = YOLO(f"{name}.pt")
            except Exception:
                self._ultra_model = YOLO(f"{name}.yaml")
        else:
            self._ultra_model = YOLO(f"{name}.yaml")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        return self._ultra_model(x, *args, **kwargs)

    def load_pretrained(self, weights: str) -> "YOLO26Model":
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
