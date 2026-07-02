"""YOLO Model Implementations - auto-registers all variants on import."""

from .yolov5_model import YOLOv5Model
from .yolov8_model import YOLOv8Model
from .yolov10_model import YOLOv10Model
from .yolov11_model import YOLOv11Model
from .yolo26_model import YOLO26Model

__all__ = ["YOLOv5Model", "YOLOv8Model", "YOLOv10Model", "YOLOv11Model", "YOLO26Model"]
