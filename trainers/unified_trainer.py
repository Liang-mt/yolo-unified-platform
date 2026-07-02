"""
Unified Trainer - single entry point for training any YOLO variant.
Wraps ultralytics trainer with additional features:
- Custom loss functions
- Multi-GPU support
- Callback system
- Experiment tracking
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from core.config import Config, load_config, merge_configs, VARIANT_DEFAULTS
from core.registry import MODEL_REGISTRY, LOSS_REGISTRY
from trainers.callbacks import CallbackManager, EarlyStopping, ModelCheckpoint


class UnifiedTrainer:
    """
    Unified training interface for all YOLO variants.

    Usage:
        trainer = UnifiedTrainer(variant="yolov8", size="s", num_classes=80)
        trainer.train(data="coco128.yaml", epochs=100, batch_size=16)
    """

    def __init__(
        self,
        variant: str = "yolov8",
        size: str = "s",
        num_classes: int = 80,
        img_size: int = 640,
        device: str = "auto",
        project: str = "runs/train",
        name: str = "exp",
        pretrained: bool = True,
        weights: Optional[str] = None,
        custom_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.variant = variant.lower()
        self.size = size
        self.num_classes = num_classes
        self.img_size = img_size
        self.project = project
        self.name = name
        self.pretrained = pretrained
        self.weights = weights
        self.custom_cfg = custom_cfg or {}

        # Resolve device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Callbacks
        self.callback_mgr = CallbackManager()

        # Model (lazy init)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._init_model()
        return self._model

    def _init_model(self):
        """Initialize the YOLO model."""
        from ultralytics import YOLO

        variant_map = {
            "yolov5": {"n": "yolov5n", "s": "yolov5s", "m": "yolov5m", "l": "yolov5l", "x": "yolov5x"},
            "yolov8": {"n": "yolov8n", "s": "yolov8s", "m": "yolov8m", "l": "yolov8l", "x": "yolov8x"},
            "yolov10": {"n": "yolov10n", "s": "yolov10s", "m": "yolov10m", "l": "yolov10l", "x": "yolov10x"},
            "yolov11": {"n": "yolo11n", "s": "yolo11s", "m": "yolo11m", "l": "yolo11l", "x": "yolo11x"},
            "yolo26": {"n": "yolo26n", "s": "yolo26s", "m": "yolo26m", "l": "yolo26l", "x": "yolo26x"},
        }

        if self.weights and Path(self.weights).exists():
            self._model = YOLO(self.weights)
        else:
            size_map = variant_map.get(self.variant, {})
            model_name = size_map.get(self.size, f"{self.variant}{self.size}")
            self._model = YOLO(f"{model_name}.pt")

    def add_callback(self, event: str, callback) -> None:
        """Add a callback for training events."""
        self.callback_mgr.register(event, callback)

    def train(
        self,
        data: str,
        epochs: int = 100,
        batch_size: int = 16,
        lr0: float = 0.01,
        lrf: float = 0.01,
        optimizer: str = "auto",
        workers: int = 8,
        cos_lr: bool = False,
        close_mosaic: int = 10,
        amp: bool = True,
        patience: int = 50,
        resume: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run training.

        Args:
            data: Dataset YAML path
            epochs: Number of training epochs
            batch_size: Batch size
            lr0: Initial learning rate
            lrf: Final learning rate (factor of lr0)
            optimizer: 'SGD', 'Adam', 'AdamW', 'auto'
            workers: DataLoader workers
            cos_lr: Use cosine LR scheduler
            close_mosaic: Disable mosaic augmentation in last N epochs
            amp: Use Automatic Mixed Precision
            patience: Early stopping patience
            resume: Resume from last checkpoint

        Returns:
            Training results dict
        """
        self._init_model()

        # Merge all training args
        train_args = {
            "data": data,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": self.img_size,
            "device": self.device,
            "project": self.project,
            "name": self.name,
            "lr0": lr0,
            "lrf": lrf,
            "optimizer": optimizer,
            "workers": workers,
            "cos_lr": cos_lr,
            "close_mosaic": close_mosaic,
            "amp": amp,
            "patience": patience,
            "resume": resume,
            "pretrained": self.pretrained,
            **self.custom_cfg,
            **kwargs,
        }

        self.callback_mgr.fire("on_train_start", train_args)

        try:
            results = self._model.train(**train_args)
            self.callback_mgr.fire("on_train_end", results)
            return self._format_results(results)
        except Exception as e:
            self.callback_mgr.fire("on_train_error", e)
            raise

    def _format_results(self, results) -> Dict[str, Any]:
        """Format training results into a standard dict."""
        try:
            return {
                "fitness": results.fitness,
                "results_dict": results.results_dict,
                "save_dir": str(results.save_dir),
            }
        except Exception:
            return {"raw": str(results)}

    def validate(self, data: Optional[str] = None, **kwargs) -> Dict[str, float]:
        """Run validation on the trained model."""
        if self._model is None:
            raise RuntimeError("No model loaded. Train first or load weights.")

        val_args = {"imgsz": self.img_size, "device": self.device, **kwargs}
        if data:
            val_args["data"] = data

        results = self._model.val(**val_args)
        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the trained model."""
        if self._model is None:
            raise RuntimeError("No model loaded.")
        result = self._model.export(format=format, **kwargs)
        return str(result)

    @staticmethod
    def list_supported_variants() -> List[str]:
        return ["yolov5", "yolov8", "yolov10", "yolov11", "yolo26"]
