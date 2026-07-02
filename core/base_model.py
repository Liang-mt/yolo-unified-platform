"""
Abstract base class for all YOLO model variants.
Ensures a unified interface across YOLOv5/v8/v10/v11/yolo26.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn


class BaseYOLOModel(ABC, nn.Module):
    """Unified interface for all YOLO model variants."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self._cfg = cfg
        self._variant = cfg.get("variant", "unknown")
        self._num_classes = cfg.get("num_classes", 80)
        self._img_size = cfg.get("img_size", 640)

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        """Forward pass. Returns model-dependent output format."""
        ...

    @abstractmethod
    def load_pretrained(self, weights: str) -> "BaseYOLOModel":
        """Load pretrained weights from path or identifier."""
        ...

    @abstractmethod
    def export_onnx(
        self,
        save_path: Union[str, Path],
        img_size: int = 640,
        opset: int = 17,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> str:
        """Export model to ONNX format. Returns saved path."""
        ...

    @abstractmethod
    def predict(
        self,
        source: Union[str, Path, torch.Tensor],
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 640,
        device: str = "auto",
    ) -> List[Any]:
        """Run inference and return detections."""
        ...

    @abstractmethod
    def val(
        self,
        data: str,
        img_size: int = 640,
        batch_size: int = 16,
        **kwargs,
    ) -> Dict[str, float]:
        """Run validation and return metrics dict."""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "variant": self._variant,
            "num_classes": self._num_classes,
            "img_size": self._img_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_params_m": total_params / 1e6,
        }

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters."""
        for name, param in self.named_parameters():
            if "head" not in name and "detect" not in name:
                param.requires_grad = not freeze

    def get_optim_param_groups(self, lr: float, weight_decay: float) -> List[Dict]:
        """Return parameter groups with bias/norm handling for optimizer."""
        pg_bn, pg_other = [], []
        for name, module in self.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_name = f"{name}.{pname}" if name else pname
                if pname == "bias":
                    pg_bn.append(param)
                else:
                    pg_other.append(param)
        return [
            {"params": pg_other, "lr": lr, "weight_decay": weight_decay},
            {"params": pg_bn, "lr": lr, "weight_decay": 0.0},
        ]
