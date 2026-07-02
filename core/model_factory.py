"""
Model Factory: unified entry point to create any YOLO variant.
Handles model instantiation, pretrained loading, and config resolution.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path

from .registry import MODEL_REGISTRY
from .config import Config, VARIANT_DEFAULTS, merge_configs


class ModelFactory:
    """
    Factory to create YOLO models across all variants.

    Usage:
        model = ModelFactory.create("yolov8", size="s", num_classes=80)
        model = ModelFactory.create("yolov5", size="m", pretrained=True)
    """

    @staticmethod
    def create(
        variant: str,
        size: str = "s",
        num_classes: int = 80,
        img_size: int = 640,
        pretrained: bool = True,
        weights: Optional[str] = None,
        device: str = "auto",
        **kwargs,
    ) -> Any:
        """
        Create a YOLO model instance.

        Args:
            variant: One of 'yolov5', 'yolov8', 'yolov10', 'yolov11', 'yolo26'
            size: Model size - 'n', 's', 'm', 'l', 'x'
            num_classes: Number of detection classes
            img_size: Input image size
            pretrained: Load pretrained weights
            weights: Path to custom weights file
            device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.

        Returns:
            Model instance (BaseYOLOModel subclass or ultralytics.YOLO)
        """
        variant = variant.lower().replace("-", "").replace("_", "")
        if variant not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {MODEL_REGISTRY.list_keys()}"
            )

        # Build config from defaults + overrides
        defaults = VARIANT_DEFAULTS.get(variant, {}).copy()
        cfg = merge_configs(defaults, {
            "variant": variant,
            "size": size,
            "num_classes": num_classes,
            "img_size": img_size,
            "pretrained": pretrained,
            **kwargs,
        })

        model_cls = MODEL_REGISTRY.get(variant)
        model = model_cls(cfg)

        # Load weights
        if weights and Path(weights).exists():
            model.load_pretrained(weights)
        elif pretrained:
            model.load_pretrained(f"{variant}{size}")

        # Move to device
        if device != "auto":
            import torch
            model = model.to(device)
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

        return model

    @staticmethod
    def list_variants() -> list:
        return MODEL_REGISTRY.list_keys()

    @staticmethod
    def from_config(cfg: Union[Config, Dict[str, Any]]) -> Any:
        """Create model from a full Config object."""
        if isinstance(cfg, Config):
            cfg = cfg.to_dict()
        variant = cfg.get("model", {}).get("variant", "unknown")
        return ModelFactory.create(
            variant=variant,
            size=cfg.get("model", {}).get("size", "s"),
            num_classes=cfg.get("model", {}).get("num_classes", 80),
            img_size=cfg.get("train", {}).get("img_size", 640),
            pretrained=cfg.get("model", {}).get("pretrained", True),
        )
