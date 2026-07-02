"""
Configuration management for YOLO Unified Platform.
Supports YAML loading, merging, and validation.
"""

import os
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """Nested attribute-access config object."""

    def __init__(self, cfg_dict: Optional[Dict[str, Any]] = None):
        self._cfg = cfg_dict or {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            val = self._cfg[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'")
        if isinstance(val, dict):
            return Config(val)
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._cfg[name] = value

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._cfg[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._cfg

    def get(self, key: str, default: Any = None) -> Any:
        return self._cfg.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)

    def update(self, other: Dict[str, Any]) -> None:
        deep_update(self._cfg, other)

    def __repr__(self) -> str:
        return f"Config({self._cfg})"


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple config dicts, later ones override earlier ones."""
    result = {}
    for cfg in configs:
        deep_update(result, cfg)
    return result


def load_config(path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load YAML config file and apply optional overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}

    if overrides:
        deep_update(cfg_dict, overrides)

    return Config(cfg_dict)


# ─── Predefined Config Templates ─────────────────────────────────────

YOLOV5_DEFAULTS = {
    "model": {"variant": "yolov5", "size": "s", "pretrained": True, "num_classes": 80},
    "train": {
        "epochs": 300, "batch_size": 16, "img_size": 640,
        "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
        "momentum": 0.937, "weight_decay": 0.0005,
        "warmup_epochs": 3.0, "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1, "workers": 8,
    },
    "augment": {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5,
        "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
        "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0,
    },
}

YOLOV8_DEFAULTS = {
    "model": {"variant": "yolov8", "size": "s", "pretrained": True, "num_classes": 80},
    "train": {
        "epochs": 100, "batch_size": 16, "img_size": 640,
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01,
        "momentum": 0.937, "weight_decay": 0.0005,
        "warmup_epochs": 3.0, "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1, "workers": 8,
        "close_mosaic": 10,
    },
}

YOLOV10_DEFAULTS = {
    "model": {"variant": "yolov10", "size": "s", "pretrained": True, "num_classes": 80},
    "train": {
        "epochs": 100, "batch_size": 16, "img_size": 640,
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01,
        "workers": 8,
    },
}

YOLOV11_DEFAULTS = {
    "model": {"variant": "yolov11", "size": "s", "pretrained": True, "num_classes": 80},
    "train": {
        "epochs": 100, "batch_size": 16, "img_size": 640,
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01,
        "workers": 8,
    },
}

YOLO26_DEFAULTS = {
    "model": {"variant": "yolo26", "size": "s", "pretrained": True, "num_classes": 80},
    "train": {
        "epochs": 100, "batch_size": 16, "img_size": 640,
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01,
        "workers": 8,
    },
}

VARIANT_DEFAULTS = {
    "yolov5": YOLOV5_DEFAULTS,
    "yolov8": YOLOV8_DEFAULTS,
    "yolov10": YOLOV10_DEFAULTS,
    "yolov11": YOLOV11_DEFAULTS,
    "yolo26": YOLO26_DEFAULTS,
}
