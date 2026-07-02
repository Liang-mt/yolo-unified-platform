"""
Universal Registry for YOLO Unified Platform.
Provides a decorator-based registration mechanism for models, trainers, losses, etc.
"""

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A universal registry to map strings to classes or functions."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: str, obj: Optional[Any] = None) -> Any:
        """Register an object. Can be used as decorator or direct call."""
        if obj is not None:
            self._registry[key] = obj
            return obj

        def decorator(func_or_cls: Any) -> Any:
            self._registry[key] = func_or_cls
            return func_or_cls
        return decorator

    def get(self, key: str) -> Any:
        """Get registered object by key."""
        if key not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{key}' not found in {self._name} registry. "
                f"Available: {available}"
            )
        return self._registry[key]

    def list_keys(self) -> list:
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={len(self._registry)})"


# ─── Global Registries ───────────────────────────────────────────────

MODEL_REGISTRY = Registry("model")
TRAINER_REGISTRY = Registry("trainer")
LOSS_REGISTRY = Registry("loss")
AUGMENTATION_REGISTRY = Registry("augmentation")
DEPLOYER_REGISTRY = Registry("deployer")
PRUNER_REGISTRY = Registry("pruner")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")
METRIC_REGISTRY = Registry("metric")
