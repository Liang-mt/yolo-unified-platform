"""
Callback System for Training Events.
Provides hooks for logging, checkpointing, early stopping, etc.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch


class Callback:
    """Base callback class."""

    def on_train_start(self, args: Dict[str, Any]) -> None:
        pass

    def on_train_end(self, results: Any) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_train_error(self, error: Exception) -> None:
        pass


class CallbackManager:
    """Manages and fires callbacks."""

    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {}

    def register(self, event: str, callback: Callable) -> None:
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def fire(self, event: str, *args, **kwargs) -> None:
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ Callback error ({event}): {e}")


class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting."""

    def __init__(self, patience: int = 50, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.stopped = False

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        score = metrics.get("mAP50-95", metrics.get("fitness", 0))

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                print(f"🛑 Early stopping at epoch {epoch} (patience={self.patience})")
        else:
            self.best_score = score
            self.counter = 0


class ModelCheckpoint(Callback):
    """Save model checkpoints at specified intervals."""

    def __init__(
        self,
        save_dir: Union[str, Path],
        monitor: str = "mAP50-95",
        mode: str = "max",
        save_best: bool = True,
        save_last: bool = True,
        every_n_epochs: int = 0,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.best_score = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        score = metrics.get(self.monitor, 0)

        if self.save_best:
            if self.best_score is None or (self.mode == "max" and score > self.best_score) or \
               (self.mode == "min" and score < self.best_score):
                self.best_score = score
                print(f"💾 Best model saved: {self.monitor}={score:.4f}")

        if self.every_n_epochs > 0 and (epoch + 1) % self.every_n_epochs == 0:
            print(f"💾 Checkpoint saved at epoch {epoch + 1}")


class WandBLogger(Callback):
    """Weights & Biases logging callback."""

    def __init__(self, project: str, name: str, config: Optional[Dict] = None):
        self.project = project
        self.name = name
        self.config = config or {}
        self._run = None

    def on_train_start(self, args: Dict[str, Any]) -> None:
        try:
            import wandb
            self._run = wandb.init(project=self.project, name=self.name, config={**self.config, **args})
        except ImportError:
            print("⚠️ wandb not installed, skipping W&B logging")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if self._run:
            self._run.log({"epoch": epoch, **metrics})

    def on_train_end(self, results: Any) -> None:
        if self._run:
            self._run.finish()


class TensorBoardLogger(Callback):
    """TensorBoard logging callback."""

    def __init__(self, log_dir: str = "runs/tb"):
        self.log_dir = log_dir
        self._writer = None

    def on_train_start(self, args: Dict[str, Any]) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("⚠️ tensorboard not installed")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if self._writer:
            for key, val in metrics.items():
                self._writer.add_scalar(key, val, epoch)

    def on_train_end(self, results: Any) -> None:
        if self._writer:
            self._writer.close()
