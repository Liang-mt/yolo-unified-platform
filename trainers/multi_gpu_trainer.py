"""
Multi-GPU Training Wrapper.
Supports DistributedDataParallel (DDP) and DataParallel (DP).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


class MultiGPUTrainer:
    """
    Multi-GPU training orchestrator.

    Usage:
        # Automatic multi-GPU via ultralytics DDP
        trainer = MultiGPUTrainer(variant="yolov8", gpus=[0, 1, 2, 3])
        trainer.train(data="coco.yaml", epochs=100)
    """

    def __init__(
        self,
        variant: str = "yolov8",
        size: str = "s",
        num_classes: int = 80,
        gpus: Optional[List[int]] = None,
        strategy: str = "ddp",
        project: str = "runs/train",
        name: str = "exp",
    ):
        self.variant = variant
        self.size = size
        self.num_classes = num_classes
        self.strategy = strategy
        self.project = project
        self.name = name

        # Auto-detect GPUs
        if gpus is None:
            self.gpus = list(range(torch.cuda.device_count()))
        else:
            self.gpus = gpus

        self.num_gpus = len(self.gpus)

    def train(self, data: str, epochs: int = 100, batch_size: int = 16, **kwargs) -> Dict[str, Any]:
        """
        Launch multi-GPU training.

        For DDP: spawns processes via ultralytics built-in DDP.
        For DP: wraps model in DataParallel.
        """
        if self.num_gpus <= 1:
            return self._single_gpu_train(data, epochs, batch_size, **kwargs)

        if self.strategy == "ddp":
            return self._ddp_train(data, epochs, batch_size, **kwargs)
        elif self.strategy == "dp":
            return self._dp_train(data, epochs, batch_size, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}. Use 'ddp' or 'dp'.")

    def _single_gpu_train(self, data, epochs, batch_size, **kwargs) -> Dict:
        """Single GPU fallback."""
        from .unified_trainer import UnifiedTrainer
        trainer = UnifiedTrainer(
            variant=self.variant, size=self.size,
            num_classes=self.num_classes, device=f"cuda:{self.gpus[0]}" if self.gpus else "cpu",
            project=self.project, name=self.name,
        )
        return trainer.train(data=data, epochs=epochs, batch_size=batch_size, **kwargs)

    def _ddp_train(self, data, epochs, batch_size, **kwargs) -> Dict:
        """DDP training via ultralytics built-in support."""
        from ultralytics import YOLO

        variant_map = {
            "yolov5": "yolov5", "yolov8": "yolov8",
            "yolov10": "yolov10", "yolov11": "yolo11", "yolo26": "yolo26",
        }
        prefix = variant_map.get(self.variant, self.variant)
        model = YOLO(f"{prefix}{self.size}.pt")

        device_str = ",".join(str(g) for g in self.gpus)
        results = model.train(
            data=data, epochs=epochs, batch=batch_size,
            device=device_str, project=self.project, name=self.name, **kwargs,
        )

        return {
            "fitness": results.fitness if hasattr(results, "fitness") else None,
            "save_dir": str(results.save_dir) if hasattr(results, "save_dir") else None,
            "gpus_used": self.num_gpus,
        }

    def _dp_train(self, data, epochs, batch_size, **kwargs) -> Dict:
        """DataParallel training (less efficient but simpler)."""
        print(f"⚠️ DataParallel mode: using {self.num_gpus} GPUs")
        # DP is handled by setting device to multiple GPUs
        return self._ddp_train(data, epochs, batch_size, **kwargs)

    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Get information about available GPUs."""
        info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info.append({
                "index": i,
                "name": props.name,
                "memory_gb": getattr(props, 'total_mem', getattr(props, 'total_memory', 0)) / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}",
            })
        return info

    @staticmethod
    def launch_ddp_script(
        script_path: str,
        gpus: List[int],
        master_port: str = "12355",
        **kwargs,
    ) -> subprocess.Popen:
        """
        Launch a DDP training script manually.

        Args:
            script_path: Path to training script
            gpus: List of GPU indices
            master_port: Port for DDP communication
        """
        env = os.environ.copy()
        env["MASTER_PORT"] = master_port
        env["WORLD_SIZE"] = str(len(gpus))
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

        cmd = [sys.executable, "-m", "torch.distributed.launch",
               "--nproc_per_node", str(len(gpus)),
               "--master_port", master_port,
               script_path] + [f"--{k}={v}" for k, v in kwargs.items()]

        return subprocess.Popen(cmd, env=env)
