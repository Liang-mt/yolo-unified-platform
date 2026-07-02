"""
Combined Loss - compose multiple loss functions with configurable weights.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .focal_loss import FocalLoss
from .ciou_loss import CIoULoss, DIoULoss, GIoULoss, SIoULoss
from .wasserstein_loss import WassersteinLoss


class CombinedLoss(nn.Module):
    """
    Composable loss function for YOLO training.

    Usage:
        loss_fn = CombinedLoss(
            cls_loss={"type": "focal", "weight": 1.0, "alpha": 0.25, "gamma": 2.0},
            box_loss={"type": "ciou", "weight": 5.0},
            dfl_loss={"type": "dfl", "weight": 1.0},
        )
        total_loss = loss_fn(cls_pred, box_pred, targets)
    """

    LOSS_REGISTRY = {
        "focal": FocalLoss,
        "quality_focal": lambda **kw: __import__('losses.focal_loss', fromlist=['QualityFocalLoss']).QualityFocalLoss(**kw),
        "ciou": CIoULoss,
        "diou": DIoULoss,
        "giou": GIoULoss,
        "siou": SIoULoss,
        "wasserstein": WassersteinLoss,
        "bce": lambda **kw: nn.BCEWithLogitsLoss(),
        "mse": lambda **kw: nn.MSELoss(),
        "smooth_l1": lambda **kw: nn.SmoothL1Loss(),
    }

    def __init__(
        self,
        cls_loss: Optional[Dict] = None,
        box_loss: Optional[Dict] = None,
        dfl_loss: Optional[Dict] = None,
    ):
        super().__init__()
        self.losses = nn.ModuleDict()

        if cls_loss:
            self._add_loss("cls", cls_loss)
        if box_loss:
            self._add_loss("box", box_loss)
        if dfl_loss:
            self._add_loss("dfl", dfl_loss)

    def _add_loss(self, name: str, cfg: Dict) -> None:
        loss_type = cfg.get("type", "bce")
        weight = cfg.get("weight", 1.0)

        if loss_type not in self.LOSS_REGISTRY:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(self.LOSS_REGISTRY.keys())}")

        loss_cls = self.LOSS_REGISTRY[loss_type]
        # Filter out 'type' and 'weight' from kwargs
        kwargs = {k: v for k, v in cfg.items() if k not in ("type", "weight")}
        self.losses[name] = loss_cls(**kwargs)
        self._weights[name] = weight

    @property
    def _weights(self):
        if not hasattr(self, "__weights"):
            self.__weights = {}
        return self.__weights

    def forward(self, cls_pred: torch.Tensor, box_pred: torch.Tensor,
                targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        total = torch.tensor(0.0, device=cls_pred.device)

        if "cls" in self.losses:
            cls_loss = self.losses["cls"](cls_pred, targets)
            losses["cls_loss"] = cls_loss
            total = total + self._weights.get("cls", 1.0) * cls_loss

        if "box" in self.losses:
            box_loss = self.losses["box"](box_pred, targets)
            losses["box_loss"] = box_loss
            total = total + self._weights.get("box", 1.0) * box_loss

        if "dfl" in self.losses:
            dfl_loss = self.losses["dfl"](cls_pred, targets)
            losses["dfl_loss"] = dfl_loss
            total = total + self._weights.get("dfl", 1.0) * dfl_loss

        losses["total_loss"] = total
        return losses


class EIoULoss(CIoULoss):
    """Efficient IoU Loss - CIoU with enhanced aspect ratio penalty."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        iou = self.bbox_iou(pred, target)

        center_dist = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2

        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        enclose_x1 = torch.min(pred_x1, tgt_x1)
        enclose_y1 = torch.min(pred_y1, tgt_y1)
        enclose_x2 = torch.max(pred_x2, tgt_x2)
        enclose_y2 = torch.max(pred_y2, tgt_y2)

        diagonal_dist = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

        # Width/height difference
        w_diff = (pred[:, 2] - target[:, 2]) ** 2
        h_diff = (pred[:, 3] - target[:, 3]) ** 2

        # Enclosing box dimensions
        enclose_w = (enclose_x2 - enclose_x1) ** 2 + 1e-7
        enclose_h = (enclose_y2 - enclose_y1) ** 2 + 1e-7

        rho_w = w_diff / enclose_w
        rho_h = h_diff / enclose_h

        eiou = iou - center_dist / diagonal_dist - rho_w - rho_h
        return (1 - eiou).mean()
