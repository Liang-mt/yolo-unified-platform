"""
Focal Loss variants for handling class imbalance in detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Balancing factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (used in FCOS, ATSS, etc.).
    Combines classification and IoU quality estimation.

    Args:
        beta: Smoothing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, beta: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, C) predicted logits
            target: (N, C) soft targets with quality scores
        """
        # Sigmoid cross entropy
        pred_sigmoid = pred.sigmoid()
        focal_weight = torch.abs(pred_sigmoid - target) ** self.beta

        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        qfl = focal_weight * bce_loss

        if self.reduction == "mean":
            return qfl.mean()
        elif self.reduction == "sum":
            return qfl.sum()
        return qfl


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss (used in VFNet).
    Similar to QFL but with asymmetric focusing on positive/negative samples.

    Args:
        alpha: Weight for positive samples (default: 0.75)
        gamma: Focusing parameter (default: 2.0)
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = pred.sigmoid()

        # Positive samples: focal weight on quality
        pos_mask = target > 0
        focal_weight = torch.zeros_like(pred)
        focal_weight[pos_mask] = target[pos_mask] * torch.abs(pred_sigmoid[pos_mask] - target[pos_mask]) ** self.gamma

        # Negative samples: standard focal
        neg_mask = target == 0
        focal_weight[neg_mask] = pred_sigmoid[neg_mask] ** self.gamma

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        vfl = focal_weight * bce_loss

        return vfl.mean()


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) for bounding box regression.
    Models box coordinates as distributions over discretized bins.

    Args:
        reg_max: Maximum regression value (default: 16)
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (N, 4*(reg_max+1)) predicted distributions
            target: (N, 4) target values (continuous)
        """
        N = target.shape[0]
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()

        # Clamp
        target_left = target_left.clamp(0, self.reg_max - 1)
        target_right = target_right.clamp(0, self.reg_max)

        # Reshape pred to (N*4, reg_max+1)
        pred = pred.view(-1, self.reg_max + 1)

        # Flatten targets to (N*4,)
        target_left = target_left.view(-1)
        target_right = target_right.view(-1)
        weight_left = weight_left.view(-1)
        weight_right = weight_right.view(-1)

        # Compute loss
        left_loss = F.cross_entropy(pred, target_left, reduction="none")
        right_loss = F.cross_entropy(pred, target_right, reduction="none")

        dfl = left_loss * weight_left + right_loss * weight_right
        return dfl.mean()
