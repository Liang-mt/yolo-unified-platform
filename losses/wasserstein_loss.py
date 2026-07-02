"""
Wasserstein Distance-based Loss for bounding box regression.
Provides smoother gradients than IoU-based losses for small objects.
"""

import torch
import torch.nn as nn


class WassersteinLoss(nn.Module):
    """
    Wasserstein Distance-based Bounding Box Loss.

    Uses the 2nd-order Wasserstein distance between Gaussian distributions
    parameterized by bounding boxes. More effective for small objects and
    non-overlapping boxes.

    Reference: "Rethinking Rotated Object Detection with Gaussian Wasserstein Distance"
    """

    def __init__(self, eps: float = 1e-6, distance_type: str = "l2"):
        super().__init__()
        self.eps = eps
        self.distance_type = distance_type

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Wasserstein loss between predicted and target boxes.

        Args:
            pred: (N, 4) predicted boxes [cx, cy, w, h]
            target: (N, 4) target boxes [cx, cy, w, h]

        Returns:
            Scalar loss
        """
        # Convert boxes to Gaussian parameters (mean, covariance diagonal)
        pred_mean = pred[:, :2]  # (cx, cy)
        pred_var = (pred[:, 2:] / 2).clamp(min=self.eps)  # (w/2, h/2) as std

        tgt_mean = target[:, :2]
        tgt_var = (target[:, 2:] / 2).clamp(min=self.eps)

        # 2nd-order Wasserstein distance for diagonal Gaussians:
        # W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2
        if self.distance_type == "l2":
            mean_dist = ((pred_mean - tgt_mean) ** 2).sum(dim=-1)
            var_dist = ((pred_var - tgt_var) ** 2).sum(dim=-1)
        else:  # l1
            mean_dist = (pred_mean - tgt_mean).abs().sum(dim=-1)
            var_dist = (pred_var - tgt_var).abs().sum(dim=-1)

        wasserstein_dist = mean_dist + var_dist

        # Normalize to [0, 1] range
        loss = 1 - torch.exp(-wasserstein_dist)
        return loss.mean()


class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance Loss.
    Normalizes the Wasserstein distance by box size for scale-invariance.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred[:, :2]
        pred_wh = pred[:, 2:].clamp(min=self.eps)
        tgt_mean = target[:, :2]
        tgt_wh = target[:, 2:].clamp(min=self.eps)

        # Center distance normalized by box diagonal
        center_dist = ((pred_mean - tgt_mean) ** 2).sum(dim=-1)
        pred_diag = (pred_wh[:, 0] ** 2 + pred_wh[:, 1] ** 2).clamp(min=self.eps)
        tgt_diag = (tgt_wh[:, 0] ** 2 + tgt_wh[:, 1] ** 2).clamp(min=self.eps)

        # Size difference normalized
        wh_dist = ((pred_wh - tgt_wh) ** 2).sum(dim=-1)

        nwd = center_dist / (pred_diag + tgt_diag) + wh_dist / (pred_wh * tgt_wh).sum(dim=-1).clamp(min=self.eps)
        return (1 - torch.exp(-nwd)).mean()
