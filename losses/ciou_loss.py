"""
IoU-based Loss Functions for Bounding Box Regression.
Includes CIoU, DIoU, GIoU, SIoU variants.
"""

import math
import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """Base IoU loss computation."""

    @staticmethod
    def bbox_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Compute IoU between pred and target boxes (cx, cy, w, h format)."""
        # Convert to x1y1x2y2
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        tgt_area = (tgt_x2 - tgt_x1) * (tgt_y2 - tgt_y1)
        union_area = pred_area + tgt_area - inter_area + eps

        iou = inter_area / union_area
        return iou


class GIoULoss(IoULoss):
    """Generalized IoU Loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        iou = self.bbox_iou(pred, target)

        # Enclosing box
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

        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        pred_area = pred[:, 2] * pred[:, 3]
        tgt_area = target[:, 2] * target[:, 3]
        union_area = pred_area + tgt_area - iou * (pred_area + tgt_area - iou * (pred_area + tgt_area)) + 1e-7

        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
        return (1 - giou).mean()


class DIoULoss(IoULoss):
    """Distance IoU Loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        iou = self.bbox_iou(pred, target)

        # Center distance
        center_dist = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2

        # Enclosing box diagonal
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

        diou = iou - center_dist / diagonal_dist
        return (1 - diou).mean()


class CIoULoss(IoULoss):
    """Complete IoU Loss - considers overlap, center distance, and aspect ratio."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        iou = self.bbox_iou(pred, target)

        # Center distance
        center_dist = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2

        # Enclosing box diagonal
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

        # Aspect ratio
        pred_w = pred[:, 2].clamp(min=1e-7)
        pred_h = pred[:, 3].clamp(min=1e-7)
        tgt_w = target[:, 2].clamp(min=1e-7)
        tgt_h = target[:, 3].clamp(min=1e-7)

        v = (4 / math.pi ** 2) * (torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - center_dist / diagonal_dist - alpha * v
        return (1 - ciou).mean()


class SIoULoss(IoULoss):
    """SCYLLA-IoU Loss - considers angle cost."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        iou = self.bbox_iou(pred, target)

        # Center distance
        cx_diff = pred[:, 0] - target[:, 0]
        cy_diff = pred[:, 1] - target[:, 1]

        # Enclosing box
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        tgt_x1 = target[:, 0] - target[:, 2] / 2
        tgt_y1 = target[:, 1] - target[:, 3] / 2
        tgt_x2 = target[:, 0] + target[:, 2] / 2
        tgt_y2 = target[:, 1] + target[:, 3] / 2

        enclose_w = torch.max(pred_x2, tgt_x2) - torch.min(pred_x1, tgt_x1)
        enclose_h = torch.max(pred_y2, tgt_y2) - torch.min(pred_y1, tgt_y1)

        # Angle cost
        gamma = 2 - torch.atan2(cx_diff.abs(), cy_diff.abs()) * (2 / math.pi)

        # Distance cost
        dist = (cx_diff ** 2 + cy_diff ** 2) / (enclose_w ** 2 + enclose_h ** 2 + 1e-7)

        # Shape cost
        w_diff = (pred[:, 2] - target[:, 2]).abs()
        h_diff = (pred[:, 3] - target[:, 3]).abs()
        omega = (w_diff / torch.max(pred[:, 2], target[:, 2]) + h_diff / torch.max(pred[:, 3], target[:, 3])) ** 2

        siou = iou - (dist + omega) * 0.5
        return (1 - siou).mean()
