"""
Data Augmentation Pipeline for YOLO training.
Supports both albumentations and built-in YOLO augmentations.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class AugmentationPipeline:
    """Configurable augmentation pipeline for object detection."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        self._transforms = self._build_pipeline()

    def _build_pipeline(self) -> list:
        """Build augmentation pipeline from config."""
        try:
            import albumentations as A
        except ImportError:
            return self._build_builtin_pipeline()

        transforms = []

        # Spatial transforms
        if self.cfg.get("horizontal_flip", 0) > 0:
            transforms.append(A.HorizontalFlip(p=self.cfg["horizontal_flip"]))
        if self.cfg.get("vertical_flip", 0) > 0:
            transforms.append(A.VerticalFlip(p=self.cfg["vertical_flip"]))

        if self.cfg.get("rotate", 0) > 0:
            transforms.append(A.Rotate(
                limit=self.cfg.get("rotate_limit", 15),
                p=self.cfg["rotate"],
            ))

        if self.cfg.get("scale", 0) > 0:
            transforms.append(A.RandomScale(
                scale_limit=self.cfg.get("scale_limit", 0.2),
                p=self.cfg["scale"],
            ))

        # Color transforms
        if self.cfg.get("brightness", 0) > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=self.cfg.get("brightness_limit", 0.2),
                contrast_limit=self.cfg.get("contrast_limit", 0.2),
                p=self.cfg["brightness"],
            ))

        if self.cfg.get("hue_saturation", 0) > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=self.cfg.get("hue_limit", 20),
                sat_shift_limit=self.cfg.get("sat_limit", 30),
                val_shift_limit=self.cfg.get("val_limit", 20),
                p=self.cfg["hue_saturation"],
            ))

        if self.cfg.get("blur", 0) > 0:
            transforms.append(A.Blur(blur_limit=self.cfg.get("blur_limit", 3), p=self.cfg["blur"]))

        if self.cfg.get("noise", 0) > 0:
            transforms.append(A.GaussNoise(p=self.cfg["noise"]))

        if self.cfg.get("clahe", 0) > 0:
            transforms.append(A.CLAHE(p=self.cfg["clahe"]))

        # Bbox-compatible format
        bbox_params = A.BboxParams(
            format="yolo",
            min_area=0,
            min_visibility=0.3,
            label_fields=["class_labels"],
        )

        if transforms:
            return [A.Compose(transforms, bbox_params=bbox_params)]
        return []

    def _build_builtin_pipeline(self) -> list:
        """Built-in augmentation when albumentations is not available."""
        return []

    def __call__(self, image: np.ndarray, bboxes: List[List[float]], class_labels: List[int]) -> Tuple:
        """
        Apply augmentations.

        Args:
            image: BGR image (H, W, 3)
            bboxes: List of [cx, cy, w, h] normalized
            class_labels: List of class IDs

        Returns:
            (augmented_image, augmented_bboxes, augmented_labels)
        """
        if not self._transforms:
            return image, bboxes, class_labels

        try:
            result = self._transforms[0](
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            return result["image"], result["bboxes"], result["class_labels"]
        except Exception:
            return image, bboxes, class_labels

    # ─── Built-in YOLO-style augmentations ───────────────────────────

    @staticmethod
    def mosaic(images: List[np.ndarray], labels_list: List[List], img_size: int = 640) -> Tuple:
        """
        Mosaic augmentation: combine 4 images into one.

        Args:
            images: List of 4 images
            labels_list: List of 4 label lists (each: [[cx, cy, w, h, cls], ...])
            img_size: Output image size

        Returns:
            (mosaic_image, mosaic_labels)
        """
        assert len(images) >= 4, "Mosaic requires at least 4 images"

        sel = random.sample(range(len(images)), 4)
        mosaic_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        cx, cy = random.randint(img_size // 4, img_size * 3 // 4), random.randint(img_size // 4, img_size * 3 // 4)

        mosaic_labels = []
        positions = [(0, 0), (img_size, 0), (0, img_size), (img_size, img_size)]

        for i, (px, py) in enumerate(positions):
            idx = sel[i]
            img = images[idx]
            h, w = img.shape[:2]

            # Resize keeping aspect ratio
            scale = min(img_size / 2 / h, img_size / 2 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img, (new_w, new_h))

            # Calculate paste coordinates
            x1 = max(cx - new_w, 0) if px == 0 else min(cx, img_size - new_w)
            y1 = max(cy - new_h, 0) if py == 0 else min(cy, img_size - new_h)
            x2 = x1 + new_w
            y2 = y1 + new_h

            # Paste
            mosaic_img[y1:y2, x1:x2] = img_resized

            # Adjust labels
            for label in labels_list[idx]:
                cls_id = label[4] if len(label) > 4 else label[0]
                cx_norm, cy_norm, w_norm, h_norm = label[0], label[1], label[2], label[3]

                # Denormalize → pixel coords in mosaic
                px_cx = cx_norm * new_w + x1
                px_cy = cy_norm * new_h + y1
                px_w = w_norm * new_w
                px_h = h_norm * new_h

                # Renormalize to mosaic
                new_cx = px_cx / img_size
                new_cy = px_cy / img_size
                new_w_n = px_w / img_size
                new_h_n = px_h / img_size

                if 0 <= new_cx <= 1 and 0 <= new_cy <= 1:
                    mosaic_labels.append([new_cx, new_cy, new_w_n, new_h_n, cls_id])

        return mosaic_img, mosaic_labels

    @staticmethod
    def mixup(img1: np.ndarray, labels1: List, img2: np.ndarray, labels2: List,
              alpha: float = 1.5) -> Tuple:
        """MixUp augmentation: blend two images."""
        beta = np.random.beta(alpha, alpha)
        beta = max(0.3, min(0.7, beta))  # Clip to reasonable range

        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        mixed = (img1 * beta + img2 * (1 - beta)).astype(np.uint8)
        mixed_labels = labels1 + labels2

        return mixed, mixed_labels

    @staticmethod
    def copy_paste(image: np.ndarray, bboxes: List, objects_pool: List[Tuple]) -> Tuple:
        """Copy-paste augmentation: paste random objects onto image."""
        h, w = image.shape[:2]
        new_bboxes = list(bboxes)

        for obj_img, obj_bbox in random.sample(objects_pool, min(3, len(objects_pool))):
            # Random position
            oh, ow = obj_img.shape[:2]
            x = random.randint(0, max(0, w - ow))
            y = random.randint(0, max(0, h - oh))

            # Paste with mask
            mask = obj_img > 0
            image[y:y+oh, x:x+ow] = np.where(mask, obj_img, image[y:y+oh, x:x+ow])

            # Add bbox
            cx = (x + ow / 2) / w
            cy = (y + oh / 2) / h
            nw = ow / w
            nh = oh / h
            new_bboxes.append([cx, cy, nw, nh, obj_bbox[4] if len(obj_bbox) > 4 else 0])

        return image, new_bboxes
