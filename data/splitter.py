"""
Dataset Splitter - train/val/test splitting with stratification support.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from tqdm import tqdm


class DatasetSplitter:
    """Split datasets into train/val/test sets with proper file management."""

    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ):
        """
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing label .txt files
            output_dir: Output root directory
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        test_ratio: float = 0.05,
        seed: int = 42,
        copy_images: bool = True,
        classes: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Split dataset into train/val/test.

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
            copy_images: If True, copy files; if False, create symlinks
            classes: Class names for YAML config generation

        Returns:
            Dict with split counts
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        random.seed(seed)

        # Find matched image-label pairs
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        image_files = sorted([
            f for f in self.image_dir.iterdir()
            if f.suffix.lower() in img_exts
        ])

        # Match with labels
        pairs = []
        for img_file in image_files:
            label_file = self.label_dir / (img_file.stem + ".txt")
            if label_file.exists():
                pairs.append((img_file, label_file))

        random.shuffle(pairs)

        n = len(pairs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": pairs[:n_train],
            "val": pairs[n_train:n_train + n_val],
            "test": pairs[n_train + n_val:],
        }

        # Create output structure
        for split_name, split_pairs in splits.items():
            img_out = self.output_dir / split_name / "images"
            lbl_out = self.output_dir / split_name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_file, lbl_file in tqdm(split_pairs, desc=f"Splitting {split_name}"):
                if copy_images:
                    shutil.copy2(img_file, img_out / img_file.name)
                    shutil.copy2(lbl_file, lbl_out / lbl_file.name)
                else:
                    self._create_symlink(img_file, img_out / img_file.name)
                    self._create_symlink(lbl_file, lbl_out / lbl_file.name)

        # Generate YAML config
        if classes is None:
            classes = self._infer_classes(pairs)

        self._generate_yaml(classes)

        counts = {k: len(v) for k, v in splits.items()}
        print(f"✅ Dataset split complete: {counts}")
        print(f"   Output: {self.output_dir}")
        return counts

    def _create_symlink(self, src: Path, dst: Path) -> None:
        """Create symlink, handling Windows."""
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

    def _infer_classes(self, pairs: List[Tuple]) -> List[str]:
        """Infer class names from label files."""
        class_ids = set()
        for _, lbl_file in pairs:
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_ids.add(int(parts[0]))
        return [f"class_{i}" for i in sorted(class_ids)]

    def _generate_yaml(self, classes: List[str]) -> None:
        """Generate YOLO dataset YAML config."""
        yaml_content = {
            "path": str(self.output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(classes),
            "names": classes,
        }
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        print(f"   YAML config saved to: {yaml_path}")

    @staticmethod
    def from_coco_split(
        coco_json: Union[str, Path],
        image_dir: Union[str, Path],
        output_dir: Union[str, Path],
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, int]:
        """Split COCO dataset by image IDs."""
        import json
        coco_json = Path(coco_json)
        output_dir = Path(output_dir)

        with open(coco_json) as f:
            coco = json.load(f)

        images = coco["images"]
        random.seed(seed)
        random.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        # Build annotation lookup
        ann_by_img = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        image_dir = Path(image_dir)

        for split_name, split_images in splits.items():
            split_coco = {
                "images": split_images,
                "annotations": [],
                "categories": coco["categories"],
            }
            for img in split_images:
                split_coco["annotations"].extend(ann_by_img.get(img["id"], []))

            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            with open(split_dir / "annotations.json", "w") as f:
                json.dump(split_coco, f, indent=2)

        counts = {k: len(v) for k, v in splits.items()}
        print(f"✅ COCO split: {counts}")
        return counts
