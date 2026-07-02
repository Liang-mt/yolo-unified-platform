"""
Dataset Cleaner - Detect and fix common issues in YOLO datasets.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm


class DatasetCleaner:
    """
    Comprehensive dataset cleaning tool for YOLO datasets.

    Detects and optionally fixes:
    - Missing labels/images
    - Empty label files
    - Invalid bounding boxes (out of range, zero-size)
    - Corrupt images
    - Duplicate images
    - Class imbalance
    - Tiny/large boxes
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        classes: Optional[List[str]] = None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.classes = classes
        self.img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.issues: List[Dict[str, Any]] = []

    def scan(self, fix: bool = False) -> Dict[str, Any]:
        """
        Run full dataset scan.

        Args:
            fix: If True, attempt to fix issues automatically

        Returns:
            Report dict with all found issues
        """
        print("🔍 Scanning dataset...")
        self.issues = []

        image_files = self._get_image_files()
        label_files = self._get_label_files()

        report = {
            "total_images": len(image_files),
            "total_labels": len(label_files),
            "issues": [],
            "stats": {},
        }

        # Check 1: Missing labels
        self._check_missing_labels(image_files, label_files, fix)
        # Check 2: Missing images
        self._check_missing_images(image_files, label_files, fix)
        # Check 3: Corrupt images
        self._check_corrupt_images(image_files, fix)
        # Check 4: Empty labels
        self._check_empty_labels(label_files, fix)
        # Check 5: Invalid bboxes
        self._check_invalid_bboxes(label_files, fix)
        # Check 6: Tiny/large boxes
        self._check_box_sizes(label_files)
        # Check 7: Class distribution
        class_dist = self._analyze_class_distribution(label_files)
        report["stats"]["class_distribution"] = class_dist
        # Check 8: Duplicate images
        dupes = self._check_duplicates(image_files, fix)
        report["stats"]["duplicates"] = len(dupes)

        report["issues"] = self.issues
        report["total_issues"] = len(self.issues)

        self._print_report(report)
        return report

    def _get_image_files(self) -> Dict[str, Path]:
        """Get all image files indexed by stem."""
        files = {}
        for f in self.image_dir.iterdir():
            if f.suffix.lower() in self.img_exts:
                files[f.stem] = f
        return files

    def _get_label_files(self) -> Dict[str, Path]:
        """Get all label files indexed by stem."""
        files = {}
        for f in self.label_dir.iterdir():
            if f.suffix.lower() == ".txt":
                files[f.stem] = f
        return files

    def _check_missing_labels(self, images: Dict, labels: Dict, fix: bool) -> None:
        missing = set(images.keys()) - set(labels.keys())
        for stem in missing:
            self.issues.append({
                "type": "missing_label",
                "severity": "error",
                "file": str(images[stem]),
                "message": f"Image '{stem}' has no label file",
            })
            if fix:
                # Create empty label file
                (self.label_dir / f"{stem}.txt").touch()

    def _check_missing_images(self, images: Dict, labels: Dict, fix: bool) -> None:
        missing = set(labels.keys()) - set(images.keys())
        for stem in missing:
            self.issues.append({
                "type": "missing_image",
                "severity": "error",
                "file": str(labels[stem]),
                "message": f"Label '{stem}' has no corresponding image",
            })
            if fix:
                labels[stem].unlink()

    def _check_corrupt_images(self, images: Dict, fix: bool) -> None:
        for stem, img_path in tqdm(images.items(), desc="Checking images"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("cv2.imread returned None")
                h, w = img.shape[:2]
                if h < 10 or w < 10:
                    raise ValueError(f"Image too small: {w}x{h}")
            except Exception as e:
                self.issues.append({
                    "type": "corrupt_image",
                    "severity": "error",
                    "file": str(img_path),
                    "message": f"Corrupt image: {e}",
                })
                if fix:
                    img_path.unlink()
                    label_path = self.label_dir / f"{stem}.txt"
                    if label_path.exists():
                        label_path.unlink()

    def _check_empty_labels(self, labels: Dict, fix: bool) -> None:
        for stem, lbl_path in labels.items():
            content = lbl_path.read_text().strip()
            if not content:
                self.issues.append({
                    "type": "empty_label",
                    "severity": "warning",
                    "file": str(lbl_path),
                    "message": f"Empty label file: '{stem}'",
                })

    def _check_invalid_bboxes(self, labels: Dict, fix: bool) -> None:
        for stem, lbl_path in labels.items():
            lines = lbl_path.read_text().strip().split("\n")
            new_lines = []
            modified = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    self.issues.append({
                        "type": "invalid_bbox",
                        "severity": "error",
                        "file": str(lbl_path),
                        "message": f"Malformed line: '{line}'",
                    })
                    modified = True
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                except ValueError:
                    self.issues.append({
                        "type": "invalid_bbox",
                        "severity": "error",
                        "file": str(lbl_path),
                        "message": f"Non-numeric values: '{line}'",
                    })
                    modified = True
                    continue

                # Check ranges
                valid = True
                if not (0 <= cx <= 1 and 0 <= cy <= 1):
                    self.issues.append({
                        "type": "bbox_out_of_range",
                        "severity": "error",
                        "file": str(lbl_path),
                        "message": f"Center out of range: cx={cx}, cy={cy}",
                    })
                    valid = False

                if not (0 < w <= 1 and 0 < h <= 1):
                    self.issues.append({
                        "type": "bbox_out_of_range",
                        "severity": "error",
                        "file": str(lbl_path),
                        "message": f"Size out of range: w={w}, h={h}",
                    })
                    valid = False

                if valid:
                    new_lines.append(line)
                else:
                    modified = True

            if fix and modified:
                with open(lbl_path, "w") as f:
                    f.write("\n".join(new_lines) + "\n" if new_lines else "")

    def _check_box_sizes(self, labels: Dict, min_area: float = 0.0001, max_area: float = 0.9) -> None:
        tiny_count = 0
        large_count = 0
        for lbl_path in labels.values():
            for line in lbl_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 5:
                    w, h = float(parts[3]), float(parts[4])
                    area = w * h
                    if area < min_area:
                        tiny_count += 1
                    elif area > max_area:
                        large_count += 1

        if tiny_count > 0:
            self.issues.append({
                "type": "tiny_boxes",
                "severity": "info",
                "file": "dataset",
                "message": f"{tiny_count} tiny boxes (area < {min_area})",
            })
        if large_count > 0:
            self.issues.append({
                "type": "large_boxes",
                "severity": "info",
                "file": "dataset",
                "message": f"{large_count} very large boxes (area > {max_area})",
            })

    def _analyze_class_distribution(self, labels: Dict) -> Dict[int, int]:
        counter = Counter()
        for lbl_path in labels.values():
            for line in lbl_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1

        total = sum(counter.values())
        if total > 0:
            min_cls = min(counter, key=counter.get)
            max_cls = max(counter, key=counter.get)
            ratio = counter[max_cls] / max(counter[min_cls], 1)
            if ratio > 100:
                self.issues.append({
                    "type": "class_imbalance",
                    "severity": "warning",
                    "file": "dataset",
                    "message": f"Severe class imbalance: max/min ratio = {ratio:.1f}",
                })

        return dict(counter)

    def _check_duplicates(self, images: Dict, fix: bool) -> List[str]:
        """Check for duplicate images using file hashes."""
        import hashlib
        seen_hashes: Dict[str, str] = {}
        duplicates = []

        for stem, img_path in tqdm(images.items(), desc="Checking duplicates"):
            try:
                with open(img_path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                if h in seen_hashes:
                    duplicates.append(stem)
                    if fix:
                        img_path.unlink()
                        lbl = self.label_dir / f"{stem}.txt"
                        if lbl.exists():
                            lbl.unlink()
                else:
                    seen_hashes[h] = stem
            except Exception:
                continue

        return duplicates

    def _print_report(self, report: Dict) -> None:
        """Print formatted report."""
        print("\n" + "=" * 60)
        print("📊 DATASET CLEANING REPORT")
        print("=" * 60)
        print(f"  Total images:  {report['total_images']}")
        print(f"  Total labels:  {report['total_labels']}")
        print(f"  Total issues:  {report['total_issues']}")

        if report["stats"].get("class_distribution"):
            print(f"\n  Class distribution:")
            dist = report["stats"]["class_distribution"]
            for cls_id, count in sorted(dist.items()):
                name = self.classes[cls_id] if self.classes and cls_id < len(self.classes) else f"class_{cls_id}"
                print(f"    {name}: {count}")

        if report["issues"]:
            print(f"\n  Issues found:")
            by_type = {}
            for issue in report["issues"]:
                by_type.setdefault(issue["type"], []).append(issue)
            for issue_type, items in by_type.items():
                print(f"    {issue_type}: {len(items)}")

        print("=" * 60 + "\n")

    def generate_report_json(self, output_path: Union[str, Path]) -> None:
        """Save scan report as JSON."""
        report = {"issues": self.issues}
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to: {output_path}")
