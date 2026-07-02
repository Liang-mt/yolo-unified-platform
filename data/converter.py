"""
Annotation Format Converter.
Converts between VOC (XML), COCO (JSON), YOLO (TXT), LabelMe (JSON) formats.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


class AnnotationConverter:
    """Universal annotation format converter."""

    SUPPORTED_FORMATS = ["voc", "coco", "yolo", "labelme"]

    # ─── VOC → YOLO ──────────────────────────────────────────────────

    @staticmethod
    def voc_to_yolo(
        voc_dir: Union[str, Path],
        output_dir: Union[str, Path],
        classes: List[str],
    ) -> Path:
        """
        Convert Pascal VOC XML annotations to YOLO TXT format.

        Args:
            voc_dir: Directory containing XML annotation files
            output_dir: Output directory for YOLO labels
            classes: List of class names (order defines class IDs)

        Returns:
            Path to output labels directory
        """
        voc_dir = Path(voc_dir)
        output_dir = Path(output_dir)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        class_to_id = {cls: i for i, cls in enumerate(classes)}
        xml_files = list(voc_dir.glob("*.xml"))

        for xml_file in tqdm(xml_files, desc="VOC → YOLO"):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)

            yolo_lines = []
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in class_to_id:
                    continue
                cls_id = class_to_id[cls_name]

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                # Convert to YOLO format (center_x, center_y, w, h) normalized
                cx = ((xmin + xmax) / 2) / img_w
                cy = ((ymin + ymax) / 2) / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            txt_name = xml_file.stem + ".txt"
            with open(labels_dir / txt_name, "w") as f:
                f.write("\n".join(yolo_lines))

        print(f"✅ Converted {len(xml_files)} VOC files → {labels_dir}")
        return labels_dir

    # ─── COCO → YOLO ─────────────────────────────────────────────────

    @staticmethod
    def coco_to_yolo(
        coco_json: Union[str, Path],
        output_dir: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Convert COCO JSON annotations to YOLO TXT format.

        Args:
            coco_json: Path to COCO annotation JSON file
            output_dir: Output directory
            image_dir: Directory containing images (if different from JSON location)

        Returns:
            Path to output labels directory
        """
        coco_json = Path(coco_json)
        output_dir = Path(output_dir)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        with open(coco_json, "r") as f:
            coco = json.load(f)

        # Build image id → info mapping
        images = {img["id"]: img for img in coco["images"]}

        # Build category id → contiguous id mapping
        categories = {cat["id"]: i for i, cat in enumerate(coco["categories"])}

        # Group annotations by image
        ann_by_image: Dict[int, list] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in ann_by_image:
                ann_by_image[img_id] = []
            ann_by_image[img_id].append(ann)

        for img_id, img_info in tqdm(images.items(), desc="COCO → YOLO"):
            img_w = img_info["width"]
            img_h = img_info["height"]
            filename = Path(img_info["file_name"]).stem

            yolo_lines = []
            for ann in ann_by_image.get(img_id, []):
                cls_id = categories[ann["category_id"]]
                x, y, w, h = ann["bbox"]  # COCO: top-left x,y,w,h

                # Normalize to YOLO format
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h

                # Clip to [0, 1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            with open(labels_dir / f"{filename}.txt", "w") as f:
                f.write("\n".join(yolo_lines))

        # Copy images if needed
        if image_dir:
            images_out = output_dir / "images"
            images_out.mkdir(exist_ok=True)
            image_dir = Path(image_dir)
            for img_info in tqdm(images.values(), desc="Copying images"):
                src = image_dir / img_info["file_name"]
                if src.exists():
                    shutil.copy2(src, images_out / Path(img_info["file_name"]).name)

        print(f"✅ Converted COCO → YOLO: {len(images)} images, {len(coco['annotations'])} annotations")
        return labels_dir

    # ─── LabelMe → YOLO ──────────────────────────────────────────────

    @staticmethod
    def labelme_to_yolo(
        labelme_dir: Union[str, Path],
        output_dir: Union[str, Path],
        classes: List[str],
    ) -> Path:
        """
        Convert LabelMe JSON annotations to YOLO TXT format.

        Args:
            labelme_dir: Directory containing LabelMe JSON files
            output_dir: Output directory
            classes: List of class names

        Returns:
            Path to output labels directory
        """
        labelme_dir = Path(labelme_dir)
        output_dir = Path(output_dir)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        class_to_id = {cls: i for i, cls in enumerate(classes)}
        json_files = list(labelme_dir.glob("*.json"))

        for json_file in tqdm(json_files, desc="LabelMe → YOLO"):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_w = data.get("imageWidth", 0)
            img_h = data.get("imageHeight", 0)
            if img_w == 0 or img_h == 0:
                continue

            yolo_lines = []
            for shape in data.get("shapes", []):
                label = shape["label"]
                if label not in class_to_id:
                    continue
                cls_id = class_to_id[label]

                points = shape["points"]
                if shape["shape_type"] == "rectangle":
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                elif shape["shape_type"] == "polygon":
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                else:
                    continue

                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            txt_name = json_file.stem + ".txt"
            with open(labels_dir / txt_name, "w") as f:
                f.write("\n".join(yolo_lines))

        print(f"✅ Converted {len(json_files)} LabelMe files → {labels_dir}")
        return labels_dir

    # ─── YOLO → COCO ─────────────────────────────────────────────────

    @staticmethod
    def yolo_to_coco(
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        classes: List[str],
        output_json: Union[str, Path],
    ) -> Path:
        """
        Convert YOLO TXT annotations to COCO JSON format.

        Args:
            image_dir: Directory containing images
            label_dir: Directory containing YOLO label .txt files
            classes: List of class names
            output_json: Output JSON file path

        Returns:
            Path to output JSON
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        output_json = Path(output_json)

        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": cls} for i, cls in enumerate(classes)],
        }

        ann_id = 0
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in img_extensions]

        for img_id, img_file in enumerate(tqdm(image_files, desc="YOLO → COCO")):
            label_file = label_dir / (img_file.stem + ".txt")

            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            coco["images"].append({
                "id": img_id,
                "file_name": img_file.name,
                "width": img_w,
                "height": img_h,
            })

            if label_file.exists():
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])

                        x = (cx - w / 2) * img_w
                        y = (cy - h / 2) * img_h
                        bw = w * img_w
                        bh = h * img_h

                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                            "area": round(bw * bh, 2),
                            "iscrowd": 0,
                        })
                        ann_id += 1

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"✅ Converted YOLO → COCO: {len(coco['images'])} images, {ann_id} annotations → {output_json}")
        return output_json

    # ─── VOC → COCO ──────────────────────────────────────────────────

    @staticmethod
    def voc_to_coco(
        voc_dir: Union[str, Path],
        output_json: Union[str, Path],
        classes: Optional[List[str]] = None,
    ) -> Path:
        """Convert VOC XML annotations to COCO JSON format."""
        voc_dir = Path(voc_dir)
        output_json = Path(output_json)

        coco = {"images": [], "annotations": [], "categories": []}
        class_set = set()
        xml_files = list(voc_dir.glob("*.xml"))

        # First pass: collect classes if not provided
        if classes is None:
            for xml_file in xml_files:
                tree = ET.parse(xml_file)
                for obj in tree.getroot().findall("object"):
                    class_set.add(obj.find("name").text)
            classes = sorted(class_set)

        coco["categories"] = [{"id": i, "name": c} for i, c in enumerate(classes)]
        class_to_id = {c: i for i, c in enumerate(classes)}

        ann_id = 0
        for img_id, xml_file in enumerate(tqdm(xml_files, desc="VOC → COCO")):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find("filename").text
            size = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)

            coco["images"].append({
                "id": img_id, "file_name": filename,
                "width": img_w, "height": img_h,
            })

            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in class_to_id:
                    continue
                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)

                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": class_to_id[cls_name],
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0,
                })
                ann_id += 1

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"✅ Converted VOC → COCO: {len(coco['images'])} images, {ann_id} annotations")
        return output_json

    # ─── Auto-detect and convert ─────────────────────────────────────

    @classmethod
    def auto_convert(
        cls,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        source_format: str,
        target_format: str,
        classes: List[str],
        **kwargs,
    ) -> Path:
        """
        Auto-detect source format and convert to target format.

        Args:
            input_path: Input file/directory path
            output_dir: Output directory
            source_format: 'voc', 'coco', 'yolo', 'labelme'
            target_format: 'voc', 'coco', 'yolo', 'labelme'
            classes: List of class names
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)

        key = f"{source_format}_to_{target_format}"
        method_map = {
            "voc_to_yolo": lambda: cls.voc_to_yolo(input_path, output_dir, classes),
            "coco_to_yolo": lambda: cls.coco_to_yolo(input_path, output_dir, **kwargs),
            "labelme_to_yolo": lambda: cls.labelme_to_yolo(input_path, output_dir, classes),
            "yolo_to_coco": lambda: cls.yolo_to_coco(
                kwargs.get("image_dir", input_path / "images"),
                kwargs.get("label_dir", input_path / "labels"),
                classes, output_dir / "annotations.json"
            ),
            "voc_to_coco": lambda: cls.voc_to_coco(input_path, output_dir / "annotations.json", classes),
        }

        if key not in method_map:
            raise ValueError(f"Unsupported conversion: {source_format} → {target_format}. Supported: {list(method_map.keys())}")

        return method_map[key]()
