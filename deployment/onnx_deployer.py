"""
ONNX Export and Inference Deployer.
Handles model export, optimization, and inference via ONNX Runtime.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


class ONNXDeployer:
    """
    ONNX model deployment toolkit.

    Features:
    - Export PyTorch models to ONNX
    - Simplify ONNX graphs
    - Run inference via ONNX Runtime
    - Post-processing (NMS)
    - Benchmark inference speed

    Usage:
        deployer = ONNXDeployer("model.onnx")
        results = deployer.inference("image.jpg")
    """

    def __init__(self, onnx_path: Optional[Union[str, Path]] = None):
        self.onnx_path = str(onnx_path) if onnx_path else None
        self.session = None

    def export(
        self,
        model: torch.nn.Module,
        save_path: Union[str, Path],
        img_size: int = 640,
        opset: int = 17,
        simplify: bool = True,
        dynamic: bool = False,
        half: bool = False,
    ) -> str:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: PyTorch model
            save_path: Output ONNX file path
            img_size: Input image size
            opset: ONNX opset version
            simplify: Simplify the ONNX graph
            dynamic: Use dynamic axes
            half: Export as FP16

        Returns:
            Path to exported ONNX model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        if half:
            dummy = dummy.half()
            model = model.half()

        # Export
        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 1: "detections"},
            }

        torch.onnx.export(
            model,
            dummy,
            str(save_path),
            opset_version=opset,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        # Simplify
        if simplify:
            self._simplify(str(save_path))

        self.onnx_path = str(save_path)
        print(f"✅ ONNX model exported to: {save_path}")
        return str(save_path)

    def _simplify(self, onnx_path: str) -> None:
        """Simplify ONNX model graph."""
        try:
            import onnx
            from onnxsim import simplify

            model = onnx.load(onnx_path)
            model_simp, check = simplify(model)
            if check:
                onnx.save(model_simp, onnx_path)
                print("  ✓ ONNX graph simplified")
            else:
                print("  ⚠ ONNX simplification check failed")
        except ImportError:
            print("  ⚠ onnxsim not installed, skipping simplification")

    def load(self, onnx_path: Optional[str] = None) -> None:
        """Load ONNX model into ONNX Runtime session."""
        import onnxruntime as ort

        path = onnx_path or self.onnx_path
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=providers)
        self.onnx_path = path
        print(f"✅ ONNX model loaded: {path}")
        print(f"   Providers: {self.session.get_providers()}")

    def inference(
        self,
        source: Union[str, Path, np.ndarray],
        img_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on an image.

        Args:
            source: Image path or numpy array
            img_size: Input size for the model
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold

        Returns:
            List of detection results
        """
        if self.session is None:
            self.load()

        # Preprocess
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
        else:
            img = source

        original_h, original_w = img.shape[:2]
        img_resized = cv2.resize(img, (img_size, img_size))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_input})

        # Post-process
        predictions = outputs[0]
        detections = self._postprocess(
            predictions, original_w, original_h,
            img_size, conf_threshold, iou_threshold,
        )

        return detections

    def _postprocess(
        self,
        predictions: np.ndarray,
        original_w: int,
        original_h: int,
        img_size: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[Dict]:
        """Post-process model output with NMS."""
        if predictions.ndim == 3:
            predictions = predictions[0]

        # Filter by confidence
        if predictions.shape[-1] > 5:
            # Format: [cx, cy, w, h, cls_scores...]
            boxes = predictions[:, :4]
            class_scores = predictions[:, 4:]
            max_scores = class_scores.max(axis=1)
            class_ids = class_scores.argmax(axis=1)

            mask = max_scores > conf_threshold
            boxes = boxes[mask]
            scores = max_scores[mask]
            class_ids = class_ids[mask]
        else:
            # Format: [cx, cy, w, h, score]
            mask = predictions[:, 4] > conf_threshold
            predictions = predictions[mask]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            class_ids = np.zeros(len(scores), dtype=int)

        if len(boxes) == 0:
            return []

        # Scale to original image size
        scale_x = original_w / img_size
        scale_y = original_h / img_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Convert to x1y1x2y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS
        keep = self._nms(boxes_xyxy, scores, iou_threshold)

        results = []
        for idx in keep:
            results.append({
                "bbox": boxes_xyxy[idx].tolist(),
                "score": float(scores[idx]),
                "class_id": int(class_ids[idx]),
            })

        return results

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def benchmark(
        self,
        img_size: int = 640,
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark ONNX inference speed."""
        if self.session is None:
            self.load()

        dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        input_name = self.session.get_inputs()[0].name

        # Warmup
        for _ in range(warmup):
            self.session.run(None, {input_name: dummy})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.session.run(None, {input_name: dummy})
            times.append(time.perf_counter() - start)

        times_ms = [t * 1000 for t in times]
        results = {
            "mean_ms": round(np.mean(times_ms), 2),
            "std_ms": round(np.std(times_ms), 2),
            "min_ms": round(np.min(times_ms), 2),
            "max_ms": round(np.max(times_ms), 2),
            "fps": round(1000 / np.mean(times_ms), 1),
        }

        print(f"📊 ONNX Benchmark ({num_runs} runs):")
        print(f"   Mean: {results['mean_ms']}ms ± {results['std_ms']}ms")
        print(f"   FPS: {results['fps']}")
        return results
