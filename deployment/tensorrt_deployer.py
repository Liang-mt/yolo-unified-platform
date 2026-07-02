"""
TensorRT Deployer - convert ONNX models to TensorRT engines and run inference.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class TensorRTDeployer:
    """
    TensorRT deployment toolkit.

    Features:
    - Convert ONNX to TensorRT engine
    - FP32/FP16/INT8 precision modes
    - Inference with TensorRT Runtime
    - Benchmark speed

    Usage:
        deployer = TensorRTDeployer()
        engine_path = deployer.build_engine("model.onnx", precision="fp16")
        deployer.load_engine(engine_path)
        results = deployer.inference(image)
    """

    def __init__(self, max_batch_size: int = 1, workspace_gb: int = 4):
        self.max_batch_size = max_batch_size
        self.workspace_gb = workspace_gb
        self.engine = None
        self.context = None

    def build_engine(
        self,
        onnx_path: Union[str, Path],
        engine_path: Optional[Union[str, Path]] = None,
        precision: str = "fp16",
        dynamic_shapes: Optional[Dict] = None,
    ) -> str:
        """
        Build TensorRT engine from ONNX model.

        Args:
            onnx_path: Path to ONNX model
            engine_path: Output engine path (default: same dir as ONNX)
            precision: 'fp32', 'fp16', or 'int8'
            dynamic_shapes: Dynamic shape profiles for variable input sizes

        Returns:
            Path to TensorRT engine file
        """
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT is not installed. Install from: "
                "https://developer.nvidia.com/tensorrt"
            )

        onnx_path = Path(onnx_path)
        if engine_path is None:
            engine_path = onnx_path.with_suffix(".engine")
        engine_path = Path(engine_path)

        logger = trt.Logger(trt.Logger.WARNING)
        builder = builder_create(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        print(f"📄 Parsing ONNX model: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ❌ Parser error: {parser.get_error(i)}")
                raise RuntimeError("ONNX parsing failed")

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_gb * (1 << 30))

        # Set precision
        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("  ✓ FP16 enabled")
            else:
                print("  ⚠ Platform does not support FP16, using FP32")
        elif precision == "int8":
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print("  ✓ INT8 enabled")
            else:
                print("  ⚠ Platform does not support INT8")

        # Dynamic shapes
        if dynamic_shapes:
            profile = builder.create_optimization_profile()
            for name, shapes in dynamic_shapes.items():
                min_shape, opt_shape, max_shape = shapes
                profile.set_shape(name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

        # Build engine
        print(f"🔧 Building TensorRT engine (precision={precision})...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Engine build failed")

        # Save engine
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

        print(f"✅ TensorRT engine saved to: {engine_path}")
        print(f"   Size: {engine_path.stat().st_size / 1024 / 1024:.1f} MB")
        return str(engine_path)

    def load_engine(self, engine_path: Union[str, Path]) -> None:
        """Load a TensorRT engine for inference."""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError("TensorRT is not installed")

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        print(f"✅ TensorRT engine loaded: {engine_path}")

    def inference(
        self,
        image: np.ndarray,
        img_size: int = 640,
        conf_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Run inference with TensorRT engine.

        Args:
            image: Input image (H, W, 3) BGR format
            img_size: Model input size
            conf_threshold: Confidence threshold

        Returns:
            List of detection results
        """
        if self.engine is None:
            raise RuntimeError("Engine not loaded. Call load_engine() first.")

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("pycuda is required for TensorRT inference")

        # Preprocess
        import cv2
        original_h, original_w = image.shape[:2]
        img_resized = cv2.resize(image, (img_size, img_size))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        img_input = np.ascontiguousarray(img_input)

        # Allocate device memory
        d_input = cuda.mem_alloc(img_input.nbytes)

        # Get output shape
        output_shape = (1, 84, 8400)  # Default YOLOv8 output shape
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        # Transfer input
        cuda.memcpy_htod(d_input, img_input)

        # Run inference
        self.context.execute_v2(bindings=[int(d_input), int(d_output)])

        # Transfer output
        cuda.memcpy_dtoh(output, d_output)

        # Post-process (simplified)
        output = output[0].T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:].max(axis=1)
        class_ids = output[:, 4:].argmax(axis=1)

        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # Scale to original size
        scale_x = original_w / img_size
        scale_y = original_h / img_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        results = []
        for i in range(len(boxes)):
            results.append({
                "bbox": boxes[i].tolist(),
                "score": float(scores[i]),
                "class_id": int(class_ids[i]),
            })

        return results

    def benchmark(
        self,
        img_size: int = 640,
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark TensorRT inference speed."""
        if self.engine is None:
            raise RuntimeError("Engine not loaded")

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("pycuda is required")

        dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        d_input = cuda.mem_alloc(dummy.nbytes)

        output_shape = (1, 84, 8400)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        # Warmup
        for _ in range(warmup):
            cuda.memcpy_htod(d_input, dummy)
            self.context.execute_v2(bindings=[int(d_input), int(d_output)])

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            cuda.memcpy_htod(d_input, dummy)
            self.context.execute_v2(bindings=[int(d_input), int(d_output)])
            cuda.Context.synchronize()
            times.append(time.perf_counter() - start)

        times_ms = [t * 1000 for t in times]
        results = {
            "mean_ms": round(np.mean(times_ms), 2),
            "std_ms": round(np.std(times_ms), 2),
            "min_ms": round(np.min(times_ms), 2),
            "max_ms": round(np.max(times_ms), 2),
            "fps": round(1000 / np.mean(times_ms), 1),
        }

        print(f"📊 TensorRT Benchmark ({num_runs} runs):")
        print(f"   Mean: {results['mean_ms']}ms ± {results['std_ms']}ms")
        print(f"   FPS: {results['fps']}")
        return results


def builder_create(logger):
    """Helper to create TensorRT builder (handles API differences)."""
    try:
        import tensorrt as trt
        return trt.Builder(logger)
    except Exception:
        raise
