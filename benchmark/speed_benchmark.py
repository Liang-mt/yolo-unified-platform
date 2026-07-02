"""
Speed Benchmark - comprehensive inference speed testing across frameworks.
Compares PyTorch, ONNX Runtime, and TensorRT performance.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


class SpeedBenchmark:
    """
    Multi-framework inference speed benchmark.

    Supports:
    - PyTorch (native)
    - ONNX Runtime
    - TensorRT

    Usage:
        bench = SpeedBenchmark()
        results = bench.compare_all(model, img_size=640)
        bench.print_report(results)
    """

    def __init__(self, device: str = "auto"):
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else device
        if self.device == "auto":
            self.device = "cpu"

    def benchmark_pytorch(
        self,
        model: torch.nn.Module,
        img_size: int = 640,
        batch_size: int = 1,
        num_runs: int = 100,
        warmup: int = 10,
        half: bool = False,
    ) -> Dict[str, float]:
        """Benchmark PyTorch model inference."""
        model.eval()
        model.to(self.device)

        dummy = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
        if half:
            dummy = dummy.half()
            model = model.half()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                model(dummy)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        return self._compute_stats(times, "PyTorch", half)

    def benchmark_onnx(
        self,
        onnx_path: Union[str, Path],
        img_size: int = 640,
        batch_size: int = 1,
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark ONNX Runtime inference."""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name

        dummy = np.random.randn(batch_size, 3, img_size, img_size).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            session.run(None, {input_name: dummy})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            times.append(time.perf_counter() - start)

        return self._compute_stats(times, "ONNX")

    def benchmark_tensorrt(
        self,
        engine_path: Union[str, Path],
        img_size: int = 640,
        batch_size: int = 1,
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark TensorRT inference."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            return {"error": "TensorRT/pycuda not installed"}

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        dummy = np.random.randn(batch_size, 3, img_size, img_size).astype(np.float32)
        d_input = cuda.mem_alloc(dummy.nbytes)
        output = np.empty((batch_size, 84, 8400), dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        # Warmup
        for _ in range(warmup):
            cuda.memcpy_htod(d_input, dummy)
            context.execute_v2(bindings=[int(d_input), int(d_output)])

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            cuda.memcpy_htod(d_input, dummy)
            context.execute_v2(bindings=[int(d_input), int(d_output)])
            cuda.Context.synchronize()
            times.append(time.perf_counter() - start)

        return self._compute_stats(times, "TensorRT")

    def benchmark_ultralytics(
        self,
        model_path: str,
        img_size: int = 640,
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Benchmark ultralytics YOLO model."""
        from ultralytics import YOLO

        model = YOLO(model_path)
        dummy = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        # Warmup
        for _ in range(warmup):
            model.predict(dummy, verbose=False)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            model.predict(dummy, verbose=False)
            times.append(time.perf_counter() - start)

        return self._compute_stats(times, f"Ultralytics({Path(model_path).stem})")

    def compare_all(
        self,
        model: torch.nn.Module,
        img_size: int = 640,
        onnx_path: Optional[str] = None,
        engine_path: Optional[str] = None,
        num_runs: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Run comprehensive comparison across all available backends."""
        results = {}

        # PyTorch
        print("🏃 Benchmarking PyTorch...")
        results["pytorch"] = self.benchmark_pytorch(model, img_size, num_runs=num_runs)

        # ONNX
        if onnx_path and Path(onnx_path).exists():
            print("🏃 Benchmarking ONNX...")
            results["onnx"] = self.benchmark_onnx(onnx_path, img_size, num_runs=num_runs)

        # TensorRT
        if engine_path and Path(engine_path).exists():
            print("🏃 Benchmarking TensorRT...")
            results["tensorrt"] = self.benchmark_tensorrt(engine_path, img_size, num_runs=num_runs)

        return results

    @staticmethod
    def _compute_stats(times: List[float], name: str, half: bool = False) -> Dict[str, float]:
        """Compute latency statistics."""
        times_ms = [t * 1000 for t in times]
        return {
            "backend": name,
            "half": half,
            "mean_ms": round(np.mean(times_ms), 2),
            "std_ms": round(np.std(times_ms), 2),
            "min_ms": round(np.min(times_ms), 2),
            "max_ms": round(np.max(times_ms), 2),
            "median_ms": round(np.median(times_ms), 2),
            "p95_ms": round(np.percentile(times_ms, 95), 2),
            "p99_ms": round(np.percentile(times_ms, 99), 2),
            "fps": round(1000 / np.mean(times_ms), 1),
        }

    @staticmethod
    def print_report(results: Dict[str, Dict[str, float]]) -> None:
        """Print formatted benchmark report."""
        print("\n" + "=" * 70)
        print("📊 INFERENCE SPEED BENCHMARK REPORT")
        print("=" * 70)
        print(f"{'Backend':<20} {'Mean(ms)':<12} {'FPS':<10} {'P95(ms)':<12} {'Min(ms)':<12}")
        print("-" * 70)

        for name, stats in results.items():
            if "error" in stats:
                print(f"{name:<20} Error: {stats['error']}")
                continue
            print(
                f"{stats.get('backend', name):<20} "
                f"{stats['mean_ms']:<12} "
                f"{stats['fps']:<10} "
                f"{stats['p95_ms']:<12} "
                f"{stats['min_ms']:<12}"
            )

        # Speedup comparison
        if "pytorch" in results and len(results) > 1:
            base = results["pytorch"]["mean_ms"]
            print("\nSpeedup vs PyTorch:")
            for name, stats in results.items():
                if name != "pytorch" and "mean_ms" in stats:
                    speedup = base / stats["mean_ms"]
                    print(f"  {name}: {speedup:.2f}x")

        print("=" * 70 + "\n")
