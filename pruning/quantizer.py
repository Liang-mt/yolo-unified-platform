"""
Model Quantization - INT8/FP16 quantization for YOLO models.
Supports dynamic quantization, static quantization, and QAT.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.quantization as quant


class ModelQuantizer:
    """
    Model quantization toolkit.

    Supports:
    - Dynamic quantization (post-training, CPU)
    - Static quantization (post-training, with calibration data)
    - FP16 (half precision, GPU)
    - INT8 quantization

    Usage:
        quantizer = ModelQuantizer(model)
        quantized = quantizer.quantize_dynamic()
        quantizer.export_onnx("quantized.onnx")
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    def quantize_dynamic(
        self,
        dtype: torch.dtype = torch.qint8,
        layers: Optional[List[type]] = None,
    ) -> nn.Module:
        """
        Dynamic quantization (post-training, no calibration needed).

        Args:
            dtype: Target dtype (torch.qint8 or torch.quint8)
            layers: Layer types to quantize (default: Linear, LSTM)

        Returns:
            Quantized model
        """
        if layers is None:
            layers = [nn.Linear, nn.Conv2d]

        quantized = torch.quantization.quantize_dynamic(
            self.model.cpu(),
            {tuple(layers): dtype},
            dtype=dtype,
        )
        print(f"✅ Dynamic quantization complete (dtype={dtype})")
        return quantized

    def quantize_static(
        self,
        calibration_data: torch.Tensor,
        backend: str = "fbgemm",
    ) -> nn.Module:
        """
        Static quantization with calibration.

        Args:
            calibration_data: Sample input tensor for calibration
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)

        Returns:
            Quantized model
        """
        model = copy.deepcopy(self.model).cpu().eval()

        # Set quantization config
        model.qconfig = quant.get_default_qconfig(backend)

        # Prepare
        prepared = quant.prepare(model)

        # Calibrate with sample data
        print("📊 Calibrating...")
        with torch.no_grad():
            if isinstance(calibration_data, torch.Tensor):
                prepared(calibration_data)
            elif isinstance(calibration_data, (list, tuple)):
                for batch in calibration_data:
                    prepared(batch)

        # Convert
        quantized = quant.convert(prepared)
        print(f"✅ Static quantization complete (backend={backend})")
        return quantized

    def quantize_fp16(self) -> nn.Module:
        """Convert model to FP16 (half precision)."""
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, falling back to CPU FP16")
        model = self.model.half()
        print("✅ FP16 conversion complete")
        return model

    def quantize_int8_onnx(
        self,
        onnx_path: Union[str, Path],
        output_path: Union[str, Path],
        calibration_data: Optional[List] = None,
    ) -> str:
        """
        INT8 quantization via ONNX Runtime.

        Args:
            onnx_path: Input ONNX model path
            output_path: Output quantized ONNX path
            calibration_data: Calibration dataset for static quantization

        Returns:
            Path to quantized ONNX model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic as ort_quantize
            from onnxruntime.quantization import QuantType
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX INT8 quantization")

        onnx_path = str(onnx_path)
        output_path = str(output_path)

        ort_quantize(
            onnx_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"✅ INT8 quantized ONNX model saved to: {output_path}")
        return output_path

    def benchmark_speedup(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """
        Compare inference speed between original and quantized models.

        Returns:
            Dict with latency and speedup stats
        """
        import time

        device = self.device

        def measure_latency(model, runs):
            model.eval()
            model.to(device)
            dummy = torch.randn(input_shape).to(device)
            if next(model.parameters()).dtype == torch.float16:
                dummy = dummy.half()

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    model(dummy)

            # Measure
            times = []
            with torch.no_grad():
                for _ in range(runs):
                    start = time.perf_counter()
                    model(dummy)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.perf_counter() - start)

            return sum(times) / len(times) * 1000  # ms

        orig_latency = measure_latency(original_model, num_runs)
        quant_latency = measure_latency(quantized_model, num_runs)

        results = {
            "original_ms": round(orig_latency, 2),
            "quantized_ms": round(quant_latency, 2),
            "speedup": round(orig_latency / max(quant_latency, 1e-6), 2),
            "original_size_mb": self._model_size_mb(original_model),
            "quantized_size_mb": self._model_size_mb(quantized_model),
        }

        print(f"📊 Speedup: {results['speedup']}x ({results['original_ms']}ms → {results['quantized_ms']}ms)")
        print(f"   Size: {results['original_size_mb']}MB → {results['quantized_size_mb']}MB")
        return results

    @staticmethod
    def _model_size_mb(model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return round((param_size + buffer_size) / 1024 / 1024, 2)
