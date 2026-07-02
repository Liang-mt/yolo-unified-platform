"""Background thread workers for training, inference, export, etc."""

import re
import sys
import logging
import traceback
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal


class InferenceWorker(QThread):
    """Run inference in background thread."""
    result_ready = Signal(np.ndarray, dict)
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.image = None
        self.conf = 0.25
        self.iou = 0.45
        self.device = None

    def run(self):
        try:
            import cv2
            img_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            results = self.model.predict(
                source=img_bgr, conf=self.conf, iou=self.iou,
                device=self.device, verbose=False,
            )
            ann = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            dets = [{"class": results[0].names[int(b.cls)], "confidence": round(float(b.conf), 3)} for b in results[0].boxes]
            self.result_ready.emit(ann, {"count": len(dets), "detections": dets})
        except Exception as e:
            self.error_occurred.emit(str(e))


class TrainWorker(QThread):
    """Run training in background, emit parsed metrics for live charts."""
    epoch_done = Signal(dict)
    log_updated = Signal(str)
    train_finished = Signal(str)
    train_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.data_path = ""
        self.epochs = 100
        self.batch = 16
        self.imgsz = 640
        self.lr = 0.01
        self.device = "0"
        self.cfg = ""
        self.hyp = ""
        self.optimizer = "auto"
        self.resume = False
        self.cache = False
        self.stop_flag = False

    def run(self):
        import io

        old_stdout, old_stderr = sys.stdout, sys.stderr
        self._lines = []
        self.stop_flag = False
        # Reset all epoch tracking
        self._cur_epoch = None
        self._cur_total = 0
        self._cur_gpu = ""
        self._cur_box = 0
        self._cur_cls = 0
        self._cur_dfl = 0
        self._cur_precision = 0
        self._cur_recall = 0
        self._cur_map50 = 0
        self._cur_map5095 = 0
        self._last_val_epoch = -1  # track last validation emission to prevent duplicates

        class EmitCapturer(io.TextIOBase):
            """Capture stdout/stderr, parse metrics from each line."""
            def __init__(self, worker, orig):
                self.w = worker
                self.orig = orig
                self._buf = ""

            @staticmethod
            def _strip(t):
                # Remove ANSI escape codes (e.g. \x1b[32m, \x1b[0m)
                t = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', t)
                # Remove orphan CSI sequences (missing ESC prefix)
                t = re.sub(r'\[\d+[A-Za-z]', '', t)
                return t

            def write(self, text):
                if not text:
                    return 0
                self.orig.write(text)
                for ch in text:
                    if ch == "\n":
                        line = self._strip(self._buf).rstrip()
                        if line.strip():
                            self.w._lines.append(line)
                            self.w.log_updated.emit("\n".join(self.w._lines[-200:]))
                            self.w._parse_metrics(line)
                        self._buf = ""
                    elif ch == "\r":
                        clean = self._strip(self._buf).rstrip()
                        if clean.strip():
                            if self.w._lines:
                                self.w._lines[-1] = clean
                            else:
                                self.w._lines.append(clean)
                            self.w.log_updated.emit("\n".join(self.w._lines[-200:]))
                            self.w._parse_metrics(clean)
                        self._buf = ""
                    else:
                        self._buf += ch
                return len(text)

            def flush(self):
                self.orig.flush()

        capturer = EmitCapturer(self, old_stdout)
        sys.stdout = capturer
        sys.stderr = capturer

        # Update ultralytics LOGGER's StreamHandler to point to our new capturer.
        # The handler was created with the original sys.stdout; we must re-point it
        # so that LOGGER output flows through our EmitCapturer for metric parsing.
        ultralytics_logger = logging.getLogger("ultralytics")
        for h in ultralytics_logger.handlers:
            if isinstance(h, logging.StreamHandler) and hasattr(h, 'stream'):
                h.stream = capturer

        try:
            project_root = str(Path(__file__).parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from ultralytics import YOLO

            self._lines.append(f"[Model] {self.model_path}")
            self._lines.append(f"[Data]  {self.data_path}")
            self.log_updated.emit("\n".join(self._lines))

            # Free GPU memory from previous training
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = YOLO(self.model_path)

            def on_train_epoch_start(trainer):
                if self.stop_flag:
                    trainer.stop = True
                    return
                # Extract LR from trainer
                try:
                    lr = trainer.optimizer.param_groups[0]["lr"]
                    self.epoch_done.emit({"lr": lr})
                except Exception:
                    pass

            def on_train_batch_end(trainer):
                """Check stop flag after every batch for responsive force-stop."""
                if self.stop_flag:
                    trainer.stop = True

            model.add_callback("on_train_epoch_start", on_train_epoch_start)
            model.add_callback("on_train_batch_end", on_train_batch_end)

            train_kwargs = dict(
                data=self.data_path, epochs=self.epochs,
                batch=self.batch, imgsz=self.imgsz,
                lr0=self.lr, device=self.device,
                optimizer=self.optimizer,
                resume=self.resume,
                cache=self.cache,
            )
            if self.cfg:
                train_kwargs["cfg"] = self.cfg
            if self.hyp:
                train_kwargs["hyp"] = self.hyp
            result = model.train(**train_kwargs)

            sys.stdout, sys.stderr = old_stdout, old_stderr
            # Reset epoch tracking so next training run starts clean
            self._cur_epoch = None
            self._last_val_epoch = -1

            if self.stop_flag:
                # Release model and free GPU memory before exiting
                del model, result
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.train_finished.emit("\nTraining stopped.")
                return

            try:
                r = result.results_dict
                summary = (
                    f"\n{'='*50}\n"
                    f"  Training Complete\n"
                    f"  mAP50:    {r.get('metrics/mAP50(B)', 0):.4f}\n"
                    f"  mAP50-95: {r.get('metrics/mAP50-95(B)', 0):.4f}\n"
                    f"  Precision: {r.get('metrics/precision(B)', 0):.4f}\n"
                    f"  Recall:    {r.get('metrics/recall(B)', 0):.4f}\n"
                    f"{'='*50}"
                )
            except Exception:
                summary = "\nTraining complete."

            # Release model and free GPU memory
            del model, result
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.train_finished.emit(summary)

        except Exception:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self.train_error.emit(traceback.format_exc())

    def _emit_val(self):
        """Emit validation metrics only (no loss). Deduplicate: once per epoch."""
        if self._last_val_epoch == self._cur_epoch:
            return
        self._last_val_epoch = self._cur_epoch
        # Emit ONLY validation metrics — loss was already emitted from the training line
        self.epoch_done.emit({
            "epoch": self._cur_epoch, "total": self._cur_total,
            "precision": self._cur_precision,
            "recall": self._cur_recall,
            "map50": self._cur_map50,
            "map5095": self._cur_map5095,
        })

    def force_stop(self):
        """Force stop training immediately."""
        self.stop_flag = True

    def _parse_metrics(self, line):
        """Parse metrics from captured stdout/stderr line."""
        try:
            # 1) Strip progress-bar block chars, percentages, speed suffixes
            clean = re.sub(r'[━─╸━]+', ' ', line)
            clean = re.sub(r'\d+%\s*', '', clean)
            clean = re.sub(r'\d+/\d+\s*(it/s|s)\s*[\d.:]*', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()

            # 2) Training epoch line — emit loss on change (TQDM overwrites via \r)
            m = re.search(
                r'(\d+)\s*/\s*(\d+)\s+'
                r'([\d.]+)\s*G\s+'
                r'([\d.]+)\s+'
                r'([\d.]+)\s+'
                r'([\d.]+)\s+'
                r'(\d+)\s+'
                r'(\d+)',
                clean,
            )
            if m:
                epoch = int(m.group(1))
                box   = float(m.group(4))
                cls   = float(m.group(5))
                dfl   = float(m.group(6))
                if epoch != self._cur_epoch or box != self._cur_box or cls != self._cur_cls or dfl != self._cur_dfl:
                    self._cur_epoch = epoch
                    self._cur_total = int(m.group(2))
                    self._cur_gpu   = m.group(3)
                    self._cur_box   = box
                    self._cur_cls   = cls
                    self._cur_dfl   = dfl
                    self.epoch_done.emit({
                        "epoch": epoch, "total": self._cur_total,
                        "gpu_mem": self._cur_gpu,
                        "box_loss": box, "cls_loss": cls, "dfl_loss": dfl,
                    })
                else:
                    self._cur_epoch = epoch
                    self._cur_gpu   = m.group(3)
                return

            # 3) Validation summary: "all 128 929 0.657 0.598 0.67 0.5"
            #    IMPORTANT: no print() here — stdout is captured by EmitCapturer,
            #    writing to it from _parse_metrics would cause infinite recursion.
            if self._cur_epoch is not None and 'all' in clean:
                m2 = re.search(
                    r'\ball\s+(\d+)\s+(\d+)\s+'
                    r'([\d.]+)\s+'
                    r'([\d.]+)\s+'
                    r'([\d.]+)\s+'
                    r'([\d.]+)',
                    clean,
                )
                if m2:
                    self._cur_precision = float(m2.group(3))
                    self._cur_recall    = float(m2.group(4))
                    self._cur_map50     = float(m2.group(5))
                    self._cur_map5095   = float(m2.group(6))
                    self._emit_val()
                    return
        except Exception:
            pass


class VideoDetectWorker(QThread):
    """Run video detection frame by frame, emit both input and output frames."""
    frame_ready = Signal(np.ndarray, np.ndarray, int, int)  # input_frame, output_frame, current, total
    video_finished = Signal(str)
    video_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.model = None
        self.video_path = ""
        self.conf = 0.25
        self.iou = 0.45
        self.device = None
        self.paused = False
        self.stop_flag = False
        self.sync_fps = True  # True=和原视频帧率对齐, False=识别多快就多块

    def run(self):
        import cv2
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.video_error.emit(f"Cannot open video: {self.video_path}")
                return

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / max(fps, 1) if self.sync_fps else 0
            count = 0

            while cap.isOpened() and not self.stop_flag:
                if self.paused:
                    time.sleep(0.1)
                    continue

                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                count += 1
                results = self.model.predict(source=frame, conf=self.conf, iou=self.iou,
                                            device=self.device, verbose=False)
                input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(input_rgb, output_rgb, count, total)

                if self.sync_fps:
                    elapsed = time.time() - start
                    sleep_time = frame_delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            cap.release()
            if self.stop_flag:
                self.video_finished.emit("Video stopped")
            else:
                self.video_finished.emit(f"Video complete: {count}/{total} frames")

        except Exception as e:
            self.video_error.emit(str(e))

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.stop_flag = True
        self.paused = False


class ExportWorker(QThread):
    """Export model to ONNX/TensorRT/etc."""
    finished = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.format = "onnx"
        self.imgsz = 640
        self.half = False
        self.dynamic = False
        self.simplify = True
        self.opset = None  # None = use ultralytics default

    def run(self):
        try:
            project_root = str(Path(__file__).parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from ultralytics import YOLO
            model = YOLO(self.model_path)

            export_kwargs = dict(
                format=self.format,
                imgsz=self.imgsz,
                half=self.half,
                dynamic=self.dynamic,
                simplify=self.simplify,
            )
            if self.opset is not None:
                export_kwargs["opset"] = self.opset

            result = model.export(**export_kwargs)

            # Build detailed export info
            lines = [f"Exported: {result}"]

            # Get exported file path and size
            export_path = Path(result) if result else None
            if export_path and export_path.exists():
                size_mb = export_path.stat().st_size / (1024 * 1024)
                lines.append(f"File: {export_path.name} ({size_mb:.1f} MB)")

            # For ONNX format, get input/output shapes and ONNX version
            if self.format == "onnx" and export_path and export_path.exists():
                try:
                    import onnx
                    onnx_model = onnx.load(str(export_path))
                    onnx_ver = onnx_model.opset_import[0].version
                    lines.append(f"ONNX opset version: {onnx_ver}")

                    # Input shape
                    inp = onnx_model.graph.input[0]
                    dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
                    lines.append(f"Input shape: {tuple(dims)} BCHW")

                    # Output shapes
                    for i, out in enumerate(onnx_model.graph.output):
                        out_dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
                        lines.append(f"Output shape[{i}]: {tuple(out_dims)}")
                except ImportError:
                    lines.append("(install 'onnx' package for detailed shape info)")
                except Exception as e:
                    lines.append(f"(ONNX inspection error: {e})")

            self.finished.emit("\n".join(lines))
        except Exception as e:
            self.error.emit(str(e))


class BenchmarkWorker(QThread):
    """Run speed benchmark."""
    finished = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.imgsz = 640
        self.runs = 100

    def run(self):
        try:
            import json
            project_root = str(Path(__file__).parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from ultralytics import YOLO
            from benchmark.speed_benchmark import SpeedBenchmark
            model = YOLO(self.model_path)
            bench = SpeedBenchmark()
            result = bench.benchmark_pytorch(model.model, img_size=self.imgsz, num_runs=self.runs)
            self.finished.emit(json.dumps(result, indent=2))
        except Exception as e:
            self.error.emit(str(e))
