"""
Gradio Web UI for YOLO Unified Platform.
Chinese / English bilingual with language switch.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    gr = None


def _t(en, zh, lang):
    return zh if lang == "zh" else en


# ── Model Cache ──────────────────────────────────────────────────────
_MODEL_CACHE = {}  # key: (model_name, device) -> YOLO model
_MODELS_DIR = Path(__file__).parent.parent / "models"
_DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def _scan_yaml() -> list:
    """Scan datasets/ folder and common locations for .yaml dataset files."""
    yamls = []
    # Scan datasets/ folder (recursive)
    _DATASETS_DIR.mkdir(exist_ok=True)
    for f in sorted(_DATASETS_DIR.rglob("*.yaml")):
        yamls.append(str(f))
    for f in sorted(_DATASETS_DIR.rglob("*.yml")):
        yamls.append(str(f))
    return yamls


def _scan_models() -> list:
    """Scan models/ folder for .pt files, return sorted list."""
    _MODELS_DIR.mkdir(exist_ok=True)
    models = sorted([f.name for f in _MODELS_DIR.glob("*.pt")])
    if not models:
        # Fallback: provide common model names for download
        models = ["yolov5n.pt", "yolov5s.pt", "yolov8n.pt", "yolov8s.pt",
                  "yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt"]
    return models


def _resolve_model(name: str) -> str:
    """Resolve model name to full path if it exists in models/ folder."""
    local = _MODELS_DIR / name
    if local.exists():
        return str(local)
    return name  # fallback: let ultralytics handle it (download or absolute path)


def _get_model(model_name: str, device: str = None):
    """Load model once, reuse from cache on subsequent calls."""
    from ultralytics import YOLO
    import numpy as np

    resolved = _resolve_model(model_name)
    cache_key = (resolved, device)

    # For custom file paths (not in models/ folder), always reload
    is_custom = not str(resolved).startswith(str(_MODELS_DIR))

    if is_custom or cache_key not in _MODEL_CACHE:
        print(f"[YOLO] Loading model: {resolved} on {device or 'auto'}...")
        model = YOLO(str(resolved))
        if device:
            model.to(device)
        # Warmup
        model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        _MODEL_CACHE[cache_key] = model
        print(f"[YOLO] Model loaded: {resolved}")
    return _MODEL_CACHE[cache_key]


def create_app(
    default_variant: str = "yolov8",
    default_size: str = "s",
    default_classes: Optional[List[str]] = None,
) -> "gr.Blocks":
    if gr is None:
        raise ImportError("gradio is required. pip install gradio")

    with gr.Blocks(
        title="YOLO Unified Training Platform",
        theme=gr.themes.Soft(),
        css="#train_output textarea { height: 65vh !important; max-height: 65vh !important; overflow-y: auto !important; resize: none !important; font-family: monospace; font-size: 13px; }",
    ) as app:

        # ── Header ──────────────────────────────────────────────────────
        header_md = gr.Markdown(
            value="# 🚀 YOLO Unified Training Platform\n"
                  "**Support / 支持:** YOLOv5 | YOLOv8 | YOLOv10 | YOLOv11 | YOLO26"
        )
        lang_dd = gr.Dropdown(
            choices=[("English", "en"), ("中文", "zh")],
            value="en", label="🌐 Language / 语言",
            scale=0, min_width=140,
        )

        # ── Tabs ────────────────────────────────────────────────────────
        with gr.Tabs():
            # ── Inference ───────────────────────────────────────────────
            with gr.Tab("🔍 Inference / 推理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i_model = gr.Dropdown(
                            choices=_scan_models(), value=_scan_models()[0] if _scan_models() else None,
                            label="Model / 模型",
                        )
                        i_custom = gr.File(
                            label="Or browse custom .pt / 或选择自定义模型",
                            file_types=[".pt"], type="filepath",
                        )
                        i_refresh = gr.Button("🔄 Refresh / 刷新列表", size="sm")
                        i_device = gr.Dropdown(
                            choices=["auto", "cpu", "0", "0,1"],
                            value="auto", label="Device / 设备",
                        )
                        i_conf = gr.Slider(0.01, 1.0, value=0.25, label="Conf / 置信度")
                        i_iou = gr.Slider(0.1, 1.0, value=0.45, label="IoU / 阈值")
                        i_btn = gr.Button("🚀 Inference / 推理", variant="primary")
                    with gr.Column(scale=2):
                        i_img_in = gr.Image(type="numpy", label="Input / 输入")
                        i_img_out = gr.Image(type="numpy", label="Result / 结果")
                        i_json = gr.JSON(label="Details / 详情")

                def fn_refresh_models():
                    models = _scan_models()
                    return gr.update(choices=models, value=models[0] if models else None)

                i_refresh.click(fn=fn_refresh_models, inputs=[], outputs=[i_model])

                def fn_infer(image, model_name, custom_file, device, conf, iou, lang):
                    if image is None:
                        return None, {"error": _t("No image", "未提供图片", lang)}
                    try:
                        # Determine model path
                        name = custom_file if custom_file else model_name
                        if not name:
                            return image, {"error": _t("Please select a model", "请选择模型", lang)}

                        # Resolve to full path
                        resolved = _resolve_model(name)
                        print(f"[Infer] model={name} -> {resolved}, device={device}")

                        # Load model
                        dev = device if device != "auto" else None
                        model = _get_model(resolved, dev)

                        # Gradio gives RGB, ultralytics expects BGR for numpy arrays
                        import cv2
                        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        results = model.predict(source=img_bgr, conf=conf, iou=iou, device=dev, verbose=False)
                        ann = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                        dets = [{"class": results[0].names[int(b.cls)],
                                 "confidence": round(float(b.conf), 3),
                                 "bbox": [round(x, 1) for x in b.xyxy[0].tolist()]}
                                for b in results[0].boxes]
                        info = {
                            "model": str(name),
                            "resolved": str(resolved),
                            "device": device,
                            "count": len(dets),
                            "detections": dets,
                        }
                        return ann, info
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        return image, {"error": str(e), "model": str(name) if 'name' in dir() else "unknown"}

                i_btn.click(fn=fn_infer, inputs=[i_img_in, i_model, i_custom, i_device, i_conf, i_iou, lang_dd],
                            outputs=[i_img_out, i_json])

            # ── Training ────────────────────────────────────────────────
            with gr.Tab("🏋️ Training / 训练"):
                with gr.Row():
                    with gr.Column():
                        t_model = gr.Dropdown(
                            choices=_scan_models(), value=_scan_models()[0] if _scan_models() else None,
                            label="Model / 模型",
                        )
                        t_custom = gr.File(
                            label="Or browse custom .pt / 或选择自定义模型",
                            file_types=[".pt"], type="filepath",
                        )
                        t_data = gr.Dropdown(
                            choices=_scan_yaml(), value=_scan_yaml()[0] if _scan_yaml() else None,
                            label="Dataset YAML / 数据集",
                            allow_custom_value=True,
                        )
                        t_data_refresh = gr.Button("🔄 Refresh YAML / 刷新数据集列表", size="sm")
                        t_epochs = gr.Number(value=100, label="Epochs / 轮数", precision=0)
                        t_batch = gr.Number(value=16, label="Batch / 批大小", precision=0)
                        t_imgsz = gr.Number(value=640, label="ImgSize / 图片尺寸", precision=0)
                        t_lr = gr.Number(value=0.01, label="LR / 学习率")
                        t_device = gr.Textbox(value="0", label="GPU / 设备")
                        t_btn = gr.Button("🚀 Train / 训练", variant="primary")
                    with gr.Column():
                        t_out = gr.Textbox(label="Output / 输出", lines=25, max_lines=25, elem_id="train_output")

                def fn_refresh_yaml():
                    yamls = _scan_yaml()
                    return gr.update(choices=yamls, value=yamls[0] if yamls else None)

                t_data_refresh.click(fn=fn_refresh_yaml, inputs=[], outputs=[t_data])

                def fn_train(model_name, custom_file, data, epochs, batch, imgsz, lr, device, lang):
                    """Generator: yields log_text in real-time."""
                    import sys, time, threading
                    if not data or not str(data).strip():
                        yield _t("Provide dataset YAML path", "请输入数据集 YAML 路径", lang)
                        return
                    name = custom_file if custom_file else model_name
                    if not name:
                        yield _t("Please select a model", "请选择模型", lang)
                        return

                    # Log capture: handles \r (overwrite) and \n (new line)
                    log_lines = []
                    buf = []
                    lock = threading.Lock()
                    done = {"ok": False, "result": "", "error": ""}

                    class LogCapture:
                        """Capture stdout+stderr, handle \\r progress bars."""
                        def __init__(self):
                            self._buf = []
                        def write(self, text):
                            if not text:
                                return 0
                            with lock:
                                for ch in text:
                                    if ch == "\n":
                                        line = "".join(self._buf).rstrip()
                                        if line:
                                            log_lines.append(line)
                                        self._buf.clear()
                                    elif ch == "\r":
                                        self._buf.clear()
                                    else:
                                        self._buf.append(ch)
                            return len(text)
                        def flush(self):
                            pass
                        @property
                        def text(self):
                            with lock:
                                cur = "".join(self._buf).rstrip()
                                lines = log_lines.copy()
                                if cur:
                                    lines.append(cur)
                                return "\n".join(lines[-500:])

                    capture = LogCapture()

                    def run_train():
                        try:
                            from ultralytics import YOLO
                            resolved = _resolve_model(name)
                            data_path = str(data).strip()
                            with lock:
                                log_lines.append(f"[Model] {name} -> {resolved}")
                                log_lines.append(f"[Data]  {data_path}")
                            model = YOLO(str(resolved))
                            # Capture BOTH stdout and stderr (tqdm writes to stderr)
                            old_stdout = sys.stdout
                            old_stderr = sys.stderr
                            sys.stdout = capture
                            sys.stderr = capture
                            try:
                                result = model.train(
                                    data=data_path, epochs=int(epochs), batch=int(batch),
                                    imgsz=int(imgsz), lr0=float(lr), device=device,
                                )
                                # Extract only key metrics, not the full object dump
                                try:
                                    r = result.results_dict
                                    done["result"] = (
                                        f"\n{'='*50}\n"
                                        f"  Training Complete\n"
                                        f"  mAP50:    {r.get('metrics/mAP50(B)', 'N/A'):.4f}\n"
                                        f"  mAP50-95: {r.get('metrics/mAP50-95(B)', 'N/A'):.4f}\n"
                                        f"  Precision: {r.get('metrics/precision(B)', 'N/A'):.4f}\n"
                                        f"  Recall:    {r.get('metrics/recall(B)', 'N/A'):.4f}\n"
                                        f"{'='*50}"
                                    )
                                except Exception:
                                    done["result"] = "Training complete."
                                done["ok"] = True
                            finally:
                                sys.stdout = old_stdout
                                sys.stderr = old_stderr
                        except Exception as e:
                            sys.stdout = sys.__stdout__
                            sys.stderr = sys.__stderr__
                            import traceback
                            done["error"] = traceback.format_exc()

                    thread = threading.Thread(target=run_train, daemon=True)
                    thread.start()

                    last_text = ""
                    while thread.is_alive():
                        time.sleep(0.3)
                        text = capture.text
                        if text != last_text:
                            yield text
                            last_text = text

                    text = capture.text
                    if done["ok"]:
                        yield text + f"\n\n{done['result']}"
                    elif done["error"]:
                        yield text + f"\n\n{done['error']}"
                    else:
                        yield text

                t_btn.click(fn=fn_train,
                            inputs=[t_model, t_custom, t_data, t_epochs, t_batch, t_imgsz, t_lr, t_device, lang_dd],
                            outputs=[t_out])

            # ── Dataset Tools ───────────────────────────────────────────
            with gr.Tab("📦 Dataset / 数据集"):
                with gr.Tabs():
                    with gr.Tab("🔄 Converter / 格式转换"):
                        with gr.Row():
                            c_src = gr.Dropdown(choices=["voc", "coco", "labelme", "yolo"], value="voc",
                                                label="Source / 源格式")
                            c_tgt = gr.Dropdown(choices=["yolo", "coco", "voc"], value="yolo",
                                                label="Target / 目标格式")
                        c_in = gr.File(label="Input Dir / 输入目录", type="filepath")
                        c_out = gr.Textbox(label="Output Dir / 输出目录", placeholder="如: datasets/output")
                        c_cls = gr.Textbox(label="Classes / 类别", placeholder="cat,dog,person")
                        c_btn = gr.Button("🔄 Convert / 转换", variant="primary")
                        c_res = gr.Textbox(label="Result / 结果", lines=5)

                        def fn_convert(src, tgt, inp, out, classes, lang):
                            try:
                                sys_path = __import__('sys').path
                                p = str(Path(__file__).parent.parent)
                                if p not in sys_path:
                                    sys_path.insert(0, p)
                                from data.converter import AnnotationConverter
                                cls_list = [c.strip() for c in classes.split(",")] if classes else []
                                result = AnnotationConverter.auto_convert(inp, out, src, tgt, cls_list)
                                return f"✅ {_t('Done', '转换完成', lang)}: {result}"
                            except Exception as e:
                                return f"❌ {e}"

                        c_btn.click(fn=fn_convert, inputs=[c_src, c_tgt, c_in, c_out, c_cls, lang_dd], outputs=[c_res])

                    with gr.Tab("🔍 Cleaner / 数据清洗"):
                        cl_img = gr.File(label="Image Dir / 图片目录", type="filepath")
                        cl_lbl = gr.File(label="Label Dir / 标注目录", type="filepath")
                        cl_btn = gr.Button("🔍 Scan / 扫描", variant="primary")
                        cl_res = gr.Textbox(label="Report / 报告", lines=15)

                        def fn_clean(img_d, lbl_d, lang):
                            try:
                                sys_path = __import__('sys').path
                                p = str(Path(__file__).parent.parent)
                                if p not in sys_path:
                                    sys_path.insert(0, p)
                                from data.cleaner import DatasetCleaner
                                import json
                                report = DatasetCleaner(img_d, lbl_d).scan(fix=False)
                                return json.dumps(report, indent=2, default=str)
                            except Exception as e:
                                return f"❌ {e}"

                        cl_btn.click(fn=fn_clean, inputs=[cl_img, cl_lbl, lang_dd], outputs=[cl_res])

            # ── Export ──────────────────────────────────────────────────
            with gr.Tab("📤 Export / 导出"):
                with gr.Row():
                    with gr.Column():
                        e_model = gr.Dropdown(
                            choices=_scan_models(), value=_scan_models()[0] if _scan_models() else None,
                            label="Model / 模型",
                        )
                        e_custom = gr.File(
                            label="Or browse custom .pt / 或选择自定义模型",
                            file_types=[".pt"], type="filepath",
                        )
                        e_fmt = gr.Dropdown(
                            choices=["onnx", "torchscript", "engine", "tflite", "coreml"],
                            value="onnx", label="Format / 格式",
                        )
                        e_imgsz = gr.Number(value=640, label="ImgSize / 尺寸")
                        e_half = gr.Checkbox(label="FP16 / 半精度")
                        e_btn = gr.Button("📤 Export / 导出", variant="primary")
                    with gr.Column():
                        e_out = gr.Textbox(label="Result / 结果", lines=10)

                        def fn_export(model_name, custom_file, fmt, imgsz, half, lang):
                            try:
                                from ultralytics import YOLO
                                name = custom_file if custom_file else model_name
                                if not name:
                                    return _t("Please select a model", "请选择模型", lang)
                                resolved = _resolve_model(name)
                                print(f"[Export] model={name} -> {resolved}")
                                result = YOLO(str(resolved)).export(format=fmt, imgsz=int(imgsz), half=half)
                                return f"✅ {_t('Exported', '导出完成', lang)}: {result}"
                            except Exception as e:
                                import traceback; traceback.print_exc()
                                return f"❌ {e}"

                        e_btn.click(fn=fn_export, inputs=[e_model, e_custom, e_fmt, e_imgsz, e_half, lang_dd],
                                    outputs=[e_out])

            # ── Log Analysis ────────────────────────────────────────────
            with gr.Tab("📊 Analysis / 分析"):
                with gr.Row():
                    with gr.Column():
                        l_dir = gr.File(label="Log Dir / 日志目录", type="filepath")
                        l_btn = gr.Button("📊 Analyze / 分析", variant="primary")
                    with gr.Column():
                        l_out = gr.Textbox(label="Report / 报告", lines=20)
                        l_plots = gr.Gallery(label="Plots / 曲线图", columns=2)

                        def fn_analyze(log_dir, lang):
                            if not log_dir:
                                return _t("Provide log dir", "请提供日志目录", lang), []
                            try:
                                sys_path = __import__('sys').path
                                p = str(Path(__file__).parent.parent)
                                if p not in sys_path:
                                    sys_path.insert(0, p)
                                from analysis.log_analyzer import LogAnalyzer
                                import json
                                analyzer = LogAnalyzer(log_dir)
                                report = analyzer.analyze()
                                plots = analyzer.plot_curves(Path(tempfile.mkdtemp()))
                                return json.dumps(report, indent=2, default=str), plots
                            except Exception as e:
                                return f"❌ {e}", []

                        l_btn.click(fn=fn_analyze, inputs=[l_dir, lang_dd], outputs=[l_out, l_plots])

            # ── Benchmark ───────────────────────────────────────────────
            with gr.Tab("⚡ Benchmark / 测速"):
                with gr.Row():
                    with gr.Column():
                        b_model = gr.Dropdown(
                            choices=_scan_models(), value=_scan_models()[0] if _scan_models() else None,
                            label="Model / 模型",
                        )
                        b_custom = gr.File(
                            label="Or browse custom .pt / 或选择自定义模型",
                            file_types=[".pt"], type="filepath",
                        )
                        b_imgsz = gr.Number(value=640, label="ImgSize / 尺寸")
                        b_runs = gr.Number(value=100, label="Runs / 轮次")
                        b_btn = gr.Button("⚡ Run / 测速", variant="primary")
                    with gr.Column():
                        b_out = gr.Textbox(label="Results / 结果", lines=15)

                        def fn_bench(model_name, custom_file, imgsz, runs, lang):
                            try:
                                from benchmark.speed_benchmark import SpeedBenchmark
                                import json
                                name = custom_file if custom_file else model_name
                                if not name:
                                    return _t("Please select a model", "请选择模型", lang)
                                resolved = _resolve_model(name)
                                print(f"[Bench] model={name} -> {resolved}")
                                model = _get_model(resolved)
                                result = SpeedBenchmark().benchmark_pytorch(model.model, img_size=int(imgsz),
                                                                            num_runs=int(runs))
                                return json.dumps(result, indent=2)
                            except Exception as e:
                                import traceback; traceback.print_exc()
                                return f"❌ {e}"

                        b_btn.click(fn=fn_bench, inputs=[b_model, b_custom, b_imgsz, b_runs, lang_dd],
                                    outputs=[b_out])

        # ── Footer ──────────────────────────────────────────────────────
        footer_md = gr.Markdown(
            "---\n**YOLO Unified Training Platform** | "
            "YOLOv5 · YOLOv8 · YOLOv10 · YOLOv11 · YOLO26"
        )

        # ── Language Switch (single event, NO gr.State) ─────────────────
        text_out = [header_md, i_btn, t_btn, c_btn, cl_btn, e_btn, l_btn, b_btn, footer_md]

        def switch_lang(lang):
            return [
                "# 🚀 YOLO Unified Training Platform\n**Support / 支持:** YOLOv5 | YOLOv8 | YOLOv10 | YOLOv11 | YOLO26"
                if lang == "en" else
                "# 🚀 YOLO 统一训练平台\n**支持:** YOLOv5 | YOLOv8 | YOLOv10 | YOLOv11 | YOLO26",
                _t("🚀 Inference", "🚀 推理", lang),
                _t("🚀 Train", "🚀 训练", lang),
                _t("🔄 Convert", "🔄 转换", lang),
                _t("🔍 Scan", "🔍 扫描", lang),
                _t("📤 Export", "📤 导出", lang),
                _t("📊 Analyze", "📊 分析", lang),
                _t("⚡ Run", "⚡ 测速", lang),
                "---\n**YOLO Unified Training Platform** | YOLOv5 · YOLOv8 · YOLOv10 · YOLOv11 · YOLO26"
                if lang == "en" else
                "---\n**YOLO 统一训练平台** | 支持 YOLOv5、YOLOv8、YOLOv10、YOLOv11、YOLO26",
            ]

        lang_dd.change(fn=switch_lang, inputs=[lang_dd], outputs=text_out)

    return app


def launch(host="0.0.0.0", port=7860, share=False, **kw):
    app = create_app()
    app.launch(server_name=host, server_port=port, share=share, **kw)


if __name__ == "__main__":
    launch()
