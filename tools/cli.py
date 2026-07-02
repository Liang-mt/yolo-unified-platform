"""
YOLO Unified Platform - Command Line Interface.
Provides CLI access to all platform features.
"""

import sys
from pathlib import Path

try:
    import click
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print("Install dependencies: pip install click rich")
    sys.exit(1)

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🚀 YOLO Unified Training Platform CLI"""
    pass


# ─── Train Command ────────────────────────────────────────────────────

@cli.command()
@click.option("--variant", "-v", type=click.Choice(["yolov5", "yolov8", "yolov10", "yolov11", "yolo26"]), default="yolov8")
@click.option("--size", "-s", type=click.Choice(["n", "s", "m", "l", "x"]), default="s")
@click.option("--data", "-d", required=True, help="Dataset YAML path")
@click.option("--epochs", "-e", default=100, help="Number of epochs")
@click.option("--batch", "-b", default=16, help="Batch size")
@click.option("--imgsz", default=640, help="Image size")
@click.option("--device", default="auto", help="Device (auto/cpu/cuda/cuda:0)")
@click.option("--project", default="runs/train", help="Project directory")
@click.option("--name", default="exp", help="Experiment name")
def train(variant, size, data, epochs, batch, imgsz, device, project, name):
    """Train a YOLO model."""
    from trainers.unified_trainer import UnifiedTrainer

    console.print(f"[bold green]🚀 Training {variant}{size}[/bold green]")
    console.print(f"   Data: {data}")
    console.print(f"   Epochs: {epochs}, Batch: {batch}, ImgSize: {imgsz}")

    trainer = UnifiedTrainer(
        variant=variant, size=size, device=device,
        project=project, name=name,
    )
    result = trainer.train(data=data, epochs=epochs, batch_size=batch, img_size=imgsz)
    console.print(f"[bold green]✅ Training complete![/bold green]")
    console.print(f"   Results: {result}")


# ─── Export Command ───────────────────────────────────────────────────

@cli.command()
@click.option("--weights", "-w", required=True, help="Model weights path")
@click.option("--format", "-f", "fmt", type=click.Choice(["onnx", "torchscript", "engine", "tflite"]), default="onnx")
@click.option("--imgsz", default=640, help="Image size")
@click.option("--half", is_flag=True, help="FP16 half precision")
@click.option("--simplify", is_flag=True, default=True, help="Simplify ONNX graph")
def export(weights, fmt, imgsz, half, simplify):
    """Export a trained model."""
    from ultralytics import YOLO

    console.print(f"[bold blue]📦 Exporting model to {fmt}[/bold blue]")
    model = YOLO(weights)
    result = model.export(format=fmt, imgsz=imgsz, half=half, simplify=simplify)
    console.print(f"[bold green]✅ Export complete: {result}[/bold green]")


# ─── Predict Command ─────────────────────────────────────────────────

@cli.command()
@click.option("--weights", "-w", required=True, help="Model weights path")
@click.option("--source", "-s", required=True, help="Image/video/directory path")
@click.option("--conf", default=0.25, help="Confidence threshold")
@click.option("--iou", default=0.45, help="IoU threshold")
@click.option("--save", is_flag=True, default=True, help="Save results")
def predict(weights, source, conf, iou, save):
    """Run inference on images/videos."""
    from ultralytics import YOLO

    console.print(f"[bold blue]🔍 Running inference[/bold blue]")
    model = YOLO(weights)
    results = model.predict(source=source, conf=conf, iou=iou, save=save)

    for r in results:
        console.print(f"   {Path(r.path).name}: {len(r.boxes)} detections")


# ─── Validate Command ────────────────────────────────────────────────

@cli.command()
@click.option("--weights", "-w", required=True, help="Model weights path")
@click.option("--data", "-d", required=True, help="Dataset YAML path")
@click.option("--imgsz", default=640, help="Image size")
@click.option("--batch", "-b", default=16, help="Batch size")
def val(weights, data, imgsz, batch):
    """Validate a trained model."""
    from ultralytics import YOLO

    console.print(f"[bold blue]📊 Validating model[/bold blue]")
    model = YOLO(weights)
    results = model.val(data=data, imgsz=imgsz, batch=batch)

    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("mAP50", f"{results.box.map50:.4f}")
    table.add_row("mAP50-95", f"{results.box.map:.4f}")
    table.add_row("Precision", f"{results.box.mp:.4f}")
    table.add_row("Recall", f"{results.box.mr:.4f}")
    console.print(table)


# ─── Dataset Commands ────────────────────────────────────────────────

@cli.group()
def data():
    """Dataset management commands."""
    pass


@data.command("convert")
@click.option("--input", "-i", "input_path", required=True, help="Input path")
@click.option("--output", "-o", "output_dir", required=True, help="Output directory")
@click.option("--source", type=click.Choice(["voc", "coco", "labelme"]), required=True)
@click.option("--target", type=click.Choice(["yolo", "coco"]), default="yolo")
@click.option("--classes", "-c", required=True, help="Comma-separated class names")
def data_convert(input_path, output_dir, source, target, classes):
    """Convert annotation formats."""
    from data.converter import AnnotationConverter

    cls_list = [c.strip() for c in classes.split(",")]
    console.print(f"🔄 Converting {source} → {target}")
    result = AnnotationConverter.auto_convert(input_path, output_dir, source, target, cls_list)
    console.print(f"[bold green]✅ Done: {result}[/bold green]")


@data.command("clean")
@click.option("--images", "-i", required=True, help="Image directory")
@click.option("--labels", "-l", required=True, help="Label directory")
@click.option("--fix", is_flag=True, help="Auto-fix issues")
def data_clean(images, labels, fix):
    """Clean and validate dataset."""
    from data.cleaner import DatasetCleaner

    console.print(f"🔍 Scanning dataset...")
    cleaner = DatasetCleaner(images, labels)
    report = cleaner.scan(fix=fix)
    console.print(f"[bold green]✅ Scan complete: {report['total_issues']} issues found[/bold green]")


@data.command("split")
@click.option("--images", "-i", required=True, help="Image directory")
@click.option("--labels", "-l", required=True, help="Label directory")
@click.option("--output", "-o", required=True, help="Output directory")
@click.option("--train-ratio", default=0.8, help="Train split ratio")
@click.option("--val-ratio", default=0.15, help="Val split ratio")
def data_split(images, labels, output, train_ratio, val_ratio):
    """Split dataset into train/val/test."""
    from data.splitter import DatasetSplitter

    splitter = DatasetSplitter(images, labels, output)
    counts = splitter.split(train_ratio=train_ratio, val_ratio=val_ratio)
    console.print(f"[bold green]✅ Split: {counts}[/bold green]")


# ─── Benchmark Command ───────────────────────────────────────────────

@cli.command()
@click.option("--weights", "-w", required=True, help="Model weights path")
@click.option("--imgsz", default=640, help="Image size")
@click.option("--runs", default=100, help="Number of runs")
def benchmark(weights, imgsz, runs):
    """Benchmark model inference speed."""
    from ultralytics import YOLO
    from benchmark.speed_benchmark import SpeedBenchmark

    model = YOLO(weights)
    bench = SpeedBenchmark()
    result = bench.benchmark_pytorch(model.model, img_size=imgsz, num_runs=runs)
    bench.print_report({"pytorch": result})


# ─── Web UI Command ──────────────────────────────────────────────────

@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=7860, help="Server port")
@click.option("--share", is_flag=True, help="Create public link")
def web(host, port, share):
    """Launch the web interface."""
    from web.app import launch
    console.print(f"[bold green]🌐 Launching web UI at http://{host}:{port}[/bold green]")
    launch(host=host, port=port, share=share)


# ─── Analyze Command ─────────────────────────────────────────────────

@cli.command()
@click.option("--log-dir", "-d", required=True, help="Training log directory")
@click.option("--output", "-o", default=None, help="Output report path")
def analyze(log_dir, output):
    """Analyze training logs."""
    from analysis.log_analyzer import LogAnalyzer

    analyzer = LogAnalyzer(log_dir)
    analyzer.print_summary()

    if output:
        analyzer.export_report(output)
        analyzer.plot_curves(Path(output).parent / "plots")


def main():
    cli()


if __name__ == "__main__":
    main()
