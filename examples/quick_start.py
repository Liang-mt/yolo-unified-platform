"""
YOLO Unified Platform - Quick Start Examples
"""

# ─── Example 1: Basic Training ───────────────────────────────────────

def example_basic_training():
    """Train YOLOv8s on a custom dataset."""
    from trainers.unified_trainer import UnifiedTrainer

    trainer = UnifiedTrainer(
        variant="yolov8",
        size="s",
        num_classes=3,
        device="0",
        project="runs/quickstart",
        name="my_first_train",
    )

    results = trainer.train(
        data="configs/custom_dataset.yaml",
        epochs=50,
        batch_size=16,
        img_size=640,
        lr0=0.01,
    )

    print(f"Training complete: {results}")


# ─── Example 2: Multi-GPU Training ───────────────────────────────────

def example_multi_gpu():
    """Train with multiple GPUs using DDP."""
    from trainers.multi_gpu_trainer import MultiGPUTrainer

    trainer = MultiGPUTrainer(
        variant="yolov8",
        size="m",
        gpus=[0, 1, 2, 3],
        strategy="ddp",
    )

    results = trainer.train(
        data="configs/coco.yaml",
        epochs=100,
        batch_size=64,  # Total batch = 64 / 4 GPUs = 16 per GPU
    )

    print(f"Multi-GPU training complete: {results}")


# ─── Example 3: Compare All Variants ─────────────────────────────────

def example_compare_variants():
    """Compare all YOLO variants on the same dataset."""
    from trainers.unified_trainer import UnifiedTrainer
    from benchmark.speed_benchmark import SpeedBenchmark

    variants = ["yolov5", "yolov8", "yolov10", "yolov11", "yolo26"]
    results = {}

    for variant in variants:
        print(f"\n{'='*50}")
        print(f"Training {variant}")
        print('='*50)

        trainer = UnifiedTrainer(
            variant=variant, size="s", device="0",
            project="runs/compare", name=variant,
        )

        result = trainer.train(
            data="configs/custom_dataset.yaml",
            epochs=20,
            batch_size=16,
        )

        results[variant] = result

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    for variant, result in results.items():
        print(f"  {variant}: {result}")


# ─── Example 4: Export and Deploy ────────────────────────────────────

def example_export_deploy():
    """Export model to ONNX and benchmark."""
    from deployment.onnx_deployer import ONNXDeployer
    from benchmark.speed_benchmark import SpeedBenchmark

    # Export to ONNX
    deployer = ONNXDeployer()
    onnx_path = deployer.export(
        model=None,  # Load your model here
        save_path="exports/model.onnx",
        img_size=640,
        simplify=True,
    )

    # Benchmark
    bench = SpeedBenchmark()
    result = bench.benchmark_onnx(onnx_path, img_size=640)
    print(f"ONNX Benchmark: {result}")


# ─── Example 5: Dataset Pipeline ────────────────────────────────────

def example_dataset_pipeline():
    """Full dataset preparation pipeline."""
    from data.converter import AnnotationConverter
    from data.cleaner import DatasetCleaner
    from data.splitter import DatasetSplitter

    # Step 1: Convert VOC to YOLO format
    print("Step 1: Converting annotations...")
    AnnotationConverter.voc_to_yolo(
        voc_dir="raw_data/annotations",
        output_dir="datasets/my_dataset",
        classes=["cat", "dog", "bird"],
    )

    # Step 2: Clean dataset
    print("\nStep 2: Cleaning dataset...")
    cleaner = DatasetCleaner(
        image_dir="datasets/my_dataset/images",
        label_dir="datasets/my_dataset/labels",
    )
    report = cleaner.scan(fix=True)

    # Step 3: Split dataset
    print("\nStep 3: Splitting dataset...")
    splitter = DatasetSplitter(
        image_dir="datasets/my_dataset/images",
        label_dir="datasets/my_dataset/labels",
        output_dir="datasets/my_dataset_split",
    )
    splitter.split(
        train_ratio=0.8,
        val_ratio=0.15,
        test_ratio=0.05,
        classes=["cat", "dog", "bird"],
    )


# ─── Example 6: Custom Loss Function ────────────────────────────────

def example_custom_loss():
    """Use custom loss functions."""
    import torch
    from losses import CIoULoss, FocalLoss, CombinedLoss

    # Individual loss
    ciou = CIoULoss()
    pred = torch.randn(10, 4)
    target = torch.randn(10, 4)
    loss = ciou(pred, target)
    print(f"CIoU Loss: {loss.item():.4f}")

    # Combined loss
    combined = CombinedLoss(
        cls_loss={"type": "focal", "weight": 1.0, "alpha": 0.25, "gamma": 2.0},
        box_loss={"type": "ciou", "weight": 5.0},
    )


# ─── Example 7: Model Pruning ───────────────────────────────────────

def example_pruning():
    """Prune a model for deployment."""
    from pruning.pruner import ModelPruner
    from pruning.quantizer import ModelQuantizer

    # Load your model
    # model = load_model(...)

    # Prune
    pruner = ModelPruner(model)
    pruned_model = pruner.prune(amount=0.3, method="magnitude")
    pruner.export_pruned("pruned_model.pt")

    # Quantize
    quantizer = ModelQuantizer(model)
    quantized = quantizer.quantize_dynamic()
    speedup = quantizer.benchmark_speedup(model, quantized)


# ─── Example 8: Log Analysis ────────────────────────────────────────

def example_log_analysis():
    """Analyze training logs."""
    from analysis.log_analyzer import LogAnalyzer

    analyzer = LogAnalyzer("runs/train/exp")
    analyzer.print_summary()

    # Generate plots
    analyzer.plot_curves("analysis_output/")

    # Export report
    analyzer.export_report("analysis_output/report.json")


# ─── Run Examples ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    examples = {
        "train": example_basic_training,
        "multigpu": example_multi_gpu,
        "compare": example_compare_variants,
        "export": example_export_deploy,
        "dataset": example_dataset_pipeline,
        "loss": example_custom_loss,
        "prune": example_pruning,
        "analyze": example_log_analysis,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Available examples:")
        for name in examples:
            print(f"  python examples/quick_start.py {name}")
