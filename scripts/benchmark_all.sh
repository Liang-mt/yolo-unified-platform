#!/bin/bash
# YOLO Unified Platform - Benchmark All Variants
# Usage: bash scripts/benchmark_all.sh --imgsz 640 --runs 100

set -e

echo "=========================================="
echo " YOLO All-Variant Benchmark"
echo "=========================================="

IMGSZ=640
RUNS=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

VARIANTS=("yolov5" "yolov8" "yolov10" "yolov11" "yolo26")
SIZES=("n" "s" "m")

echo ""
echo "Config: imgsz=$IMGSZ, runs=$RUNS"
echo "Variants: ${VARIANTS[*]}"
echo "Sizes: ${SIZES[*]}"
echo ""

for variant in "${VARIANTS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "----------------------------------------"
        echo "Benchmarking: ${variant}${size}"
        echo "----------------------------------------"
        python -c "
from benchmark.speed_benchmark import SpeedBenchmark
try:
    from ultralytics import YOLO
    variant_map = {
        'yolov5': 'yolov5${size}', 'yolov8': 'yolov8${size}',
        'yolov10': 'yolov10${size}', 'yolov11': 'yolo11${size}',
        'yolo26': 'yolo26${size}',
    }
    model = YOLO(f'{variant_map['${variant}']}.pt')
    bench = SpeedBenchmark()
    result = bench.benchmark_pytorch(model.model, img_size=$IMGSZ, num_runs=$RUNS)
    bench.print_report({'${variant}${size}': result})
except Exception as e:
    print(f'Error: {e}')
"
        echo ""
    done
done

echo "=========================================="
echo " ✅ Benchmark complete!"
echo "=========================================="
