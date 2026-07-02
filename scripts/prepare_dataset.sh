#!/bin/bash
# YOLO Unified Platform - Dataset Preparation Script
# Usage: bash scripts/prepare_dataset.sh --source voc --input data/raw --output data/processed

set -e

echo "=========================================="
echo " YOLO Dataset Preparation Script"
echo "=========================================="

# Default values
SOURCE_FORMAT="voc"
TARGET_FORMAT="yolo"
INPUT_DIR=""
OUTPUT_DIR="datasets/my_dataset"
CLASSES=""
TRAIN_RATIO=0.8
VAL_RATIO=0.15

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --source) SOURCE_FORMAT="$2"; shift 2 ;;
        --target) TARGET_FORMAT="$2"; shift 2 ;;
        --input) INPUT_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --classes) CLASSES="$2"; shift 2 ;;
        --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
        --val-ratio) VAL_RATIO="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_DIR" ]; then
    echo "❌ Error: --input is required"
    echo "Usage: bash scripts/prepare_dataset.sh --source voc --input data/raw --output data/processed --classes 'cat,dog,bird'"
    exit 1
fi

if [ -z "$CLASSES" ]; then
    echo "❌ Error: --classes is required"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  Source format: $SOURCE_FORMAT"
echo "  Target format: $TARGET_FORMAT"
echo "  Input: $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Classes: $CLASSES"
echo "  Train/Val ratio: $TRAIN_RATIO/$VAL_RATIO"
echo ""

# Step 1: Convert annotations
echo "📦 Step 1: Converting annotations ($SOURCE_FORMAT → $TARGET_FORMAT)..."
python -c "
from data.converter import AnnotationConverter
cls_list = [c.strip() for c in '$CLASSES'.split(',')]
AnnotationConverter.auto_convert('$INPUT_DIR', '$OUTPUT_DIR', '$SOURCE_FORMAT', '$TARGET_FORMAT', cls_list)
"
echo "✅ Conversion complete"
echo ""

# Step 2: Clean dataset
echo "🔍 Step 2: Cleaning dataset..."
python -c "
from data.cleaner import DatasetCleaner
cleaner = DatasetCleaner('$OUTPUT_DIR/images', '$OUTPUT_DIR/labels')
report = cleaner.scan(fix=True)
"
echo "✅ Cleaning complete"
echo ""

# Step 3: Split dataset
echo "✂️ Step 3: Splitting dataset..."
python -c "
from data.splitter import DatasetSplitter
cls_list = [c.strip() for c in '$CLASSES'.split(',')]
splitter = DatasetSplitter('$OUTPUT_DIR/images', '$OUTPUT_DIR/labels', '${OUTPUT_DIR}_split')
splitter.split(train_ratio=$TRAIN_RATIO, val_ratio=$VAL_RATIO, classes=cls_list)
"
echo "✅ Splitting complete"
echo ""

echo "=========================================="
echo " ✅ Dataset preparation complete!"
echo " Output: ${OUTPUT_DIR}_split"
echo "=========================================="
