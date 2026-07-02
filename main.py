"""
YOLO Unified Training Platform - Main Entry Point

Usage:
    python main.py train --variant yolov8 --data coco.yaml --epochs 100
    python main.py export --weights best.pt --format onnx
    python main.py web --port 7860
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Main entry point - delegates to CLI."""
    from tools.cli import cli
    cli()


if __name__ == "__main__":
    main()
