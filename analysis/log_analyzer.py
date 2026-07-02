"""
Training Log Analyzer - parse, visualize, and analyze training logs.
Supports ultralytics logs, TensorBoard events, and custom CSV logs.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np


class LogAnalyzer:
    """
    Training log analysis toolkit.

    Features:
    - Parse ultralytics training logs
    - Parse TensorBoard events
    - Generate training curves (loss, mAP, LR)
    - Detect overfitting
    - Export analysis reports

    Usage:
        analyzer = LogAnalyzer("runs/train/exp")
        report = analyzer.analyze()
        analyzer.plot_curves("plots/")
    """

    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.data = defaultdict(list)
        self._parsed = False

    def parse(self) -> "LogAnalyzer":
        """Parse all available log files."""
        # Try ultralytics results.csv
        csv_file = self.log_dir / "results.csv"
        if csv_file.exists():
            self._parse_csv(csv_file)

        # Try TensorBoard events
        tb_dir = self.log_dir
        if any(tb_dir.glob("events.out.tfevents.*")):
            self._parse_tensorboard(tb_dir)

        # Try custom JSON log
        json_file = self.log_dir / "training_log.json"
        if json_file.exists():
            self._parse_json(json_file)

        self._parsed = True
        return self

    def _parse_csv(self, csv_path: Path) -> None:
        """Parse ultralytics results.csv."""
        import pandas as pd

        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]

        for col in df.columns:
            self.data[col] = df[col].dropna().tolist()

        print(f"✅ Parsed CSV: {len(df)} epochs, {len(df.columns)} columns")

    def _parse_tensorboard(self, log_dir: Path) -> None:
        """Parse TensorBoard event files."""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            for event_file in log_dir.glob("events.out.tfevents.*"):
                ea = EventAccumulator(str(log_dir))
                ea.Reload()

                for tag in ea.Tags().get("scalars", []):
                    events = ea.Scalars(tag)
                    self.data[tag] = [e.value for e in events]

                print(f"✅ Parsed TensorBoard: {len(ea.Tags().get('scalars', []))} scalar tags")
                break
        except ImportError:
            print("⚠️ tensorboard not installed, skipping TB parsing")

    def _parse_json(self, json_path: Path) -> None:
        """Parse custom JSON training log."""
        with open(json_path, "r") as f:
            log_data = json.load(f)

        if isinstance(log_data, list):
            for key in log_data[0].keys():
                self.data[key] = [entry[key] for entry in log_data if key in entry]
        elif isinstance(log_data, dict):
            for key, values in log_data.items():
                self.data[key] = values

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze training logs and generate report.

        Returns:
            Analysis report dict
        """
        if not self._parsed:
            self.parse()

        report = {
            "total_epochs": self._get_total_epochs(),
            "best_metrics": self._get_best_metrics(),
            "training_stability": self._analyze_stability(),
            "overfitting_analysis": self._detect_overfitting(),
            "convergence_analysis": self._analyze_convergence(),
            "learning_rate_analysis": self._analyze_lr(),
        }

        return report

    def _get_total_epochs(self) -> int:
        """Get total number of training epochs."""
        for key in ["epoch", "epochs"]:
            if key in self.data:
                return len(self.data[key])
        # Try inferring from loss data
        for key in self.data:
            if "loss" in key.lower():
                return len(self.data[key])
        return 0

    def _get_best_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get best values for key metrics."""
        best = {}
        metric_keys = {
            "mAP50": "max", "mAP50-95": "max",
            "metrics/mAP50(B)": "max", "metrics/mAP50-95(B)": "max",
            "train/box_loss": "min", "train/cls_loss": "min", "train/dfl_loss": "min",
            "val/box_loss": "min", "val/cls_loss": "min", "val/dfl_loss": "min",
        }

        for key, values in self.data.items():
            clean_key = key.strip()
            if clean_key in metric_keys or any(m in clean_key for m in ["loss", "mAP", "precision", "recall"]):
                numeric = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
                if numeric:
                    best[clean_key] = {
                        "best": max(numeric) if "mAP" in clean_key or "precision" in clean_key else min(numeric),
                        "last": numeric[-1],
                        "epoch": numeric.index(max(numeric) if "mAP" in clean_key else min(numeric)),
                    }

        return best

    def _analyze_stability(self) -> Dict[str, Any]:
        """Analyze training stability (loss variance, spikes)."""
        stability = {}

        for key, values in self.data.items():
            if "loss" not in key.lower():
                continue
            numeric = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
            if len(numeric) < 5:
                continue

            arr = np.array(numeric)
            std = np.std(arr)
            mean = np.mean(arr)

            # Detect spikes (>3 std from mean)
            spikes = np.where(np.abs(arr - mean) > 3 * std)[0]

            stability[key] = {
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
                "cv": round(float(std / (mean + 1e-8)), 4),
                "num_spikes": len(spikes),
                "spike_epochs": spikes.tolist()[:10],
            }

        return stability

    def _detect_overfitting(self) -> Dict[str, Any]:
        """Detect overfitting by comparing train/val loss divergence."""
        result = {"detected": False, "details": []}

        # Find train and val loss keys
        train_loss_keys = [k for k in self.data if "train" in k.lower() and "loss" in k.lower()]
        val_loss_keys = [k for k in self.data if "val" in k.lower() and "loss" in k.lower()]

        for tkey in train_loss_keys:
            # Find matching val key
            suffix = tkey.replace("train", "").strip()
            matching_vkey = None
            for vkey in val_loss_keys:
                if suffix in vkey:
                    matching_vkey = vkey
                    break

            if matching_vkey is None:
                continue

            train_vals = [v for v in self.data[tkey] if isinstance(v, (int, float))]
            val_vals = [v for v in self.data[matching_vkey] if isinstance(v, (int, float))]

            min_len = min(len(train_vals), len(val_vals))
            if min_len < 10:
                continue

            train_vals = np.array(train_vals[:min_len])
            val_vals = np.array(val_vals[:min_len])

            # Check if val loss is increasing while train loss is decreasing
            window = min(10, min_len // 3)
            if window < 3:
                continue

            train_trend = np.polyfit(range(window), train_vals[-window:], 1)[0]
            val_trend = np.polyfit(range(window), val_vals[-window:], 1)[0]

            if train_trend < 0 and val_trend > 0:
                result["detected"] = True
                result["details"].append({
                    "loss_type": suffix,
                    "train_trend": round(float(train_trend), 6),
                    "val_trend": round(float(val_trend), 6),
                    "gap": round(float(val_vals[-1] - train_vals[-1]), 4),
                })

        return result

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence speed."""
        result = {}

        for key in self.data:
            if "mAP" not in key:
                continue
            values = [v for v in self.data[key] if isinstance(v, (int, float))]
            if len(values) < 5:
                continue

            arr = np.array(values)
            max_val = np.max(arr)
            threshold_90 = max_val * 0.9
            threshold_95 = max_val * 0.95

            epoch_90 = int(np.argmax(arr >= threshold_90)) if np.any(arr >= threshold_90) else -1
            epoch_95 = int(np.argmax(arr >= threshold_95)) if np.any(arr >= threshold_95) else -1

            result[key] = {
                "max_value": round(float(max_val), 4),
                "max_epoch": int(np.argmax(arr)),
                "converge_90_epoch": epoch_90,
                "converge_95_epoch": epoch_95,
            }

        return result

    def _analyze_lr(self) -> Dict[str, Any]:
        """Analyze learning rate schedule."""
        lr_keys = [k for k in self.data if "lr" in k.lower() or "learning_rate" in k.lower()]
        result = {}

        for key in lr_keys:
            values = [v for v in self.data[key] if isinstance(v, (int, float))]
            if values:
                result[key] = {
                    "initial": round(float(values[0]), 8),
                    "final": round(float(values[-1]), 8),
                    "max": round(float(max(values)), 8),
                    "min": round(float(min(values)), 8),
                }

        return result

    def plot_curves(self, output_dir: Union[str, Path], dpi: int = 150) -> List[str]:
        """
        Generate training curve plots.

        Args:
            output_dir: Directory to save plots
            dpi: Image DPI

        Returns:
            List of saved plot paths
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not installed")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        if not self._parsed:
            self.parse()

        # Plot loss curves
        loss_keys = [k for k in self.data if "loss" in k.lower()]
        if loss_keys:
            fig, ax = plt.subplots(figsize=(12, 6))
            for key in loss_keys:
                values = [v for v in self.data[key] if isinstance(v, (int, float))]
                ax.plot(values, label=key.strip(), linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training & Validation Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = output_dir / "loss_curves.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(path))

        # Plot mAP curves
        map_keys = [k for k in self.data if "map" in k.lower()]
        if map_keys:
            fig, ax = plt.subplots(figsize=(12, 6))
            for key in map_keys:
                values = [v for v in self.data[key] if isinstance(v, (int, float))]
                ax.plot(values, label=key.strip(), linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("mAP")
            ax.set_title("Validation mAP")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = output_dir / "map_curves.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(path))

        # Plot learning rate
        lr_keys = [k for k in self.data if "lr" in k.lower()]
        if lr_keys:
            fig, ax = plt.subplots(figsize=(12, 4))
            for key in lr_keys:
                values = [v for v in self.data[key] if isinstance(v, (int, float))]
                ax.plot(values, label=key.strip(), linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = output_dir / "lr_schedule.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(path))

        print(f"📊 Plots saved to: {output_dir} ({len(saved)} files)")
        return saved

    def export_report(self, output_path: Union[str, Path]) -> str:
        """Export analysis report as JSON."""
        report = self.analyze()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"📄 Report saved to: {output_path}")
        return str(output_path)

    def print_summary(self) -> None:
        """Print a formatted summary of the analysis."""
        report = self.analyze()

        print("\n" + "=" * 60)
        print("📊 TRAINING LOG ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"  Total Epochs: {report['total_epochs']}")

        if report["best_metrics"]:
            print("\n  Best Metrics:")
            for key, info in report["best_metrics"].items():
                print(f"    {key}: {info['best']:.4f} (epoch {info['epoch']})")

        if report["overfitting_analysis"]["detected"]:
            print("\n  ⚠️ OVERFITTING DETECTED:")
            for detail in report["overfitting_analysis"]["details"]:
                print(f"    {detail['loss_type']}: train↓ val↑ (gap={detail['gap']})")
        else:
            print("\n  ✅ No overfitting detected")

        if report["convergence_analysis"]:
            print("\n  Convergence:")
            for key, info in report["convergence_analysis"].items():
                print(f"    {key}: max={info['max_value']:.4f} at epoch {info['max_epoch']}")

        print("=" * 60 + "\n")
