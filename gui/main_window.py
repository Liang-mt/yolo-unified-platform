"""
YOLO Unified Platform - PySide6 Main Window
Layout reference: YOLOv8-PySide6-GUI-main
"""

import os
import sys
import json
import time
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QSlider, QProgressBar, QFileDialog, QTextEdit, QLineEdit,
    QSplitter, QTabWidget, QCheckBox, QScrollArea, QGroupBox,
    QSpacerItem, QSizePolicy, QGridLayout, QApplication,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QPropertyAnimation,
    QEasingCurve, QParallelAnimationGroup,
)
from PySide6.QtGui import (
    QColor, QImage, QPixmap, QIcon,
)
from PySide6.QtWidgets import QGraphicsDropShadowEffect

from .styles import (
    MAIN_STYLE, MENU_BUTTON, CONTENT_BUTTON, SETTINGS_BUTTON,
    SLIDER_STYLE, SPINBOX_STYLE, COMBOBOX_STYLE, PROGRESS_BAR,
    PARAM_FRAME, TEXTEDIT_STYLE, LINEEDIT_STYLE, TAB_STYLE,
)
from .workers import InferenceWorker, TrainWorker, VideoDetectWorker, ExportWorker, BenchmarkWorker
from . import settings


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Unified Training Platform")
        self.resize(1280, 800)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.dragPos = None

        # Model cache
        self._model_cache = {}
        self._models_dir = Path(__file__).parent.parent / "models"
        self._models_dir.mkdir(exist_ok=True)

        self._setup_ui()
        self._connect_signals()
        self._refresh_models()
        self._refresh_yaml()

    # ════════════════════════════════════════════════════════════════
    #  UI Setup
    # ════════════════════════════════════════════════════════════════

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main frame with gradient background
        self.main_frame = QFrame()
        self.main_frame.setObjectName("Main_QF")
        self.main_frame.setStyleSheet(MAIN_STYLE)
        frame_layout = QHBoxLayout(self.main_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        # ── Left Menu ──────────────────────────────────────────────
        self._build_left_menu(frame_layout)

        # ── Content Area ───────────────────────────────────────────
        self._build_content(frame_layout)

        main_layout.addWidget(self.main_frame)

    def _build_left_menu(self, parent_layout):
        """Left sidebar with navigation buttons."""
        self.left_menu = QFrame()
        self.left_menu.setObjectName("LeftMenuBg")
        self.left_menu.setMinimumWidth(68)
        self.left_menu.setMaximumWidth(68)

        layout = QVBoxLayout(self.left_menu)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo area
        logo_frame = QFrame()
        logo_frame.setFixedHeight(70)
        logo_label = QLabel("YOLO")
        logo_label.setStyleSheet("font: 700 16pt 'Nirmala UI'; color: white !important;")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_layout = QVBoxLayout(logo_frame)
        logo_layout.addWidget(logo_label)
        layout.addWidget(logo_frame)

        # Toggle button
        self.toggle_btn = QPushButton("  Menu")
        self.toggle_btn.setFixedHeight(45)
        self.toggle_btn.setStyleSheet(MENU_BUTTON)
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.clicked.connect(self._toggle_menu)
        layout.addWidget(self.toggle_btn)

        # Navigation buttons
        self.nav_buttons = {}
        nav_items = [
            ("detect", "Detect", "Detection / Inference"),
            ("train", "Train", "Model Training"),
            ("export", "Export", "Model Export"),
            ("bench", "Bench", "Speed Benchmark"),
        ]

        for key, text, tooltip in nav_items:
            btn = QPushButton(f"  {text}")
            btn.setFixedHeight(45)
            btn.setStyleSheet(MENU_BUTTON)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            layout.addWidget(btn)
            self.nav_buttons[key] = btn

        # First button checked by default
        self.nav_buttons["detect"].setChecked(True)

        # Spacer
        layout.addStretch()

        # Version
        ver_label = QLabel("v2.0")
        ver_label.setStyleSheet("font: 900 italic 10pt 'Segoe UI'; color: rgba(255,255,255,199);")
        ver_label.setAlignment(Qt.AlignCenter)
        ver_label.setFixedHeight(20)
        layout.addWidget(ver_label)

        parent_layout.addWidget(self.left_menu)

    def _build_content(self, parent_layout):
        """Main content area."""
        content = QFrame()
        content.setObjectName("ContentBox")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # ── Title Bar ──────────────────────────────────────────────
        self._build_title_bar(content_layout)

        # ── Tab Pages ──────────────────────────────────────────────
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(TAB_STYLE)

        self._build_detect_page()
        self._build_train_page()
        self._build_export_page()
        self._build_bench_page()


        content_layout.addWidget(self.tab_widget, 1)

        # ── Bottom Status Bar ──────────────────────────────────────
        self._build_bottom_bar(content_layout)

        parent_layout.addWidget(content, 1)

    def _build_title_bar(self, parent_layout):
        """Custom title bar with window controls."""
        top = QFrame()
        top.setFixedHeight(35)
        top.setStyleSheet("QFrame { background: transparent; }")
        layout = QHBoxLayout(top)
        layout.setContentsMargins(20, 0, 10, 0)

        title = QLabel("YOLO Unified Training Platform")
        title.setStyleSheet("font: 700 italic 11pt 'Segoe UI'; color: white !important; background: qlineargradient(x0:0, y0:0, x1:1, y1:0, stop:0 rgb(107,128,210), stop:0.5 rgb(180,140,255), stop:1 rgb(180,140,255)); border-radius: 5px; padding: 2px 10px;")
        layout.addWidget(title)
        layout.addStretch()

        for color, tooltip in [
            ("rgb(4,180,0)", "Minimize"),
            ("rgb(227,199,0)", "Maximize"),
            ("rgb(240,108,96)", "Close"),
        ]:
            btn = QPushButton()
            btn.setFixedSize(14, 14)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border: 1px solid rgba(113,17,15,50);
                    border-radius: 7px;
                }}
                QPushButton:hover {{ background-color: rgb(139,29,31); }}
            """)
            layout.addWidget(btn)

            if tooltip == "Minimize":
                btn.clicked.connect(self.showMinimized)
            elif tooltip == "Maximize":
                btn.clicked.connect(self._max_restore)
                self.max_btn = btn
            else:
                btn.clicked.connect(self.close)

        parent_layout.addWidget(top)

    def _build_settings_panel(self):
        """Right settings panel (expandable)."""
        panel = QFrame()
        panel.setObjectName("prm_page")
        panel.setMinimumWidth(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(15, 15, 10, 15)
        layout.setSpacing(12)

        title = QLabel("Settings")
        title.setStyleSheet("color: white !important; font: 700 italic 16pt 'Segoe UI'; background: transparent; border: none;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Model selector
        self.model_box = QComboBox()
        self.model_box.setStyleSheet(COMBOBOX_STYLE)
        self.model_box.setFixedHeight(30)
        layout.addWidget(self._param_frame("Model", self.model_box))

        # Device selector
        self.device_box = QComboBox()
        self.device_box.addItems(["auto", "cpu", "0", "0,1"])
        self.device_box.setStyleSheet(COMBOBOX_STYLE)
        self.device_box.setFixedHeight(30)
        layout.addWidget(self._param_frame("Device", self.device_box))

        # IOU slider
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setStyleSheet(SPINBOX_STYLE)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.setStyleSheet(SLIDER_STYLE)
        self.iou_spin.valueChanged.connect(lambda v: self.iou_slider.setValue(int(v * 100)))
        self.iou_slider.valueChanged.connect(lambda v: self.iou_spin.setValue(v / 100))
        layout.addWidget(self._slider_frame("IOU", self.iou_spin, self.iou_slider))

        # Conf slider
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setStyleSheet(SPINBOX_STYLE)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(25)
        self.conf_slider.setStyleSheet(SLIDER_STYLE)
        self.conf_spin.valueChanged.connect(lambda v: self.conf_slider.setValue(int(v * 100)))
        self.conf_slider.valueChanged.connect(lambda v: self.conf_spin.setValue(v / 100))
        layout.addWidget(self._slider_frame("Conf", self.conf_spin, self.conf_slider))

        # Custom model file
        self.custom_model_btn = QPushButton("Browse .pt")
        self.custom_model_btn.setFixedHeight(32)
        self.custom_model_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 80);
                color: rgb(40, 40, 40) !important;
                border: 1px solid rgba(255, 255, 255, 60);
                border-radius: 10px;
                padding: 5px 14px;
                font: 700 9pt "Segoe UI";
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 120);
            }
        """)
        self.custom_model_btn.setCursor(Qt.PointingHandCursor)

        self.custom_model_label = QLabel("")
        self.custom_model_label.setStyleSheet(
            "color: rgb(40,40,40) !important; font: 700 9pt 'Segoe UI'; background: transparent; border: none;"
        )
        self.custom_model_label.setAlignment(Qt.AlignCenter)
        self.custom_model_label.setWordWrap(True)
        self.custom_model_label.setMinimumHeight(22)

        custom_inner = QWidget()
        custom_layout = QVBoxLayout(custom_inner)
        custom_layout.setContentsMargins(10, 10, 10, 10)
        custom_layout.setSpacing(14)
        custom_layout.addWidget(self.custom_model_btn, 0, Qt.AlignHCenter)
        custom_layout.addWidget(self.custom_model_label, 0, Qt.AlignHCenter)

        custom_frame = self._param_frame("Custom", custom_inner, dark=True)
        custom_frame.setMinimumHeight(110)
        layout.addWidget(custom_frame)

        # Track custom model path
        self._custom_model_path = None

        layout.addStretch()
        scroll.setWidget(inner)

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)

        return panel

    # ── Detect Page ────────────────────────────────────────────────

    def _build_detect_page(self):
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(10, 5, 10, 5)
        outer.setSpacing(10)

        # Left: images + controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Image pair
        img_row = QHBoxLayout()
        img_row.setSpacing(10)

        self.detect_input = QLabel("Input Image")
        self.detect_input.setStyleSheet("background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1, stop:0 rgb(228,232,250), stop:1 rgb(238,242,255)); border: 1px solid rgba(200,205,220,120); border-radius: 15px; color: rgb(100,100,130); font: 14pt 'Segoe UI';")
        self.detect_input.setAlignment(Qt.AlignCenter)
        self.detect_input.setMinimumSize(200, 200)

        self.detect_output = QLabel("Detection Result")
        self.detect_output.setStyleSheet("background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1, stop:0 rgb(228,232,250), stop:1 rgb(238,242,255)); border: 1px solid rgba(200,205,220,120); border-radius: 15px; color: rgb(100,100,130); font: 14pt 'Segoe UI';")
        self.detect_output.setAlignment(Qt.AlignCenter)
        self.detect_output.setMinimumSize(200, 200)

        img_row.addWidget(self.detect_input)
        img_row.addWidget(self.detect_output)
        left_layout.addLayout(img_row, 1)

        # Control bar - Row 1: buttons
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.detect_browse_btn = QPushButton("Image")
        self.detect_browse_btn.setFixedHeight(35)
        self.detect_browse_btn.setStyleSheet(CONTENT_BUTTON)
        self.detect_browse_btn.setCursor(Qt.PointingHandCursor)
        ctrl_row.addWidget(self.detect_browse_btn)

        self.detect_video_btn = QPushButton("Video")
        self.detect_video_btn.setFixedHeight(35)
        self.detect_video_btn.setStyleSheet(CONTENT_BUTTON)
        self.detect_video_btn.setCursor(Qt.PointingHandCursor)
        ctrl_row.addWidget(self.detect_video_btn)

        self.detect_run_btn = QPushButton("Run Detect")
        self.detect_run_btn.setFixedHeight(35)
        self.detect_run_btn.setStyleSheet(CONTENT_BUTTON + "QPushButton { font: 700 11pt 'Segoe UI'; }")
        self.detect_run_btn.setCursor(Qt.PointingHandCursor)
        ctrl_row.addWidget(self.detect_run_btn)

        self.detect_pause_btn = QPushButton("Pause")
        self.detect_pause_btn.setFixedHeight(35)
        self.detect_pause_btn.setStyleSheet(CONTENT_BUTTON)
        self.detect_pause_btn.setCursor(Qt.PointingHandCursor)
        self.detect_pause_btn.setEnabled(False)
        ctrl_row.addWidget(self.detect_pause_btn)

        self.detect_stop_btn = QPushButton("Stop")
        self.detect_stop_btn.setFixedHeight(35)
        self.detect_stop_btn.setStyleSheet("QPushButton { background-color: rgb(220, 80, 80); color: white; border-radius: 8px; font: 700 11pt 'Segoe UI'; } QPushButton:hover { background-color: rgb(200, 60, 60); }")
        self.detect_stop_btn.setCursor(Qt.PointingHandCursor)
        self.detect_stop_btn.setEnabled(False)
        ctrl_row.addWidget(self.detect_stop_btn)

        ctrl_row.addStretch()
        left_layout.addLayout(ctrl_row)

        # Control bar - Row 2: info + progress
        info_row = QHBoxLayout()
        info_row.setSpacing(10)

        self.detect_info_label = QLabel("No media loaded")
        self.detect_info_label.setStyleSheet("color: rgb(50,50,50) !important; font: 10pt 'Segoe UI'; background: transparent;")
        info_row.addWidget(self.detect_info_label)

        self.detect_progress = QProgressBar()
        self.detect_progress.setMaximum(100)
        self.detect_progress.setValue(0)
        self.detect_progress.setFormat("%v%")
        self.detect_progress.setStyleSheet(PROGRESS_BAR)
        self.detect_progress.setVisible(False)
        info_row.addWidget(self.detect_progress)

        left_layout.addLayout(info_row)
        outer.addWidget(left, 1)

        # Right: settings panel
        self.settings_panel = self._build_settings_panel()
        self.settings_panel.setMinimumWidth(200)
        self.settings_panel.setMaximumWidth(280)
        outer.addWidget(self.settings_panel)

        self._detect_image = None
        self._detect_video_path = None
        self.tab_widget.addTab(page, "Detect")

    def _content_label(self, text):
        """Create a label for content area (dark text on light background)."""
        lbl = QLabel(text)
        lbl.setStyleSheet("color: rgb(50, 50, 50); font: 10pt 'Segoe UI'; background: transparent;")
        return lbl

    def _param_frame(self, label_text, widget, dark=True):
        """Create a labeled parameter frame."""
        frame = QFrame()
        frame.setMinimumHeight(65)
        frame.setMaximumHeight(65)
        frame.setStyleSheet(PARAM_FRAME)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignCenter)
        if dark:
            lbl.setStyleSheet("color: white !important; font: 700 12pt 'Nirmala UI'; border: none; background: transparent;")
        else:
            lbl.setStyleSheet("color: rgb(40,40,40) !important; font: 700 12pt 'Segoe UI'; border: none;")
        layout.addWidget(lbl)
        layout.addWidget(widget)
        return frame

    def _slider_frame(self, label_text, spinbox, slider, dark=True):
        """Create a slider+spinbox parameter frame."""
        frame = QFrame()
        frame.setMinimumHeight(75)
        frame.setMaximumHeight(75)
        frame.setStyleSheet(PARAM_FRAME)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 5, 10, 5)

        lbl = QLabel(label_text)
        if dark:
            lbl.setStyleSheet("color: white !important; font: 700 12pt 'Nirmala UI'; border: none; background: transparent;")
        else:
            lbl.setStyleSheet("color: rgb(40,40,40) !important; font: 700 12pt 'Segoe UI'; border: none;")
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.addWidget(spinbox)
        row.addWidget(slider)
        layout.addLayout(row)
        return frame

    # ── Train Page ─────────────────────────────────────────────────

    def _build_train_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(10, 5, 10, 5)

        # ── Left: controls ──
        left = QFrame()
        left.setMaximumWidth(350)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self.train_model_box = QComboBox()
        self.train_model_box.setMinimumWidth(180)
        self.train_model_box.setStyleSheet(COMBOBOX_STYLE)
        self.train_scale_box = QComboBox()
        self.train_scale_box.addItems(["n", "s", "m", "l", "x"])
        self.train_scale_box.setFixedWidth(50)
        self.train_scale_box.setStyleSheet(COMBOBOX_STYLE)
        self.train_scale_box.setToolTip("Model scale (only for .yaml)")
        self.train_custom_btn = QPushButton("Browse")
        self.train_custom_btn.setFixedHeight(30)
        self.train_custom_btn.setStyleSheet(CONTENT_BUTTON)
        self.train_custom_btn.setCursor(Qt.PointingHandCursor)
        train_browse_row = QHBoxLayout()
        train_browse_row.setSpacing(4)
        train_browse_row.addWidget(self.train_model_box, 1)
        train_browse_row.addWidget(self.train_scale_box)
        train_browse_row.addWidget(self.train_custom_btn)
        left_layout.addLayout(train_browse_row)

        self.train_yaml_box = QComboBox()
        self.train_yaml_box.setEditable(True)
        self.train_yaml_box.setStyleSheet(COMBOBOX_STYLE)
        left_layout.addWidget(self._param_frame("Dataset YAML", self.train_yaml_box, dark=False))

        yaml_row = QHBoxLayout()
        yaml_row.setSpacing(5)

        self.train_yaml_refresh = QPushButton("Refresh")
        self.train_yaml_refresh.setFixedHeight(28)
        self.train_yaml_refresh.setStyleSheet(CONTENT_BUTTON)
        self.train_yaml_refresh.setCursor(Qt.PointingHandCursor)
        yaml_row.addWidget(self.train_yaml_refresh)

        self.train_yaml_browse = QPushButton("Browse .yaml")
        self.train_yaml_browse.setFixedHeight(28)
        self.train_yaml_browse.setStyleSheet(CONTENT_BUTTON)
        self.train_yaml_browse.setCursor(Qt.PointingHandCursor)
        yaml_row.addWidget(self.train_yaml_browse)

        left_layout.addLayout(yaml_row)

        # 基本参数 2×3 网格
        basic_grid = QGridLayout()
        basic_grid.setSpacing(6)
        basic_grid.setColumnStretch(0, 1)
        basic_grid.setColumnStretch(1, 1)

        self.train_epochs = QSpinBox()
        self.train_epochs.setRange(1, 9999)
        self.train_epochs.setValue(100)
        self.train_epochs.setStyleSheet(SPINBOX_STYLE)
        basic_grid.addWidget(self._param_frame("Epochs", self.train_epochs, dark=False), 0, 0)

        self.train_batch = QSpinBox()
        self.train_batch.setRange(1, 256)
        self.train_batch.setValue(16)
        self.train_batch.setStyleSheet(SPINBOX_STYLE)
        basic_grid.addWidget(self._param_frame("Batch Size", self.train_batch, dark=False), 0, 1)

        self.train_imgsz = QSpinBox()
        self.train_imgsz.setRange(32, 2048)
        self.train_imgsz.setValue(640)
        self.train_imgsz.setSingleStep(32)
        self.train_imgsz.setStyleSheet(SPINBOX_STYLE)
        basic_grid.addWidget(self._param_frame("Image Size", self.train_imgsz, dark=False), 1, 0)

        self.train_lr = QDoubleSpinBox()
        self.train_lr.setRange(0.0001, 1.0)
        self.train_lr.setValue(0.01)
        self.train_lr.setSingleStep(0.001)
        self.train_lr.setDecimals(4)
        self.train_lr.setStyleSheet(SPINBOX_STYLE)
        basic_grid.addWidget(self._param_frame("Learning Rate (lr0)", self.train_lr, dark=False), 1, 1)

        self.train_device = QLineEdit("0")
        self.train_device.setStyleSheet(LINEEDIT_STYLE)
        basic_grid.addWidget(self._param_frame("GPU Device", self.train_device, dark=False), 2, 0)

        self.train_optimizer = QComboBox()
        self.train_optimizer.addItems(["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
        self.train_optimizer.setStyleSheet(COMBOBOX_STYLE)
        basic_grid.addWidget(self._param_frame("Optimizer", self.train_optimizer, dark=False), 2, 1)

        left_layout.addLayout(basic_grid)

        # 超参数 2×2 网格
        hyp_grid = QGridLayout()
        hyp_grid.setSpacing(6)
        hyp_grid.setColumnStretch(0, 1)
        hyp_grid.setColumnStretch(1, 1)

        self.train_lrf = QDoubleSpinBox()
        self.train_lrf.setRange(0.0001, 1.0)
        self.train_lrf.setValue(0.01)
        self.train_lrf.setSingleStep(0.001)
        self.train_lrf.setDecimals(4)
        self.train_lrf.setStyleSheet(SPINBOX_STYLE)
        self.train_lrf.setToolTip("Final LR = lr0 × lrf")
        hyp_grid.addWidget(self._param_frame("LR Factor (lrf)", self.train_lrf, dark=False), 0, 0)

        self.train_momentum = QDoubleSpinBox()
        self.train_momentum.setRange(0.0, 1.0)
        self.train_momentum.setValue(0.937)
        self.train_momentum.setSingleStep(0.001)
        self.train_momentum.setDecimals(3)
        self.train_momentum.setStyleSheet(SPINBOX_STYLE)
        hyp_grid.addWidget(self._param_frame("Momentum", self.train_momentum, dark=False), 0, 1)

        self.train_weight_decay = QDoubleSpinBox()
        self.train_weight_decay.setRange(0.0, 0.1)
        self.train_weight_decay.setValue(0.0005)
        self.train_weight_decay.setSingleStep(0.0001)
        self.train_weight_decay.setDecimals(4)
        self.train_weight_decay.setStyleSheet(SPINBOX_STYLE)
        hyp_grid.addWidget(self._param_frame("Weight Decay", self.train_weight_decay, dark=False), 1, 0)

        self.train_warmup_epochs = QDoubleSpinBox()
        self.train_warmup_epochs.setRange(0.0, 50.0)
        self.train_warmup_epochs.setValue(3.0)
        self.train_warmup_epochs.setSingleStep(0.5)
        self.train_warmup_epochs.setDecimals(1)
        self.train_warmup_epochs.setStyleSheet(SPINBOX_STYLE)
        hyp_grid.addWidget(self._param_frame("Warmup Epochs", self.train_warmup_epochs, dark=False), 1, 1)

        left_layout.addLayout(hyp_grid)

        check_row = QHBoxLayout()
        check_row.setSpacing(15)
        self.train_cache = QCheckBox("Cache Images")
        self.train_cache.setStyleSheet("QCheckBox { color: rgb(50, 50, 50) !important; font: 10pt 'Segoe UI'; }")
        check_row.addWidget(self.train_cache)
        self.train_resume = QCheckBox("Resume Training")
        self.train_resume.setStyleSheet("QCheckBox { color: rgb(50, 50, 50) !important; font: 10pt 'Segoe UI'; }")
        check_row.addWidget(self.train_resume)
        check_row.addStretch()
        left_layout.addLayout(check_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(5)

        self.train_btn = QPushButton("Start Training")
        self.train_btn.setFixedHeight(35)
        self.train_btn.setStyleSheet(CONTENT_BUTTON + "QPushButton { font: 700 12pt 'Segoe UI'; }")
        self.train_btn.setCursor(Qt.PointingHandCursor)
        btn_row.addWidget(self.train_btn)

        self.train_stop_btn = QPushButton("Force Stop")
        self.train_stop_btn.setFixedHeight(35)
        self.train_stop_btn.setStyleSheet("QPushButton { background-color: rgb(220, 80, 80); color: white; border-radius: 8px; font: 700 12pt 'Segoe UI'; } QPushButton:hover { background-color: rgb(200, 60, 60); }")
        self.train_stop_btn.setCursor(Qt.PointingHandCursor)
        self.train_stop_btn.setEnabled(False)
        btn_row.addWidget(self.train_stop_btn)

        left_layout.addLayout(btn_row)
        left_layout.addStretch()

        # ── Right: charts + info ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        # Charts area (2x2 grid)
        self._build_train_charts(right_layout)

        # Bottom info bar
        self._build_train_info_bar(right_layout)

        right_layout.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([350, 700])
        layout.addWidget(splitter)

        # Training metrics storage
        self._train_epochs_data = []
        self._train_box_losses = []
        self._train_cls_losses = []
        self._train_dfl_losses = []
        self._train_lr_values = []
        self._train_lr_epochs = []
        self._train_map50_values = []
        self._train_map5095_values = []
        self._train_map_epochs = []
        self._train_precision_values = []
        self._train_recall_values = []
        self._train_pr_epochs = []
        self._train_pending_metrics = []
        self._train_custom_model_path = None
        self._export_custom_model_path = None
        self._bench_custom_model_path = None

        # Timer for chart updates (runs in main thread)
        self._chart_timer = QTimer()
        self._chart_timer.timeout.connect(self._flush_charts)
        self._chart_timer.setInterval(500)  # 500ms refresh

        self.tab_widget.addTab(page, "Train")

    def _build_train_charts(self, parent_layout):
        """Create 2x2 chart grid for training metrics."""
        self._chart_figures = {}
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure

            grid = QGridLayout()
            grid.setSpacing(5)

            self._chart_figures = {}
            charts = [
                ("box_loss", "box_loss", (0, 0)),
                ("cls_loss", "cls_loss", (0, 1)),
                ("dfl_loss", "dfl_loss", (0, 2)),
                ("lr", "learning rate", (1, 0)),
                ("map", "mAP", (1, 1)),
                ("precision", "precision", (1, 2)),
            ]

            for key, title, (row, col) in charts:
                fig = Figure(figsize=(3.5, 2.2), dpi=80)
                fig.patch.set_facecolor((0.925, 0.937, 0.98))
                ax = fig.add_subplot(111)
                ax.set_title(title, fontsize=9, fontweight='bold', color=(0.235, 0.235, 0.314))
                ax.set_facecolor((0.961, 0.976, 0.996))
                ax.tick_params(labelsize=7)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(True, alpha=0.3)
                self._chart_figures[key] = (fig, ax, [])

                canvas = FigureCanvas(fig)
                canvas.setStyleSheet("background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1, stop:0 rgb(228,232,250), stop:1 rgb(238,242,255)); border-radius: 8px;")
                grid.addWidget(canvas, row, col)

            right_widget = QWidget()
            right_widget.setLayout(grid)
            right_widget.setStyleSheet("background: transparent;")
            parent_layout.addWidget(right_widget, 1)

        except ImportError:
            fallback = QLabel("matplotlib not installed\npip install matplotlib")
            fallback.setStyleSheet("color: rgb(150,150,150); font: 12pt 'Segoe UI'; padding: 40px;")
            fallback.setAlignment(Qt.AlignCenter)
            parent_layout.addWidget(fallback)

    def _build_train_info_bar(self, parent_layout):
        """Bottom info bar: epoch progress, total progress, time, GPU."""
        info_frame = QFrame()
        info_frame.setFixedHeight(60)
        info_frame.setStyleSheet("QFrame { background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1, stop:0 rgb(228,232,250), stop:1 rgb(238,242,255)); border-radius: 10px; }")
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(15, 5, 15, 5)

        # Epoch progress
        self.train_epoch_progress = QProgressBar()
        self.train_epoch_progress.setMaximum(100)
        self.train_epoch_progress.setValue(0)
        self.train_epoch_progress.setFormat("Epoch: %v%")
        self.train_epoch_progress.setStyleSheet(PROGRESS_BAR)
        info_layout.addWidget(self.train_epoch_progress)


        # Time info
        for label_text in ["Elapsed", "ETA", "Remaining"]:
            box = QVBoxLayout()
            box.setSpacing(0)
            lbl = QLabel(label_text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: rgb(80,80,100); font: 8pt 'Segoe UI'; background: transparent; border: none;")
            val = QLabel("--")
            val.setAlignment(Qt.AlignCenter)
            val.setStyleSheet("color: rgb(40,40,50); font: 700 10pt 'Segoe UI'; background: transparent; border: none;")
            box.addWidget(lbl)
            box.addWidget(val)
            info_layout.addLayout(box)

            if label_text == "Elapsed":
                self.train_time_elapsed = val
            elif label_text == "ETA":
                self.train_time_eta = val
            else:
                self.train_time_remaining = val

        # GPU info
        gpu_box = QVBoxLayout()
        gpu_box.setSpacing(0)
        gpu_lbl = QLabel("GPU")
        gpu_lbl.setAlignment(Qt.AlignCenter)
        gpu_lbl.setStyleSheet("color: rgb(80,80,100); font: 8pt 'Segoe UI'; background: transparent; border: none;")
        self.train_gpu_info = QLabel("--")
        self.train_gpu_info.setAlignment(Qt.AlignCenter)
        self.train_gpu_info.setStyleSheet("color: rgb(40,40,50); font: 700 10pt 'Segoe UI'; background: transparent; border: none;")
        gpu_box.addWidget(gpu_lbl)
        gpu_box.addWidget(self.train_gpu_info)
        info_layout.addLayout(gpu_box)

        # Status
        status_box = QVBoxLayout()
        status_box.setSpacing(0)
        status_lbl = QLabel("Status")
        status_lbl.setAlignment(Qt.AlignCenter)
        status_lbl.setStyleSheet("color: rgb(80,80,100); font: 8pt 'Segoe UI'; background: transparent; border: none;")
        self.train_status_label = QLabel("Ready")
        self.train_status_label.setAlignment(Qt.AlignCenter)
        self.train_status_label.setStyleSheet("color: rgb(40,40,50); font: 700 10pt 'Segoe UI'; background: transparent; border: none;")
        status_box.addWidget(status_lbl)
        status_box.addWidget(self.train_status_label)
        info_layout.addLayout(status_box)

        parent_layout.addWidget(info_frame)

    # ── Export Page ────────────────────────────────────────────────

    def _build_export_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(10, 5, 10, 5)

        left = QFrame()
        left.setMaximumWidth(350)
        left_layout = QVBoxLayout(left)

        self.export_model_box = QComboBox()
        self.export_model_box.setMinimumWidth(220)
        self.export_model_box.setStyleSheet(COMBOBOX_STYLE)
        self.export_custom_btn = QPushButton("Browse")
        self.export_custom_btn.setFixedHeight(30)
        self.export_custom_btn.setStyleSheet(CONTENT_BUTTON)
        self.export_custom_btn.setCursor(Qt.PointingHandCursor)
        export_browse_row = QHBoxLayout()
        export_browse_row.setSpacing(6)
        export_browse_row.addWidget(self.export_model_box, 1)
        export_browse_row.addWidget(self.export_custom_btn)
        left_layout.addLayout(export_browse_row)

        self.export_format = QComboBox()
        self.export_format.addItems(["onnx", "torchscript", "engine", "tflite", "coreml", "paddle"])
        self.export_format.setStyleSheet(COMBOBOX_STYLE)
        left_layout.addWidget(self._param_frame("Format", self.export_format, dark=False))

        self.export_imgsz = QSpinBox()
        self.export_imgsz.setRange(32, 2048)
        self.export_imgsz.setValue(640)
        self.export_imgsz.setSingleStep(32)
        self.export_imgsz.setStyleSheet(SPINBOX_STYLE)
        left_layout.addWidget(self._param_frame("Image Size", self.export_imgsz, dark=False))

        self.export_opset = QSpinBox()
        self.export_opset.setRange(7, 20)
        self.export_opset.setValue(17)
        self.export_opset.setStyleSheet(SPINBOX_STYLE)
        left_layout.addWidget(self._param_frame("ONNX Opset", self.export_opset, dark=False))

        self.export_half = QCheckBox("FP16 Half Precision")
        self.export_half.setStyleSheet("QCheckBox { color: rgb(50, 50, 50) !important; font: 10pt 'Segoe UI'; }")
        left_layout.addWidget(self.export_half)

        self.export_dynamic = QCheckBox("Dynamic Batch (ONNX)")
        self.export_dynamic.setStyleSheet("QCheckBox { color: rgb(50, 50, 50) !important; font: 10pt 'Segoe UI'; }")
        left_layout.addWidget(self.export_dynamic)

        self.export_simplify = QCheckBox("Simplify ONNX")
        self.export_simplify.setChecked(True)
        self.export_simplify.setStyleSheet("QCheckBox { color: rgb(50, 50, 50) !important; font: 10pt 'Segoe UI'; }")
        left_layout.addWidget(self.export_simplify)

        self.export_btn = QPushButton("Export Model")
        self.export_btn.setFixedHeight(40)
        self.export_btn.setStyleSheet(CONTENT_BUTTON + "QPushButton { font: 700 14pt 'Segoe UI'; }")
        self.export_btn.setCursor(Qt.PointingHandCursor)
        left_layout.addWidget(self.export_btn)

        left_layout.addStretch()

        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setStyleSheet(TEXTEDIT_STYLE)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.export_log)
        splitter.setSizes([350, 700])
        layout.addWidget(splitter)

        self.tab_widget.addTab(page, "Export")

    # ── Benchmark Page ─────────────────────────────────────────────

    def _build_bench_page(self):
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(10, 5, 10, 5)

        left = QFrame()
        left.setMaximumWidth(350)
        left_layout = QVBoxLayout(left)

        self.bench_model_box = QComboBox()
        self.bench_model_box.setMinimumWidth(220)
        self.bench_model_box.setStyleSheet(COMBOBOX_STYLE)
        self.bench_custom_btn = QPushButton("Browse")
        self.bench_custom_btn.setFixedHeight(30)
        self.bench_custom_btn.setStyleSheet(CONTENT_BUTTON)
        self.bench_custom_btn.setCursor(Qt.PointingHandCursor)
        bench_browse_row = QHBoxLayout()
        bench_browse_row.setSpacing(6)
        bench_browse_row.addWidget(self.bench_model_box, 1)
        bench_browse_row.addWidget(self.bench_custom_btn)
        left_layout.addLayout(bench_browse_row)

        self.bench_imgsz = QSpinBox()
        self.bench_imgsz.setRange(32, 2048)
        self.bench_imgsz.setValue(640)
        self.bench_imgsz.setSingleStep(32)
        self.bench_imgsz.setStyleSheet(SPINBOX_STYLE)
        left_layout.addWidget(self._param_frame("Image Size", self.bench_imgsz, dark=False))

        self.bench_runs = QSpinBox()
        self.bench_runs.setRange(10, 10000)
        self.bench_runs.setValue(100)
        self.bench_runs.setStyleSheet(SPINBOX_STYLE)
        left_layout.addWidget(self._param_frame("Runs", self.bench_runs, dark=False))

        self.bench_btn = QPushButton("Run Benchmark")
        self.bench_btn.setFixedHeight(40)
        self.bench_btn.setStyleSheet(CONTENT_BUTTON + "QPushButton { font: 700 14pt 'Segoe UI'; }")
        self.bench_btn.setCursor(Qt.PointingHandCursor)
        left_layout.addWidget(self.bench_btn)

        left_layout.addStretch()

        self.bench_log = QTextEdit()
        self.bench_log.setReadOnly(True)
        self.bench_log.setStyleSheet(TEXTEDIT_STYLE)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.bench_log)
        splitter.setSizes([300, 700])
        layout.addWidget(splitter)

        self.tab_widget.addTab(page, "Benchmark")

    # ── Bottom Bar ─────────────────────────────────────────────────

    def _build_bottom_bar(self, parent_layout):
        bottom = QFrame()
        bottom.setFixedHeight(35)
        layout = QHBoxLayout(bottom)
        layout.setContentsMargins(20, 2, 10, 4)

        self.status_label = QLabel("Welcome")
        self.status_label.setStyleSheet("font: 700 11pt 'Segoe UI'; color: rgba(0,0,0,140);")
        layout.addWidget(self.status_label)

        layout.addStretch()
        parent_layout.addWidget(bottom)

    # ════════════════════════════════════════════════════════════════
    #  Signals & Slots
    # ════════════════════════════════════════════════════════════════

    def _connect_signals(self):
        # Navigation
        for key, btn in self.nav_buttons.items():
            btn.clicked.connect(lambda checked, k=key: self._switch_page(k))

        # Detect
        self.custom_model_btn.clicked.connect(self._browse_model)
        self.model_box.currentTextChanged.connect(self._on_model_selected)
        self.detect_browse_btn.clicked.connect(self._browse_detect_image)
        self.detect_video_btn.clicked.connect(self._browse_detect_video)
        self.detect_run_btn.clicked.connect(self._run_detect)
        self.detect_pause_btn.clicked.connect(self._pause_detect)
        self.detect_stop_btn.clicked.connect(self._stop_detect)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Train
        self.train_btn.clicked.connect(self._run_train)
        self.train_stop_btn.clicked.connect(self._stop_train)
        self.train_yaml_browse.clicked.connect(self._browse_yaml)
        self.train_custom_btn.clicked.connect(self._browse_train_model)
        self.train_yaml_refresh.clicked.connect(self._refresh_yaml)
        self.train_model_box.currentTextChanged.connect(self._on_train_model_selected)

        # Export
        self.export_btn.clicked.connect(self._run_export)
        self.export_custom_btn.clicked.connect(self._browse_export_model)
        self.export_model_box.currentTextChanged.connect(self._on_export_model_selected)

        # Benchmark
        self.bench_btn.clicked.connect(self._run_benchmark)
        self.bench_custom_btn.clicked.connect(self._browse_bench_model)
        self.bench_model_box.currentTextChanged.connect(self._on_bench_model_selected)

    def _switch_page(self, key):
        page_map = {"detect": 0, "train": 1, "export": 2, "bench": 3}
        self.tab_widget.setCurrentIndex(page_map.get(key, 0))
        for k, btn in self.nav_buttons.items():
            btn.setChecked(k == key)

    def _on_tab_changed(self, index):
        keys = list(self.nav_buttons.keys())
        if index < len(keys):
            for k, btn in self.nav_buttons.items():
                btn.setChecked(k == keys[index])

    # ── Detect ─────────────────────────────────────────────────────

    def _browse_detect_image(self):
        """Select image file for detection."""
        # Stop any running video detection
        if hasattr(self, '_video_worker') and self._video_worker.isRunning():
            self._video_worker.stop()
            self._video_worker.wait(2000)
        self._detect_video_path = None
        self.detect_pause_btn.setEnabled(False)
        self.detect_stop_btn.setEnabled(False)
        self.detect_progress.setVisible(False)

        import cv2
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if f:
            img = cv2.imread(f)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._detect_image = img_rgb
                self._show_image(img_rgb, self.detect_input)
                self.detect_output.clear()
                self.detect_output.setText("Detection Result")
                self.detect_info_label.setText(f"Loaded: {Path(f).name} ({img.shape[1]}x{img.shape[0]})")
                self.status_label.setText(f"Image loaded: {Path(f).name}")

    def _on_detect_result(self, annotated_img, info):
        """Handle detection result."""
        self._show_image(annotated_img, self.detect_output)
        count = info.get("count", 0)
        self.detect_info_label.setText(f"Detected: {count} objects")
        self.detect_run_btn.setEnabled(True)
        self.status_label.setText("Detection complete")

    def _on_detect_error(self, error):
        """Handle detection error."""
        self.detect_info_label.setText(f"Error: {error[:50]}")
        self.detect_run_btn.setEnabled(True)
        self.status_label.setText(f"Error: {error[:80]}")

    def _browse_detect_video(self):
        """Select video file for detection."""
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.flv);;All Files (*)"
        )
        if f:
            import cv2
            cap = cv2.VideoCapture(f)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self._detect_video_path = f
                    self._detect_image = None
                    self._show_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), self.detect_input)
                    self.detect_output.clear()
                    self.detect_output.setText("Detection Result")
                    self.detect_info_label.setText(f"Video: {Path(f).name} ({total} frames, {fps:.1f}fps)")
                    self.status_label.setText(f"Video loaded: {Path(f).name}")
            else:
                self.status_label.setText(f"Cannot open video: {Path(f).name}")

    def _run_detect(self):
        """Run detection on loaded image or video."""
        model_path = self._get_model_path(self.model_box, self._custom_model_path)
        if not model_path:
            self.status_label.setText("Please select a model")
            return

        from ultralytics import YOLO
        dev = self.device_box.currentText()
        device = None if dev == "auto" else dev
        model = self._load_model(model_path, device)

        # Video detection
        if hasattr(self, '_detect_video_path') and self._detect_video_path:
            self.detect_run_btn.setEnabled(False)
            self.detect_pause_btn.setEnabled(True)
            self.detect_stop_btn.setEnabled(True)
            self.detect_progress.setVisible(True)
            self.detect_progress.setValue(0)
            self.detect_progress.setFormat("%v%")
            self.status_label.setText("Video detecting...")

            self._video_worker = VideoDetectWorker()
            self._video_worker.model = model
            self._video_worker.video_path = self._detect_video_path
            self._video_worker.conf = self.conf_spin.value()
            self._video_worker.sync_fps = settings.VIDEO_SYNC_FPS
            self._video_worker.iou = self.iou_spin.value()
            self._video_worker.device = device

            self._video_worker.frame_ready.connect(self._on_video_frame)
            self._video_worker.video_finished.connect(self._on_video_done)
            self._video_worker.video_error.connect(self._on_video_error)
            self._video_worker.start()
            return

        # Image detection
        if self._detect_image is None:
            self.status_label.setText("Please load an image or video first")
            return

        self.detect_run_btn.setEnabled(False)
        self.detect_info_label.setText("Detecting...")
        self.status_label.setText("Detecting...")

        try:
            self._detect_worker = InferenceWorker()
            self._detect_worker.model = model
            self._detect_worker.image = self._detect_image.copy()
            self._detect_worker.conf = self.conf_spin.value()
            self._detect_worker.iou = self.iou_spin.value()
            self._detect_worker.device = device

            self._detect_worker.result_ready.connect(self._on_detect_result)
            self._detect_worker.error_occurred.connect(self._on_detect_error)
            self._detect_worker.start()
        except Exception as e:
            self.detect_info_label.setText(f"Error: {str(e)[:50]}")
            self.detect_run_btn.setEnabled(True)
            self.status_label.setText(f"Error: {str(e)[:80]}")

    def _on_video_frame(self, input_frame, output_frame, current, total):
        """Handle video frame: show input on left, output on right."""
        self._show_image(input_frame, self.detect_input)
        self._show_image(output_frame, self.detect_output)
        self.detect_progress.setValue(int(current / max(total, 1) * 100))
        self.detect_info_label.setText(f"Frame {current}/{total}")

    def _on_video_done(self, msg):
        """Handle video detection completion."""
        self.detect_run_btn.setEnabled(True)
        self.detect_pause_btn.setEnabled(False)
        self.detect_stop_btn.setEnabled(False)
        self.detect_progress.setValue(100)
        self.detect_info_label.setText(msg)
        self.status_label.setText(msg)

    def _on_video_error(self, error):
        """Handle video detection error."""
        self.detect_info_label.setText(f"Error: {error[:50]}")
        self.detect_run_btn.setEnabled(True)
        self.detect_pause_btn.setEnabled(False)
        self.detect_stop_btn.setEnabled(False)
        self.detect_progress.setVisible(False)
        self.status_label.setText(f"Error: {error[:80]}")

    def _pause_detect(self):
        """Pause video detection."""
        if hasattr(self, '_video_worker') and self._video_worker.isRunning():
            if self._video_worker.paused:
                self._video_worker.resume()
                self.detect_pause_btn.setText("Pause")
                self.status_label.setText("Detecting...")
            else:
                self._video_worker.pause()
                self.detect_pause_btn.setText("Resume")
                self.status_label.setText("Paused")

    def _stop_detect(self):
        """Stop video detection."""
        if hasattr(self, '_video_worker') and self._video_worker.isRunning():
            self._video_worker.stop()
            self.status_label.setText("Stopping...")

    @staticmethod
    def _show_image(img_rgb, label):
        """Display RGB image on a QLabel."""
        import cv2
        import numpy as np
        h, w = img_rgb.shape[:2]
        label_w = label.width()
        label_h = label.height()
        # Use minimum size if label hasn't been rendered yet
        if label_w < 10:
            label_w = max(label.minimumWidth(), 300)
        if label_h < 10:
            label_h = max(label.minimumHeight(), 200)
        if w / label_w > h / label_h:
            nw = label_w
            nh = int(h * label_w / w)
        else:
            nh = label_h
            nw = int(w * label_h / h)
        nw = max(nw, 1)
        nh = max(nh, 1)
        img_resized = cv2.resize(img_rgb, (nw, nh))
        h2, w2 = img_resized.shape[:2]
        # Ensure data is contiguous
        img_resized = np.ascontiguousarray(img_resized)
        qimg = QImage(img_resized.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg))
        label.update()

    # ── Menu Animation ─────────────────────────────────────────────

    def _toggle_menu(self):
        width = self.left_menu.width()
        target = 180 if width < 100 else 68

        # Min width animation
        self._anim_min = QPropertyAnimation(self.left_menu, b"minimumWidth")
        self._anim_min.setDuration(150)
        self._anim_min.setStartValue(width)
        self._anim_min.setEndValue(target)
        self._anim_min.setEasingCurve(QEasingCurve.OutQuad)

        # Max width animation
        self._anim_max = QPropertyAnimation(self.left_menu, b"maximumWidth")
        self._anim_max.setDuration(150)
        self._anim_max.setStartValue(width)
        self._anim_max.setEndValue(target)
        self._anim_max.setEasingCurve(QEasingCurve.OutQuad)

        self._anim_min.start()
        self._anim_max.start()

    def _max_restore(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # ── Model Management ───────────────────────────────────────────

    def _refresh_models(self):
        models = sorted([f.name for f in self._models_dir.glob("*.pt")])
        if not models:
            models = ["yolov5n.pt", "yolov8n.pt", "yolo11n.pt", "yolo26n.pt"]

        for box in [self.model_box, self.train_model_box, self.export_model_box, self.bench_model_box]:
            current = box.currentText()
            box.clear()
            box.addItems(models)
            if current in models:
                box.setCurrentText(current)

    def _refresh_yaml(self):
        datasets_dir = Path(__file__).parent.parent / "datasets"
        datasets_dir.mkdir(exist_ok=True)
        yamls = [str(f) for f in sorted(datasets_dir.rglob("*.yaml"))]
        yamls += [str(f) for f in sorted(datasets_dir.rglob("*.yml"))]
        # Add coco128.yaml as default if not already in list
        coco128 = "C:/Users/mc/Desktop/yolov8_1_0/coco128.yaml"
        if coco128 not in yamls:
            yamls.insert(0, coco128)
        current = self.train_yaml_box.currentText()
        self.train_yaml_box.clear()
        self.train_yaml_box.addItems(yamls)
        if current:
            self.train_yaml_box.setCurrentText(current)
        else:
            self.train_yaml_box.setCurrentText(coco128)

    def _get_model_path(self, box=None, custom_path=None):
        """Get model path: custom file > combobox > default."""
        # Priority 1: custom file path
        if custom_path and Path(custom_path).exists():
            return custom_path
        # Priority 2: combobox selection
        if box:
            path = box.currentText()
            if not path:
                return None
            # Check local models dir
            full = self._models_dir / path
            if full.exists():
                return str(full)
            if Path(path).exists():
                return path
            # Return as-is (ultralytics handles .yaml lookup internally)
            return path
        return None

    def _set_custom_model(self, box, name):
        """Insert custom model name at top of combobox and select it."""
        self._remove_custom_model(box)
        box.insertItem(0, name)
        box.setCurrentIndex(0)

    def _remove_custom_model(self, box):
        """Remove custom model item (index 0) if it's not a built-in model."""
        if box.count() > 0:
            first = box.itemText(0)
            built_in = [box.itemText(i) for i in range(box.count())]
            # If first item is not in the built-in list (i.e. it was a custom insert)
            models_dir = self._models_dir
            if not (models_dir / first).exists():
                box.removeItem(0)

    def _resolve_model(self, box, custom_path, scale_box=None):
        """Resolve model path: custom > dropdown > None.
        If scale_box is provided and model is .yaml, append scale suffix."""
        if custom_path and Path(custom_path).exists():
            path = custom_path
        else:
            path = self._get_model_path(box)
        if path and scale_box and (path.endswith('.yaml') or path.endswith('.yml')):
            scale = scale_box.currentText()
            base = Path(path).stem  # e.g. "yolov8" from "yolov8.yaml"
            # Only add suffix if not already present
            if not base.endswith(scale):
                path = str(Path(path).parent / f"{base}{scale}.yaml")
        return path

    def _load_model(self, model_path, device=None):
        """Load and cache YOLO model."""
        from ultralytics import YOLO
        cache_key = (model_path, device)
        if cache_key not in self._model_cache:
            model = YOLO(model_path)
            if device:
                model.to(device)
            self._model_cache[cache_key] = model
        return self._model_cache[cache_key]

    # ── File Browsing ──────────────────────────────────────────────

    def _on_model_selected(self, text):
        """When user selects from dropdown, clear custom model."""
        if text:
            self._custom_model_path = None
            self.custom_model_label.setText("")

    def _on_train_model_selected(self, text):
        if text and self._train_custom_model_path:
            if (self._models_dir / text).exists():
                self._train_custom_model_path = None
                self._remove_custom_model(self.train_model_box)

    def _on_export_model_selected(self, text):
        if text and self._export_custom_model_path:
            if (self._models_dir / text).exists():
                self._export_custom_model_path = None
                self._remove_custom_model(self.export_model_box)

    def _on_bench_model_selected(self, text):
        if text and self._bench_custom_model_path:
            if (self._models_dir / text).exists():
                self._bench_custom_model_path = None
                self._remove_custom_model(self.bench_model_box)

    def _browse_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt *.yaml *.yml)")
        if f:
            name = Path(f).name
            self.custom_model_label.setText(name)
            self._custom_model_path = f
            self.model_box.setCurrentText("")  # clear dropdown to show custom is active
            self.status_label.setText(f"Custom model: {name}")

    def _browse_train_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt *.yaml *.yml)")
        if f:
            self._train_custom_model_path = f
            self._set_custom_model(self.train_model_box, Path(f).name)
            self.status_label.setText(f"Train model: {Path(f).name}")

    def _browse_export_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt *.yaml *.yml)")
        if f:
            self._export_custom_model_path = f
            self._set_custom_model(self.export_model_box, Path(f).name)
            self.status_label.setText(f"Export model: {Path(f).name}")

    def _browse_bench_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt *.yaml *.yml)")
        if f:
            self._bench_custom_model_path = f
            self._set_custom_model(self.bench_model_box, Path(f).name)
            self.status_label.setText(f"Bench model: {Path(f).name}")

    # ── Inference ──────────────────────────────────────────────────

    # (Detection is handled by showing images in labels - to be connected
    #  with the InferenceWorker when image source is selected)

    # ── Training ───────────────────────────────────────────────────

    def _browse_yaml(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset YAML", "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if f:
            self.train_yaml_box.setCurrentText(f)
            self.status_label.setText(f"Dataset: {Path(f).name}")

    def _run_train(self):
        model_path = self._resolve_model(self.train_model_box, self._train_custom_model_path, self.train_scale_box)
        data_path = self.train_yaml_box.currentText().strip()

        if not data_path:
            self.status_label.setText("Please provide dataset YAML path")
            return

        # Clean up previous training session
        if hasattr(self, '_train_worker') and self._train_worker is not None:
            if self._train_worker.isRunning():
                self._train_worker.force_stop()
                self._train_worker.wait(3000)
            # Disconnect signals to prevent stale references
            try:
                self._train_worker.epoch_done.disconnect()
                self._train_worker.train_finished.disconnect()
                self._train_worker.train_error.disconnect()
                self._train_worker.log_updated.disconnect()
            except RuntimeError:
                pass  # already disconnected
            self._train_worker.deleteLater()
            self._train_worker = None
        if hasattr(self, '_chart_timer'):
            self._chart_timer.stop()

        # Free GPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reset all data
        for attr in ['_train_epochs_data', '_train_box_losses', '_train_cls_losses',
                      '_train_dfl_losses', '_train_lr_values', '_train_lr_epochs',
                      '_train_map50_values', '_train_map5095_values', '_train_map_epochs',
                      '_train_precision_values', '_train_recall_values', '_train_pr_epochs',
                      '_train_pending_metrics']:
            getattr(self, attr).clear()

        # Reset UI
        self.train_btn.setEnabled(False)
        self.train_stop_btn.setEnabled(True)
        self.train_status_label.setText("Starting...")
        self.train_epoch_progress.setValue(0)
        self.train_epoch_progress.setFormat("Epoch 0/?")
        self.train_time_elapsed.setText("--")
        self.train_time_eta.setText("--")
        self.train_time_remaining.setText("--")
        self.train_gpu_info.setText("--")
        self.status_label.setText("Training...")

        # Reset charts
        titles = {"box_loss": "box_loss", "cls_loss": "cls_loss", "dfl_loss": "dfl_loss",
                  "lr": "learning rate", "map": "mAP", "precision": "precision"}
        for key in ["box_loss", "cls_loss", "dfl_loss", "lr", "map", "precision"]:
            if key in self._chart_figures:
                fig, ax, _ = self._chart_figures[key]
                ax.clear()
                ax.set_title(titles.get(key, key), fontsize=9, fontweight='bold', color=(0.235, 0.235, 0.314))
                ax.set_facecolor((0.961, 0.976, 0.996))
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3)
                fig.canvas.draw()

        # Free GPU memory from previous training
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._train_start_time = time.time()
        self._chart_timer.start()

        self._train_worker = TrainWorker()
        self._train_worker.model_path = model_path
        self._train_worker.data_path = data_path
        self._train_worker.epochs = self.train_epochs.value()
        self._train_worker.batch = self.train_batch.value()
        self._train_worker.imgsz = self.train_imgsz.value()
        self._train_worker.lr = self.train_lr.value()
        self._train_worker.lrf = self.train_lrf.value()
        self._train_worker.momentum = self.train_momentum.value()
        self._train_worker.weight_decay = self.train_weight_decay.value()
        self._train_worker.warmup_epochs = self.train_warmup_epochs.value()
        self._train_worker.device = self.train_device.text().strip()
        self._train_worker.optimizer = self.train_optimizer.currentText()
        self._train_worker.resume = self.train_resume.isChecked()
        self._train_worker.cache = self.train_cache.isChecked()

        self._train_worker.epoch_done.connect(self._on_train_metrics)
        self._train_worker.train_finished.connect(self._on_train_done)
        self._train_worker.train_error.connect(self._on_train_error)
        self._train_worker.start()

    def _on_train_metrics(self, metrics):
        """Buffer metrics for main-thread chart update."""
        self._train_pending_metrics.append(metrics)

    def _flush_charts(self):
        """Process buffered metrics and update charts (main thread)."""
        # Process pending metrics
        while self._train_pending_metrics:
            metrics = self._train_pending_metrics.pop(0)
            self._apply_metrics(metrics)
        # Always update time (every 500ms via timer)
        self._update_time_display()

    def _update_time_display(self):
        """Update elapsed/ETA/remaining time display."""
        if not hasattr(self, '_train_start_time'):
            return
        elapsed = time.time() - self._train_start_time
        self.train_time_elapsed.setText(self._format_time(elapsed))
        total_epochs = self.train_epochs.value()
        if self._train_epochs_data and self._train_epochs_data[-1] > 0:
            done = self._train_epochs_data[-1]
            speed = elapsed / done
            remaining = speed * (total_epochs - done)
            self.train_time_remaining.setText(self._format_time(remaining))
            self.train_time_eta.setText(self._format_time(elapsed + remaining))
        else:
            self.train_time_eta.setText("--")
            self.train_time_remaining.setText("--")

    def _apply_metrics(self, metrics):
        """Apply metrics to charts and info bar."""
        # Training loss
        if "box_loss" in metrics:
            epoch = metrics.get("epoch", 0)
            total = metrics.get("total", 100)
            box = metrics.get("box_loss", 0)
            cls = metrics.get("cls_loss", 0)
            dfl = metrics.get("dfl_loss", 0)

            if settings.LOSS_UPDATE_MODE == "batch":
                # Batch mode: update chart on every batch (real-time, may jitter)
                if self._train_epochs_data and self._train_epochs_data[-1] == epoch:
                    self._train_box_losses[-1] = box
                    self._train_cls_losses[-1] = cls
                    self._train_dfl_losses[-1] = dfl
                else:
                    self._train_epochs_data.append(epoch)
                    self._train_box_losses.append(box)
                    self._train_cls_losses.append(cls)
                    self._train_dfl_losses.append(dfl)
                self._update_chart("box_loss", self._train_epochs_data, self._train_box_losses)
                self._update_chart("cls_loss", self._train_epochs_data, self._train_cls_losses)
                self._update_chart("dfl_loss", self._train_epochs_data, self._train_dfl_losses)
            else:
                # Epoch mode: buffer loss, plot only when validation arrives
                self._pending_box = box
                self._pending_cls = cls
                self._pending_dfl = dfl
                self._pending_epoch = epoch
                self._pending_total = total

        # Validation arrives = epoch complete — commit loss + val to charts
        if "map50" in metrics:
            epoch = metrics.get("epoch", 0)
            total = metrics.get("total", 100)

            # Commit loss (only in epoch mode; batch mode already plotted)
            if settings.LOSS_UPDATE_MODE == "epoch" and hasattr(self, '_pending_epoch') and self._pending_epoch == epoch:
                self._train_epochs_data.append(epoch)
                self._train_box_losses.append(self._pending_box)
                self._train_cls_losses.append(self._pending_cls)
                self._train_dfl_losses.append(self._pending_dfl)
                self._update_chart("box_loss", self._train_epochs_data, self._train_box_losses)
                self._update_chart("cls_loss", self._train_epochs_data, self._train_cls_losses)
                self._update_chart("dfl_loss", self._train_epochs_data, self._train_dfl_losses)

            # Commit validation data
            self._train_map_epochs.append(epoch)
            self._train_map50_values.append(metrics["map50"])
            self._train_map5095_values.append(metrics.get("map5095", 0))
            self._train_pr_epochs.append(epoch)
            self._train_precision_values.append(metrics.get("precision", 0))
            self._train_recall_values.append(metrics.get("recall", 0))
            self._update_chart("map", self._train_map_epochs,
                               self._train_map50_values, self._train_map5095_values)
            self._update_chart("precision", self._train_pr_epochs,
                               self._train_precision_values, self._train_recall_values)

            # Update progress
            self.train_epoch_progress.setValue(int(epoch / max(total, 1) * 100))
            self.train_epoch_progress.setFormat(f"Epoch {epoch}/{total}")

        # Learning rate
        if "lr" in metrics:
            self._train_lr_epochs.append(len(self._train_lr_values) + 1)
            self._train_lr_values.append(metrics["lr"])
            self._update_chart("lr", self._train_lr_epochs, self._train_lr_values)

        # GPU
        gpu_mem = metrics.get("gpu_mem")
        if gpu_mem:
            self.train_gpu_info.setText(f"{gpu_mem}G")

        # Time
        epoch = metrics.get("epoch", 0)
        elapsed = time.time() - self._train_start_time
        if epoch > 0:
            eta = elapsed / epoch * (self.train_epochs.value() - epoch)
            self.train_time_elapsed.setText(self._format_time(elapsed))
            self.train_time_remaining.setText(self._format_time(eta))

        self.train_status_label.setText(f"Epoch {epoch}/{metrics.get('total', '?')}")

    def _update_chart(self, key, x_data, y_data, y_data2=None):
        if key not in self._chart_figures or not x_data:
            return
        fig, ax, _ = self._chart_figures[key]
        ax.clear()

        labels = {"map": ("mAP50", "mAP50-95"), "precision": ("Precision", "Recall")}
        if y_data2 is not None and key in labels:
            ax.plot(x_data, y_data, linewidth=1.5, label=labels[key][0])
            ax.plot(x_data, y_data2, linewidth=1.5, label=labels[key][1])
            ax.legend(fontsize=7, loc='upper left')
        else:
            ax.plot(x_data, y_data, linewidth=1.5, label=key)

        ax.set_title(key, fontsize=9, fontweight='bold', color=(0.235, 0.235, 0.314))
        ax.set_facecolor((0.961, 0.976, 0.996))
        ax.tick_params(labelsize=7)
        ax.set_xlabel("epoch", fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.canvas.draw()
        fig.canvas.flush_events()

    @staticmethod
    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _on_train_done(self, summary):
        self._chart_timer.stop()
        self._flush_charts()
        self.train_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.train_status_label.setText("Training Complete")
        self.train_epoch_progress.setValue(100)
        self.train_time_remaining.setText("00:00:00")
        self.status_label.setText("Training Complete")
        # Free GPU memory after training
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _on_train_error(self, err):
        self._chart_timer.stop()
        self.train_btn.setEnabled(True)
        self.train_stop_btn.setEnabled(False)
        self.train_status_label.setText("Training Failed")
        # Free GPU memory on error
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Show full error in a message box
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Training Error", err[-2000:])

    def _stop_train(self):
        """Force stop training immediately."""
        if hasattr(self, '_train_worker') and self._train_worker.isRunning():
            self._train_worker.force_stop()
            self.train_status_label.setText("Force stopping...")

    # ── Export ─────────────────────────────────────────────────────

    def _run_export(self):
        model_path = self._resolve_model(self.export_model_box, self._export_custom_model_path)
        if not model_path:
            self.status_label.setText("Please select a model")
            return

        self.export_log.clear()
        self.export_btn.setEnabled(False)
        self.status_label.setText("Exporting...")

        self._export_worker = ExportWorker()
        self._export_worker.model_path = model_path
        self._export_worker.format = self.export_format.currentText()
        self._export_worker.imgsz = self.export_imgsz.value()
        self._export_worker.half = self.export_half.isChecked()
        self._export_worker.dynamic = self.export_dynamic.isChecked()
        self._export_worker.simplify = self.export_simplify.isChecked()
        self._export_worker.opset = self.export_opset.value()

        self._export_worker.finished.connect(lambda msg: (
            self.export_log.append(msg),
            self.export_btn.setEnabled(True),
            self.status_label.setText("Export Complete"),
        ))
        self._export_worker.error.connect(lambda err: (
            self.export_log.append(f"❌ {err}"),
            self.export_btn.setEnabled(True),
            self.status_label.setText("Export Failed"),
        ))
        self._export_worker.start()

    # ── Dataset Tools ──────────────────────────────────────────────

    # ── Benchmark ──────────────────────────────────────────────────

    def _run_benchmark(self):
        model_path = self._resolve_model(self.bench_model_box, self._bench_custom_model_path)
        if not model_path:
            self.status_label.setText("Please select a model")
            return

        self.bench_log.clear()
        self.bench_btn.setEnabled(False)
        self.status_label.setText("Benchmarking...")

        self._bench_worker = BenchmarkWorker()
        self._bench_worker.model_path = model_path
        self._bench_worker.imgsz = self.bench_imgsz.value()
        self._bench_worker.runs = self.bench_runs.value()

        self._bench_worker.finished.connect(lambda msg: (
            self.bench_log.append(msg),
            self.bench_btn.setEnabled(True),
            self.status_label.setText("Benchmark Complete"),
        ))
        self._bench_worker.error.connect(lambda err: (
            self.bench_log.append(f"❌ {err}"),
            self.bench_btn.setEnabled(True),
            self.status_label.setText("Benchmark Failed"),
        ))
        self._bench_worker.start()

    # ── Window Events ──────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragPos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.dragPos and event.buttons() == Qt.LeftButton:
            if self.isMaximized():
                self.showNormal()
            self.move(self.pos() + event.globalPosition().toPoint() - self.dragPos)
            self.dragPos = event.globalPosition().toPoint()
