"""QSS Styles for the application."""

MAIN_STYLE = """
QFrame#Main_QF {
    background-color: qlineargradient(x0:0, y0:1, x1:1, y1:1,
        stop:0.4 rgb(107, 128, 210), stop:1 rgb(180, 140, 255));
    border: 0px solid red;
    border-radius: 30px;
}

QFrame#LeftMenuBg {
    background-color: rgba(255, 255, 255, 0);
    border: 0px solid red;
    border-radius: 30px;
}

QFrame#ContentBox {
    background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1,
        stop:0 rgb(180, 185, 220), stop:0.5 rgb(210, 215, 240), stop:1 rgb(230, 235, 250));
    border: 0px solid red;
    border-radius: 30px;
}

QFrame#prm_page {
    background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1,
        stop:0 rgb(243, 175, 189), stop:1 rgb(155, 118, 218));
    border-top-left-radius: 30px;
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
    border-bottom-left-radius: 30px;
}

/* ===== Force ALL text to dark on light backgrounds ===== */
QMainWindow, QWidget {
    color: rgb(40, 40, 40);
}
QLabel {
    color: rgb(40, 40, 40);
}
QComboBox {
    color: rgb(40, 40, 40);
}
QDoubleSpinBox, QSpinBox {
    color: rgb(40, 40, 40);
}
QLineEdit {
    color: rgb(40, 40, 40);
}
QPushButton {
    color: rgb(50, 50, 50);
}
QTextEdit {
    color: rgb(30, 30, 30);
}
QCheckBox {
    color: rgb(50, 50, 50);
}
QProgressBar {
    color: rgb(50, 50, 50);
}
"""

MENU_BUTTON = """
QPushButton {
    background-repeat: no-repeat;
    background-position: center;
    border: none;
    text-align: center;
    padding: 0px;
    color: rgba(255, 255, 255, 199) !important;
    font: 700 12pt "Nirmala UI";
}
QPushButton:hover {
    background-color: rgba(114, 129, 214, 59);
}
QPushButton:checked {
    background-color: rgba(114, 129, 214, 80);
}
"""

CONTENT_BUTTON = """
QPushButton {
    background-color: rgb(230, 235, 250);
    color: rgb(30, 30, 40) !important;
    border: 1px solid rgb(180, 180, 200);
    border-radius: 8px;
    padding: 6px 16px;
    font: 700 11pt "Segoe UI";
}
QPushButton:hover {
    background-color: rgb(200, 210, 240);
}
QPushButton:pressed {
    background-color: rgb(180, 195, 230);
}
QPushButton:disabled {
    background-color: rgb(220, 220, 220);
    color: rgb(150, 150, 150) !important;
}
"""

SETTINGS_BUTTON = """
QPushButton {
    background-color: rgba(255, 255, 255, 80);
    color: rgba(255, 255, 255, 220) !important;
    border: 1px solid rgba(255, 255, 255, 60);
    border-radius: 12px;
    padding: 8px 16px;
    font: 700 10pt "Segoe UI";
}
QPushButton:hover {
    background-color: rgba(255, 255, 255, 120);
}
QPushButton:pressed {
    background-color: rgba(255, 255, 255, 150);
}
"""

STAT_CARD = """
QFrame {{
    color: rgb(255, 255, 255);
    border-radius: 15px;
    background-color: qradialgradient(cx:0, cy:0, radius:1, fx:0.1, fy:0.1,
        stop:0 {color1}, stop:1 {color2});
    border: 1px outset {border};
}}
"""

SLIDER_STYLE = """
QSlider::groove:horizontal {
    border: none;
    height: 10px;
    background-color: rgba(255,255,255,90);
    border-radius: 5px;
}
QSlider::handle:horizontal {
    width: 10px;
    margin: -1px 0px -1px 0px;
    border-radius: 3px;
    background-color: white;
}
QSlider::sub-page:horizontal {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #59969b, stop:1 #04e7fa);
    border-radius: 5px;
}
"""

SPINBOX_STYLE = """
QDoubleSpinBox, QSpinBox {
    border: 0px solid lightgray;
    border-radius: 2px;
    background-color: rgba(255,255,255,90);
    font: 600 9pt "Segoe UI";
}
"""

COMBOBOX_STYLE = """
QComboBox {
    background-color: rgba(255,255,255,90);
    color: rgba(0, 0, 0, 200);
    font: 600 9pt "Segoe UI";
    border: 1px solid lightgray;
    border-radius: 10px;
    padding-left: 15px;
}
QComboBox::drop-down {
    width: 22px;
    border-left: 1px solid lightgray;
    border-top-right-radius: 15px;
    border-bottom-right-radius: 15px;
}
"""

PROGRESS_BAR = """
QProgressBar {
    font: 700 10pt "Microsoft YaHei UI";
    color: rgb(253, 143, 134);
    text-align: center;
    border: 3px solid rgb(255, 255, 255);
    border-radius: 10px;
    background-color: rgba(215, 215, 215, 100);
}
QProgressBar::chunk {
    border-radius: 7px;
    background: rgba(119, 111, 252, 200);
}
"""

VIDEO_LABEL = """
QLabel {
    background-color: rgb(238, 242, 255);
    border: 2px solid rgb(255, 255, 255);
    border-radius: 15px;
}
"""

PARAM_FRAME = """
QFrame {
    background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1,
        stop:0 rgba(225, 230, 245, 180), stop:1 rgba(235, 240, 252, 180));
    border: 1px solid rgba(200, 205, 220, 120);
    border-radius: 15px;
}
"""

CONTENT_LABEL = """
QLabel {
    color: rgb(50, 50, 50);
    font: 10pt "Segoe UI";
}
"""

SECTION_TITLE = """
QLabel {
    color: rgb(60, 60, 80);
    font: 700 12pt "Segoe UI";
}
"""

TEXTEDIT_STYLE = """
QTextEdit {
    background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1, stop:0 rgb(225,230,248), stop:1 rgb(235,240,252));
    color: rgb(40, 40, 50);
    border: 1px solid rgba(180, 185, 210, 150);
    border-radius: 10px;
    font: 10pt "Consolas";
    padding: 5px;
}
"""

LINEEDIT_STYLE = """
QLineEdit {
    border: 1px solid lightgray;
    border-radius: 10px;
    padding: 5px 10px;
    background-color: rgba(255,255,255,90);
    font: 10pt "Segoe UI";
}
"""

CHECKBOX_STYLE = """
QCheckBox {
    color: rgba(255, 255, 255, 199);
    font: 590 10pt "Nirmala UI";
}
"""

TAB_STYLE = """
QTabWidget::pane {
    border: none;
    background: transparent;
}
QTabBar {
    background: qlineargradient(x0:0, y0:0, x1:1, y1:0,
        stop:0 rgba(162,129,247,80), stop:0.33 rgba(253,139,133,80),
        stop:0.66 rgba(243,175,189,80), stop:1 rgba(66,226,192,80));
    border-radius: 8px;
    padding: 4px;
}
QTabBar::tab {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(162,129,247,200), stop:1 rgba(119,111,252,200));
    color: white;
    padding: 8px 20px;
    margin: 2px;
    border-radius: 8px;
    font: 700 11pt "Nirmala UI";
}
QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgb(119,111,252), stop:1 rgb(98,91,213));
    color: white;
}
QTabBar::tab:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(162,129,247,240), stop:1 rgba(119,111,252,240));
    color: white;
}
"""
