# frontend/pyqt_app/app.py
import sys
import requests
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QTabWidget,
    QSlider,
    QSpinBox,
    QComboBox,
    QProgressBar,
    QGroupBox,
    QGridLayout,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QFont, QIcon, QDragEnterEvent, QDropEvent
from PIL import Image
import io
from ui.main_window import VisionQuestMainWindow


class VisionQuestApp(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("VisionQuest")
        self.setApplicationVersion("0.1.0")
        self.setOrganizationName("VisionQuest Team")

        # Set application style
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 15px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #667eea;
            }
            QPushButton {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a67d8;
            }
            QPushButton:pressed {
                background-color: #4c51bf;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
            }
        """
        )


def main():
    app = VisionQuestApp(sys.argv)

    # Create main window
    window = VisionQuestMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
