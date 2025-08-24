# frontend/pyqt_app/ui/main_window.py
import sys
import requests
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PIL import Image
import io


class APIWorker(QThread):
    """Background worker for API calls"""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, endpoint, files=None, data=None):
        super().__init__()
        self.endpoint = endpoint
        self.files = files
        self.data = data
        self.api_base = "http://localhost:8000/api/v1"

    def run(self):
        try:
            url = f"{self.api_base}/{self.endpoint}"

            if self.files:
                response = requests.post(
                    url, files=self.files, data=self.data, timeout=30
                )
            else:
                response = requests.get(url, timeout=10)

            if response.status_code == 200:
                self.finished.emit(response.json())
            else:
                error_msg = f"API 錯誤 {response.status_code}"
                try:
                    error_detail = response.json().get("detail", error_msg)
                    self.error.emit(error_detail)
                except:
                    self.error.emit(error_msg)

        except requests.exceptions.RequestException as e:
            self.error.emit(f"連接錯誤: {str(e)}")
        except Exception as e:
            self.error.emit(f"未知錯誤: {str(e)}")


class ImageDropLabel(QLabel):
    """Label that accepts drag and drop images"""

    imageDropped = pyqtSignal(str)

    def __init__(self, text=""):
        super().__init__(text)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                min-height: 200px;
            }
            QLabel:hover {
                border-color: #667eea;
                background-color: #f0f4ff;
            }
        """
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(
                """
                QLabel {
                    border: 2px solid #667eea;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #e6f3ff;
                    min-height: 200px;
                }
            """
            )
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                min-height: 200px;
            }
        """
        )

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files and files[0].lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        ):
            self.imageDropped.emit(files[0])

        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
                min-height: 200px;
            }
        """
        )


class VisionQuestMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.api_worker = None

        self.setWindowTitle("VisionQuest - 多模態 AI 工具箱")
        self.setWindowIcon(QIcon("icon.png"))  # Add icon if available
        self.resize(1000, 700)

        # Center window
        self.center_window()

        self.setup_ui()
        self.setup_connections()

        # Auto-check health on startup
        QTimer.singleShot(1000, self.check_health)

    def center_window(self):
        """Center the window on screen"""
        screen = self.screen().availableGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2
        )

    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Header
        header_layout = self.create_header()
        main_layout.addLayout(header_layout)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_caption_tab(), "📸 圖像描述")
        self.tab_widget.addTab(self.create_vqa_tab(), "🤔 視覺問答")
        self.tab_widget.addTab(self.create_chat_tab(), "💬 文字聊天")
        self.tab_widget.addTab(self.create_health_tab(), "🔍 系統狀態")

        main_layout.addWidget(self.tab_widget)

        # Status bar
        self.statusBar().showMessage("就緒")

    def create_header(self):
        """Create header section"""
        layout = QVBoxLayout()

        title = QLabel("🎯 VisionQuest - 多模態 AI 工具箱")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #333; margin: 10px;")

        subtitle = QLabel("Phase 2: 圖像理解與視覺問答系統")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; margin-bottom: 10px;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        return layout

    def create_caption_tab(self):
        """Create caption generation tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left panel - Image and controls
        left_panel = QVBoxLayout()

        # Image display
        self.image_label = ImageDropLabel("拖放圖片到此處\n或點擊下方按鈕選擇圖片")
        self.image_label.imageDropped.connect(self.load_image)
        left_panel.addWidget(self.image_label)

        # File selection button
        self.select_file_btn = QPushButton("📁 選擇圖片")
        self.select_file_btn.clicked.connect(self.select_image_file)
        left_panel.addWidget(self.select_file_btn)

        # Parameters
        params_group = QGroupBox("參數設定")
        params_layout = QGridLayout(params_group)

        # Max length
        params_layout.addWidget(QLabel("最大長度:"), 0, 0)
        self.max_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_length_slider.setRange(10, 200)
        self.max_length_slider.setValue(50)
        self.max_length_value = QLabel("50")
        self.max_length_slider.valueChanged.connect(
            lambda v: self.max_length_value.setText(str(v))
        )
        params_layout.addWidget(self.max_length_slider, 0, 1)
        params_layout.addWidget(self.max_length_value, 0, 2)

        # Num beams
        params_layout.addWidget(QLabel("束搜索:"), 1, 0)
        self.num_beams_spin = QSpinBox()
        self.num_beams_spin.setRange(1, 10)
        self.num_beams_spin.setValue(5)
        params_layout.addWidget(self.num_beams_spin, 1, 1, 1, 2)

        # Temperature
        params_layout.addWidget(QLabel("溫度:"), 2, 0)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(1, 20)  # 0.1 to 2.0, scaled by 10
        self.temperature_slider.setValue(10)
        self.temperature_value = QLabel("1.0")
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temperature_value.setText(f"{v/10:.1f}")
        )
        params_layout.addWidget(self.temperature_slider, 2, 1)
        params_layout.addWidget(self.temperature_value, 2, 2)

        left_panel.addWidget(params_group)

        # Generate button
        self.generate_caption_btn = QPushButton("🚀 生成描述")
        self.generate_caption_btn.clicked.connect(self.generate_caption)
        self.generate_caption_btn.setEnabled(False)
        left_panel.addWidget(self.generate_caption_btn)

        # Progress bar
        self.caption_progress = QProgressBar()
        self.caption_progress.setVisible(False)
        left_panel.addWidget(self.caption_progress)

        # Right panel - Results
        right_panel = QVBoxLayout()

        result_group = QGroupBox("生成結果")
        result_layout = QVBoxLayout(result_group)

        self.caption_result = QTextEdit()
        self.caption_result.setPlaceholderText("AI 生成的圖像描述將顯示在這裡...")
        self.caption_result.setMaximumHeight(150)
        result_layout.addWidget(self.caption_result)

        # Metadata
        meta_layout = QGridLayout()
        meta_layout.addWidget(QLabel("信心度:"), 0, 0)
        self.confidence_label = QLabel("--")
        meta_layout.addWidget(self.confidence_label, 0, 1)

        meta_layout.addWidget(QLabel("使用模型:"), 1, 0)
        self.model_label = QLabel("--")
        meta_layout.addWidget(self.model_label, 1, 1)

        meta_layout.addWidget(QLabel("處理時間:"), 2, 0)
        self.time_label = QLabel("--")
        meta_layout.addWidget(self.time_label, 2, 1)

        result_layout.addLayout(meta_layout)
        right_panel.addWidget(result_group)

        # Add to main layout
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 1)

        return widget

    def create_vqa_tab(self):
        """Create VQA tab - simplified for now"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("🚧 視覺問答功能開發中..."))
        layout.addWidget(QLabel("將在 Phase 2 完整實作"))

        return widget

    def create_chat_tab(self):
        """Create chat tab - placeholder"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("🚧 聊天功能將在 Phase 3 推出"))

        return widget

    def create_health_tab(self):
        """Create system health tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Refresh button
        refresh_btn = QPushButton("🔄 刷新狀態")
        refresh_btn.clicked.connect(self.check_health)
        layout.addWidget(refresh_btn)

        # Health display
        self.health_display = QTextEdit()
        self.health_display.setReadOnly(True)
        layout.addWidget(self.health_display)

        return widget

    def setup_connections(self):
        """Setup signal connections"""
        pass  # Additional connections if needed

    def select_image_file(self):
        """Open file dialog to select image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)"
        )

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        """Load and display image"""
        try:
            self.current_image_path = file_path

            # Display image
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                400,
                300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setText("")

            # Enable generate button
            self.generate_caption_btn.setEnabled(True)

            self.statusBar().showMessage(f"已載入圖片: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法載入圖片: {str(e)}")

    def generate_caption(self):
        """Generate caption for current image"""
        if not self.current_image_path:
            return

        try:
            # Prepare image file
            with open(self.current_image_path, "rb") as f:
                files = {"file": ("image.jpg", f, "image/jpeg")}

                data = {
                    "max_length": self.max_length_slider.value(),
                    "num_beams": self.num_beams_spin.value(),
                    "temperature": self.temperature_slider.value() / 10.0,
                }

                # Show progress
                self.caption_progress.setVisible(True)
                self.caption_progress.setRange(0, 0)  # Indeterminate
                self.generate_caption_btn.setEnabled(False)

                # Start API call
                self.api_worker = APIWorker("caption/", files=files, data=data)
                self.api_worker.finished.connect(self.on_caption_result)
                self.api_worker.error.connect(self.on_api_error)
                self.api_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法讀取圖片: {str(e)}")
            self.reset_caption_ui()

    def on_caption_result(self, result):
        """Handle caption generation result"""
        self.caption_result.setText(result.get("caption", ""))
        self.confidence_label.setText(f"{result.get('confidence', 0) * 100:.1f}%")
        self.model_label.setText(result.get("model_used", "Unknown"))
        self.time_label.setText(f"{result.get('processing_time_ms', 0):.0f}ms")

        self.reset_caption_ui()
        self.statusBar().showMessage("圖像描述生成完成")

    def on_api_error(self, error_msg):
        """Handle API error"""
        QMessageBox.warning(self, "API 錯誤", error_msg)
        self.reset_caption_ui()
        self.statusBar().showMessage("API 調用失敗")

    def reset_caption_ui(self):
        """Reset caption UI state"""
        self.caption_progress.setVisible(False)
        self.generate_caption_btn.setEnabled(bool(self.current_image_path))

    def check_health(self):
        """Check system health"""
        self.health_display.setText("檢查中...")

        self.api_worker = APIWorker("health/")
        self.api_worker.finished.connect(self.on_health_result)
        self.api_worker.error.connect(self.on_health_error)
        self.api_worker.start()

    def on_health_result(self, result):
        """Display health check result"""
        import json

        formatted_result = json.dumps(result, indent=2, ensure_ascii=False)
        self.health_display.setText(formatted_result)

        status = result.get("status", "unknown")
        if status == "healthy":
            self.statusBar().showMessage("系統狀態正常")
        else:
            self.statusBar().showMessage("系統狀態異常")

    def on_health_error(self, error_msg):
        """Handle health check error"""
        self.health_display.setText(f"健康檢查失敗: {error_msg}")
        self.statusBar().showMessage("無法連接到後端服務")
