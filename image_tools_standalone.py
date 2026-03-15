import sys, os, random, math, zipfile, logging, gc
import subprocess
import urllib.request
from collections import Counter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QStackedWidget, QLineEdit, QFileDialog,
    QComboBox, QSlider, QCheckBox, QScrollArea, QColorDialog,
    QProgressBar, QRadioButton, QButtonGroup, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QPlainTextEdit, QSplitter,
    QSizePolicy, QMessageBox, QGroupBox, QTabWidget
)
from PySide6.QtCore import Qt, Signal, QThread, QUrl, QTimer, QSettings, QObject
from PySide6.QtGui import QPixmap, QIcon, QColor, QImage, QDesktopServices, QPainter, QPen, QBrush

# Windows dark title bar support
try:
    import ctypes
    from ctypes import wintypes
    HAS_WINDOWS_TITLEBAR = True
except ImportError:
    HAS_WINDOWS_TITLEBAR = False

if getattr(sys, "frozen", False):
    ROOT_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = getattr(sys, "_MEIPASS", ROOT_DIR)
else:
    BUNDLE_DIR = os.path.dirname(os.path.dirname(__file__))
    ROOT_DIR = BUNDLE_DIR

# ============================
# LOGGING SETUP
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ROOT_DIR, 'pixelforge.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================
# CONFIGURATION
# ============================
ICON_PATH = os.path.join(BUNDLE_DIR, "pixelforge.ico")
if not os.path.exists(ICON_PATH):
    fallback_icon = os.path.join(BUNDLE_DIR, "dnk.ico")
    if os.path.exists(fallback_icon):
        ICON_PATH = fallback_icon

APP_VERSION = "3.1.0"
UPDATE_CHECK_URL = "https://raw.githubusercontent.com/Orvlyn/PixelForge/main/version.json"
ICON_GITHUB_URL = "https://raw.githubusercontent.com/Orvlyn/PixelForge/main/pixelforge.ico"
ICON_CACHE_PATH = os.path.join(ROOT_DIR, ".pixelforge_icon_cache.ico")


def get_cached_icon() -> QIcon:
    for candidate in [ICON_PATH, ICON_CACHE_PATH]:
        if candidate and os.path.exists(candidate):
            icon = QIcon(candidate)
            if not icon.isNull():
                return icon
    try:
        urllib.request.urlretrieve(ICON_GITHUB_URL, ICON_CACHE_PATH)
        icon = QIcon(ICON_CACHE_PATH)
        if not icon.isNull():
            return icon
    except Exception:
        pass
    return QIcon()

from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageSequence, ImageFilter
import numpy as np
import colorsys
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    # Fallback using PIL
    def gaussian_filter(arr, sigma):
        from PIL import Image as PILImage
        # Convert numpy array to PIL Image, apply filter, convert back
        if arr.ndim == 2:
            pil_img = PILImage.fromarray((arr * 255).astype(np.uint8))
            filtered = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
            return np.array(filtered).astype(np.float32) / 255.0
        pil_img = PILImage.fromarray((arr * 255).astype(np.uint8))
        filtered = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(filtered).astype(np.float32) / 255.0

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def load_cube_lut(path):
    """Load a .cube LUT file into a numpy array with domain metadata."""
    size = None
    data = []
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("TITLE"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                parts = line.split()
                if len(parts) >= 2:
                    size = int(parts[1])
                continue
            if line.startswith("DOMAIN_MIN"):
                parts = line.split()
                if len(parts) >= 4:
                    domain_min = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                continue
            if line.startswith("DOMAIN_MAX"):
                parts = line.split()
                if len(parts) >= 4:
                    domain_max = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
                continue
            if line[0].isalpha():
                continue
            parts = line.split()
            if len(parts) == 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if not size or len(data) != size ** 3:
        raise ValueError("Invalid .cube LUT format")

    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    idx = 0
    for r in range(size):
        for g in range(size):
            for b in range(size):
                lut[r, g, b] = data[idx]
                idx += 1
    return {"lut": lut, "domain_min": domain_min, "domain_max": domain_max}


def apply_lut_to_array(img_arr, lut_data, swap_rb=False, linearize=False):
    """Apply 3D LUT to an image array (float32 0..1)."""
    if isinstance(lut_data, dict):
        lut = lut_data.get("lut")
        domain_min = lut_data.get("domain_min", np.array([0.0, 0.0, 0.0], dtype=np.float32))
        domain_max = lut_data.get("domain_max", np.array([1.0, 1.0, 1.0], dtype=np.float32))
    else:
        lut = lut_data
        domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    size = lut.shape[0]
    if swap_rb:
        img_arr = img_arr[..., ::-1]

    if linearize:
        img_arr = srgb_to_linear(img_arr)

    denom = np.maximum(domain_max - domain_min, 1e-6)
    scaled_in = (img_arr - domain_min) / denom
    scaled_in = np.clip(scaled_in, 0.0, 1.0)
    scaled = scaled_in * (size - 1)
    idx0 = np.floor(scaled).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, size - 1)
    frac = scaled - idx0

    r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
    r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]
    fr, fg, fb = frac[..., 0], frac[..., 1], frac[..., 2]

    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    c00 = c000 * (1 - fb[..., None]) + c001 * fb[..., None]
    c01 = c010 * (1 - fb[..., None]) + c011 * fb[..., None]
    c10 = c100 * (1 - fb[..., None]) + c101 * fb[..., None]
    c11 = c110 * (1 - fb[..., None]) + c111 * fb[..., None]

    c0 = c00 * (1 - fg[..., None]) + c01 * fg[..., None]
    c1 = c10 * (1 - fg[..., None]) + c11 * fg[..., None]

    out = c0 * (1 - fr[..., None]) + c1 * fr[..., None]
    out = np.clip(out, 0, 1)
    if linearize:
        out = linear_to_srgb(out)
    if swap_rb:
        out = out[..., ::-1]
    return out


def srgb_to_linear(arr):
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(arr):
    return np.where(arr <= 0.0031308, arr * 12.92, 1.055 * (arr ** (1 / 2.4)) - 0.055)


# Custom Slider - blocks wheel events
class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()

class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()

# Zoom Slider - allows wheel events for zooming
class ZoomSlider(QSlider):
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        current = self.value()
        self.setValue(current + (10 if delta > 0 else -10))
        event.accept()

# Zoomable Label - allows wheel events to control zoom
class ZoomableLabel(QLabel):
    def __init__(self, zoom_slider):
        super().__init__()
        self.zoom_slider = zoom_slider
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(current + (10 if delta > 0 else -10))
        event.accept()


# Color Wheel Widget for Color Grading
class ColorWheelWidget(QWidget):
    colorChanged = Signal(QColor, float, float)  # color, hue shift, saturation shift
    
    def __init__(self, title="Color Wheel", size=120):
        super().__init__()
        self.title = title
        self.wheel_size = size
        self.selected_point = None
        self.setFixedSize(size, size)
        self.setToolTip("Click to adjust color")
        
        # Pre-render the color wheel to improve performance
        self._wheel_pixmap = None
        self._render_wheel()
        
    def _render_wheel(self):
        """Pre-render the color wheel for better performance"""
        self._wheel_pixmap = QPixmap(self.wheel_size, self.wheel_size)
        self._wheel_pixmap.fill(Qt.transparent)
        
        painter = QPainter(self._wheel_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center = self.wheel_size // 2
        radius = center - 5
        
        # Draw color wheel with better coverage
        for r in range(radius):
            sat = int((r / radius) * 255)
            for angle in range(360):
                color = QColor.fromHsv(angle, sat, 255)
                painter.setPen(color)
                x = center + int(r * math.cos(math.radians(angle)))
                y = center + int(r * math.sin(math.radians(angle)))
                painter.drawPoint(x, y)
        
        # Draw center neutral circle
        painter.setBrush(QColor(128, 128, 128))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center - 8, center - 8, 16, 16)
        
        painter.end()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw pre-rendered wheel
        if self._wheel_pixmap:
            painter.drawPixmap(0, 0, self._wheel_pixmap)
        
        # Draw selected point if any
        if self.selected_point:
            painter.setPen(QPen(Qt.white, 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self.selected_point[0] - 6, self.selected_point[1] - 6, 12, 12)
            painter.setPen(QPen(Qt.black, 1))
            painter.drawEllipse(self.selected_point[0] - 6, self.selected_point[1] - 6, 12, 12)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            center = self.wheel_size // 2
            radius = center - 5
            
            # Calculate distance from center
            dx = x - center
            dy = y - center
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= radius:
                self.selected_point = (x, y)
                
                # Calculate hue and saturation shifts
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0:
                    angle += 360
                
                hue_shift = angle
                sat_shift = distance / radius
                
                color = QColor.fromHsv(int(angle), int(sat_shift * 255), 255)
                self.colorChanged.emit(color, hue_shift, sat_shift)
                self.update()
            elif distance <= 15:  # Click on center to reset
                self.selected_point = None
                self.colorChanged.emit(QColor(128, 128, 128), 0, 0)
                self.update()
    
    def reset(self):
        self.selected_point = None
        self.update()
    
    def wheelEvent(self, event):
        # Ignore wheel events to prevent scrolling the container
        event.ignore()


# Collapsible Group Box
class CollapsibleGroupBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Header button
        self.toggle_button = QPushButton(f"▼ {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_content)
        self.layout.addWidget(self.toggle_button)
        
        # Content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 5, 10, 5)
        self.layout.addWidget(self.content_widget)
        
    def toggle_content(self):
        is_visible = self.content_widget.isVisible()
        self.content_widget.setVisible(not is_visible)
        arrow = "▼" if not is_visible else "▶"
        current_text = self.toggle_button.text()
        new_text = arrow + current_text[1:]
        self.toggle_button.setText(new_text)
    
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        self.content_layout.addLayout(layout)


# ============================
# UTILITY FUNCTIONS
# ============================

def is_valid_image_file(file_path: str, max_size_mb: int = 500) -> tuple[bool, str]:
    """
    Validate if a file is a readable image within size constraints.
    Returns (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if not os.path.isfile(file_path):
        return False, f"Not a file: {file_path}"
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.1f}MB (max {max_size_mb}MB)"
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    if not file_path.lower().endswith(valid_extensions):
        return False, f"Unsupported file type: {os.path.splitext(file_path)[1]}"
    
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, f"Corrupted or invalid image: {e}"

def safe_load_image(file_path: str, max_size_mb: int = 500) -> Image.Image | None:
    """
    Safely load an image with validation and error logging.
    Returns None if loading fails.
    """
    is_valid, error_msg = is_valid_image_file(file_path, max_size_mb)
    if not is_valid:
        logger.warning(f"Image validation failed for {file_path}: {error_msg}")
        return None
    
    try:
        return Image.open(file_path)
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        return None

def fast_resize_image(img: Image.Image, size: tuple[int, int], resample=Image.Resampling.LANCZOS) -> Image.Image:
    """
    Resize image with OpenCV acceleration if available, otherwise use Pillow.
    OpenCV can be 2-3x faster for large images.
    """
    if HAS_CV2 and img.size[0] > 1000 and img.size[1] > 1000:
        try:
            import cv2
            # Convert PIL to OpenCV format
            cv_img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            # Resize with OpenCV
            cv_resized = cv2.resize(cv_img, size, interpolation=cv2.INTER_LANCZOS4)
            # Convert back to PIL
            result = Image.fromarray(cv2.cvtColor(cv_resized, cv2.COLOR_BGR2RGB))
            return result
        except Exception as e:
            logger.debug(f"OpenCV resize failed, falling back to Pillow: {e}")
    
    return img.resize(size, resample)

# Worker Thread

class Worker(QThread):
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        try:
            self.func(self.progress.emit, *self.args)
            self.finished.emit("Done")
        except Exception as e:
            self.finished.emit(str(e))


# Base Card Page
class CardPage(QWidget):
    def __init__(self, title=""):
        super().__init__()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.NoFrame)

        self.card = QWidget()
        self.card.setObjectName("Card")
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setSpacing(15)
        self.card_layout.setContentsMargins(20, 20, 20, 20)

        self.scroll_area.setWidget(self.card)
        main_layout.addWidget(self.scroll_area)

class ImageResizerPage(CardPage):
    def __init__(self):
        super().__init__("Image Resizer")
        self.setAcceptDrops(True)

        self.input_dir = ""
        self.output_dir = ""
        self.bg_fill_color = QColor("#ffffff")
        self._preview_pixmap = QPixmap()

        split = QHBoxLayout()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.input_btn = QPushButton("📁 Select Input File/Folder")
        self.output_btn = QPushButton("📂 Select Output Folder")
        self.start_btn = QPushButton("▶ Start Batch Resize")
        left_layout.addWidget(self.input_btn)
        self.input_mode_combo = NoWheelComboBox()
        self.input_mode_combo.addItems(["Input File", "Input Folder"])
        left_layout.addWidget(self.input_mode_combo)
        left_layout.addWidget(self.output_btn)
        left_layout.addWidget(self.start_btn)

        row1 = QHBoxLayout()
        self.width_input = QLineEdit("1920")
        self.height_input = QLineEdit("1080")
        row1.addWidget(QLabel("W"))
        row1.addWidget(self.width_input)
        row1.addWidget(QLabel("H"))
        row1.addWidget(self.height_input)
        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.scale_input = QLineEdit("")
        self.scale_input.setPlaceholderText("Scale % (optional)")
        row2.addWidget(QLabel("Scale"))
        row2.addWidget(self.scale_input)
        left_layout.addLayout(row2)

        self.keep_ratio = QCheckBox("Keep Aspect Ratio")
        self.keep_ratio.setChecked(True)
        self.no_upscale = QCheckBox("Do Not Upscale")
        self.no_upscale.setChecked(True)
        self.orientation_check = QCheckBox("Auto-rotate from EXIF")
        self.orientation_check.setChecked(True)
        left_layout.addWidget(self.keep_ratio)
        left_layout.addWidget(self.no_upscale)
        left_layout.addWidget(self.orientation_check)

        self.format_combo = NoWheelComboBox()
        self.format_combo.addItems(["JPEG", "PNG", "WEBP", "GIF"])
        left_layout.addWidget(QLabel("Output Format"))
        left_layout.addWidget(self.format_combo)

        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(90)
        left_layout.addWidget(QLabel("JPEG/WebP Quality"))
        left_layout.addWidget(self.quality_slider)

        self.target_size_input = QLineEdit("")
        self.target_size_input.setPlaceholderText("Target KB (JPEG only, optional)")
        left_layout.addWidget(self.target_size_input)

        self.dpi_combo = NoWheelComboBox()
        self.dpi_combo.addItems(["72", "96", "150", "300"])
        self.dpi_combo.setCurrentText("96")
        left_layout.addWidget(QLabel("DPI"))
        left_layout.addWidget(self.dpi_combo)

        self.sharpen_combo = NoWheelComboBox()
        self.sharpen_combo.addItems(["None", "Low", "Medium", "High"])
        left_layout.addWidget(QLabel("Sharpen"))
        left_layout.addWidget(self.sharpen_combo)

        self.stats_label = QLabel("Load an image to preview resize settings.")
        self.stats_label.setWordWrap(True)
        left_layout.addWidget(self.stats_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        left_layout.addWidget(self.progress)
        left_layout.addStretch()

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.preview_scroll)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom"))
        self.zoom_slider = ZoomSlider(Qt.Horizontal)
        self.zoom_slider.setRange(30, 400)
        self.zoom_slider.setValue(100)
        zoom_row.addWidget(self.zoom_slider)
        right_layout.addLayout(zoom_row)

        split.addWidget(left, 35)
        split.addWidget(right, 65)
        self.card_layout.addLayout(split)

        self.input_btn.clicked.connect(self.select_input)
        self.output_btn.clicked.connect(self.select_output)
        self.start_btn.clicked.connect(self.start_process)
        self.zoom_slider.valueChanged.connect(self.apply_resize_zoom)

        for widget in [
            self.width_input, self.height_input, self.scale_input,
            self.target_size_input
        ]:
            widget.textChanged.connect(self.refresh_resize_preview)

        self.keep_ratio.stateChanged.connect(self.refresh_resize_preview)
        self.no_upscale.stateChanged.connect(self.refresh_resize_preview)
        self.orientation_check.stateChanged.connect(self.refresh_resize_preview)
        self.format_combo.currentIndexChanged.connect(self.refresh_resize_preview)
        self.quality_slider.valueChanged.connect(self.refresh_resize_preview)
        self.dpi_combo.currentIndexChanged.connect(self.refresh_resize_preview)
        self.sharpen_combo.currentIndexChanged.connect(self.refresh_resize_preview)

    def select_input(self):
        if self.input_mode_combo.currentText() == "Input Folder":
            folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
            if folder:
                self.input_dir = folder
                self.refresh_resize_preview()
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",
            "Images (*.png *.jpg *.jpeg *.gif *.webp *.bmp);;All Files (*.*)"
        )
        if file_path:
            self.input_dir = file_path
            self.refresh_resize_preview()

    def select_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        local_paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if not local_paths:
            return

        path = local_paths[0]
        image_ext = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
        if os.path.isdir(path):
            self.input_mode_combo.setCurrentText("Input Folder")
            self.input_dir = path
            self.refresh_resize_preview()
            event.acceptProposedAction()
        elif os.path.isfile(path) and path.lower().endswith(image_ext):
            self.input_mode_combo.setCurrentText("Input File")
            self.input_dir = path
            self.refresh_resize_preview()
            event.acceptProposedAction()

    def apply_resize_zoom(self):
        if self._preview_pixmap.isNull():
            return
        zoom = self.zoom_slider.value() / 100.0
        sw = int(self._preview_pixmap.width() * zoom)
        sh = int(self._preview_pixmap.height() * zoom)
        avail_w = self.preview_scroll.viewport().width() or sw
        avail_h = self.preview_scroll.viewport().height() or sh
        fit_scale = min(1.0, avail_w / sw if sw else 1.0, avail_h / sh if sh else 1.0)
        sw2 = int(sw * fit_scale)
        sh2 = int(sh * fit_scale)
        scaled_pix = self._preview_pixmap.scaled(sw2, sh2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pix)
        self.preview_label.setFixedSize(sw2, sh2)
        self.preview_label.setAlignment(Qt.AlignCenter)

    def refresh_resize_preview(self):
        if not self.input_dir:
            return
        # support file path as well as folder
        if os.path.isdir(self.input_dir):
            files = [f for f in os.listdir(self.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
            if not files:
                return
            path = os.path.join(self.input_dir, files[0])
        else:
            path = self.input_dir
        try:
            with Image.open(path) as img:
                # orientation fix
                if self.orientation_check.isChecked():
                    try:
                        from PIL import ImageOps
                        img = ImageOps.exif_transpose(img)
                    except Exception as e:
                        logger.warning(f"Could not apply EXIF transpose: {e}")

                orig_w, orig_h = img.width, img.height
                orig_size = os.path.getsize(path)

                w = int(self.width_input.text() or 1920)
                h = int(self.height_input.text() or 1080)
                scale = float(self.scale_input.text()) if self.scale_input.text().strip() else 0
                keep = self.keep_ratio.isChecked()

                # compute target size
                if scale > 0:
                    nw = int(img.width * scale / 100)
                    nh = int(img.height * scale / 100)
                else:
                    if keep:
                        # preserve ratio when resizing to fit box
                        ratio = min(w / img.width, h / img.height)
                        nw = int(img.width * ratio)
                        nh = int(img.height * ratio)
                    else:
                        nw, nh = w, h

                # don't upscale
                if self.no_upscale.isChecked():
                    nw = min(nw, img.width)
                    nh = min(nh, img.height)

                resized = fast_resize_image(img, (nw, nh), Image.Resampling.LANCZOS)

                # apply preview sharpening
                sharp = self.sharpen_combo.currentText()
                if sharp != "None":
                    from PIL import ImageFilter
                    factor = {"Low":1, "Medium":2, "High":3}.get(sharp, 0)
                    if factor:
                        resized = resized.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

                # compute estimate byte size
                from io import BytesIO
                temp = resized.copy()
                fmt = self.format_combo.currentText()
                if fmt.upper() in ("JPG","JPEG"):
                    # fill background if transparency exists
                    if temp.mode in ("RGBA","LA") and self.bg_fill_color:
                        bg = Image.new("RGB", temp.size, self.bg_fill_color.name())
                        bg.paste(temp, mask=temp.split()[-1])
                        temp = bg
                buf = BytesIO()
                save_args = {}
                if fmt.upper() in ("JPG","JPEG"):
                    save_args["quality"] = self.quality_slider.value()
                try:
                    dpi = int(self.dpi_combo.currentText())
                    save_args["dpi"] = (dpi, dpi)
                except Exception as e:
                    logger.debug(f"Could not parse DPI setting: {e}")
                temp.save(buf, fmt, **save_args)
                new_est = buf.tell()
                percent = (1 - new_est / orig_size) * 100 if orig_size > 0 else 0
                self.stats_label.setText(f"Old: {orig_w}x{orig_h} ({orig_size//1024}KB)\nNew: {resized.width}x{resized.height} ({new_est//1024}KB) ({percent:.1f}% reduction)")

                data = resized.convert("RGB").tobytes()
                qimg = QImage(data, resized.width, resized.height, resized.width * 3, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)

                # cache pixmap and delegate zooming to helper
                self._preview_pixmap = pix
                self.apply_resize_zoom()
        except Exception:
            pass

    # start_process and perform_batch_resize unchanged (same as before)
    def start_process(self):
        if not self.input_dir or not self.output_dir: return
        try:
            width = int(self.width_input.text() or 1920)
            height = int(self.height_input.text() or 1080)
            scale = float(self.scale_input.text()) if self.scale_input.text().strip() else 0
            keep_ratio = self.keep_ratio.isChecked()
            output_format = self.format_combo.currentText()
            quality = self.quality_slider.value()
            target_kb = int(self.target_size_input.text() or 0) if self.target_size_input.text().strip() else 0
            no_upscale = self.no_upscale.isChecked()
            orient = self.orientation_check.isChecked()
            dpi = int(self.dpi_combo.currentText())
            sharpen = self.sharpen_combo.currentText()
            bg_color = self.bg_fill_color.name()
        except:
            return

        def worker_func(progress_callback):
            self.perform_batch_resize(
                progress_callback, self.input_dir, self.output_dir,
                width, height, scale, keep_ratio,
                output_format, quality, target_kb,
                no_upscale, orient, dpi, sharpen, bg_color
            )
        self.worker = Worker(worker_func)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(lambda: self.progress.setVisible(False))
        self.worker.start()

    def perform_batch_resize(self, progress_callback, input_dir, output_dir, width, height, scale, keep_ratio, output_format, quality, target_kb, no_upscale, orient, dpi, sharpen, bg_color):
        # allow input_dir to be a single file or a folder
        if os.path.isdir(input_dir):
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            base_dir = input_dir
        else:
            files = [os.path.basename(input_dir)]
            base_dir = os.path.dirname(input_dir)
        total = len(files) or 1
        for i, filename in enumerate(files):
            in_path = os.path.join(base_dir, filename)
            base, ext = os.path.splitext(filename)
            out_ext = '.' + output_format.lower()
            out_filename = f"{base}_resized{out_ext}"
            out_path = os.path.join(output_dir, out_filename)
            try:
                with Image.open(in_path) as img:
                    if orient:
                        try:
                            from PIL import ImageOps
                            img = ImageOps.exif_transpose(img)
                        except Exception as e:
                            logger.debug(f"EXIF transpose failed for {in_path}: {e}")

                    if scale > 0:
                        new_w = int(img.width * scale / 100)
                        new_h = int(img.height * scale / 100)
                    else:
                        if keep_ratio:
                            ratio = min(width / img.width, height / img.height)
                            new_w = int(img.width * ratio)
                            new_h = int(img.height * ratio)
                        else:
                            new_w, new_h = width, height

                    if no_upscale:
                        new_w = min(new_w, img.width)
                        new_h = min(new_h, img.height)

                    resized = fast_resize_image(img, (new_w, new_h), Image.Resampling.LANCZOS)

                    # apply sharpening if requested
                    if sharpen != "None":
                        from PIL import ImageFilter
                        factor_map = {"Low":1, "Medium":2, "High":3}
                        factor = factor_map.get(sharpen, 0)
                        if factor:
                            resized = resized.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

                    # handle save
                    save_kwargs = {}
                    if output_format.upper() in ("JPG", "JPEG"):
                        # fill bg for transparency
                        if resized.mode in ("RGBA", "LA") and bg_color:
                            bg = Image.new("RGB", resized.size, bg_color)
                            bg.paste(resized, mask=resized.split()[-1])
                            resized = bg
                        # DPI
                        save_kwargs["dpi"] = (dpi, dpi)
                        if target_kb > 0:
                            q = quality
                            tmp = out_path + ".tmp.jpg"
                            max_iterations = 20  # Safety limit to prevent infinite loop
                            iteration = 0
                            while iteration < max_iterations:
                                resized.save(tmp, "JPEG", quality=q, optimize=True, **save_kwargs)
                                if os.path.getsize(tmp) / 1024 <= target_kb or q <= 20:
                                    break
                                q = max(20, q - 5)
                                iteration += 1
                            os.replace(tmp, out_path)
                        else:
                            resized.save(out_path, "JPEG", quality=quality, optimize=True, **save_kwargs)
                    else:
                        if output_format.upper() in ("PNG","GIF"):
                            save_kwargs["dpi"] = (dpi, dpi)
                        resized.save(out_path, output_format.upper(), optimize=True, **save_kwargs)
                    
                    # Free memory after processing each image
                    del resized
            except Exception as e:
                logger.warning(f"Failed to process {os.path.basename(in_path)}: {e}")
            
            progress_callback(int((i + 1) / total * 100))
            
            # Periodic garbage collection every 10 images
            if (i + 1) % 10 == 0:
                gc.collect()


# ===================================================================
# BATCH WATERMARK - split + zoom + wheel + GIF
# ===================================================================

class PreviewWorker(QObject):
    finished = Signal(QPixmap)
    
    def __init__(self, original_pil, slider_vals, before_after, zoom_val, lut_data=None, lut_strength=60, lut_swap_rb=False, color_grading=None):
        super().__init__()
        self.original_pil = original_pil
        self.slider_vals = slider_vals
        self.before_after = before_after
        self.zoom_val = zoom_val
        self.lut_data = lut_data
        self.lut_strength = lut_strength
        self.lut_swap_rb = lut_swap_rb
        self.color_grading = color_grading or {
            'global': {'hue': 0, 'sat': 0, 'intensity': 0},
            'shadows': {'hue': 0, 'sat': 0, 'intensity': 0},
            'midtones': {'hue': 0, 'sat': 0, 'intensity': 0},
            'highlights': {'hue': 0, 'sat': 0, 'intensity': 0}
        }
    
    def process(self):
        """Process image in background thread"""
        try:
            # Create thumbnail for preview
            thumb = self.original_pil.copy()
            thumb.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            
            # Apply adjustments using optimized method
            edited = self._apply_adjustments_optimized(thumb)
            
            # Create comparison if needed
            if self.before_after:
                comp_width = edited.width * 2 + 4
                comp_height = edited.height
                comparison = Image.new("RGB", (comp_width, comp_height), (20, 26, 34))
                comparison.paste(thumb, (0, 0))
                comparison.paste(edited, (edited.width + 4, 0))
                display_img = comparison
            else:
                display_img = edited
            
            # Convert to QPixmap efficiently
            data = display_img.tobytes()
            qimg = QImage(data, display_img.width, display_img.height, 
                         display_img.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Emit the result
            self.finished.emit(pixmap)
        
        except Exception as e:
            print(f"Preview worker error: {e}")
    
    def _apply_adjustments_optimized(self, img):
        """Optimized adjustment application using PIL for speed"""
        if not img:
            return img.copy() if img else None
        
        img = img.copy().convert("RGB")
        
        # Get slider values
        exp = self.slider_vals.get("Exposure", 0) / 70.0
        con = 1.0 + self.slider_vals.get("Contrast", 0) / 80.0
        highlights = self.slider_vals.get("Highlights", 0) / 150.0
        shadows = self.slider_vals.get("Shadows", 0) / 150.0
        bright = self.slider_vals.get("Brightness", 0) / 100.0
        sat = 1.0 + self.slider_vals.get("Saturation", 0) / 70.0
        vib = self.slider_vals.get("Vibrance", 0) / 100.0
        temp = self.slider_vals.get("Temperature", 0) / 300.0
        tint = self.slider_vals.get("Tint", 0) / 400.0
        clarity = self.slider_vals.get("Clarity", 0) / 200.0
        dehaze = self.slider_vals.get("Dehaze", 0) / 300.0
        tone = self.slider_vals.get("ToneCurve", 0) / 50.0
        grain = self.slider_vals.get("Grain", 0) / 100.0
        sharpness_val = self.slider_vals.get("Sharpness", 100) / 100.0
        
        # Apply brightness directly using PIL
        if bright != 0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1 + bright * 0.3)
        
        # Apply exposure using brightness
        if exp != 0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1 + exp * 0.5)
        
        # Apply contrast using PIL
        if con != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(con)
        
        # Apply saturation using PIL (clamp to prevent negative values)
        if sat != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(max(0.0, sat))
        
        # Apply sharpness
        if sharpness_val != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness_val)
        
        # Apply temperature and tint with numpy (fast approach)
        if temp != 0 or tint != 0 or highlights != 0 or shadows != 0 or vib != 0 or clarity != 0 or dehaze != 0 or tone != 0:
            img_arr = np.array(img, dtype=np.float32) / 255.0
            
            # Highlights/Shadows using brightness mask
            if highlights != 0 or shadows != 0:
                brightness = np.mean(img_arr, axis=2)
                highlight_mask = brightness > 0.5
                if highlights != 0:
                    img_arr[highlight_mask] = np.clip(img_arr[highlight_mask] * (1 + highlights * 0.15), 0, 1)
                if shadows != 0:
                    img_arr[~highlight_mask] = np.clip(img_arr[~highlight_mask] * (1 + shadows * 0.15), 0, 1)
            
            # Temperature adjustment
            if temp > 0:
                img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + temp * 0.15, 0, 1)  # Red
            elif temp < 0:
                img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] - temp * 0.15, 0, 1)  # Blue
            
            # Tint adjustment
            if tint > 0:
                img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] + tint * 0.15, 0, 1)  # Green
            elif tint < 0:
                img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + tint * 0.15, 0, 1)  # Red
            
            # Vibrance (boost saturation for less saturated colors)
            if vib != 0:
                mean = np.mean(img_arr, axis=2, keepdims=True)
                img_arr = mean + (img_arr - mean) * (1 + vib * 0.3)
                img_arr = np.clip(img_arr, 0, 1)
            
            # Clarity (boost local contrast)
            if clarity != 0:
                pass  # Will use PIL enhancement after conversion
            
            # Dehaze (use the same contrast reduction approach)
            if dehaze != 0:
                pass  # Will use PIL enhancement after conversion
            
            # Tone curve
            if tone != 0:
                brightness = np.mean(img_arr, axis=2, keepdims=True)
                adjusted = 0.5 + (brightness - 0.5) * (1 + tone * 0.2)
                ratio = np.divide(adjusted, brightness, where=brightness > 0.001, out=np.ones_like(brightness))
                img_arr = np.clip(img_arr * ratio, 0, 1)
            
            img = Image.fromarray((img_arr * 255).astype(np.uint8), mode='RGB')
        
        # Clarity (local contrast) using PIL
        if clarity != 0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1 + clarity * 0.3)
        
        # Dehaze (reduce contrast)
        if dehaze != 0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1 - dehaze * 0.2)
        
        # Grain effect (add noise)
        if grain > 0:
            img_arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, grain * 1.5, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255)
            img = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')

        # COLOR GRADING - Apply color wheels (global, shadows, midtones, highlights)
        has_color_grading = any(
            self.color_grading[tr]['intensity'] > 0 
            for tr in ['global', 'shadows', 'midtones', 'highlights']
        )
        
        if has_color_grading:
            img_arr = np.array(img, dtype=np.float32) / 255.0
            
            # Apply global color grading first (affects entire image uniformly)
            global_grading = self.color_grading['global']
            global_intensity = global_grading['intensity'] / 100.0
            
            if global_intensity > 0 and global_grading['sat'] > 0:
                hue = global_grading['hue']
                sat = global_grading['sat']
                color = colorsys.hsv_to_rgb(hue / 360.0, sat, 1.0)
                color_shift = np.array(color, dtype=np.float32)
                
                for c in range(3):
                    shift = (color_shift[c] - 0.5) * global_intensity * 0.25
                    img_arr[:,:,c] = np.clip(img_arr[:,:,c] + shift, 0, 1)
            
            # Calculate luminance for each pixel
            luminance = 0.299 * img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 * img_arr[:,:,2]
            
            # Apply color grading for each tonal range
            for tone_range, lum_min, lum_max in [
                ('shadows', 0.0, 0.33),
                ('midtones', 0.33, 0.67),
                ('highlights', 0.67, 1.0)
            ]:
                grading = self.color_grading[tone_range]
                intensity = grading['intensity'] / 100.0
                
                if intensity > 0:
                    # Calculate mask for this tonal range
                    mask = np.zeros_like(luminance)
                    in_range = (luminance >= lum_min) & (luminance < lum_max)
                    mask[in_range] = 1.0
                    
                    # Apply smooth falloff
                    falloff = 0.1
                    if tone_range == 'shadows':
                        transition = (luminance - lum_max) / falloff
                        mask = np.clip(mask * (1 - np.clip(transition, 0, 1)), 0, 1)
                    elif tone_range == 'highlights':
                        transition = (lum_min - luminance) / falloff
                        mask = np.clip(mask * (1 - np.clip(transition, 0, 1)), 0, 1)
                    
                    # Convert hue (0-360) and saturation (0-1) to RGB color shift
                    hue = grading['hue']
                    sat = grading['sat']
                    
                    # Create color from hue/sat
                    if sat > 0:
                        color = colorsys.hsv_to_rgb(hue / 360.0, sat, 1.0)
                        color_shift = np.array(color, dtype=np.float32)
                        
                        # Apply color shift with mask and intensity
                        for c in range(3):
                            shift = (color_shift[c] - 0.5) * intensity * 0.3
                            img_arr[:,:,c] = img_arr[:,:,c] + shift * mask[:,:,np.newaxis].squeeze()
            
            img_arr = np.clip(img_arr, 0, 1)
            img = Image.fromarray((img_arr * 255).astype(np.uint8), mode='RGB')

        # LUT (3D .cube)
        if self.lut_data is not None:
            img_arr = np.array(img, dtype=np.float32) / 255.0
            lut_arr = apply_lut_to_array(img_arr, self.lut_data, self.lut_swap_rb)
            strength = max(0.0, min(1.0, self.lut_strength / 100.0))
            img_arr = img_arr * (1 - strength) + lut_arr * strength
            img = Image.fromarray((img_arr * 255).astype(np.uint8), mode='RGB')
        
        return img


# ===================================================================
# PHOTO EDITING - Professional image editor with split preview
# ===================================================================
class PhotoEditingPage(CardPage):
    def __init__(self):
        super().__init__("Photo Editing")

        # Preview update worker and signals
        self.preview_worker = None
        self.preview_update_timer = QTimer()
        self.preview_update_timer.setSingleShot(True)
        self.preview_update_timer.setInterval(100)  # 100ms debounce
        self.preview_update_timer.timeout.connect(self._start_preview_worker)

        # allow user to resize panels
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(8)
        self.splitter.setStyleSheet("QSplitter::handle { background: #00ffc6; border-radius: 3px; }")
        
        # LEFT PANEL - Container with scrollable area and fixed buttons
        left_container = QWidget()
        left_container_layout = QVBoxLayout(left_container)
        left_container_layout.setContentsMargins(0, 0, 0, 0)
        left_container_layout.setSpacing(0)
        
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Enable smooth scrolling
        left_scroll.verticalScrollBar().setSingleStep(20)
        left_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
        
        left_widget = QWidget()
        self.left_layout = QVBoxLayout(left_widget)
        self.left_layout.setContentsMargins(30, 10, 30, 10)
        self.left_layout.setSpacing(10)
        
        left_scroll.setWidget(left_widget)
        left_container_layout.addWidget(left_scroll, 1)
        
        # Fixed bottom controls (outside scroll area)
        fixed_controls = QWidget()
        fixed_controls_layout = QVBoxLayout(fixed_controls)
        fixed_controls_layout.setContentsMargins(30, 10, 30, 10)
        fixed_controls_layout.setSpacing(8)
        
        self.before_after_check = QCheckBox("Before/After")
        self.before_after_check.setChecked(False)
        self.before_after_check.setStyleSheet("font-size: 13px;")
        fixed_controls_layout.addWidget(self.before_after_check)
        
        btn_h = QHBoxLayout()
        btn_h.setSpacing(8)
        self.export_btn = QPushButton("💾 Export")
        self.export_btn.setMinimumHeight(40)
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.setStyleSheet("font-size: 13px; padding: 10px;")
        btn_h.addWidget(self.export_btn)
        btn_h.addWidget(self.reset_btn)
        fixed_controls_layout.addLayout(btn_h)
        
        left_container_layout.addWidget(fixed_controls, 0)
        
        # RIGHT PANEL
        right = QWidget()
        self.right_layout = QVBoxLayout(right)
        
        self.splitter.addWidget(left_container)
        self.splitter.addWidget(right)
        self.splitter.setSizes([400, 600])
        self.card_layout.addWidget(self.splitter, 1)

        # TOP - FILE SELECTION (compact)
        top_h = QHBoxLayout()
        top_h.setSpacing(6)
        self.input_btn = QPushButton("📷 Photo")
        self.output_btn = QPushButton("💾 Output")
        self.input_btn.setMaximumWidth(225)
        self.output_btn.setMaximumWidth(225)
        self.input_btn.setMinimumHeight(40)
        self.output_btn.setMinimumHeight(40)
        self.input_btn.setStyleSheet("font-size: 13px; padding: 8px;")
        self.output_btn.setStyleSheet("font-size: 13px; padding: 8px;")
        top_h.addWidget(self.input_btn)
        top_h.addWidget(self.output_btn)
        self.left_layout.addLayout(top_h)

        # PRESETS - PROMINENT
        presets_row = QHBoxLayout()
        self.filter_combo = NoWheelComboBox()
        self.presets = {
            "None": {},
            
            # === PORTRAIT & PEOPLE ===
            "Wedding": {
                "exposure": 10, "saturation": 5, "vibrance": 15, "contrast": 10, "temperature": 5,
                "color_grading": {"highlights": {"hue": 30, "sat": 0.3, "intensity": 20}}
            },
            "Bright & Airy": {
                "exposure": 25, "highlights": 30, "saturation": 10, "vibrance": 15, "shadows": 10,
                "color_grading": {"global": {"hue": 45, "sat": 0.15, "intensity": 12}}
            },
            "Portrait Classic": {
                "saturation": 10, "vibrance": 20, "shadows": 15, "clarity": -10, "sharpness": 120,
                "color_grading": {"midtones": {"hue": 20, "sat": 0.2, "intensity": 18}}
            },
            "Soft Skin": {"clarity": -25, "sharpness": 75, "vibrance": 5, "saturation": 5, "temperature": 5},
            "Fashion Editorial": {
                "contrast": 20, "clarity": 15, "vibrance": 25, "saturation": 15, "sharpness": 130,
                "color_grading": {
                    "shadows": {"hue": 200, "sat": 0.4, "intensity": 25},
                    "highlights": {"hue": 40, "sat": 0.3, "intensity": 20}
                }
            },
            "Beauty Portrait": {
                "exposure": 8, "clarity": -15, "sharpness": 110, "vibrance": 10, "temperature": 3,
                "color_grading": {"highlights": {"hue": 25, "sat": 0.25, "intensity": 15}}
            },
            
            # === LANDSCAPE & NATURE ===
            "Landscape Pro": {
                "saturation": 20, "contrast": 20, "vibrance": 25, "clarity": 15, "sharpness": 130,
                "color_grading": {
                    "shadows": {"hue": 220, "sat": 0.3, "intensity": 20},
                    "highlights": {"hue": 50, "sat": 0.25, "intensity": 18}
                }
            },
            "Vibrant Nature": {
                "saturation": 35, "vibrance": 35, "contrast": 20, "clarity": 20, "sharpness": 140,
                "color_grading": {
                    "midtones": {"hue": 120, "sat": 0.3, "intensity": 25}
                }
            },
            "Mountain Vista": {
                "contrast": 25, "clarity": 25, "dehaze": 20, "saturation": 18, "vibrance": 20,
                "color_grading": {
                    "shadows": {"hue": 230, "sat": 0.35, "intensity": 28},
                    "highlights": {"hue": 200, "sat": 0.2, "intensity": 15}
                }
            },
            "Forest Green": {
                "saturation": 25, "vibrance": 30, "contrast": 15, "clarity": 10,
                "color_grading": {
                    "midtones": {"hue": 120, "sat": 0.4, "intensity": 30},
                    "shadows": {"hue": 140, "sat": 0.25, "intensity": 20}
                }
            },
            "Ocean Blue": {
                "saturation": 20, "vibrance": 25, "contrast": 18, "clarity": 12,
                "color_grading": {
                    "shadows": {"hue": 200, "sat": 0.5, "intensity": 35},
                    "midtones": {"hue": 190, "sat": 0.35, "intensity": 25}
                }
            },
            
            # === TIME OF DAY ===
            "Golden Hour": {
                "temperature": 25, "exposure": 5, "saturation": 15, "vibrance": 20,
                "color_grading": {
                    "highlights": {"hue": 35, "sat": 0.5, "intensity": 35},
                    "midtones": {"hue": 40, "sat": 0.3, "intensity": 20}
                }
            },
            "Blue Hour": {
                "temperature": -20, "tint": -10, "contrast": 15, "saturation": 10,
                "color_grading": {
                    "shadows": {"hue": 220, "sat": 0.5, "intensity": 35},
                    "midtones": {"hue": 210, "sat": 0.3, "intensity": 20}
                }
            },
            "Warm Sunset": {
                "temperature": 35, "exposure": 10, "saturation": 20, "vibrance": 25,
                "color_grading": {
                    "shadows": {"hue": 280, "sat": 0.4, "intensity": 28},
                    "highlights": {"hue": 25, "sat": 0.6, "intensity": 40}
                }
            },
            "Sunrise Magic": {
                "temperature": 20, "exposure": 8, "saturation": 18, "vibrance": 22, "contrast": 12,
                "color_grading": {
                    "shadows": {"hue": 290, "sat": 0.35, "intensity": 25},
                    "highlights": {"hue": 35, "sat": 0.5, "intensity": 35}
                }
            },
            "Midday Bright": {"exposure": 15, "highlights": 25, "saturation": 15, "vibrance": 18, "clarity": 10},
            "Twilight": {
                "temperature": -15, "exposure": -5, "contrast": 20, "saturation": 12,
                "color_grading": {
                    "shadows": {"hue": 240, "sat": 0.45, "intensity": 32},
                    "highlights": {"hue": 30, "sat": 0.25, "intensity": 18}
                }
            },
            
            # === CINEMATIC & MOODY ===
            "Cinematic": {
                "exposure": 5, "contrast": 15, "temperature": 10, "tint": -5,
                "color_grading": {
                    "shadows": {"hue": 180, "sat": 0.5, "intensity": 35},
                    "highlights": {"hue": 30, "sat": 0.4, "intensity": 30}
                }
            },
            "Blockbuster": {
                "contrast": 20, "saturation": 12, "vibrance": 15, "clarity": 18,
                "color_grading": {
                    "shadows": {"hue": 190, "sat": 0.6, "intensity": 40},
                    "highlights": {"hue": 28, "sat": 0.5, "intensity": 35}
                }
            },
            "Warm & Moody": {
                "temperature": 20, "exposure": -5, "contrast": 20, "saturation": 15,
                "color_grading": {"shadows": {"hue": 25, "sat": 0.35, "intensity": 28}}
            },
            "Cold Moody": {
                "temperature": -25, "contrast": 25, "saturation": 5, "shadows": -15,
                "color_grading": {
                    "shadows": {"hue": 200, "sat": 0.5, "intensity": 35},
                    "midtones": {"hue": 210, "sat": 0.25, "intensity": 18}
                }
            },
            "Dark Cinematic": {
                "exposure": -12, "contrast": 28, "shadows": -18, "clarity": 15,
                "color_grading": {
                    "shadows": {"hue": 195, "sat": 0.45, "intensity": 30},
                    "highlights": {"hue": 35, "sat": 0.35, "intensity": 25}
                }
            },
            "Noir Mood": {
                "exposure": -8, "contrast": 30, "saturation": -30, "clarity": 12, "shadows": -12,
                "color_grading": {"shadows": {"hue": 240, "sat": 0.25, "intensity": 15}}
            },
            
            # === VINTAGE & FILM ===
            "Vintage Color": {
                "exposure": 5, "saturation": -10, "contrast": -5, "temperature": 15,
                "color_grading": {
                    "shadows": {"hue": 240, "sat": 0.3, "intensity": 22},
                    "highlights": {"hue": 50, "sat": 0.4, "intensity": 28}
                }
            },
            "70s Film": {
                "temperature": 18, "saturation": -8, "contrast": -8, "vibrance": 5,
                "color_grading": {
                    "shadows": {"hue": 25, "sat": 0.35, "intensity": 25},
                    "highlights": {"hue": 45, "sat": 0.3, "intensity": 20}
                }
            },
            "Polaroid": {
                "exposure": 8, "saturation": -12, "contrast": -10, "temperature": 12, "vibrance": -5,
                "color_grading": {
                    "global": {"hue": 180, "sat": 0.2, "intensity": 15},
                    "highlights": {"hue": 50, "sat": 0.25, "intensity": 18}
                }
            },
            "Kodachrome": {
                "saturation": 25, "vibrance": 20, "contrast": 15, "temperature": 8,
                "color_grading": {
                    "shadows": {"hue": 350, "sat": 0.3, "intensity": 20},
                    "highlights": {"hue": 45, "sat": 0.35, "intensity": 25}
                }
            },
            "Retro Faded": {
                "saturation": -20, "temperature": 10, "tint": 5, "highlights": 10, "contrast": -12,
                "color_grading": {"midtones": {"hue": 55, "sat": 0.25, "intensity": 18}}
            },
            
            # === MODERN & CLEAN ===
            "Crisp Modern": {"contrast": 25, "clarity": 20, "sharpness": 140, "vibrance": 15},
            "Clean Minimal": {
                "exposure": 12, "highlights": 18, "saturation": -8, "contrast": 8, "clarity": 5,
                "color_grading": {"global": {"hue": 0, "sat": 0.05, "intensity": 8}}
            },
            "Pure White": {"exposure": 28, "highlights": 30, "saturation": -15, "brightness": 18, "contrast": -5},
            "Vivid Pop": {
                "saturation": 30, "vibrance": 30, "contrast": 20, "clarity": 20,
                "color_grading": {"midtones": {"hue": 200, "sat": 0.15, "intensity": 12}}
            },

            # === NO COLOR GRADING (Clean Looks) ===
            "Natural Standard": {"contrast": 8, "vibrance": 10, "sharpness": 110},
            "Studio Neutral": {"exposure": 4, "contrast": 6, "clarity": 6, "saturation": -2},
            "Clean Portrait": {"exposure": 6, "contrast": 8, "clarity": -8, "sharpness": 105, "temperature": 3},
            "Detail Boost": {"contrast": 18, "clarity": 22, "sharpness": 145, "dehaze": 10},
            "Soft Natural": {"contrast": -4, "clarity": -12, "saturation": 4, "vibrance": 8, "highlights": 6},
            "Commercial Bright": {"exposure": 14, "highlights": 18, "contrast": 10, "clarity": 8, "sharpness": 120},
            "Documentary": {"contrast": 14, "clarity": 12, "vibrance": 6, "sharpness": 125},
            "Flat Base": {"contrast": -15, "saturation": -10, "vibrance": -8, "highlights": -6, "shadows": 10},
            "Print Ready": {"contrast": 12, "saturation": 6, "vibrance": 8, "sharpness": 115, "dehaze": 4},
            "Web Crisp": {"contrast": 16, "clarity": 14, "sharpness": 135, "vibrance": 12},
            
            # === SPECIAL EFFECTS ===
            "High Key": {"exposure": 30, "highlights": 20, "saturation": -5, "brightness": 15},
            "Low Key": {
                "exposure": -20, "contrast": 25, "shadows": -20,
                "color_grading": {"shadows": {"hue": 240, "sat": 0.2, "intensity": 15}}
            },
            "Film Noir": {"saturation": -100, "contrast": 30, "shadows": -10, "clarity": 10},
            "Matte Finish": {
                "contrast": -10, "saturation": -5, "clarity": -5,
                "color_grading": {"global": {"hue": 0, "sat": 0.1, "intensity": 8}}
            },
            "Dreamy Haze": {"clarity": -20, "sharpness": 70, "saturation": 10, "highlights": 15, "exposure": 5},
            "Deep Shadow": {
                "shadows": -25, "contrast": 20, "clarity": 15,
                "color_grading": {"shadows": {"hue": 240, "sat": 0.3, "intensity": 20}}
            },
            "Soft Focus": {"clarity": -30, "sharpness": 60, "highlights": 12, "vibrance": 8},
            
            # === BLACK & WHITE ===
            "B&W Classic": {"saturation": -100, "contrast": 15},
            "B&W High Contrast": {"saturation": -100, "contrast": 35, "clarity": 10},
            "B&W Dramatic": {"saturation": -100, "contrast": 40, "shadows": -15, "clarity": 20},
            "B&W Soft": {"saturation": -100, "contrast": 5, "clarity": -10, "highlights": 10},
            
            # === INSTAGRAM-STYLE FILTERS (Enhanced) ===
            "Nashville": {
                "temperature": 20, "tint": -10, "saturation": -15, "contrast": -10, "exposure": 5,
                "color_grading": {"midtones": {"hue": 45, "sat": 0.25, "intensity": 18}}
            },
            "Valencia": {
                "temperature": 10, "exposure": 8, "contrast": 8, "saturation": 20, "vibrance": 15,
                "color_grading": {"highlights": {"hue": 35, "sat": 0.3, "intensity": 20}}
            },
            "X-Pro II": {
                "temperature": 15, "shadows": -20, "highlights": 10, "vibrance": 20, "contrast": 15,
                "color_grading": {"shadows": {"hue": 180, "sat": 0.35, "intensity": 25}}
            },
            "Lo-Fi": {
                "saturation": 30, "contrast": 20, "temperature": 5, "shadows": -15,
                "color_grading": {"shadows": {"hue": 280, "sat": 0.3, "intensity": 22}}
            },
            "Earlybird": {
                "temperature": 20, "saturation": 15, "contrast": 15, "tint": 10, "exposure": 5,
                "color_grading": {"highlights": {"hue": 40, "sat": 0.4, "intensity": 28}}
            },
            "Clarendon": {
                "saturation": 20, "contrast": 20, "highlights": 15, "shadows": 15, "vibrance": 20,
                "color_grading": {"highlights": {"hue": 200, "sat": 0.2, "intensity": 15}}
            },
            
            # === FOOD & PRODUCT ===
            "Food Bright": {
                "exposure": 12, "saturation": 25, "vibrance": 28, "contrast": 15, "clarity": 10,
                "color_grading": {"highlights": {"hue": 30, "sat": 0.3, "intensity": 20}}
            },
            "Food Moody": {
                "exposure": -8, "contrast": 22, "saturation": 20, "shadows": -10, "clarity": 15,
                "color_grading": {"shadows": {"hue": 25, "sat": 0.35, "intensity": 25}}
            },
            "Product Clean": {"exposure": 10, "highlights": 15, "saturation": 10, "clarity": 18, "sharpness": 135},
            
            # === TRAVEL & ADVENTURE ===
            "Travel Warm": {
                "temperature": 15, "saturation": 20, "vibrance": 22, "contrast": 12, "clarity": 10,
                "color_grading": {"highlights": {"hue": 35, "sat": 0.3, "intensity": 22}}
            },
            "Travel Cool": {
                "temperature": -12, "saturation": 18, "vibrance": 20, "contrast": 15, "clarity": 12,
                "color_grading": {"midtones": {"hue": 200, "sat": 0.25, "intensity": 18}}
            },
            "Adventure": {
                "saturation": 25, "vibrance": 28, "contrast": 20, "clarity": 18, "sharpness": 130,
                "color_grading": {
                    "shadows": {"hue": 220, "sat": 0.3, "intensity": 20},
                    "highlights": {"hue": 45, "sat": 0.25, "intensity": 18}
                }
            },
            
            # === URBAN & STREET ===
            "Urban Grit": {
                "contrast": 25, "clarity": 20, "saturation": 10, "shadows": -10, "dehaze": 15,
                "color_grading": {"shadows": {"hue": 200, "sat": 0.25, "intensity": 18}}
            },
            "Street Photography": {
                "contrast": 20, "clarity": 15, "saturation": 12, "vibrance": 15,
                "color_grading": {"midtones": {"hue": 30, "sat": 0.2, "intensity": 15}}
            },
            "Neon Night": {
                "exposure": -10, "contrast": 30, "saturation": 35, "vibrance": 40, "shadows": -15,
                "color_grading": {
                    "shadows": {"hue": 280, "sat": 0.5, "intensity": 35},
                    "highlights": {"hue": 180, "sat": 0.6, "intensity": 40}
                }
            },
            
            # === ARTISTIC ===
            "Desaturated": {
                "saturation": -40, "contrast": 12, "clarity": 8,
                "color_grading": {"global": {"hue": 200, "sat": 0.15, "intensity": 10}}
            },
            "Cross Process": {
                "temperature": 10, "contrast": 25, "saturation": 20,
                "color_grading": {
                    "shadows": {"hue": 180, "sat": 0.4, "intensity": 30},
                    "highlights": {"hue": 60, "sat": 0.45, "intensity": 32}
                }
            },
            "Bleach Bypass": {
                "saturation": -25, "contrast": 30, "clarity": 15,
                "color_grading": {"highlights": {"hue": 200, "sat": 0.2, "intensity": 15}}
            },
            "Cyberpunk": {
                "contrast": 25, "saturation": 30, "vibrance": 35,
                "color_grading": {
                    "shadows": {"hue": 280, "sat": 0.7, "intensity": 50},
                    "highlights": {"hue": 180, "sat": 0.6, "intensity": 45}
                }
            },
            "Pastel Dream": {
                "exposure": 8, "saturation": -15, "contrast": -12, "highlights": 15,
                "color_grading": {
                    "shadows": {"hue": 280, "sat": 0.3, "intensity": 20},
                    "midtones": {"hue": 180, "sat": 0.25, "intensity": 18},
                    "highlights": {"hue": 45, "sat": 0.3, "intensity": 20}
                }
            }
        }
        self.filter_combo.addItems(self.presets.keys())
        save_preset_btn = QPushButton("💾 Save Preset")
        load_preset_btn = QPushButton("📂 Load Preset")
        save_preset_btn.setMaximumWidth(100)
        load_preset_btn.setMaximumWidth(100)
        presets_row.addWidget(self.filter_combo)
        presets_row.addWidget(save_preset_btn)
        presets_row.addWidget(load_preset_btn)
        preset_label = QLabel("Presets")
        preset_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.filter_combo.setStyleSheet("font-size: 12px; padding: 6px;")
        self.left_layout.addWidget(preset_label)
        self.left_layout.addLayout(presets_row)
        save_preset_btn.clicked.connect(self.save_preset)
        load_preset_btn.clicked.connect(self.load_preset)

        # LUT Controls - Collapsible
        self.lut_data = None
        self.lut_path = ""
        self.lut_strength = 60
        self.lut_swap_rb = False
        
        self.lut_group = CollapsibleGroupBox("LUT (3D .cube)")
        self.lut_label = QLabel("No LUT loaded")
        self.lut_label.setStyleSheet("font-size: 11px;")
        self.lut_load_btn = QPushButton("Load LUT")
        self.lut_clear_btn = QPushButton("Clear LUT")
        self.lut_swap_check = QCheckBox("Swap R/B (fix blue tint)")
        self.lut_strength_slider = NoWheelSlider(Qt.Horizontal)
        self.lut_strength_slider.setRange(0, 100)
        self.lut_strength_slider.setValue(self.lut_strength)
        self.lut_strength_label = QLabel(f"{self.lut_strength}%")
        lut_strength_row = QHBoxLayout()
        lut_strength_row.addWidget(QLabel("LUT Strength"))
        lut_strength_row.addWidget(self.lut_strength_slider)
        lut_strength_row.addWidget(self.lut_strength_label)
        
        self.lut_group.add_widget(self.lut_label)
        self.lut_group.add_widget(self.lut_load_btn)
        self.lut_group.add_widget(self.lut_clear_btn)
        self.lut_group.add_widget(self.lut_swap_check)
        self.lut_group.add_layout(lut_strength_row)
        
        self.left_layout.addWidget(self.lut_group)

        self.lut_load_btn.clicked.connect(self.load_lut)
        self.lut_clear_btn.clicked.connect(self.clear_lut)
        self.lut_strength_slider.valueChanged.connect(self._update_lut_strength)
        self.lut_swap_check.stateChanged.connect(self._update_lut_swap)

        # COLOR GRADING - Color Wheels (Lightroom-style)
        self.color_grading_group = CollapsibleGroupBox("Color Grading")
        
        # Color Grading Presets
        self.color_grading_presets = {
            "None": {},
            "Teal & Orange": {
                'shadows': {'hue': 180, 'sat': 0.6, 'intensity': 40},
                'highlights': {'hue': 30, 'sat': 0.5, 'intensity': 35}
            },
            "Cinematic Cool": {
                'shadows': {'hue': 200, 'sat': 0.4, 'intensity': 30},
                'midtones': {'hue': 210, 'sat': 0.2, 'intensity': 15}
            },
            "Warm Sunset": {
                'shadows': {'hue': 280, 'sat': 0.3, 'intensity': 25},
                'highlights': {'hue': 25, 'sat': 0.6, 'intensity': 40}
            },
            "Vintage Film": {
                'shadows': {'hue': 240, 'sat': 0.3, 'intensity': 20},
                'midtones': {'hue': 45, 'sat': 0.2, 'intensity': 15},
                'highlights': {'hue': 50, 'sat': 0.4, 'intensity': 25}
            },
            "Cyberpunk": {
                'shadows': {'hue': 280, 'sat': 0.7, 'intensity': 50},
                'highlights': {'hue': 180, 'sat': 0.6, 'intensity': 45}
            },
            "Nordic Blue": {
                'global': {'hue': 200, 'sat': 0.3, 'intensity': 25},
                'shadows': {'hue': 220, 'sat': 0.4, 'intensity': 30}
            },
            "Golden Hour": {
                'midtones': {'hue': 35, 'sat': 0.5, 'intensity': 30},
                'highlights': {'hue': 40, 'sat': 0.6, 'intensity': 40}
            },
            "Horror Green": {
                'shadows': {'hue': 160, 'sat': 0.5, 'intensity': 40},
                'midtones': {'hue': 120, 'sat': 0.3, 'intensity': 20}
            },
            "Purple Dream": {
                'shadows': {'hue': 270, 'sat': 0.6, 'intensity': 45},
                'highlights': {'hue': 300, 'sat': 0.4, 'intensity': 30}
            },
            "Bleach Bypass": {
                'highlights': {'hue': 200, 'sat': 0.2, 'intensity': 15}
            },
            
            # Tri-Tone presets (shadows + midtones + highlights)
            "Tri-Tone Teal Amber Magenta": {
                'shadows': {'hue': 190, 'sat': 0.55, 'intensity': 35},
                'midtones': {'hue': 35, 'sat': 0.35, 'intensity': 24},
                'highlights': {'hue': 320, 'sat': 0.30, 'intensity': 20}
            },
            "Tri-Tone Forest Gold Cyan": {
                'shadows': {'hue': 135, 'sat': 0.45, 'intensity': 30},
                'midtones': {'hue': 48, 'sat': 0.35, 'intensity': 22},
                'highlights': {'hue': 188, 'sat': 0.32, 'intensity': 20}
            },
            "Tri-Tone Indigo Copper Cream": {
                'shadows': {'hue': 238, 'sat': 0.52, 'intensity': 34},
                'midtones': {'hue': 24, 'sat': 0.36, 'intensity': 24},
                'highlights': {'hue': 52, 'sat': 0.24, 'intensity': 16}
            },
            "Tri-Tone Rose Olive Sky": {
                'shadows': {'hue': 342, 'sat': 0.40, 'intensity': 26},
                'midtones': {'hue': 92, 'sat': 0.33, 'intensity': 20},
                'highlights': {'hue': 206, 'sat': 0.38, 'intensity': 25}
            },
            "Tri-Tone Neon Split": {
                'shadows': {'hue': 278, 'sat': 0.68, 'intensity': 44},
                'midtones': {'hue': 184, 'sat': 0.55, 'intensity': 34},
                'highlights': {'hue': 36, 'sat': 0.62, 'intensity': 38}
            },
            "Tri-Tone Silver Teal Gold": {
                'shadows': {'hue': 220, 'sat': 0.30, 'intensity': 20},
                'midtones': {'hue': 180, 'sat': 0.26, 'intensity': 18},
                'highlights': {'hue': 42, 'sat': 0.40, 'intensity': 26}
            }
        }
        
        preset_row = QHBoxLayout()
        self.color_preset_combo = NoWheelComboBox()
        self.color_preset_combo.addItems(self.color_grading_presets.keys())
        self.color_preset_combo.setStyleSheet("font-size: 11px; padding: 4px;")
        self.color_preset_combo.currentIndexChanged.connect(self.apply_color_grading_preset)
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self.color_preset_combo)
        self.color_grading_group.add_layout(preset_row)
        
        # Store color grading values
        self.color_grading = {
            'global': {'hue': 0, 'sat': 0, 'intensity': 0},
            'shadows': {'hue': 0, 'sat': 0, 'intensity': 0},
            'midtones': {'hue': 0, 'sat': 0, 'intensity': 0},
            'highlights': {'hue': 0, 'sat': 0, 'intensity': 0}
        }
        
        # Create color wheels and intensity sliders for each tonal range
        for tone_range in ['global', 'shadows', 'midtones', 'highlights']:
            section_label = QLabel(tone_range.capitalize())
            section_label.setStyleSheet("font-size: 11px; font-weight: bold; margin-top: 10px;")
            self.color_grading_group.add_widget(section_label)
            
            # Color wheel
            wheel = ColorWheelWidget(tone_range.capitalize(), size=100)
            wheel.setObjectName(f"{tone_range}_wheel")
            
            # Intensity slider
            intensity_layout = QHBoxLayout()
            intensity_label = QLabel("Intensity")
            intensity_label.setStyleSheet("font-size: 10px;")
            intensity_slider = NoWheelSlider(Qt.Horizontal)
            intensity_slider.setRange(0, 100)
            intensity_slider.setValue(0)
            intensity_slider.setObjectName(f"{tone_range}_intensity")
            intensity_value_label = QLabel("0%")
            intensity_value_label.setStyleSheet("font-size: 10px;")
            intensity_value_label.setObjectName(f"{tone_range}_intensity_label")
            intensity_layout.addWidget(intensity_label)
            intensity_layout.addWidget(intensity_slider)
            intensity_layout.addWidget(intensity_value_label)
            
            # Center the wheel
            wheel_container = QHBoxLayout()
            wheel_container.addStretch()
            wheel_container.addWidget(wheel)
            wheel_container.addStretch()
            
            self.color_grading_group.add_layout(wheel_container)
            self.color_grading_group.add_layout(intensity_layout)
            
            # Connect signals
            def make_color_changed(tr=tone_range):
                def on_color_changed(color, hue, sat):
                    self.color_grading[tr]['hue'] = hue
                    self.color_grading[tr]['sat'] = sat
                    self.update_live_preview()
                return on_color_changed
            
            def make_intensity_changed(tr=tone_range, lbl=intensity_value_label):
                def on_intensity_changed(value):
                    self.color_grading[tr]['intensity'] = value
                    lbl.setText(f"{value}%")
                    self.update_live_preview()
                return on_intensity_changed
            
            wheel.colorChanged.connect(make_color_changed())
            intensity_slider.valueChanged.connect(make_intensity_changed())
        
        # Reset button for color grading
        reset_color_btn = QPushButton("Reset Color Grading")
        reset_color_btn.clicked.connect(self.reset_color_grading)
        self.color_grading_group.add_widget(reset_color_btn)
        
        self.left_layout.addWidget(self.color_grading_group)

        # ADJUSTMENTS - GROUPED & COMPACT
        self.sliders = {}
        self.slider_labels = {}
        self.group_labels = []
        
        # Add adjustments label
        adjust_label = QLabel("Adjustments")
        adjust_label.setStyleSheet("font-size: 13px; font-weight: bold; margin-top: 10px;")
        self.left_layout.addWidget(adjust_label)

        slider_groups = [
            ("Light", [("Exposure", -100, 100, 0), ("Brightness", -100, 100, 0), ("Highlights", -100, 100, 0), ("Shadows", -100, 100, 0)]),
            ("Color", [("Temperature", -100, 100, 0), ("Tint", -100, 100, 0), ("Saturation", -100, 100, 0), ("Vibrance", -100, 100, 0)]),
            ("Detail", [("Contrast", -100, 100, 0), ("Clarity", -100, 100, 0), ("Sharpness", 0, 200, 100), ("Dehaze", -100, 100, 0)]),
            ("Effects", [("ToneCurve", -50, 50, 0), ("Grain", 0, 100, 0)])
        ]

        for group_name, sliders in slider_groups:
            group_label = QLabel(group_name)
            group_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 15px; margin-bottom: 8px;")
            self.left_layout.addWidget(group_label)
            self.group_labels.append(group_label)
            
            for label, min_val, max_val, default in sliders:
                # Container for each slider
                lbl = QLabel(label)
                lbl.setStyleSheet("font-size: 11px; font-weight: bold;")
                self.left_layout.addWidget(lbl)
                
                slider = NoWheelSlider(Qt.Horizontal)
                slider.setRange(min_val, max_val)
                slider.setValue(default)
                slider.default = default
                slider.setMinimumHeight(24)
                # allow right-click reset
                slider.setContextMenuPolicy(Qt.CustomContextMenu)
                def make_menu(sl=slider, lab=label):
                    def show_menu(pos):
                        menu = sl.createStandardContextMenu()
                        reset_action = menu.addAction("Reset")
                        def do_reset():
                            sl.setValue(sl.default)
                        reset_action.triggered.connect(do_reset)
                        menu.exec_(sl.mapToGlobal(pos))
                    return show_menu
                slider.customContextMenuRequested.connect(make_menu())
                self.left_layout.addWidget(slider)
                
                val_lbl = QLabel(str(default))
                val_lbl.setStyleSheet("font-size: 10px; text-align: center; margin-bottom: 8px;")
                val_lbl.setAlignment(Qt.AlignCenter)
                self.left_layout.addWidget(val_lbl)
                
                def make_update(s, v):
                    def update_val():
                        v.setText(str(s.value()))
                    return update_val
                slider.valueChanged.connect(make_update(slider, val_lbl))
                
                self.sliders[label] = slider
                self.slider_labels[label] = val_lbl

        self.left_layout.addStretch()

        # Note: Before/After, Export and Reset buttons are added to fixed_controls outside the scroll area

        # RIGHT - PREVIEW
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)  # Don't auto-resize, let image control size
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        # Enable smooth scrolling
        self.preview_scroll.horizontalScrollBar().setSingleStep(20)
        self.preview_scroll.verticalScrollBar().setSingleStep(20)
        self.preview_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
            }
            QScrollBar:horizontal, QScrollBar:vertical {
                background: transparent;
            }
            QScrollBar:horizontal {
                height: 10px;
            }
            QScrollBar:vertical {
                width: 10px;
            }
            QScrollBar::handle:horizontal, QScrollBar::handle:vertical {
                background: rgba(0, 255, 198, 0.3);
                border-radius: 5px;
                min-height: 20px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {
                background: rgba(0, 255, 198, 0.5);
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                height: 0px;
                width: 0px;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background: transparent;
            }
        """)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        self.right_layout.addWidget(self.preview_scroll, 1)

        # Zoom Control
        zoom_h = QHBoxLayout()
        zoom_h.addWidget(QLabel("Zoom"))
        self.zoom_slider = ZoomSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        zoom_h.addWidget(self.zoom_slider)
        self.right_layout.addLayout(zoom_h)
        self.right_layout.addStretch()
        
        # Connect button signals AFTER they're created
        self.reset_btn.clicked.connect(self.reset_sliders)

        # State
        self.input_file = ""
        self.output_dir = ""
        self.original_pil = None
        self._original_shown = False  # Flag to show original image on first load
        self._cached_preview = QPixmap()  # Cache for zoom-only scaling
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_live_preview)
        self.preview_worker = None
        self.preview_worker_obj = None

        # Connect Signals
        self.input_btn.clicked.connect(self.select_input)
        self.output_btn.clicked.connect(self.select_output)
        self.export_btn.clicked.connect(self.export_photo)
        self.zoom_slider.valueChanged.connect(self._apply_zoom_only)
        self.filter_combo.currentIndexChanged.connect(self.apply_preset)
        self.before_after_check.stateChanged.connect(self.update_live_preview)

        for slider in self.sliders.values():
            slider.valueChanged.connect(lambda: (self.preview_update_timer.stop(), self.preview_update_timer.start()))

        self.preview_label.wheelEvent = self.wheel_zoom

    def _apply_zoom_only(self):
        """Apply zoom to cached preview without reprocessing"""
        if not hasattr(self, '_cached_preview') or self._cached_preview.isNull():
            return
        
        zoom = self.zoom_slider.value() / 100.0
        sw = int(self._cached_preview.width() * zoom)
        sh = int(self._cached_preview.height() * zoom)
        scaled_pix = self._cached_preview.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.preview_label.setPixmap(scaled_pix)
        self.preview_label.setFixedSize(sw, sh)

    def save_preset(self):
        if not self.sliders: return
        vals = {k: s.value() for k, s in self.sliders.items()}
        file, _ = QFileDialog.getSaveFileName(self, "Save Preset", "", "JSON (*.json)")
        if file:
            try:
                import json
                with open(file, 'w') as f:
                    json.dump(vals, f)
            except Exception as e:
                print("Failed to save preset", e)

    def load_preset(self):
        file, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON (*.json)")
        if file:
            try:
                import json
                with open(file) as f:
                    vals = json.load(f)
                for k, v in vals.items():
                    if k in self.sliders:
                        self.sliders[k].setValue(v)
                        self.slider_labels[k].setText(str(v))
                self.update_live_preview()
            except Exception as e:
                print("Failed to load preset", e)

    def apply_preset(self):
        """Apply preset settings to sliders and color grading"""
        preset_name = self.filter_combo.currentText()
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            
            # Reset all sliders first
            for label, slider in self.sliders.items():
                slider.blockSignals(True)
                default_val = 0 if label != "Sharpness" else 100
                slider.setValue(default_val)
                slider.blockSignals(False)
                # Update label
                self.slider_labels[label].setText(str(default_val))
            
            # Reset color grading
            for tone_range in ['global', 'shadows', 'midtones', 'highlights']:
                self.color_grading[tone_range] = {'hue': 0, 'sat': 0, 'intensity': 0}
                
                # Reset wheel
                wheel = self.color_grading_group.findChild(ColorWheelWidget, f"{tone_range}_wheel")
                if wheel:
                    wheel.reset()
                
                # Reset intensity slider
                slider = self.color_grading_group.findChild(NoWheelSlider, f"{tone_range}_intensity")
                if slider:
                    slider.blockSignals(True)
                    slider.setValue(0)
                    slider.blockSignals(False)
                
                # Reset intensity label
                label = self.color_grading_group.findChild(QLabel, f"{tone_range}_intensity_label")
                if label:
                    label.setText("0%")
            
            # Apply preset values
            for key, value in preset.items():
                if key == "color_grading":
                    # Apply color grading values
                    for tone_range, grading_values in value.items():
                        if tone_range in self.color_grading:
                            # Update data
                            self.color_grading[tone_range] = grading_values.copy()
                            
                            # Update wheel position
                            wheel = self.color_grading_group.findChild(ColorWheelWidget, f"{tone_range}_wheel")
                            if wheel:
                                hue = grading_values.get('hue', 0)
                                sat = grading_values.get('sat', 0)
                                
                                if sat > 0:
                                    # Calculate position on wheel
                                    center = wheel.wheel_size // 2
                                    radius = (center - 5) * sat
                                    angle_rad = math.radians(hue)
                                    x = int(center + radius * math.cos(angle_rad))
                                    y = int(center + radius * math.sin(angle_rad))
                                    wheel.selected_point = (x, y)
                                    wheel.update()
                            
                            # Update intensity slider
                            intensity = grading_values.get('intensity', 0)
                            slider = self.color_grading_group.findChild(NoWheelSlider, f"{tone_range}_intensity")
                            if slider:
                                slider.blockSignals(True)
                                slider.setValue(intensity)
                                slider.blockSignals(False)
                            
                            # Update intensity label
                            label = self.color_grading_group.findChild(QLabel, f"{tone_range}_intensity_label")
                            if label:
                                label.setText(f"{intensity}%")
                
                elif key == "saturation":
                    self.sliders["Saturation"].blockSignals(True)
                    self.sliders["Saturation"].setValue(value)
                    self.slider_labels["Saturation"].setText(str(value))
                    self.sliders["Saturation"].blockSignals(False)
                elif key == "exposure":
                    self.sliders["Exposure"].blockSignals(True)
                    self.sliders["Exposure"].setValue(value)
                    self.slider_labels["Exposure"].setText(str(value))
                    self.sliders["Exposure"].blockSignals(False)
                elif key in self.sliders:
                    self.sliders[key].blockSignals(True)
                    self.sliders[key].setValue(value)
                    self.slider_labels[key].setText(str(value))
                    self.sliders[key].blockSignals(False)
        
        self.update_live_preview()

    def wheel_zoom(self, event):
        delta = event.angleDelta().y()
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(current + (10 if delta > 0 else -10))
        event.accept()

    def select_input(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("PhotoEditingPage_last_input_folder", "")
        self.input_file, _ = QFileDialog.getOpenFileName(
            self, "Select Photo", last_folder,
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp);;All Files (*.*)"
        )
        if self.input_file:
            settings.setValue("PhotoEditingPage_last_input_folder", os.path.dirname(self.input_file))
            # Load image in a deferred way to avoid freezing UI
            try:
                self.original_pil = Image.open(self.input_file).convert("RGB")
                self._original_shown = False  # Reset original shown flag for new image
                self.zoom_slider.setValue(100)
                # Use a timer to defer preview update, giving the UI time to respond
                QTimer.singleShot(50, self.update_live_preview)
            except Exception as e:
                self.preview_label.setText(f"Error loading image: {str(e)}")
                self.original_pil = None

    def select_output(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("PhotoEditingPage_last_output_folder", "")
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder", last_folder)
        if self.output_dir:
            settings.setValue("PhotoEditingPage_last_output_folder", self.output_dir)

    def load_lut(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("PhotoEditingPage_last_lut_folder", "")
        path, _ = QFileDialog.getOpenFileName(self, "Load LUT", last_folder, "LUT Files (*.cube)")
        if not path:
            return
        settings.setValue("PhotoEditingPage_last_lut_folder", os.path.dirname(path))
        try:
            self.lut_data = load_cube_lut(path)
            self.lut_path = path
            self.lut_label.setText(os.path.basename(path))
            self.update_live_preview()
        except Exception as e:
            self.lut_data = None
            self.lut_path = ""
            self.lut_label.setText("Invalid LUT")
            QMessageBox.warning(self, "LUT Error", f"Failed to load LUT: {e}")

    def clear_lut(self):
        self.lut_data = None
        self.lut_path = ""
        self.lut_label.setText("No LUT loaded")
        self.update_live_preview()

    def _update_lut_strength(self, value):
        self.lut_strength = value
        self.lut_strength_label.setText(f"{value}%")
        self.update_live_preview()

    def _update_lut_swap(self, state):
        self.lut_swap_rb = (state == Qt.Checked)
        self.update_live_preview()

    def reset_sliders(self):
        for label, slider in self.sliders.items():
            slider.blockSignals(True)
            default_val = 0 if label != "Sharpness" else 100
            slider.setValue(default_val)
            self.slider_labels[label].setText(str(default_val))
            slider.blockSignals(False)
        self.filter_combo.blockSignals(True)
        self.filter_combo.setCurrentIndex(0)
        self.filter_combo.blockSignals(False)
        
        # Also reset color grading
        self.reset_color_grading()
        
        # Don't call update_live_preview twice - reset_color_grading already calls it

    def reset_color_grading(self):
        """Reset all color grading wheels and intensities"""
        for tone_range in ['global', 'shadows', 'midtones', 'highlights']:
            self.color_grading[tone_range] = {'hue': 0, 'sat': 0, 'intensity': 0}
            
            # Reset wheel
            wheel = self.color_grading_group.findChild(ColorWheelWidget, f"{tone_range}_wheel")
            if wheel:
                wheel.reset()
            
            # Reset intensity slider
            slider = self.color_grading_group.findChild(NoWheelSlider, f"{tone_range}_intensity")
            if slider:
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)
            
            # Reset intensity label
            label = self.color_grading_group.findChild(QLabel, f"{tone_range}_intensity_label")
            if label:
                label.setText("0%")
        
        # Reset preset combo
        self.color_preset_combo.blockSignals(True)
        self.color_preset_combo.setCurrentIndex(0)
        self.color_preset_combo.blockSignals(False)
        
        self.update_live_preview()
    
    def apply_color_grading_preset(self):
        """Apply a color grading preset"""
        preset_name = self.color_preset_combo.currentText()
        if preset_name not in self.color_grading_presets:
            return
        
        preset = self.color_grading_presets[preset_name]
        
        # Reset all first (both data and UI)
        for tone_range in ['global', 'shadows', 'midtones', 'highlights']:
            self.color_grading[tone_range] = {'hue': 0, 'sat': 0, 'intensity': 0}
            
            # Reset wheel
            wheel = self.color_grading_group.findChild(ColorWheelWidget, f"{tone_range}_wheel")
            if wheel:
                wheel.reset()
            
            # Reset intensity slider
            slider = self.color_grading_group.findChild(NoWheelSlider, f"{tone_range}_intensity")
            if slider:
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)
            
            # Reset intensity label
            label = self.color_grading_group.findChild(QLabel, f"{tone_range}_intensity_label")
            if label:
                label.setText("0%")
        
        # Apply preset values
        for tone_range, values in preset.items():
            if tone_range in self.color_grading:
                # Update data
                self.color_grading[tone_range] = values.copy()
                
                # Update wheel position
                wheel = self.color_grading_group.findChild(ColorWheelWidget, f"{tone_range}_wheel")
                if wheel:
                    hue = values.get('hue', 0)
                    sat = values.get('sat', 0)
                    
                    if sat > 0:
                        # Calculate position on wheel
                        center = wheel.wheel_size // 2
                        radius = (center - 5) * sat
                        angle_rad = math.radians(hue)
                        x = int(center + radius * math.cos(angle_rad))
                        y = int(center + radius * math.sin(angle_rad))
                        wheel.selected_point = (x, y)
                        wheel.update()
                
                # Update intensity slider
                intensity = values.get('intensity', 0)
                slider = self.color_grading_group.findChild(NoWheelSlider, f"{tone_range}_intensity")
                if slider:
                    slider.blockSignals(True)
                    slider.setValue(intensity)
                    slider.blockSignals(False)
                
                # Update intensity label
                label = self.color_grading_group.findChild(QLabel, f"{tone_range}_intensity_label")
                if label:
                    label.setText(f"{intensity}%")
        
        self.update_live_preview()

    def apply_adjustments(self, img):
        """Apply all slider adjustments for full-resolution export - optimized version"""
        if not img:
            return img.copy() if img else None

        img = img.copy().convert("RGB")

        # Get slider values
        exp = self.sliders["Exposure"].value() / 70.0
        con = 1.0 + self.sliders["Contrast"].value() / 80.0
        highlights = self.sliders["Highlights"].value() / 150.0
        shadows = self.sliders["Shadows"].value() / 150.0
        bright = self.sliders["Brightness"].value() / 100.0
        sat = 1.0 + self.sliders["Saturation"].value() / 70.0
        vib = self.sliders["Vibrance"].value() / 100.0
        temp = self.sliders["Temperature"].value() / 300.0
        tint = self.sliders["Tint"].value() / 400.0
        clarity = self.sliders["Clarity"].value() / 200.0
        dehaze = self.sliders["Dehaze"].value() / 300.0
        tone = self.sliders.get("ToneCurve", QSlider()).value() / 50.0
        grain = self.sliders.get("Grain", QSlider()).value() / 100.0
        sharpness_val = self.sliders["Sharpness"].value() / 100.0

        # Apply brightness using PIL (fast)
        if bright != 0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1 + bright * 0.3)
        
        # Apply exposure using brightness
        if exp != 0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1 + exp * 0.5)
        
        # Apply contrast using PIL (fast)
        if con != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(con)
        
        # Apply saturation using PIL (fast, clamp to prevent negative values)
        if sat != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(max(0.0, sat))
        
        # Apply sharpness
        if sharpness_val != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness_val)
        
        # Apply numpy-based adjustments for temp, tint, highlights/shadows, vibrance, etc.
        if temp != 0 or tint != 0 or highlights != 0 or shadows != 0 or vib != 0 or tone != 0:
            img_arr = np.array(img, dtype=np.float32) / 255.0
            
            # Highlights/Shadows using brightness mask
            if highlights != 0 or shadows != 0:
                brightness = np.mean(img_arr, axis=2)
                highlight_mask = brightness > 0.5
                if highlights != 0:
                    img_arr[highlight_mask] = np.clip(img_arr[highlight_mask] * (1 + highlights * 0.15), 0, 1)
                if shadows != 0:
                    img_arr[~highlight_mask] = np.clip(img_arr[~highlight_mask] * (1 + shadows * 0.15), 0, 1)
            
            # Temperature adjustment
            if temp > 0:
                img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + temp * 0.15, 0, 1)  # Red
            elif temp < 0:
                img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] - temp * 0.15, 0, 1)  # Blue
            
            # Tint adjustment
            if tint > 0:
                img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] + tint * 0.15, 0, 1)  # Green
            elif tint < 0:
                img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + tint * 0.15, 0, 1)  # Red
            
            # Vibrance (boost saturation for less saturated colors)
            if vib != 0:
                mean = np.mean(img_arr, axis=2, keepdims=True)
                img_arr = mean + (img_arr - mean) * (1 + vib * 0.3)
                img_arr = np.clip(img_arr, 0, 1)
            
            # Tone curve
            if tone != 0:
                brightness = np.mean(img_arr, axis=2, keepdims=True)
                adjusted = 0.5 + (brightness - 0.5) * (1 + tone * 0.2)
                ratio = np.divide(adjusted, brightness, where=brightness > 0.001, out=np.ones_like(brightness))
                img_arr = np.clip(img_arr * ratio, 0, 1)
            
            img = Image.fromarray((img_arr * 255).astype(np.uint8), mode='RGB')
        
        # Clarity (local contrast) using PIL
        if clarity != 0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1 + clarity * 0.3)
        
        # Dehaze (reduce contrast)
        if dehaze != 0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1 - dehaze * 0.2)

        # Grain effect - add realistic film grain
        if grain > 0:
            img_arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, grain * 1.5, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255)
            img = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')

        # COLOR GRADING - Apply color wheels (shadows, midtones, highlights)
        has_color_grading = any(
            self.color_grading[tr]['intensity'] > 0 
            for tr in ['shadows', 'midtones', 'highlights']
        )
        
        if has_color_grading:
            img_arr = np.array(img, dtype=np.float32) / 255.0
            
            # Calculate luminance for each pixel
            luminance = 0.299 * img_arr[:,:,0] + 0.587 * img_arr[:,:,1] + 0.114 * img_arr[:,:,2]
            
            # Apply color grading for each tonal range
            for tone_range, lum_min, lum_max in [
                ('shadows', 0.0, 0.33),
                ('midtones', 0.33, 0.67),
                ('highlights', 0.67, 1.0)
            ]:
                grading = self.color_grading[tone_range]
                intensity = grading['intensity'] / 100.0
                
                if intensity > 0:
                    # Calculate mask for this tonal range
                    mask = np.zeros_like(luminance)
                    in_range = (luminance >= lum_min) & (luminance < lum_max)
                    mask[in_range] = 1.0
                    
                    # Apply smooth falloff
                    falloff = 0.1
                    if tone_range == 'shadows':
                        transition = (luminance - lum_max) / falloff
                        mask = np.clip(mask * (1 - np.clip(transition, 0, 1)), 0, 1)
                    elif tone_range == 'highlights':
                        transition = (lum_min - luminance) / falloff
                        mask = np.clip(mask * (1 - np.clip(transition, 0, 1)), 0, 1)
                    
                    # Convert hue (0-360) and saturation (0-1) to RGB color shift
                    hue = grading['hue']
                    sat = grading['sat']
                    
                    # Create color from hue/sat
                    if sat > 0:
                        color = colorsys.hsv_to_rgb(hue / 360.0, sat, 1.0)
                        color_shift = np.array(color, dtype=np.float32)
                        
                        # Apply color shift with mask and intensity
                        for c in range(3):
                            shift = (color_shift[c] - 0.5) * intensity * 0.3
                            img_arr[:,:,c] = img_arr[:,:,c] + shift * mask[:,:,np.newaxis].squeeze()
            
            img_arr = np.clip(img_arr, 0, 1)
            img = Image.fromarray((img_arr * 255).astype(np.uint8), mode='RGB')

        # LUT (3D .cube)
        if self.lut_data is not None:
            img_arr = np.array(img, dtype=np.float32) / 255.0
            lut_arr = apply_lut_to_array(img_arr, self.lut_data, self.lut_swap_rb)
            strength = max(0.0, min(1.0, self.lut_strength / 100.0))
            img_arr = img_arr * (1 - strength) + lut_arr * strength
            img = Image.fromarray((img_arr * 255).astype(np.uint8), mode='RGB')

        return img

    def update_live_preview(self):
        """Schedule a preview update with debouncing"""
        if self.original_pil and not self._original_shown:
            # For first load, immediately show the original unprocessed image
            self._show_original_thumbnail()
            self._original_shown = True
        
        # Restart the debounce timer
        self.preview_update_timer.stop()
        self.preview_update_timer.start()
    
    def _show_original_thumbnail(self):
        """Show the original image as thumbnail for quick feedback"""
        if not self.original_pil:
            return
        
        thumb = self.original_pil.copy()
        thumb.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        
        qimg = QImage(thumb.tobytes(), thumb.width, thumb.height, 3 * thumb.width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._cached_preview = pixmap
        
        # Apply current zoom
        zoom = self.zoom_slider.value() / 100.0
        sw = int(pixmap.width() * zoom)
        sh = int(pixmap.height() * zoom)
        scaled_pix = pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.preview_label.setPixmap(scaled_pix)
        self.preview_label.setFixedSize(sw, sh)
        self.preview_label.setStyleSheet("")
    
    def _start_preview_worker(self):
        """Start the background worker thread for preview generation"""
        if not self.original_pil:
            return
        
        # Don't blank the preview - keep current image visible while processing
        # The image will update seamlessly when processing completes
        
        # Stop any existing worker
        if hasattr(self, 'preview_worker') and self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.quit()
            self.preview_worker.wait()
        
        # Create new worker with current settings
        slider_vals = {label: slider.value() for label, slider in self.sliders.items()}
        before_after = self.before_after_check.isChecked()
        zoom_val = self.zoom_slider.value()
        
        self.preview_worker = QThread()
        # Keep a reference to worker_obj so it doesn't get garbage collected
        self.preview_worker_obj = PreviewWorker(
            self.original_pil, 
            slider_vals, 
            before_after, 
            zoom_val, 
            self.lut_data, 
            self.lut_strength, 
            self.lut_swap_rb,
            self.color_grading
        )
        self.preview_worker_obj.moveToThread(self.preview_worker)
        
        # Connect signals
        self.preview_worker.started.connect(self.preview_worker_obj.process)
        self.preview_worker_obj.finished.connect(self._on_preview_ready)
        self.preview_worker_obj.finished.connect(self.preview_worker.quit)
        
        self.preview_worker.start()
    
    def _on_preview_ready(self, pixmap):
        """Display the processed preview"""
        if not pixmap or pixmap.isNull():
            return
        
        # Cache the pixmap
        self._cached_preview = pixmap
        
        # Apply zoom
        zoom = self.zoom_slider.value() / 100.0
        sw = int(pixmap.width() * zoom)
        sh = int(pixmap.height() * zoom)
        scaled_pix = pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Update the preview seamlessly
        self.preview_label.setPixmap(scaled_pix)
        self.preview_label.setFixedSize(sw, sh)
        self.preview_label.setStyleSheet("")
        self.preview_label.update()

    def export_photo(self):
        if not self.input_file or not self.output_dir or not self.original_pil:
            return

        def save_image(path, img):
            try:
                ext = os.path.splitext(path)[1].lower()
                if ext in (".jpg", ".jpeg"):
                    img.convert("RGB").save(path, "JPEG", quality=95, optimize=True)
                else:
                    img.save(path, optimize=True)
                print(f"✓ Photo exported to {path}")
            except Exception as e:
                print("Export failed", e)

        # single-image export only
        edited = self.apply_adjustments(self.original_pil)
        base, ext = os.path.splitext(os.path.basename(self.input_file))
        out_path = os.path.join(self.output_dir, f"{base}_edited{ext}")
        save_image(out_path, edited)

    def update_theme_colors(self):
        """Update colors based on current theme"""
        window = self.window()
        if not hasattr(window, 'current_theme_colors'):
            return
        
        colors = window.current_theme_colors
        accent = colors.get('accent', '#00FFC6')
        primary_bg = colors.get('primary_bg', '#070A0E')
        
        # Update splitter handle
        self.splitter.setStyleSheet(f"QSplitter::handle {{ background: {accent}; border-radius: 3px; }}")
        
        # Update group labels to use theme accent color
        for lbl in self.group_labels:
            lbl.setStyleSheet(f"color: {accent}; font-weight: bold; font-size: 12px; margin-top: 15px; margin-bottom: 8px;")
        
        # Update value labels to use theme accent color
        for val_lbl in self.slider_labels.values():
            val_lbl.setStyleSheet(f"font-size: 10px; color: {accent}; text-align: center; margin-bottom: 8px;")
        
        # Update export button
        self.export_btn.setStyleSheet(f"background: {accent}; color: {primary_bg}; font-weight: bold; font-size: 13px; padding: 10px;")



# ===================================================================
# RENAME TOOL - clear titles
# ===================================================================

class BatchWatermarkPage(CardPage):
    def __init__(self):
        super().__init__("Watermark")

        self.split = QHBoxLayout()
        
        # Left panel with sticky bottom button
        left = QWidget()
        left_main_layout = QVBoxLayout(left)
        left_main_layout.setContentsMargins(0, 0, 0, 0)
        left_main_layout.setSpacing(0)
        
        # Scrollable controls area
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_controls_widget = QWidget()
        self.left_layout = QVBoxLayout(left_controls_widget)
        left_scroll.setWidget(left_controls_widget)
        left_main_layout.addWidget(left_scroll, 1)
        
        # Sticky bottom section for apply button and progress
        left_bottom_widget = QWidget()
        self.left_bottom_layout = QVBoxLayout(left_bottom_widget)
        self.left_bottom_layout.setContentsMargins(8, 8, 8, 8)
        left_main_layout.addWidget(left_bottom_widget)
        
        # Right panel with sticky bottom zoom
        right = QWidget()
        right_main_layout = QVBoxLayout(right)
        right_main_layout.setContentsMargins(0, 0, 0, 0)
        right_main_layout.setSpacing(0)
        
        # Preview scroll area
        self.right_layout = QVBoxLayout()
        right_main_layout.addLayout(self.right_layout, 1)
        
        # Sticky zoom controls at bottom
        right_bottom_widget = QWidget()
        self.right_bottom_layout = QHBoxLayout(right_bottom_widget)
        self.right_bottom_layout.setContentsMargins(8, 8, 8, 8)
        right_main_layout.addWidget(right_bottom_widget)
        
        self.split.addWidget(left, 40)
        self.split.addWidget(right, 60)
        self.card_layout.addLayout(self.split)

        # allow processing a single file if desired
        self.single_check = QCheckBox("Single Image")
        self.input_btn = QPushButton("Select Input Folder")
        self.left_layout.addWidget(self.single_check)
        # change button label depending on mode
        self.single_check.toggled.connect(lambda chk: self.input_btn.setText("Select Input File" if chk else "Select Input Folder"))
        self.output_btn = QPushButton("Select Output Folder")
        self.left_layout.addWidget(self.input_btn)
        self.left_layout.addWidget(self.output_btn)

        self.watermark_type_group = QButtonGroup()
        self.text_radio = QRadioButton("Text Watermark")
        self.image_radio = QRadioButton("Image Watermark")
        self.text_radio.setChecked(True)
        self.watermark_type_group.addButton(self.text_radio)
        self.watermark_type_group.addButton(self.image_radio)
        h = QHBoxLayout()
        h.addWidget(self.text_radio)
        h.addWidget(self.image_radio)
        self.left_layout.addLayout(h)

        self.watermark_text = QLineEdit("PixelForge")
        self.left_layout.addWidget(QLabel("Watermark Text"))
        self.left_layout.addWidget(self.watermark_text)

        self.text_color_btn = QPushButton("🎨 Pick Text Color")
        self.text_color_btn.clicked.connect(self.pick_text_color)
        self.left_layout.addWidget(self.text_color_btn)
        self.text_color = QColor(0, 255, 198)

        self.watermark_image_btn = QPushButton("Select Watermark Image")
        self.watermark_image_btn.clicked.connect(self.select_watermark_image)
        self.left_layout.addWidget(self.watermark_image_btn)

        # watermark presets
        self.wm_preset_combo = NoWheelComboBox()
        self.wm_preset_combo.addItems(["Custom", "Bottom right subtle", "Tiled light protection", "Center bold"])
        self.left_layout.addWidget(QLabel("Watermark Presets"))
        self.left_layout.addWidget(self.wm_preset_combo)

        self.position_combo = NoWheelComboBox()
        self.position_combo.addItems(["Top Left","Top Center","Top Right",
                                     "Middle Left","Center","Middle Right",
                                     "Bottom Left","Bottom Center","Bottom Right",
                                     "Custom (Slider)"])
        self.left_layout.addWidget(QLabel("Position (when not tiled)"))
        self.left_layout.addWidget(self.position_combo)

        custom_pos_layout = QGridLayout()
        custom_pos_layout.addWidget(QLabel("Custom X"), 0, 0)
        self.custom_x_slider = NoWheelSlider(Qt.Horizontal)
        self.custom_x_slider.setRange(0, 100)
        self.custom_x_slider.setValue(50)
        custom_pos_layout.addWidget(self.custom_x_slider, 0, 1)

        custom_pos_layout.addWidget(QLabel("Custom Y"), 1, 0)
        self.custom_y_slider = NoWheelSlider(Qt.Horizontal)
        self.custom_y_slider.setRange(0, 100)
        self.custom_y_slider.setValue(50)
        custom_pos_layout.addWidget(self.custom_y_slider, 1, 1)
        self.left_layout.addLayout(custom_pos_layout)

        self.rotation_slider = NoWheelSlider(Qt.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(0)
        self.left_layout.addWidget(QLabel("Rotation (°)"))
        self.left_layout.addWidget(self.rotation_slider)

        self.tile_check = QCheckBox("🟦 Tile / Grid")
        self.tile_check.setChecked(False)
        self.left_layout.addWidget(self.tile_check)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 200)
        self.padding_spin.setValue(20)
        self.left_layout.addWidget(QLabel("Padding / Spacing (px)"))
        self.left_layout.addWidget(self.padding_spin)

        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 200)
        self.margin_spin.setValue(10)
        self.left_layout.addWidget(QLabel("Margin Safety (px)"))
        self.left_layout.addWidget(self.margin_spin)

        self.opacity_slider = NoWheelSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(60)
        self.left_layout.addWidget(QLabel("Opacity"))
        self.left_layout.addWidget(self.opacity_slider)

        self.scale_slider = NoWheelSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 300)
        self.scale_slider.setValue(80)
        self.left_layout.addWidget(QLabel("Scale %"))
        self.left_layout.addWidget(self.scale_slider)

        self.stroke_check = QCheckBox("Stroke Outline")
        self.stroke_color_btn = QPushButton("Stroke Color")
        self.stroke_color_btn.clicked.connect(self.pick_stroke_color)
        self.stroke_color = QColor(0,0,0)
        self.left_layout.addWidget(self.stroke_check)
        self.left_layout.addWidget(self.stroke_color_btn)

        self.shadow_check = QCheckBox("Shadow")
        self.left_layout.addWidget(self.shadow_check)

        self.auto_shrink_check = QCheckBox("Auto shrink if too large")
        self.left_layout.addWidget(self.auto_shrink_check)
        
        # Add spacing before sticky bottom section
        self.left_layout.addStretch()

        # Sticky bottom section (apply button and progress)
        self.start_btn = QPushButton("Apply Watermark")
        self.progress = QProgressBar()
        self.left_bottom_layout.addWidget(self.start_btn)
        self.left_bottom_layout.addWidget(self.progress)

        # Preview scroll area (fills remaining space)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.preview_scroll)
        
        # Sticky zoom controls at bottom of right panel
        self.right_bottom_layout.addWidget(QLabel("Zoom"))
        self.zoom_slider = ZoomSlider(Qt.Horizontal)
        self.zoom_slider.setRange(30, 400)
        self.zoom_slider.setValue(100)
        self.right_bottom_layout.addWidget(self.zoom_slider)

        self.input_dir = ""
        self.output_dir = ""
        self.watermark_image_path = ""
        self.custom_watermark_pos = (0.5, 0.5)  # normalized center (x, y)

        self.single_check.stateChanged.connect(self.update_preview)

        # connect controls to update_preview
        self.text_radio.toggled.connect(self.update_preview)
        self.image_radio.toggled.connect(self.update_preview)
        
        # Debounce updates for better performance
        self.preview_update_timer = QTimer()
        self.preview_update_timer.setSingleShot(True)
        self.preview_update_timer.timeout.connect(self.update_preview)
        
        self.watermark_text.textChanged.connect(lambda: self.preview_update_timer.start(300))
        self.position_combo.currentIndexChanged.connect(lambda: self.preview_update_timer.start(150))
        self.opacity_slider.valueChanged.connect(lambda: self.preview_update_timer.start(100))
        self.scale_slider.valueChanged.connect(lambda: self.preview_update_timer.start(100))
        self.padding_spin.valueChanged.connect(lambda: self.preview_update_timer.start(150))
        self.tile_check.stateChanged.connect(lambda: self.preview_update_timer.start(150))
        self.rotation_slider.valueChanged.connect(lambda: self.preview_update_timer.start(100))
        self.margin_spin.valueChanged.connect(lambda: self.preview_update_timer.start(150))
        self.stroke_check.stateChanged.connect(lambda: self.preview_update_timer.start(150))
        self.shadow_check.stateChanged.connect(lambda: self.preview_update_timer.start(150))
        self.auto_shrink_check.stateChanged.connect(lambda: self.preview_update_timer.start(150))
        self.wm_preset_combo.currentIndexChanged.connect(self.apply_wm_preset)
        self.wm_preset_combo.currentIndexChanged.connect(lambda: self.preview_update_timer.start(150))
        self.position_combo.currentTextChanged.connect(self._on_position_mode_changed)
        self.custom_x_slider.valueChanged.connect(self._on_custom_slider_changed)
        self.custom_y_slider.valueChanged.connect(self._on_custom_slider_changed)
        self.custom_x_slider.sliderReleased.connect(self._apply_custom_slider_position)
        self.custom_y_slider.sliderReleased.connect(self._apply_custom_slider_position)
        self.stroke_color_btn.clicked.connect(self.update_preview)
        self.single_check.stateChanged.connect(lambda: self.preview_update_timer.start(150))

        self.zoom_slider.valueChanged.connect(self.apply_zoom)

        self.input_btn.clicked.connect(self.select_input)
        self.output_btn.clicked.connect(self.select_output)
        self.start_btn.clicked.connect(self.start_process)

        self.preview_label.wheelEvent = self.wheel_zoom

    def wheel_zoom(self, event):
        delta = event.angleDelta().y()
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(current + (10 if delta > 0 else -10))
        event.accept()

    def _on_position_mode_changed(self, text):
        is_custom = text == "Custom (Slider)"
        self.custom_x_slider.setEnabled(is_custom)
        self.custom_y_slider.setEnabled(is_custom)
        self.update_preview()

    def _set_custom_position(self, nx, ny, sync_sliders=True):
        nx = max(0.0, min(1.0, float(nx)))
        ny = max(0.0, min(1.0, float(ny)))
        self.custom_watermark_pos = (nx, ny)

        if sync_sliders:
            self.custom_x_slider.blockSignals(True)
            self.custom_y_slider.blockSignals(True)
            self.custom_x_slider.setValue(int(round(nx * 100)))
            self.custom_y_slider.setValue(int(round(ny * 100)))
            self.custom_x_slider.blockSignals(False)
            self.custom_y_slider.blockSignals(False)

    def _on_custom_slider_changed(self):
        # Update stored custom position only; keep preview static while user drags slider
        nx = self.custom_x_slider.value() / 100.0
        ny = self.custom_y_slider.value() / 100.0
        self._set_custom_position(nx, ny, sync_sliders=False)

        if self.position_combo.currentText() != "Custom (Slider)":
            self.position_combo.setCurrentText("Custom (Slider)")

    def _apply_custom_slider_position(self):
        nx = self.custom_x_slider.value() / 100.0
        ny = self.custom_y_slider.value() / 100.0
        self._set_custom_position(nx, ny, sync_sliders=False)
        if self.position_combo.currentText() != "Custom (Slider)":
            self.position_combo.setCurrentText("Custom (Slider)")
        self.update_preview()

    def pick_text_color(self):
        color = QColorDialog.getColor(self.text_color)
        if color.isValid():
            self.text_color = color
            self.update_preview()

    def pick_stroke_color(self):
        color = QColorDialog.getColor(self.stroke_color)
        if color.isValid():
            self.stroke_color = color
            self.update_preview()

    def apply_wm_preset(self):
        preset = self.wm_preset_combo.currentText()
        if preset == "Bottom right subtle":
            self.position_combo.setCurrentText("Bottom Right")
            self.opacity_slider.setValue(40)
            self.scale_slider.setValue(50)
            self.tile_check.setChecked(False)
        elif preset == "Tiled light protection":
            self.tile_check.setChecked(True)
            self.opacity_slider.setValue(20)
            self.scale_slider.setValue(30)
        elif preset == "Center bold":
            self.position_combo.setCurrentText("Center")
            self.opacity_slider.setValue(80)
            self.scale_slider.setValue(100)
            self.tile_check.setChecked(False)
        # custom does not change
        self.update_preview()

    def select_input(self):
        # if single mode, choose an image file, otherwise folder
        if self.single_check.isChecked():
            fpath, _ = QFileDialog.getOpenFileName(self, "Select Input Image", "", "Images (*.png *.jpg *.jpeg *.gif *.webp);;All Files (*.*)")
            if fpath:
                self.input_dir = fpath
        else:
            fpath = QFileDialog.getExistingDirectory(self, "Select Input Folder")
            if fpath:
                self.input_dir = fpath
        if self.input_dir:
            self.update_preview()

    def select_output(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")

    def select_watermark_image(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Watermark Image", "", "Images (*.png *.jpg *.jpeg)")
        if p:
            self.watermark_image_path = p
            self.update_preview()

    def update_preview(self):
        if not self.input_dir: return
        # support files as well as folders
        if os.path.isdir(self.input_dir):
            files = [f for f in os.listdir(self.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not files: return
            path = os.path.join(self.input_dir, files[0])
        else:
            path = self.input_dir
        try:
            with Image.open(path) as pil_img:
                source = pil_img.copy()
                if source.mode not in ("RGB", "RGBA"):
                    source = source.convert("RGBA")

                # Accurate full-resolution preview path (matches export behavior)
                watermarked_full = self.apply_watermark(source)

                # Downscale only for UI display (does not affect watermark rendering logic)
                display = watermarked_full.copy()
                display.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
                display = display.convert("RGB")

                data = display.tobytes()
                qimg = QImage(data, display.width, display.height, display.width * 3, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)

                # cache raw preview pixmap so zooming is cheap
                self._preview_pixmap = pix
                self.apply_zoom()
                self.preview_label.setAlignment(Qt.AlignCenter)
        except:
            pass

    def apply_watermark(self, img):
        if self.text_radio.isChecked() and self.watermark_text.text().strip():
            text = self.watermark_text.text()
            opacity = self.opacity_slider.value() / 100.0
            scale_factor = self.scale_slider.value() / 100.0
            padding = self.padding_spin.value()

            txt_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(txt_layer)
            # Use larger base font size for tiled mode
            base_size = 120 if self.tile_check.isChecked() else 48
            font_size = int(base_size * scale_factor)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            color = (self.text_color.red(), self.text_color.green(), self.text_color.blue(), int(255 * opacity))

            if self.tile_check.isChecked():
                step_x = max(tw + padding * 2, 150)
                step_y = max(th + padding * 2, 100)
                for yy in range(0, img.height, step_y):
                    for xx in range(0, img.width, step_x):
                        draw.text((xx + padding, yy + padding), text, fill=color, font=font)
            else:
                pos = self.position_combo.currentText()
                # anchor grid
                if pos == "Custom (Slider)":
                    cx, cy = self.custom_watermark_pos
                    x = int(cx * img.width - tw / 2)
                    y = int(cy * img.height - th / 2)
                elif pos == "Top Left": x, y = padding, padding
                elif pos == "Top Center": x, y = (img.width - tw) // 2, padding
                elif pos == "Top Right": x, y = img.width - tw - padding, padding
                elif pos == "Middle Left": x, y = padding, (img.height - th) // 2
                elif pos == "Center": x, y = (img.width - tw) // 2, (img.height - th) // 2
                elif pos == "Middle Right": x, y = img.width - tw - padding, (img.height - th) // 2
                elif pos == "Bottom Left": x, y = padding, img.height - th - padding
                elif pos == "Bottom Center": x, y = (img.width - tw) // 2, img.height - th - padding
                elif pos == "Bottom Right": x, y = img.width - tw - padding, img.height - th - padding
                else: x, y = padding, padding

                # margin safety
                m = self.margin_spin.value()
                x = max(m, min(x, img.width - tw - m))
                y = max(m, min(y, img.height - th - m))

                # shadow
                if self.shadow_check.isChecked():
                    shadow_color = (0,0,0,int(255*opacity*0.5))
                    draw.text((x+2, y+2), text, fill=shadow_color, font=font)

                # stroke effect
                if self.stroke_check.isChecked():
                    sc = (self.stroke_color.red(), self.stroke_color.green(), self.stroke_color.blue(), int(255*opacity))
                    # draw outline by multiple offsets
                    for dx in (-1,0,1):
                        for dy in (-1,0,1):
                            if dx==0 and dy==0: continue
                            draw.text((x+dx, y+dy), text, fill=sc, font=font)

                draw.text((x, y), text, fill=color, font=font)

            img = Image.alpha_composite(img.convert("RGBA"), txt_layer)
            # rotation
            angle = self.rotation_slider.value()
            if angle != 0:
                img = img.rotate(angle, expand=1, resample=Image.Resampling.BICUBIC)
            return img.convert("RGB")

        elif self.image_radio.isChecked() and self.watermark_image_path:
            try:
                with Image.open(self.watermark_image_path) as wm_img:
                    wm = wm_img.convert("RGBA")
                scale_factor = self.scale_slider.value() / 100.0
                new_w = int(wm.width * scale_factor)
                new_h = int(wm.height * scale_factor)
                if self.auto_shrink_check.isChecked():
                    max_w = img.width - self.margin_spin.value()*2
                    max_h = img.height - self.margin_spin.value()*2
                    if new_w > max_w or new_h > max_h:
                        ratio = min(max_w / new_w, max_h / new_h)
                        new_w = int(new_w * ratio)
                        new_h = int(new_h * ratio)
                wm = wm.resize((new_w, new_h), Image.Resampling.LANCZOS)
                opacity = self.opacity_slider.value() / 100.0
                padding = self.padding_spin.value()
                if wm.mode in ("RGBA","LA"):
                    alpha = wm.split()[-1].point(lambda p: int(p*opacity))
                    wm.putalpha(alpha)
                
                base = img.convert("RGBA")
                
                # Tiled mode
                if self.tile_check.isChecked():
                    step_x = max(wm.width + padding * 2, 150)
                    step_y = max(wm.height + padding * 2, 100)
                    for yy in range(0, img.height, step_y):
                        for xx in range(0, img.width, step_x):
                            base.paste(wm, (xx + padding, yy + padding), wm)
                else:
                    # Single position
                    pos = self.position_combo.currentText()
                    if pos == "Custom (Slider)":
                        cx, cy = self.custom_watermark_pos
                        x = int(cx * img.width - wm.width / 2)
                        y = int(cy * img.height - wm.height / 2)
                    elif pos == "Top Left": x, y = padding, padding
                    elif pos == "Top Center": x, y = (img.width - wm.width) // 2, padding
                    elif pos == "Top Right": x, y = img.width - wm.width - padding, padding
                    elif pos == "Middle Left": x, y = padding, (img.height - wm.height) // 2
                    elif pos == "Center": x, y = (img.width - wm.width) // 2, (img.height - wm.height) // 2
                    elif pos == "Middle Right": x, y = img.width - wm.width - padding, (img.height - wm.height) // 2
                    elif pos == "Bottom Left": x, y = padding, img.height - wm.height - padding
                    elif pos == "Bottom Center": x, y = (img.width - wm.width) // 2, img.height - wm.height - padding
                    elif pos == "Bottom Right": x, y = img.width - wm.width - padding, img.height - wm.height - padding
                    else: x, y = padding, padding
                    # margin safety
                    m = self.margin_spin.value()
                    x = max(m, min(x, img.width - wm.width - m))
                    y = max(m, min(y, img.height - wm.height - m))
                    base.paste(wm, (x, y), wm)
                
                img = base
                # rotation of whole image if requested
                angle = self.rotation_slider.value()
                if angle != 0:
                    img = img.rotate(angle, expand=1, resample=Image.Resampling.BICUBIC)
                return img.convert("RGB")
            except:
                return img.convert("RGB")
        return img.convert("RGB")

    def start_process(self):
        if not self.input_dir or not self.output_dir: return

        def worker_func(progress_callback):
            self.perform_batch_watermark(progress_callback, self.input_dir, self.output_dir)
        self.worker = Worker(worker_func)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(lambda: self.progress.setVisible(False))
        self.worker.start()

    def apply_zoom(self):
        # Apply zoom without viewport size restriction - scroll area handles overflow
        if not getattr(self, '_preview_pixmap', None):
            return
        zoom = self.zoom_slider.value() / 100.0
        sw = int(self._preview_pixmap.width() * zoom)
        sh = int(self._preview_pixmap.height() * zoom)
        # No viewport capping - allow full zoom range
        scaled_pix = self._preview_pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pix)
        self.preview_label.setFixedSize(sw, sh)
        self.preview_label.setAlignment(Qt.AlignCenter)

    def perform_batch_watermark(self, progress_callback, input_dir, output_dir):
        # single file support
        if os.path.isdir(input_dir):
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            base_dir = input_dir
        else:
            files = [os.path.basename(input_dir)]
            base_dir = os.path.dirname(input_dir)
        total = len(files) or 1
        for i, filename in enumerate(files):
            in_path = os.path.join(base_dir, filename)
            base, ext = os.path.splitext(filename)
            out_path = os.path.join(output_dir, f"{base}_watermarked{ext}")
            try:
                with Image.open(in_path) as img:
                    if ext.lower() == ".gif" and getattr(img, "is_animated", False):
                        frames = []
                        durations = []
                        for frame in ImageSequence.Iterator(img):
                            frame_copy = frame.copy().convert("RGB")
                            marked = self.apply_watermark(frame_copy)
                            frames.append(marked)
                            durations.append(frame.info.get("duration", 100))
                        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                                       loop=img.info.get("loop", 0), duration=durations, disposal=2, optimize=True)
                    else:
                        if img.mode != "RGB": img = img.convert("RGB")
                        marked = self.apply_watermark(img)
                        if ext.lower() in (".jpg", ".jpeg"):
                            marked.save(out_path, "JPEG", quality=92, optimize=True)
                        else:
                            marked.save(out_path, optimize=True)
            except:
                pass
            progress_callback(int((i + 1) / total * 100))


# ===================================================================
# BACKGROUND TOOLS - 3 separate tabs
# ===================================================================

class BackgroundToolsPage(CardPage):
    def __init__(self):
        super().__init__("Background Tools")
        self.input_file = ""
        self.output_dir = ""
        self.original = None
        self._preview_pixmap = None
        
        # Common file selection area (works for all tabs)
        file_group = QGroupBox("📁 Image & Output (Shared across all tabs)")
        file_layout = QHBoxLayout()
        self.input_btn = QPushButton("📤 Load Image")
        self.output_btn = QPushButton("💾 Select Output Folder")
        file_layout.addWidget(self.input_btn)
        file_layout.addWidget(self.output_btn)
        file_group.setLayout(file_layout)
        self.card_layout.addWidget(file_group)
        
        # Create tabs for 3 sections
        self.tabs = QTabWidget()
        self.tabs.setObjectName("BackgroundTabs")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        # TAB 1: Background Removal
        tab1 = QWidget()
        tab1_layout = QHBoxLayout(tab1)
        self._setup_removal_tab(tab1_layout)
        self.tabs.addTab(tab1, "🗑️ Remove Background")
        
        # TAB 2: Specific Color Changer
        tab2 = QWidget()
        tab2_layout = QHBoxLayout(tab2)
        self._setup_color_changer_tab(tab2_layout)
        self.tabs.addTab(tab2, "🎨 Change Specific Color")
        
        # TAB 3: Full Recoloring
        tab3 = QWidget()
        tab3_layout = QHBoxLayout(tab3)
        self._setup_recolor_tab(tab3_layout)
        self.tabs.addTab(tab3, "🌈 Full Recolor")
        
        self.card_layout.addWidget(self.tabs)
        
        # Connect common file selection buttons
        self.input_btn.clicked.connect(self.select_input_common)
        self.output_btn.clicked.connect(self.select_output_common)
    
    def _on_tab_changed(self, index):
        """Update preview when switching tabs"""
        if self.original:
            if index == 0:
                self.removal_update_preview()
            elif index == 1:
                self.changer_show_original()
            elif index == 2:
                self.recolor_update_preview()
    
    def select_input_common(self):
        """Load image for all tabs"""
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "", 
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All (*.*)"
        )
        if fpath:
            try:
                # Clear old image data
                if hasattr(self, 'original') and self.original:
                    del self.original
                gc.collect()
                
                self.input_file = fpath
                self.original = Image.open(fpath).convert("RGBA")
                # Update current tab preview
                self._on_tab_changed(0)  # Triggers appropriate preview
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")
    
    def select_output_common(self):
        """Set output folder for all tabs"""
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("VectorizationPage_last_output_folder", "")
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder", last_folder)
        if self.output_dir:
            settings.setValue("VectorizationPage_last_output_folder", self.output_dir)
    
    #=================================================================
    # TAB 1: BACKGROUND REMOVAL
    #=================================================================
    def _setup_removal_tab(self, parent_layout):
        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)
        
        # Background removal controls
        removal_group = QGroupBox("Colors to Remove")
        removal_layout = QVBoxLayout()
        
        # Color buttons
        self.removal_color_list = []
        self.removal_color_layout = QHBoxLayout()
        button_h = QHBoxLayout()
        self.add_color_btn = QPushButton("+ Pick Color")
        self.pick_from_image_btn = QPushButton("🎯 Pick from Image")
        remove_white = QPushButton("+ White")
        remove_black = QPushButton("+ Black")
        button_h.addWidget(self.add_color_btn)
        button_h.addWidget(self.pick_from_image_btn)
        button_h.addWidget(remove_white)
        button_h.addWidget(remove_black)
        removal_layout.addLayout(button_h)
        removal_layout.addLayout(self.removal_color_layout)
        
        # Tolerance & Feather
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Tolerance:"))
        self.removal_tolerance_slider = NoWheelSlider(Qt.Horizontal)
        self.removal_tolerance_slider.setRange(0, 100)
        self.removal_tolerance_slider.setValue(20)
        self.removal_tol_label = QLabel("20")
        tol_layout.addWidget(self.removal_tolerance_slider)
        tol_layout.addWidget(self.removal_tol_label)
        removal_layout.addLayout(tol_layout)
        
        feat_layout = QHBoxLayout()
        feat_layout.addWidget(QLabel("Feather:"))
        self.removal_feather_slider = NoWheelSlider(Qt.Horizontal)
        self.removal_feather_slider.setRange(0, 50)
        self.removal_feather_slider.setValue(0)
        self.removal_feat_label = QLabel("0")
        feat_layout.addWidget(self.removal_feather_slider)
        feat_layout.addWidget(self.removal_feat_label)
        removal_layout.addLayout(feat_layout)
        
        self.removal_mask_check = QCheckBox("Show Mask Preview")
        removal_layout.addWidget(self.removal_mask_check)
        
        removal_group.setLayout(removal_layout)
        left_layout.addWidget(removal_group)
        
        # Export button
        export_h = QHBoxLayout()
        self.export_removal_btn = QPushButton("💾 Export")
        self.reset_removal_btn = QPushButton("Reset")
        export_h.addWidget(self.export_removal_btn)
        export_h.addWidget(self.reset_removal_btn)
        left_layout.addLayout(export_h)
        
        left_layout.addStretch()
        
        # RIGHT: Preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.removal_preview_scroll = QScrollArea()
        self.removal_preview_scroll.setWidgetResizable(True)
        self.removal_preview_scroll.setAlignment(Qt.AlignCenter)
        self.removal_preview_label = QLabel()
        self.removal_preview_label.setAlignment(Qt.AlignCenter)
        self.removal_preview_scroll.setWidget(self.removal_preview_label)
        right_layout.addWidget(self.removal_preview_scroll, 1)
        
        # Zoom
        zoom_h = QHBoxLayout()
        zoom_h.addWidget(QLabel("Zoom:"))
        self.removal_zoom_slider = ZoomSlider(Qt.Horizontal)
        self.removal_zoom_slider.setRange(50, 200)
        self.removal_zoom_slider.setValue(100)
        self.removal_zoom_label = QLabel("100%")
        zoom_h.addWidget(self.removal_zoom_slider)
        zoom_h.addWidget(self.removal_zoom_label)
        right_layout.addLayout(zoom_h)
        
        parent_layout.addWidget(left, 35)
        parent_layout.addWidget(right, 65)
        
        # Connect tab-specific signals
        self.add_color_btn.clicked.connect(self.removal_add_color)
        self.pick_from_image_btn.clicked.connect(self.removal_toggle_picker)
        remove_white.clicked.connect(lambda: self.removal_add_preset_color((255, 255, 255)))
        remove_black.clicked.connect(lambda: self.removal_add_preset_color((0, 0, 0)))
        self.removal_tolerance_slider.valueChanged.connect(self.removal_update_params)
        self.removal_feather_slider.valueChanged.connect(self.removal_update_params)
        self.removal_mask_check.stateChanged.connect(self.removal_update_preview)
        self.removal_zoom_slider.valueChanged.connect(self.removal_apply_zoom)
        self.export_removal_btn.clicked.connect(self.export_removal)
        self.reset_removal_btn.clicked.connect(self.reset_removal)
        
        self.removal_preview_label.wheelEvent = self.removal_wheel_zoom
    
    #=================================================================
    # TAB 2: SPECIFIC COLOR CHANGER
    #=================================================================
    def _setup_color_changer_tab(self, parent_layout):
        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)
        
        # Color selection
        changer_group = QGroupBox("Color Replacement")
        changer_layout = QVBoxLayout()
        
        # Source color
        source_h = QHBoxLayout()
        source_h.addWidget(QLabel("Color to Replace:"))
        self.changer_source_btn = QPushButton("🎯 Pick from Image")
        self.changer_source_btn.setFixedHeight(50)
        self.changer_source_color = QColor(255, 255, 255)
        source_h.addWidget(self.changer_source_btn)
        changer_layout.addLayout(source_h)
        
        # Target color
        target_h = QHBoxLayout()
        target_h.addWidget(QLabel("Replace With:"))
        self.changer_target_btn = QPushButton("🎨 Pick New Color")
        self.changer_target_btn.setFixedHeight(50)
        self.changer_target_color = QColor(0, 255, 200)
        target_h.addWidget(self.changer_target_btn)
        changer_layout.addLayout(target_h)
        
        # Tolerance
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Tolerance:"))
        self.changer_tolerance_slider = NoWheelSlider(Qt.Horizontal)
        self.changer_tolerance_slider.setRange(0, 100)
        self.changer_tolerance_slider.setValue(20)
        self.changer_tol_label = QLabel("20")
        tol_layout.addWidget(self.changer_tolerance_slider)
        tol_layout.addWidget(self.changer_tol_label)
        changer_layout.addLayout(tol_layout)
        
        # Feather
        feat_layout = QHBoxLayout()
        feat_layout.addWidget(QLabel("Blend/Feather:"))
        self.changer_feather_slider = NoWheelSlider(Qt.Horizontal)
        self.changer_feather_slider.setRange(0, 20)
        self.changer_feather_slider.setValue(0)
        self.changer_feat_label = QLabel("0")
        feat_layout.addWidget(self.changer_feather_slider)
        feat_layout.addWidget(self.changer_feat_label)
        changer_layout.addLayout(feat_layout)
        
        changer_group.setLayout(changer_layout)
        left_layout.addWidget(changer_group)
        
        # Preview & Export
        preview_btn = QPushButton("🔄 Preview Changes")
        preview_btn.clicked.connect(self.changer_update_preview)
        left_layout.addWidget(preview_btn)
        
        export_h = QHBoxLayout()
        self.export_changer_btn = QPushButton("💾 Export")
        self.reset_changer_btn = QPushButton("Reset")
        export_h.addWidget(self.export_changer_btn)
        export_h.addWidget(self.reset_changer_btn)
        left_layout.addLayout(export_h)
        
        left_layout.addStretch()
        
        # RIGHT: Preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.changer_preview_scroll = QScrollArea()
        self.changer_preview_scroll.setWidgetResizable(True)
        self.changer_preview_scroll.setAlignment(Qt.AlignCenter)
        self.changer_preview_label = QLabel()
        self.changer_preview_label.setAlignment(Qt.AlignCenter)
        self.changer_preview_scroll.setWidget(self.changer_preview_label)
        right_layout.addWidget(self.changer_preview_scroll, 1)
        
        # Zoom
        zoom_h = QHBoxLayout()
        zoom_h.addWidget(QLabel("Zoom:"))
        self.changer_zoom_slider = ZoomSlider(Qt.Horizontal)
        self.changer_zoom_slider.setRange(50, 200)
        self.changer_zoom_slider.setValue(100)
        self.changer_zoom_label = QLabel("100%")
        zoom_h.addWidget(self.changer_zoom_slider)
        zoom_h.addWidget(self.changer_zoom_label)
        right_layout.addLayout(zoom_h)
        
        parent_layout.addWidget(left, 35)
        parent_layout.addWidget(right, 65)
        
        # Connect tab-specific signals
        self.changer_source_btn.clicked.connect(self.changer_pick_source)
        self.changer_target_btn.clicked.connect(self.changer_pick_target)
        self.changer_tolerance_slider.valueChanged.connect(self.changer_update_params)
        self.changer_feather_slider.valueChanged.connect(self.changer_update_params)
        self.changer_zoom_slider.valueChanged.connect(self.changer_apply_zoom)
        self.export_changer_btn.clicked.connect(self.export_changer)
        self.reset_changer_btn.clicked.connect(self.reset_changer)
        
        self.changer_preview_label.wheelEvent = self.changer_wheel_zoom
    
    #=================================================================
    # TAB 3: FULL RECOLORING
    #=================================================================
    def _setup_recolor_tab(self, parent_layout):
        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)
        
        # Recoloring controls
        recolor_group = QGroupBox("Recoloring")
        recolor_layout = QVBoxLayout()
        
        self.recolor_base_color_btn = QPushButton("🎨 Pick Target Hue")
        self.recolor_base_color_btn.setFixedHeight(50)
        self.recolor_base_color = QColor(100, 200, 255)
        recolor_layout.addWidget(self.recolor_base_color_btn)
        
        # Intensity
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity:"))
        self.recolor_intensity_slider = NoWheelSlider(Qt.Horizontal)
        self.recolor_intensity_slider.setRange(0, 100)
        self.recolor_intensity_slider.setValue(100)
        self.recolor_intensity_label = QLabel("100%")
        intensity_layout.addWidget(self.recolor_intensity_slider)
        intensity_layout.addWidget(self.recolor_intensity_label)
        recolor_layout.addLayout(intensity_layout)
        
        recolor_group.setLayout(recolor_layout)
        left_layout.addWidget(recolor_group)
        
        # Export button
        export_h = QHBoxLayout()
        self.export_recolor_btn = QPushButton("💾 Export")
        self.reset_recolor_btn = QPushButton("Reset")
        export_h.addWidget(self.export_recolor_btn)
        export_h.addWidget(self.reset_recolor_btn)
        left_layout.addLayout(export_h)
        
        left_layout.addStretch()
        
        # RIGHT: Preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.recolor_preview_scroll = QScrollArea()
        self.recolor_preview_scroll.setWidgetResizable(True)
        self.recolor_preview_scroll.setAlignment(Qt.AlignCenter)
        self.recolor_preview_label = QLabel()
        self.recolor_preview_label.setAlignment(Qt.AlignCenter)
        self.recolor_preview_scroll.setWidget(self.recolor_preview_label)
        right_layout.addWidget(self.recolor_preview_scroll, 1)
        
        # Zoom
        zoom_h = QHBoxLayout()
        zoom_h.addWidget(QLabel("Zoom:"))
        self.recolor_zoom_slider = ZoomSlider(Qt.Horizontal)
        self.recolor_zoom_slider.setRange(50, 200)
        self.recolor_zoom_slider.setValue(100)
        self.recolor_zoom_label = QLabel("100%")
        zoom_h.addWidget(self.recolor_zoom_slider)
        zoom_h.addWidget(self.recolor_zoom_label)
        right_layout.addLayout(zoom_h)
        
        parent_layout.addWidget(left, 35)
        parent_layout.addWidget(right, 65)
        
        # Connect tab-specific signals
        self.recolor_base_color_btn.clicked.connect(self.recolor_pick_color)
        self.recolor_intensity_slider.valueChanged.connect(self.recolor_update_params)
        self.recolor_zoom_slider.valueChanged.connect(self.recolor_apply_zoom)
        self.export_recolor_btn.clicked.connect(self.export_recolor)
        self.reset_recolor_btn.clicked.connect(self.reset_recolor)
        
        self.recolor_preview_label.wheelEvent = self.recolor_wheel_zoom
    
    # ================================================================
    # TAB 1: REMOVAL METHODS
    # ================================================================
    def removal_wheel_zoom(self, event):
        delta = event.angleDelta().y()
        current = self.removal_zoom_slider.value()
        self.removal_zoom_slider.setValue(current + (10 if delta > 0 else -10))
        event.accept()
    
    def removal_add_color(self):
        col = QColorDialog.getColor()
        if col.isValid():
            self.removal_add_preset_color((col.red(), col.green(), col.blue()))
    
    def removal_add_preset_color(self, rgb):
        self.removal_color_list.append(rgb)
        btn = QPushButton()
        btn.setFixedSize(28, 28)
        btn.setStyleSheet(f"background: #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}; border-radius: 4px;")
        btn.setToolTip("Click to remove")
        btn.clicked.connect(lambda: self.removal_remove_color(rgb))
        self.removal_color_layout.addWidget(btn)
        self.removal_update_preview()
    
    def removal_remove_color(self, rgb):
        if rgb in self.removal_color_list:
            self.removal_color_list.remove(rgb)
        for i in reversed(range(self.removal_color_layout.count())):
            self.removal_color_layout.itemAt(i).widget().deleteLater()
        for rgb in self.removal_color_list:
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            btn.setStyleSheet(f"background: #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}; border-radius: 4px;")
            btn.clicked.connect(lambda _, r=rgb: self.removal_remove_color(r))
            self.removal_color_layout.addWidget(btn)
        self.removal_update_preview()
    
    def removal_toggle_picker(self):
        if not self.original:
            QMessageBox.warning(self, "Error", "Load an image first")
            return
        self.removal_picker_mode = not getattr(self, 'removal_picker_mode', False)
        if self.removal_picker_mode:
            self.pick_from_image_btn.setStyleSheet("background: #2b8a3e; color: white;")
            self.pick_from_image_btn.setText("🎯 PICKING - Click Image")
            self.removal_preview_label.setCursor(Qt.CrossCursor)
            self.removal_preview_label.mousePressEvent = self.removal_on_click
        else:
            self.pick_from_image_btn.setStyleSheet("")
            self.pick_from_image_btn.setText("🎯 Pick from Image")
            self.removal_preview_label.setCursor(Qt.ArrowCursor)
            self.removal_preview_label.mousePressEvent = lambda e: e.ignore()
    
    def removal_on_click(self, event):
        if not self.original or not getattr(self, 'removal_picker_mode', False):
            return
        # Get pixel color from original image
        pos = event.position() if hasattr(event, 'position') else event.pos()
        x, y = int(pos.x()), int(pos.y())
        
        # Account for zoom and centering
        pixmap = self.removal_preview_label.pixmap()
        if not pixmap:
            return
        
        label_w, label_h = self.removal_preview_label.width(), self.removal_preview_label.height()
        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
        offset_x = (label_w - pixmap_w) // 2
        offset_y = (label_h - pixmap_h) // 2
        
        img_x = x - offset_x
        img_y = y - offset_y
        
        if img_x < 0 or img_y < 0 or img_x >= pixmap_w or img_y >= pixmap_h:
            return
        
        # Map to original image
        zoom = self.removal_zoom_slider.value() / 100.0
        # Get preview dimensions
        if hasattr(self, '_removal_preview_size'):
            pw, ph = self._removal_preview_size
            orig_x = int((img_x / zoom) * (self.original.width / pw))
            orig_y = int((img_y / zoom) * (self.original.height / ph))
        else:
            orig_x = int(img_x / zoom)
            orig_y = int(img_y / zoom)
        
        orig_x = max(0, min(orig_x, self.original.width - 1))
        orig_y = max(0, min(orig_y, self.original.height - 1))
        
        pixel = self.original.getpixel((orig_x, orig_y))
        if isinstance(pixel, tuple) and len(pixel) >= 3:
            rgb = (pixel[0], pixel[1], pixel[2])
            self.removal_add_preset_color(rgb)
    
    def removal_update_params(self):
        self.removal_tol_label.setText(str(self.removal_tolerance_slider.value()))
        self.removal_feat_label.setText(str(self.removal_feather_slider.value()))
        self.removal_update_preview()
    
    def removal_update_preview(self):
        if not self.original:
            return
        
        try:
            # Create preview (max 600x600)
            img_arr = np.array(self.original, dtype=np.float32) / 255.0
            h, w = img_arr.shape[:2]
            self._removal_preview_size = (w, h)
            
            if h > 1200 or w > 1200:
                scale = min(1200/h, 1200/w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_pil = self.original.copy()
                img_pil.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
                img_arr = np.array(img_pil, dtype=np.float32) / 255.0
                h, w = img_arr.shape[:2]
                self._removal_preview_size = (new_w, new_h)
            
            r, g, b, a = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2], img_arr[..., 3]
            
            # Create mask
            tol = self.removal_tolerance_slider.value() / 255.0
            mask = np.ones((h, w), dtype=np.float32)
            
            for cr, cg, cb in self.removal_color_list:
                cr, cg, cb = cr/255.0, cg/255.0, cb/255.0
                dist_sq = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
                mask[dist_sq <= tol*tol] = 0
            
            # Feather
            if self.removal_feather_slider.value() > 0:
                mask = gaussian_filter(mask, sigma=self.removal_feather_slider.value())
            
            # Apply mask to alpha
            img_arr[..., 3] = mask * a
            
            # Display
            if self.removal_mask_check.isChecked():
                display = Image.fromarray((mask * 255).astype(np.uint8), mode='L').convert('RGB')
                data = display.tobytes()
                qimg = QImage(data, display.width, display.height, display.width*3, QImage.Format_RGB888)
            else:
                img_uint8 = (img_arr * 255).astype(np.uint8)
                display = Image.fromarray(img_uint8, 'RGBA')
                data = display.tobytes('raw', 'RGBA')
                qimg = QImage(data, display.width, display.height, display.width*4, QImage.Format_RGBA8888)
            
            self._removal_preview_pixmap = QPixmap.fromImage(qimg)
            self.removal_apply_zoom()
        except Exception as e:
            print(f"Preview error: {e}")
        finally:
            # Clean up arrays
            if 'img_arr' in locals():
                del img_arr
            if 'r' in locals():
                del r, g, b, a
            if 'mask' in locals():
                del mask
            if 'img_uint8' in locals():
                del img_uint8
            gc.collect()
    
    def removal_apply_zoom(self):
        if not hasattr(self, '_removal_preview_pixmap') or not self._removal_preview_pixmap:
            return
        self.removal_zoom_label.setText(f"{self.removal_zoom_slider.value()}%")
        zoom = self.removal_zoom_slider.value() / 100.0
        sw = int(self._removal_preview_pixmap.width() * zoom)
        sh = int(self._removal_preview_pixmap.height() * zoom)
        scaled = self._removal_preview_pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.removal_preview_label.setPixmap(scaled)
        self.removal_preview_label.setFixedSize(sw, sh)
    
    def export_removal(self):
        if not self.original or not self.output_dir:
            QMessageBox.warning(self, "Error", "Load image and select output folder")
            return
        
        try:
            # Process full resolution
            img_arr = np.array(self.original, dtype=np.float32) / 255.0
            h, w = img_arr.shape[:2]
            r, g, b, a = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2], img_arr[..., 3]
            
            tol = self.removal_tolerance_slider.value() / 255.0
            mask = np.ones((h, w), dtype=np.float32)
            
            for cr, cg, cb in self.removal_color_list:
                cr, cg, cb = cr/255.0, cg/255.0, cb/255.0
                dist_sq = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
                mask[dist_sq <= tol*tol] = 0
            
            if self.removal_feather_slider.value() > 0:
                mask = gaussian_filter(mask, sigma=self.removal_feather_slider.value())
            
            img_arr[..., 3] = mask * a
            
            img_uint8 = (img_arr * 255).astype(np.uint8)
            result = Image.fromarray(img_uint8, 'RGBA')
            
            base = os.path.splitext(os.path.basename(self.input_file))[0]
            output_path = os.path.join(self.output_dir, f"{base}_bg_removed.png")
            result.save(output_path, 'PNG')
            QMessageBox.information(self, "Success", f"Saved to {output_path}")
            
            del img_arr, r, g, b, a, mask, img_uint8, result
            gc.collect()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def reset_removal(self):
        self.removal_color_list = []
        for i in reversed(range(self.removal_color_layout.count())):
            self.removal_color_layout.itemAt(i).widget().deleteLater()
        self.removal_tolerance_slider.setValue(20)
        self.removal_feather_slider.setValue(0)
        self.removal_mask_check.setChecked(False)
        self.removal_preview_label.clear()
    
    # ================================================================
    # TAB 2: COLOR CHANGER METHODS
    # ================================================================
    def changer_wheel_zoom(self, event):
        delta = event.angleDelta().y()
        current = self.changer_zoom_slider.value()
        self.changer_zoom_slider.setValue(current + (10 if delta > 0 else -10))
        event.accept()
    
    def changer_pick_source(self):
        if not self.original:
            QMessageBox.warning(self, "Error", "Load an image first")
            return
        
        # Enable picker mode
        self.changer_source_btn.setText("🎯 PICKING - Click Image")
        self.changer_source_btn.setStyleSheet("background: #2b8a3e; color: white;")
        self.changer_preview_label.setCursor(Qt.CrossCursor)
        self.changer_preview_label.mousePressEvent = self.changer_on_source_click
    
    def changer_on_source_click(self, event):
        if not self.original:
            return
        
        # Get pixel
        pos = event.position() if hasattr(event, 'position') else event.pos()
        x, y = int(pos.x()), int(pos.y())
        
        pixmap = self.changer_preview_label.pixmap()
        if not pixmap:
            return
        
        label_w, label_h = self.changer_preview_label.width(), self.changer_preview_label.height()
        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
        offset_x = (label_w - pixmap_w) // 2
        offset_y = (label_h - pixmap_h) // 2
        
        img_x = x - offset_x
        img_y = y - offset_y
        
        if img_x < 0 or img_y < 0 or img_x >= pixmap_w or img_y >= pixmap_h:
            return
        
        zoom = self.changer_zoom_slider.value() / 100.0
        if hasattr(self, '_changer_preview_size'):
            pw, ph = self._changer_preview_size
            orig_x = int((img_x / zoom) * (self.original.width / pw))
            orig_y = int((img_y / zoom) * (self.original.height / ph))
        else:
            orig_x = int(img_x / zoom)
            orig_y = int(img_y / zoom)
        
        orig_x = max(0, min(orig_x, self.original.width - 1))
        orig_y = max(0, min(orig_y, self.original.height - 1))
        
        pixel = self.original.getpixel((orig_x, orig_y))
        if isinstance(pixel, tuple) and len(pixel) >= 3:
            self.changer_source_color = QColor(pixel[0], pixel[1], pixel[2])
            self.changer_source_btn.setText(f"Source: #{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}")
            self.changer_source_btn.setStyleSheet(f"background: #{pixel[0]:02x}{pixel[1]:02x}{pixel[2]:02x}; color: white;")
            self.changer_preview_label.setCursor(Qt.ArrowCursor)
            self.changer_preview_label.mousePressEvent = lambda e: e.ignore()
    
    def changer_pick_target(self):
        col = QColorDialog.getColor(self.changer_target_color)
        if col.isValid():
            self.changer_target_color = col
            self.changer_target_btn.setText(f"Target: {col.name()}")
            self.changer_target_btn.setStyleSheet(f"background: {col.name()}; color: white;")
    
    def changer_update_params(self):
        self.changer_tol_label.setText(str(self.changer_tolerance_slider.value()))
        self.changer_feat_label.setText(str(self.changer_feather_slider.value()))
    
    def changer_show_original(self):
        if not self.original:
            return
        
        # Show original
        img_arr = np.array(self.original, dtype=np.float32) / 255.0
        h, w = img_arr.shape[:2]
        self._changer_preview_size = (w, h)
        
        if h > 1200 or w > 1200:
            scale = min(1200/h, 1200/w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_pil = self.original.copy()
            img_pil.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
            self._changer_preview_size = (new_w, new_h)
        else:
            img_pil = self.original.copy()
        
        data = img_pil.tobytes('raw', 'RGBA')
        qimg = QImage(data, img_pil.width, img_pil.height, img_pil.width*4, QImage.Format_RGBA8888)
        self._changer_preview_pixmap = QPixmap.fromImage(qimg)
        self.changer_apply_zoom()
    
    def changer_update_preview(self):
        if not self.original:
            return
        
        try:
            # Create preview
            img_arr = np.array(self.original, dtype=np.float32) / 255.0
            h, w = img_arr.shape[:2]
            self._changer_preview_size = (w, h)
            
            if h > 1200 or w > 1200:
                scale = min(1200/h, 1200/w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_pil = self.original.copy()
                img_pil.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
                img_arr = np.array(img_pil, dtype=np.float32) / 255.0
                h, w = img_arr.shape[:2]
                self._changer_preview_size = (new_w, new_h)
            
            r, g, b, a = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2], img_arr[..., 3]
            
            # Find pixels matching source color
            sr = self.changer_source_color.red() / 255.0
            sg = self.changer_source_color.green() / 255.0
            sb = self.changer_source_color.blue() / 255.0
            
            tol = self.changer_tolerance_slider.value() / 255.0
            dist_sq = (r - sr)**2 + (g - sg)**2 + (b - sb)**2
            mask = dist_sq <= tol*tol
            
            # Replace with target color
            tr = self.changer_target_color.red() / 255.0
            tg = self.changer_target_color.green() / 255.0
            tb = self.changer_target_color.blue() / 255.0
            
            # Apply feathering
            feather = self.changer_feather_slider.value()
            if feather > 0:
                # Create blend mask
                blend_mask = np.zeros((h, w), dtype=np.float32)
                blend_mask[mask] = 1.0
                blend_mask = gaussian_filter(blend_mask, sigma=feather)
                
                # Blend colors
                r = r * (1 - blend_mask) + tr * blend_mask
                g = g * (1 - blend_mask) + tg * blend_mask
                b = b * (1 - blend_mask) + tb * blend_mask
            else:
                r[mask] = tr
                g[mask] = tg
                b[mask] = tb
            
            img_arr[..., 0] = np.clip(r, 0, 1)
            img_arr[..., 1] = np.clip(g, 0, 1)
            img_arr[..., 2] = np.clip(b, 0, 1)
            
            img_uint8 = (img_arr * 255).astype(np.uint8)
            display = Image.fromarray(img_uint8, 'RGBA')
            data = display.tobytes('raw', 'RGBA')
            qimg = QImage(data, display.width, display.height, display.width*4, QImage.Format_RGBA8888)
            
            self._changer_preview_pixmap = QPixmap.fromImage(qimg)
            self.changer_apply_zoom()
        except Exception as e:
            print(f"Changer preview error: {e}")
    
    def changer_apply_zoom(self):
        if not hasattr(self, '_changer_preview_pixmap') or not self._changer_preview_pixmap:
            return
        self.changer_zoom_label.setText(f"{self.changer_zoom_slider.value()}%")
        zoom = self.changer_zoom_slider.value() / 100.0
        sw = int(self._changer_preview_pixmap.width() * zoom)
        sh = int(self._changer_preview_pixmap.height() * zoom)
        scaled = self._changer_preview_pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.changer_preview_label.setPixmap(scaled)
        self.changer_preview_label.setFixedSize(sw, sh)
    
    def export_changer(self):
        if not self.original or not self.output_dir:
            QMessageBox.warning(self, "Error", "Load image and select output folder")
            return
        
        try:
            # Process full resolution
            img_arr = np.array(self.original, dtype=np.float32) / 255.0
            h, w = img_arr.shape[:2]
            r, g, b, a = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2], img_arr[..., 3]
            
            sr = self.changer_source_color.red() / 255.0
            sg = self.changer_source_color.green() / 255.0
            sb = self.changer_source_color.blue() / 255.0
            
            tol = self.changer_tolerance_slider.value() / 255.0
            dist_sq = (r - sr)**2 + (g - sg)**2 + (b - sb)**2
            mask = dist_sq <= tol*tol
            
            tr = self.changer_target_color.red() / 255.0
            tg = self.changer_target_color.green() / 255.0
            tb = self.changer_target_color.blue() / 255.0
            
            feather = self.changer_feather_slider.value()
            if feather > 0:
                blend_mask = np.zeros((h, w), dtype=np.float32)
                blend_mask[mask] = 1.0
                blend_mask = gaussian_filter(blend_mask, sigma=feather)
                r = r * (1 - blend_mask) + tr * blend_mask
                g = g * (1 - blend_mask) + tg * blend_mask
                b = b * (1 - blend_mask) + tb * blend_mask
            else:
                r[mask] = tr
                g[mask] = tg
                b[mask] = tb
            
            img_arr[..., 0] = np.clip(r, 0, 1)
            img_arr[..., 1] = np.clip(g, 0, 1)
            img_arr[..., 2] = np.clip(b, 0, 1)
            
            img_uint8 = (img_arr * 255).astype(np.uint8)
            result = Image.fromarray(img_uint8, 'RGBA')
            
            base = os.path.splitext(os.path.basename(self.input_file))[0]
            output_path = os.path.join(self.output_dir, f"{base}_color_changed.png")
            result.save(output_path, 'PNG')
            QMessageBox.information(self, "Success", f"Saved to {output_path}")
            
            del img_arr, r, g, b, a, mask
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def reset_changer(self):
        self.changer_source_color = QColor(255, 255, 255)
        self.changer_target_color = QColor(0, 255, 200)
        self.changer_source_btn.setText("🎯 Pick from Image")
        self.changer_source_btn.setStyleSheet("")
        self.changer_target_btn.setText("🎨 Pick New Color")
        self.changer_target_btn.setStyleSheet("")
        self.changer_tolerance_slider.setValue(20)
        self.changer_feather_slider.setValue(0)
        if self.original:
            self.changer_show_original()
    
    # ================================================================
    # TAB 3: RECOLOR METHODS
    # ================================================================
    def recolor_wheel_zoom(self, event):
        delta = event.angleDelta().y()
        current = self.recolor_zoom_slider.value()
        self.recolor_zoom_slider.setValue(current + (10 if delta > 0 else -10))
        event.accept()
    
    def recolor_pick_color(self):
        col = QColorDialog.getColor(self.recolor_base_color)
        if col.isValid():
            self.recolor_base_color = col
            self.recolor_base_color_btn.setStyleSheet(f"background: {col.name()}; color: white;")
            self.recolor_update_preview()
    
    def recolor_update_params(self):
        self.recolor_intensity_label.setText(f"{self.recolor_intensity_slider.value()}%")
        self.recolor_update_preview()
    
    def recolor_update_preview(self):
        if not self.original:
            return
        
        try:
            # Create preview
            img_arr = np.array(self.original, dtype=np.float32) / 255.0
            h, w = img_arr.shape[:2]
            
            if h > 600 or w > 600:
                scale = min(600/h, 600/w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_pil = self.original.copy()
                img_pil.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
                img_arr = np.array(img_pil, dtype=np.float32) / 255.0
                h, w = img_arr.shape[:2]
            
            r, g, b, a = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2], img_arr[..., 3]
            
            # Apply recoloring (HSV hue shift)
            bc_r = self.recolor_base_color.red() / 255.0
            bc_g = self.recolor_base_color.green() / 255.0
            bc_b = self.recolor_base_color.blue() / 255.0
            bh, bs, bv = colorsys.rgb_to_hsv(bc_r, bc_g, bc_b)
            
            intensity = self.recolor_intensity_slider.value() / 100.0
            
            for y in range(h):
                for x in range(w):
                    if a[y, x] > 0.1:  # Skip transparent pixels
                        ch, cs, cv = colorsys.rgb_to_hsv(r[y, x], g[y, x], b[y, x])
                        # Blend hue
                        new_h = ch * (1 - intensity) + bh * intensity
                        nr, ng, nb = colorsys.hsv_to_rgb(new_h, cs, cv)
                        r[y, x], g[y, x], b[y, x] = nr, ng, nb
            
            img_arr[..., 0] = np.clip(r, 0, 1)
            img_arr[..., 1] = np.clip(g, 0, 1)
            img_arr[..., 2] = np.clip(b, 0, 1)
            
            img_uint8 = (img_arr * 255).astype(np.uint8)
            display = Image.fromarray(img_uint8, 'RGBA')
            data = display.tobytes('raw', 'RGBA')
            qimg = QImage(data, display.width, display.height, display.width*4, QImage.Format_RGBA8888)
            
            self._recolor_preview_pixmap = QPixmap.fromImage(qimg)
            self.recolor_apply_zoom()
        except Exception as e:
            print(f"Recolor preview error: {e}")
    
    def recolor_apply_zoom(self):
        if not hasattr(self, '_recolor_preview_pixmap') or not self._recolor_preview_pixmap:
            return
        self.recolor_zoom_label.setText(f"{self.recolor_zoom_slider.value()}%")
        zoom = self.recolor_zoom_slider.value() / 100.0
        sw = int(self._recolor_preview_pixmap.width() * zoom)
        sh = int(self._recolor_preview_pixmap.height() * zoom)
        scaled = self._recolor_preview_pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.recolor_preview_label.setPixmap(scaled)
        self.recolor_preview_label.setFixedSize(sw, sh)
    
    def export_recolor(self):
        if not self.original or not self.output_dir:
            QMessageBox.warning(self, "Error", "Load image and select output folder")
            return
        
        try:
            # Process full resolution
            img_arr = np.array(self.original, dtype=np.float32) / 255.0
            h, w = img_arr.shape[:2]
            r, g, b, a = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2], img_arr[..., 3]
            
            bc_r = self.recolor_base_color.red() / 255.0
            bc_g = self.recolor_base_color.green() / 255.0
            bc_b = self.recolor_base_color.blue() / 255.0
            bh, bs, bv = colorsys.rgb_to_hsv(bc_r, bc_g, bc_b)
            
            intensity = self.recolor_intensity_slider.value() / 100.0
            
            for y in range(h):
                for x in range(w):
                    if a[y, x] > 0.1:
                        ch, cs, cv = colorsys.rgb_to_hsv(r[y, x], g[y, x], b[y, x])
                        new_h = ch * (1 - intensity) + bh * intensity
                        nr, ng, nb = colorsys.hsv_to_rgb(new_h, cs, cv)
                        r[y, x], g[y, x], b[y, x] = nr, ng, nb
            
            img_arr[..., 0] = np.clip(r, 0, 1)
            img_arr[..., 1] = np.clip(g, 0, 1)
            img_arr[..., 2] = np.clip(b, 0, 1)
            
            img_uint8 = (img_arr * 255).astype(np.uint8)
            result = Image.fromarray(img_uint8, 'RGBA')
            
            base = os.path.splitext(os.path.basename(self.input_file))[0]
            output_path = os.path.join(self.output_dir, f"{base}_recolored.png")
            result.save(output_path, 'PNG')
            QMessageBox.information(self, "Success", f"Saved to {output_path}")
            
            del img_arr, r, g, b, a
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def reset_recolor(self):
        self.recolor_base_color = QColor(100, 200, 255)
        self.recolor_base_color_btn.setStyleSheet("")
        self.recolor_intensity_slider.setValue(100)
        self.recolor_preview_label.clear()
    
    def update_theme_colors(self):
        pass


# ===================================================================
# ===================================================================
# PALETTE EXTRACTOR - split (preview right)
# ===================================================================

class VectorizationPage(CardPage):
    def __init__(self):
        super().__init__("Vectorization Tool")
        self.input_file = ""
        self.output_dir = ""
        self.original = None
        self._preview_pixmap = None
        self._original_preview = None  # Store original (non-vectorized) for color picking
        self.colors_to_remove = []  # List of colors to remove
        self.picking_color = False
        
        # Layout
        split = QHBoxLayout()
        
        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)
        
        # File selection
        file_group = QGroupBox("Image")
        file_layout = QVBoxLayout()
        self.input_btn = QPushButton("📤 Load Image")
        self.output_btn = QPushButton("💾 Select Output Folder")
        file_layout.addWidget(self.input_btn)
        file_layout.addWidget(self.output_btn)
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # Vectorization controls
        vec_group = QGroupBox("Vectorization Settings")
        vec_layout = QVBoxLayout()
        
        # Number of colors (posterization)
        colors_h = QHBoxLayout()
        colors_h.addWidget(QLabel("Colors:"))
        self.colors_slider = NoWheelSlider(Qt.Horizontal)
        self.colors_slider.setRange(2, 32)
        self.colors_slider.setValue(8)
        self.colors_label = QLabel("8")
        colors_h.addWidget(self.colors_slider)
        colors_h.addWidget(self.colors_label)
        vec_layout.addLayout(colors_h)
        
        # Color removal section
        self.remove_color_check = QCheckBox("Remove Selected Colors")
        self.remove_color_check.setChecked(False)
        vec_layout.addWidget(self.remove_color_check)
        
        # Color picker buttons
        picker_h = QHBoxLayout()
        self.pick_color_btn = QPushButton("🎯 Pick from Image")
        self.pick_color_btn.setCheckable(True)
        add_white_btn = QPushButton("+ White")
        add_black_btn = QPushButton("+ Black")
        add_white_btn.clicked.connect(lambda: self.add_color_to_remove((255, 255, 255)))
        add_black_btn.clicked.connect(lambda: self.add_color_to_remove((0, 0, 0)))
        picker_h.addWidget(self.pick_color_btn)
        picker_h.addWidget(add_white_btn)
        picker_h.addWidget(add_black_btn)
        vec_layout.addLayout(picker_h)
        
        # Color list display
        self.color_buttons_layout = QHBoxLayout()
        vec_layout.addLayout(self.color_buttons_layout)
        
        # Tolerance for color removal
        tolerance_h = QHBoxLayout()
        tolerance_h.addWidget(QLabel("Color Tolerance:"))
        self.tolerance_slider = NoWheelSlider(Qt.Horizontal)
        self.tolerance_slider.setRange(0, 100)
        self.tolerance_slider.setValue(30)
        self.tolerance_label = QLabel("30")
        tolerance_h.addWidget(self.tolerance_slider)
        tolerance_h.addWidget(self.tolerance_label)
        vec_layout.addLayout(tolerance_h)
        
        # Edge detection
        self.edge_check = QCheckBox("Add Subtle Edge Darkening")
        self.edge_check.setChecked(False)
        self.edge_check.setToolTip("Slightly darkens color boundaries for definition (not harsh outlines)")
        vec_layout.addWidget(self.edge_check)
        
        # Edge thickness
        thick_h = QHBoxLayout()
        thick_h.addWidget(QLabel("Edge Width:"))
        self.thickness_slider = NoWheelSlider(Qt.Horizontal)
        self.thickness_slider.setRange(1, 6)
        self.thickness_slider.setValue(2)
        self.thick_label = QLabel("2")
        thick_h.addWidget(self.thickness_slider)
        thick_h.addWidget(self.thick_label)
        vec_layout.addLayout(thick_h)
        
        # Blur (smoothing)
        blur_h = QHBoxLayout()
        blur_h.addWidget(QLabel("Smoothing:"))
        self.blur_slider = NoWheelSlider(Qt.Horizontal)
        self.blur_slider.setRange(0, 10)
        self.blur_slider.setValue(0)
        self.blur_label = QLabel("0")
        blur_h.addWidget(self.blur_slider)
        blur_h.addWidget(self.blur_label)
        vec_layout.addLayout(blur_h)
        
        # Contrast boost
        contrast_h = QHBoxLayout()
        contrast_h.addWidget(QLabel("Edge Strength:"))
        self.contrast_slider = NoWheelSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 100)
        self.contrast_slider.setValue(50)
        self.contrast_label = QLabel("50")
        contrast_h.addWidget(self.contrast_slider)
        contrast_h.addWidget(self.contrast_label)
        vec_layout.addLayout(contrast_h)
        
        vec_group.setLayout(vec_layout)
        left_layout.addWidget(vec_group)
        
        # Presets
        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout()
        
        preset1_btn = QPushButton("🎨 High Fidelity Photo")
        preset2_btn = QPushButton("🖼️ Low Poly Art")
        preset3_btn = QPushButton("✏️ Comic/Cartoon")
        preset4_btn = QPushButton("🎭 Silhouette")
        preset5_btn = QPushButton("🎪 Pop Art")
        preset6_btn = QPushButton("🌈 Vibrant Colors")
        preset7_btn = QPushButton("📺 Retro/Vintage")
        preset8_btn = QPushButton("🖨️ Print/Stencil")
        
        preset_layout.addWidget(preset1_btn)
        preset_layout.addWidget(preset2_btn)
        preset_layout.addWidget(preset3_btn)
        preset_layout.addWidget(preset4_btn)
        preset_layout.addWidget(preset5_btn)
        preset_layout.addWidget(preset6_btn)
        preset_layout.addWidget(preset7_btn)
        preset_layout.addWidget(preset8_btn)
        preset_group.setLayout(preset_layout)
        left_layout.addWidget(preset_group)
        
        # Progress bars
        self.single_progress = QProgressBar()
        self.single_progress.setVisible(False)
        self.single_progress.setMaximum(100)
        left_layout.addWidget(self.single_progress)
        
        # Export
        export_h = QHBoxLayout()
        self.export_btn = QPushButton("💾 Export")
        self.reset_btn = QPushButton("Reset")
        export_h.addWidget(self.export_btn)
        export_h.addWidget(self.reset_btn)
        left_layout.addLayout(export_h)
        
        left_layout.addStretch()
        
        # RIGHT: Preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        right_layout.addWidget(self.preview_scroll, 1)
        
        # Zoom
        zoom_h = QHBoxLayout()
        zoom_h.addWidget(QLabel("Zoom:"))
        self.zoom_slider = ZoomSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_label = QLabel("100%")
        zoom_h.addWidget(self.zoom_slider)
        zoom_h.addWidget(self.zoom_label)
        right_layout.addLayout(zoom_h)
        
        split.addWidget(left, 35)
        split.addWidget(right, 65)
        self.card_layout.addLayout(split)
        
        # Connect signals
        self.input_btn.clicked.connect(self.select_input)
        self.output_btn.clicked.connect(self.select_output)
        
        # Debounce slider updates for smooth performance
        self.params_update_timer = QTimer()
        self.params_update_timer.setSingleShot(True)
        self.params_update_timer.timeout.connect(self.update_params)
        
        self.colors_slider.valueChanged.connect(lambda: self.params_update_timer.start(100))
        self.tolerance_slider.valueChanged.connect(lambda: self.params_update_timer.start(100))
        self.thickness_slider.valueChanged.connect(lambda: self.params_update_timer.start(100))
        self.blur_slider.valueChanged.connect(lambda: self.params_update_timer.start(100))
        self.contrast_slider.valueChanged.connect(lambda: self.params_update_timer.start(100))
        self.edge_check.stateChanged.connect(self.update_preview)
        self.remove_color_check.stateChanged.connect(self.update_preview)
        self.zoom_slider.valueChanged.connect(self.apply_zoom)
        self.export_btn.clicked.connect(self.export_image)
        self.reset_btn.clicked.connect(self.reset_all)
        self.pick_color_btn.clicked.connect(self.toggle_color_picker)
        
        preset1_btn.clicked.connect(lambda: self.apply_preset(32, 1, 0, 40, False))  # High Detail - many colors, no edges
        preset2_btn.clicked.connect(lambda: self.apply_preset(6, 2, 2, 60, True))   # Low Poly - few colors, subtle edges
        preset3_btn.clicked.connect(lambda: self.apply_preset(12, 3, 1, 75, True))  # Comic - medium colors, defined edges
        preset4_btn.clicked.connect(lambda: self.apply_preset(2, 2, 0, 80, False))  # Duo-tone - 2 colors, clean
        preset5_btn.clicked.connect(lambda: self.apply_preset(16, 2, 1, 70, True))  # Pop Art - vibrant with edges
        preset6_btn.clicked.connect(lambda: self.apply_preset(24, 1, 2, 30, False))  # Smooth - many colors, smoothed, no edges
        preset7_btn.clicked.connect(lambda: self.apply_preset(8, 3, 3, 65, True))   # Retro - medium smoothing with edges
        preset8_btn.clicked.connect(lambda: self.apply_preset(4, 4, 0, 85, True))   # Bold Graphics - few colors, strong edges
        
        self.preview_label.wheelEvent = self.wheel_zoom
    
    def wheel_zoom(self, event):
        delta = event.angleDelta().y()
        current = self.zoom_slider.value()
        self.zoom_slider.setValue(current + (10 if delta > 0 else -10))
        event.accept()
    
    def toggle_color_picker(self):
        if not self.original:
            QMessageBox.warning(self, "Error", "Load an image first")
            self.pick_color_btn.setChecked(False)
            return
        
        self.picking_color = self.pick_color_btn.isChecked()
        if self.picking_color:
            self.pick_color_btn.setStyleSheet("background: #2b8a3e; color: white;")
            self.pick_color_btn.setText("🎯 PICKING - Click Image")
            self.preview_label.setCursor(Qt.CrossCursor)
            self.preview_label.mousePressEvent = self.pick_color_from_preview
        else:
            self.pick_color_btn.setStyleSheet("")
            self.pick_color_btn.setText("🎯 Pick from Image")
            self.preview_label.setCursor(Qt.ArrowCursor)
            self.preview_label.mousePressEvent = lambda e: None
    
    def pick_color_from_preview(self, event):
        if not self.picking_color or not self._original_preview:
            return
        
        # Get click position relative to the label
        label_pos = event.pos()
        
        # Get the actual displayed preview size
        if not self._preview_pixmap:
            return
            
        # Account for zoom
        zoom = self.zoom_slider.value() / 100.0
        original_size = self._original_preview.size()
        displayed_w = int(original_size.width() * zoom)
        displayed_h = int(original_size.height() * zoom)
        
        # Calculate position offset if image is centered
        label_size = self.preview_label.size()
        offset_x = max(0, (label_size.width() - displayed_w) // 2)
        offset_y = max(0, (label_size.height() - displayed_h) // 2)
        
        # Get adjusted position
        adj_x = label_pos.x() - offset_x
        adj_y = label_pos.y() - offset_y
        
        # Convert to original preview coordinates
        if displayed_w > 0 and displayed_h > 0:
            orig_x = int(adj_x / zoom)
            orig_y = int(adj_y / zoom)
            
            # Get color from original preview image
            if 0 <= orig_x < original_size.width() and 0 <= orig_y < original_size.height():
                color = self._original_preview.toImage().pixelColor(orig_x, orig_y)
                rgb = (color.red(), color.green(), color.blue())
                
                # Add to color list
                self.add_color_to_remove(rgb)
    
    def add_color_to_remove(self, rgb):
        """Add a color to the removal list"""
        if rgb not in self.colors_to_remove:
            self.colors_to_remove.append(rgb)
            
            # Create color button
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            btn.setStyleSheet(f"background: #{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}; border-radius: 4px;")
            btn.setToolTip(f"RGB: {rgb}\\nClick to remove")
            btn.clicked.connect(lambda: self.remove_color_from_list(rgb))
            self.color_buttons_layout.addWidget(btn)
            
            # Enable color removal checkbox
            if not self.remove_color_check.isChecked():
                self.remove_color_check.setChecked(True)
            else:
                self.update_preview()
            
            # Deactivate picker
            if self.picking_color:
                self.pick_color_btn.setChecked(False)
                self.picking_color = False
                self.pick_color_btn.setStyleSheet("")
                self.pick_color_btn.setText("🎯 Pick from Image")
                self.preview_label.setCursor(Qt.ArrowCursor)
                self.preview_label.mousePressEvent = lambda e: None
    
    def remove_color_from_list(self, rgb):
        """Remove a color from the removal list"""
        if rgb in self.colors_to_remove:
            self.colors_to_remove.remove(rgb)
        
        # Rebuild color buttons
        for i in reversed(range(self.color_buttons_layout.count())):
            self.color_buttons_layout.itemAt(i).widget().deleteLater()
        
        for color in self.colors_to_remove:
            btn = QPushButton()
            btn.setFixedSize(28, 28)
            btn.setStyleSheet(f"background: #{color[0]:02x}{color[1]:02x}{color[2]:02x}; border-radius: 4px;")
            btn.setToolTip(f"RGB: {color}\\nClick to remove")
            btn.clicked.connect(lambda _, c=color: self.remove_color_from_list(c))
            self.color_buttons_layout.addWidget(btn)
        
        self.update_preview()
    
    def select_input(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("VectorizationPage_last_input_folder", "")
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Load Image", last_folder, 
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All (*.*)"
        )
        if fpath:
            try:
                # Clear old image data first
                if self.original:
                    del self.original
                if self._preview_pixmap:
                    del self._preview_pixmap
                if self._original_preview:
                    del self._original_preview
                gc.collect()
                
                self.input_file = fpath
                settings.setValue("VectorizationPage_last_input_folder", os.path.dirname(fpath))
                # Load image and preserve transparency if present
                self.original = Image.open(fpath)
                # Ensure it's in a format we can work with
                if self.original.mode not in ('RGB', 'RGBA'):
                    self.original = self.original.convert('RGB')
                self.update_preview()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")
    
    def select_output(self):
        self.output_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
    
    def apply_preset(self, colors, thickness, blur, contrast, edges=True):
        self.colors_slider.setValue(colors)
        self.thickness_slider.setValue(thickness)
        self.blur_slider.setValue(blur)
        self.contrast_slider.setValue(contrast)
        self.edge_check.setChecked(edges)
    
    def update_params(self):
        self.colors_label.setText(str(self.colors_slider.value()))
        self.thick_label.setText(str(self.thickness_slider.value()))
        self.blur_label.setText(str(self.blur_slider.value()))
        self.contrast_label.setText(str(self.contrast_slider.value()))
        self.tolerance_label.setText(str(self.tolerance_slider.value()))
        self.update_preview()
    
    def update_preview(self):
        if not self.original:
            return
        
        try:
            # Create preview (max 1200x1200)
            img = self.original.copy()
            w, h = img.size
            
            if h > 1200 or w > 1200:
                scale = min(1200/h, 1200/w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Store original preview for color picking
            if img.mode == 'RGBA':
                data = img.tobytes()
                qimg = QImage(data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
            else:
                rgb_img = img.convert('RGB')
                data = rgb_img.tobytes()
                qimg = QImage(data, rgb_img.width, rgb_img.height, rgb_img.width * 3, QImage.Format_RGB888)
            self._original_preview = QPixmap.fromImage(qimg)
            
            # Apply vectorization effect
            vectorized = self.vectorize_image(img)
            
            # Convert to QPixmap (handle both RGB and RGBA)
            if vectorized.mode == 'RGBA':
                data = vectorized.tobytes()
                qimg = QImage(data, vectorized.width, vectorized.height, 
                             vectorized.width * 4, QImage.Format_RGBA8888)
            else:
                data = vectorized.convert('RGB').tobytes()
                qimg = QImage(data, vectorized.width, vectorized.height, 
                             vectorized.width * 3, QImage.Format_RGB888)
            self._preview_pixmap = QPixmap.fromImage(qimg)
            self.apply_zoom()
        except Exception as e:
            print(f"Preview error: {e}")
        finally:
            # Clean up intermediate variables
            gc.collect()
    
    def vectorize_image(self, img):
        """Apply clean vectorization with optional color removal"""
        # Step 1: Remove selected colors if enabled
        if self.remove_color_check.isChecked() and len(self.colors_to_remove) > 0:
            img_arr = np.array(img)
            has_alpha = img.mode == 'RGBA'
            
            # Get tolerance
            tolerance = self.tolerance_slider.value()
            
            # Calculate color distance
            if has_alpha:
                rgb = img_arr[:, :, :3]
            else:
                rgb = img_arr
            
            # Convert to RGBA if not already
            if not has_alpha:
                img = img.convert('RGBA')
                img_arr = np.array(img)
            
            # Create combined mask for all colors to remove
            combined_mask = np.zeros(rgb.shape[:2], dtype=bool)
            
            for target_color in self.colors_to_remove:
                # Color difference calculation
                target_arr = np.array(target_color)
                diff = np.abs(rgb.astype(np.int16) - target_arr.astype(np.int16))
                distance = np.sum(diff, axis=2)
                
                # Add to mask
                mask = distance <= (tolerance * 7.65)  # Scale tolerance appropriately
                combined_mask = combined_mask | mask
            
            # Set alpha to 0 for matched pixels
            img_arr[:, :, 3] = np.where(combined_mask, 0, img_arr[:, :, 3])
            img = Image.fromarray(img_arr, mode='RGBA')
            
            # Free memory
            del rgb, img_arr, combined_mask, diff, distance, mask, target_arr
        
        # Convert to RGB for processing (keep alpha separate if present)
        has_alpha = img.mode == 'RGBA'
        alpha_channel = None
        if has_alpha:
            alpha_channel = img.split()[3]
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Step 2: Apply blur/smoothing if requested
        blur_val = self.blur_slider.value()
        if blur_val > 0:
            if HAS_CV2:
                cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                cv2_img = cv2.bilateralFilter(cv2_img, d=blur_val*2+1, sigmaColor=75, sigmaSpace=75)
                img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
            else:
                img = img.filter(ImageFilter.MedianFilter(size=min(5, blur_val+2)))
        
        # Step 3: Posterize colors
        num_colors = self.colors_slider.value()
        img = img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        img = img.convert('RGB')
        
        # Step 4: Aggressive smoothing for clean edges (especially angles)
        if HAS_CV2:
            cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # Bilateral filter with larger neighborhood for better smoothing
            cv2_img = cv2.bilateralFilter(cv2_img, d=9, sigmaColor=75, sigmaSpace=75)
            # Additional Gaussian blur for extra smoothness on angled edges
            cv2_img = cv2.GaussianBlur(cv2_img, (3, 3), 0.8)
            img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
        else:
            # Multiple passes of smoothing for non-CV2 users
            img = img.filter(ImageFilter.SMOOTH_MORE)
            img = img.filter(ImageFilter.SMOOTH)
        
        # Step 5: Light sharpening to restore detail without making edges blocky
        img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=50, threshold=3))
        
        # Step 6: Optional edge darkening
        if self.edge_check.isChecked():
            img_arr = np.array(img, dtype=np.uint8)
            h, w = img_arr.shape[:2]
            
            # Simple edge detection by color differences
            edges = np.zeros((h, w), dtype=np.float32)
            
            # Horizontal edges
            diff_h = np.sum(np.abs(img_arr[:, 1:, :].astype(np.int16) - img_arr[:, :-1, :].astype(np.int16)), axis=2)
            edges[:, :-1] += (diff_h > 5).astype(np.float32)
            edges[:, 1:] += (diff_h > 5).astype(np.float32)
            
            # Vertical edges
            diff_v = np.sum(np.abs(img_arr[1:, :, :].astype(np.int16) - img_arr[:-1, :, :].astype(np.int16)), axis=2)
            edges[:-1, :] += (diff_v > 5).astype(np.float32)
            edges[1:, :] += (diff_v > 5).astype(np.float32)
            
            edges = np.clip(edges / 4.0, 0, 1)
            
            # Apply thickness
            thickness = self.thickness_slider.value()
            if thickness > 1 and HAS_CV2:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
                edges = cv2.dilate((edges * 255).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255
                edges = cv2.GaussianBlur(edges, (3, 3), 0.5)
            elif thickness > 1:
                edges_img = Image.fromarray((edges * 255).astype(np.uint8), mode='L')
                for _ in range(thickness - 1):
                    edges_img = edges_img.filter(ImageFilter.MaxFilter(3))
                edges_img = edges_img.filter(ImageFilter.SMOOTH)
                edges = np.array(edges_img, dtype=np.float32) / 255
            
            # Darken edges
            contrast_strength = self.contrast_slider.value() / 100.0
            edge_darken = 0.5 * contrast_strength
            
            img_arr_float = img_arr.astype(np.float32)
            for c in range(3):
                img_arr_float[:, :, c] *= (1 - edges * edge_darken)
            
            img = Image.fromarray(np.clip(img_arr_float, 0, 255).astype(np.uint8), mode='RGB')
            
            # Free memory
            del img_arr, edges, diff_h, diff_v, img_arr_float
        
        # Step 7: Restore alpha if present
        if has_alpha and alpha_channel:
            img = img.convert('RGBA')
            img.putalpha(alpha_channel)
        
        return img
    
    def apply_zoom(self):
        if not self._preview_pixmap:
            return
        self.zoom_label.setText(f"{self.zoom_slider.value()}%")
        zoom = self.zoom_slider.value() / 100.0
        sw = int(self._preview_pixmap.width() * zoom)
        sh = int(self._preview_pixmap.height() * zoom)
        scaled = self._preview_pixmap.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        self.preview_label.setFixedSize(sw, sh)
    
    def export_image(self):
        if not self.original or not self.output_dir:
            QMessageBox.warning(self, "Error", "Load image and select output folder")
            return
        
        try:
            # Show progress bar
            self.single_progress.setValue(0)
            self.single_progress.setVisible(True)
            
            # Process full resolution
            self.single_progress.setValue(30)
            vectorized = self.vectorize_image(self.original)
            
            self.single_progress.setValue(70)
            base = os.path.splitext(os.path.basename(self.input_file))[0]
            output_path = os.path.join(self.output_dir, f"{base}_vectorized.png")
            vectorized.save(output_path, 'PNG')
            
            # Free memory
            del vectorized
            gc.collect()
            
            self.single_progress.setValue(100)
            QMessageBox.information(self, "Success", f"Saved to {output_path}")
            
            # Hide progress bar after short delay
            QTimer.singleShot(1000, lambda: self.single_progress.setVisible(False))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def reset_all(self):
        self.colors_slider.setValue(8)
        self.edge_check.setChecked(False)
        self.thickness_slider.setValue(2)
        self.blur_slider.setValue(0)
        self.contrast_slider.setValue(50)
        self.tolerance_slider.setValue(30)
        self.remove_color_check.setChecked(False)
        self.zoom_slider.setValue(100)
        self.color_to_remove = (255, 255, 255)
        self.color_preview.setStyleSheet("background-color: white; border: 1px solid black;")
        if self.original:
            self.update_preview()
    
    def update_theme_colors(self):
        pass


# ===================================================================
# HEX TOOL - Designer Palette Generator
# ===================================================================


class CategoryWindow(QMainWindow):
    THEME_PRESETS = {
        "Dark (Original)": {"accent": "#00FFC6", "primary": "#070A0E", "secondary": "#0B0F15", "tertiary": "#141A22", "text": "#E6EAF0"},
        "Ocean Blue": {"accent": "#42a5f5", "primary": "#0a1929", "secondary": "#0d2136", "tertiary": "#1565c0", "text": "#e3f2fd"},
        "Purple Dream": {"accent": "#f4d35e", "primary": "#120018", "secondary": "#1f0030", "tertiary": "#360052", "text": "#f5e9ff"},
        "Sunset Fire": {"accent": "#ff9800", "primary": "#1f0f00", "secondary": "#331a00", "tertiary": "#e65100", "text": "#fff3e0"},
        "Cherry Blossom": {"accent": "#f06292", "primary": "#150b12", "secondary": "#24131d", "tertiary": "#4a2035", "text": "#ffe4ef"},
    }

    def __init__(self):
        super().__init__()
        self.settings = QSettings("PixelForge", "PixelForgeImageTools")
        self.setWindowTitle("PixelForge - Image Tools")
        self.setMinimumSize(1480, 860)
        icon = get_cached_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setSpacing(5)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)

        self.tool_names = [
            "Image Resizer",
            "Photo Editing",
            "Watermark",
            "Background Tools",
            "Vectorization",
        ]

        self.stack = QStackedWidget()
        self.pages = {
            "Home": self._build_home_page(),
            "Image Resizer": ImageResizerPage(),
            "Photo Editing": PhotoEditingPage(),
            "Watermark": BatchWatermarkPage(),
            "Background Tools": BackgroundToolsPage(),
            "Vectorization": VectorizationPage()
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self.sidebar_buttons = {}
        category_label = QLabel("Image Tools")
        category_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px 4px;")
        self.sidebar_layout.addWidget(category_label)

        for name in ["Home", *self.tool_names]:
            button = QPushButton(name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, n=name: self.switch_page(n))
            self.sidebar_layout.addWidget(button)
            self.sidebar_buttons[name] = button
        self.sidebar_layout.addStretch()

        footer_widget = QWidget()
        footer_layout = QHBoxLayout(footer_widget)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        self.made_label = QLabel("Made by Orvlyn")
        self.made_label.setStyleSheet("font-size: 11px; color: #00FFC6; padding: 10px;")
        footer_layout.addWidget(self.made_label)
        footer_layout.addStretch()
        self.sidebar_layout.addWidget(footer_widget)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar_layout)
        sidebar_widget.setFixedWidth(340)
        sidebar_widget.setObjectName("Sidebar")

        layout.addWidget(sidebar_widget)
        layout.addWidget(self.stack)

        self.apply_theme(self.settings.value("theme", "Dark (Original)"))
        self.switch_page("Home")

    def _build_home_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(14)

        self.home_title = QLabel("Image Tools Workspace")
        self.home_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FFC6;")
        subtitle = QLabel(
            "Resize, retouch, watermark, remove backgrounds, and vectorize images in one focused toolkit."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size: 13px;")

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme"))
        self.home_theme_combo = QComboBox()
        self.home_theme_combo.addItems(list(self.THEME_PRESETS.keys()))
        self.home_theme_combo.currentTextChanged.connect(self.apply_theme)
        theme_row.addWidget(self.home_theme_combo)
        theme_row.addStretch(1)

        stats = QLabel(
            "Included: Image Resizer, Photo Editing, Watermark, Background Tools, Vectorization"
        )
        stats.setWordWrap(True)

        quick = QGridLayout()
        quick.setHorizontalSpacing(10)
        quick.setVerticalSpacing(10)
        self.home_quick_buttons = []
        for idx, name in enumerate(self.tool_names):
            btn = QPushButton(name)
            btn.setMinimumHeight(44)
            btn.clicked.connect(lambda checked=False, n=name: self.switch_page(n))
            quick.addWidget(btn, idx // 2, idx % 2)
            self.home_quick_buttons.append(btn)

        actions = QHBoxLayout()
        check_updates_btn = QPushButton("Check For Updates")
        check_updates_btn.clicked.connect(self._check_for_updates)
        github_btn = QPushButton("Open PixelForge GitHub")
        github_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/Orvlyn/PixelForge")))
        actions.addWidget(check_updates_btn)
        actions.addWidget(github_btn)
        actions.addStretch(1)

        layout.addWidget(self.home_title)
        layout.addWidget(subtitle)
        layout.addLayout(theme_row)
        layout.addWidget(stats)
        layout.addLayout(quick)
        layout.addLayout(actions)
        layout.addStretch(1)
        return page

    def _version_tuple(self, value: str) -> tuple:
        parts = []
        for token in str(value or "").replace("-", ".").split("."):
            if token.isdigit():
                parts.append(int(token))
        return tuple(parts) if parts else (0,)

    def _check_for_updates(self) -> None:
        import json

        try:
            with urllib.request.urlopen(UPDATE_CHECK_URL, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            latest = str(payload.get("version") or "").strip()
            download = str(payload.get("download_url") or "").strip()
            notes = str(payload.get("notes") or "").strip()
        except Exception as exc:
            QMessageBox.warning(self, "Update Check", f"Could not check updates.\n\n{exc}")
            return

        if latest and self._version_tuple(latest) > self._version_tuple(APP_VERSION):
            message = [f"Current: {APP_VERSION}", f"Latest: {latest}"]
            if notes:
                message.extend(["", "Release notes:", notes])
            if download:
                message.extend(["", f"Download: {download}"])
            QMessageBox.information(self, "Update Available", "\n".join(message))
        else:
            QMessageBox.information(self, "Up To Date", f"You are on the latest version ({APP_VERSION}).")

    def apply_theme(self, name=None):
        if not name or name not in self.THEME_PRESETS:
            name = "Dark (Original)"
        palette = self.THEME_PRESETS[name]
        accent = palette["accent"]
        primary = palette["primary"]
        secondary = palette["secondary"]
        tertiary = palette["tertiary"]
        text = palette["text"]
        css = """
            QMainWindow {{ background: {primary}; }}
            QWidget {{ background: {primary}; color: {text}; }}
            #Sidebar QPushButton {{
                background: {secondary};
                border: none;
                padding: 12px;
                border-radius: 6px;
                text-align: left;
                color: {text};
            }}
            #Sidebar QPushButton:checked {{
                background: {accent};
                color: {primary};
                font-weight: bold;
            }}
            #Sidebar QPushButton:hover {{
                background: {tertiary};
            }}
            QWidget#Card {{
                background: {secondary};
                border-radius: 12px;
                padding: 20px;
            }}
            QPushButton {{
                background: {secondary};
                border: 1px solid {tertiary};
                border-radius: 8px;
                padding: 8px 12px;
                color: {text};
                font-weight: 500;
            }}
            QPushButton:hover {{
                border: 1px solid {accent};
                background: {tertiary};
            }}
            QPushButton:pressed {{
                background: {accent};
                color: {primary};
            }}
            QLineEdit, QComboBox, QSpinBox, QPlainTextEdit {{
                background: {secondary};
                border: 1px solid {tertiary};
                border-radius: 6px;
                padding: 6px;
                color: {text};
                selection-background-color: {accent};
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 2px solid {accent};
            }}
            QSlider::groove:horizontal {{
                background: {tertiary};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {accent};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QTabWidget::pane {{
                background: {secondary};
                border: 1px solid {tertiary};
                border-radius: 8px;
            }}
            QTabBar::tab {{
                background: {secondary};
                color: {text};
                border: 1px solid {tertiary};
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background: {accent};
                color: {primary};
                font-weight: bold;
            }}
            QProgressBar {{
                background: {secondary};
                border-radius: 6px;
                text-align: center;
                color: {accent};
                border: 1px solid {tertiary};
            }}
            QProgressBar::chunk {{
                background: {accent};
                border-radius: 4px;
            }}
            QLabel {{
                color: {text};
                font: 13px 'Segoe UI';
            }}
            QScrollArea {{
                border: none;
                background: {secondary};
            }}
            QTableWidget {{
                background: {secondary};
                gridline-color: {tertiary};
                color: {text};
            }}
            QTableWidget::item:selected {{
                background-color: {accent};
                color: {primary};
            }}
            QHeaderView::section {{
                background-color: {tertiary};
                color: {text};
                padding: 5px;
                border: 1px solid {tertiary};
            }}
            QSplitter::handle {{
                background: {accent};
                width: 8px;
                border-radius: 3px;
            }}
        """.format(
            accent=accent,
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            text=text,
        )
        self.setStyleSheet(css)

        self.current_theme_name = name
        self.current_theme_colors = {
            "accent": accent,
            "primary_bg": primary,
            "secondary_bg": secondary,
            "tertiary_bg": tertiary,
            "text": text,
        }
        self.settings.setValue("theme", name)

        if hasattr(self, "home_theme_combo"):
            self.home_theme_combo.blockSignals(True)
            self.home_theme_combo.setCurrentText(name)
            self.home_theme_combo.blockSignals(False)
        if hasattr(self, "home_title"):
            self.home_title.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {accent};")
        if hasattr(self, "home_quick_buttons"):
            for btn in self.home_quick_buttons:
                btn.setStyleSheet(f"font-weight: bold; background: {secondary}; color: {accent};")
        if hasattr(self, "made_label"):
            self.made_label.setStyleSheet(f"font-size: 11px; color: {accent}; padding: 10px;")

    def switch_page(self, name):
        for btn in self.sidebar_buttons.values():
            btn.setChecked(False)
        self.sidebar_buttons[name].setChecked(True)
        self.stack.setCurrentWidget(self.pages[name])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CategoryWindow()
    window.show()
    sys.exit(app.exec())
