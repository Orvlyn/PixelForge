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

class PaletteExtractorPage(CardPage):
    def __init__(self):
        super().__init__("Palette Extractor")

        self.split = QHBoxLayout()
        left = QWidget()
        self.left_layout = QVBoxLayout(left)
        right = QWidget()
        self.right_layout = QVBoxLayout(right)
        self.split.addWidget(left, 35)
        self.split.addWidget(right, 65)
        self.card_layout.addLayout(self.split)

        self.input_btn = QPushButton("📤 Upload Image")
        self.input_btn.setStyleSheet("font-size: 16px; padding: 14px;")
        self.left_layout.addWidget(self.input_btn)

        self.copy_all_btn = QPushButton("📋 Copy All HEX")
        self.copy_all_btn.clicked.connect(self.copy_all_hex)
        self.left_layout.addWidget(self.copy_all_btn)
        self.clear_btn = QPushButton("🗑️ Clear Palette")
        self.clear_btn.clicked.connect(self.clear_palette)
        self.left_layout.addWidget(self.clear_btn)

        # new controls
        ctrl_h = QHBoxLayout()
        self.num_colors_spin = QSpinBox()
        self.num_colors_spin.setRange(3, 12)
        self.num_colors_spin.setValue(8)
        ctrl_h.addWidget(QLabel("Count"))
        ctrl_h.addWidget(self.num_colors_spin)
        self.sort_combo = NoWheelComboBox()
        self.sort_combo.addItems(["Frequency","Brightness"])
        ctrl_h.addWidget(QLabel("Sort by"))
        ctrl_h.addWidget(self.sort_combo)
        self.left_layout.addLayout(ctrl_h)

        self.gradient_btn = QPushButton("🎨 Generate Gradient")
        self.gradient_btn.clicked.connect(self.copy_gradient)
        self.left_layout.addWidget(self.gradient_btn)

        self.export_btn = QPushButton("💾 Export Palette PNG")
        self.export_btn.clicked.connect(self.export_palette)
        self.left_layout.addWidget(self.export_btn)

        self.export_css_btn = QPushButton("Export CSS vars")
        self.export_css_btn.clicked.connect(self.export_css_vars)
        self.left_layout.addWidget(self.export_css_btn)
        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(self.export_json)
        self.left_layout.addWidget(self.export_json_btn)
        self.export_scss_btn = QPushButton("Export SCSS map")
        self.export_scss_btn.clicked.connect(self.export_scss)
        self.left_layout.addWidget(self.export_scss_btn)

        self.left_layout.addStretch()

        # GRADIENT PREVIEW
        self.gradient_label = QLabel("Gradient Preview")
        self.gradient_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        self.left_layout.addWidget(self.gradient_label)
        
        self.gradient_preview = QLabel()
        self.gradient_preview.setMinimumHeight(60)
        self.gradient_preview.setMaximumHeight(60)
        self.gradient_preview.setStyleSheet("background: #0B0F15; border-radius: 8px; border: 2px solid #141A22;")
        self.left_layout.addWidget(self.gradient_preview)

        self.preview_label = QLabel("Image Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedHeight(300)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_layout.addWidget(self.preview_label)

        self.palette_title = QLabel("🎨 Dominant Palette")
        self.palette_title.setStyleSheet("font-size: 19px; font-weight: bold; color: #00ffc6; margin: 15px 0;")
        self.right_layout.addWidget(self.palette_title)

        self.colors_widget = QWidget()
        self.colors_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.colors_layout = QGridLayout(self.colors_widget)
        self.colors_layout.setSpacing(4)
        self.colors_layout.setContentsMargins(0,0,0,0)
        self.right_layout.addWidget(self.colors_widget)

        self.input_file = ""
        self.dominant_colors = []
        self.input_btn.clicked.connect(self.select_input)
        
        # Debounce rapid changes to prevent widget conflicts
        self.palette_update_timer = QTimer()
        self.palette_update_timer.setSingleShot(True)
        self.palette_update_timer.timeout.connect(self.extract_and_display_palette)
        self.num_colors_spin.valueChanged.connect(lambda: self.palette_update_timer.start(150))
        self.sort_combo.currentIndexChanged.connect(lambda: self.palette_update_timer.start(150))

    def select_input(self):
        self.input_file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.gif *.webp);;All Files (*.*)")
        if self.input_file:
            try:
                # Check file size before loading to prevent crashes
                file_size = os.path.getsize(self.input_file) / (1024 * 1024)  # MB
                if file_size > 100:  # Warn on files > 100MB
                    reply = QMessageBox.question(self, "Large File", 
                        f"This file is {file_size:.1f}MB. Loading may be slow. Continue?",
                        QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.No:
                        return
                
                pixmap = QPixmap(self.input_file).scaled(520, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(pixmap)
                self.extract_and_display_palette()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load image: {str(e)}")

    def clear_palette(self):
        self.dominant_colors = []
        # Immediate widget removal to prevent conflicts
        while self.colors_layout.count():
            item = self.colors_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.gradient_preview.setPixmap(QPixmap())
        self.gradient_preview.setStyleSheet("background: #0B0F15; border-radius: 8px; border: 2px solid #141A22;")

    def extract_and_display_palette(self):
        if not self.input_file:
            return
        
        # Check file still exists
        if not os.path.exists(self.input_file):
            return
            
        try:
            # Use context manager to ensure image is closed
            with Image.open(self.input_file) as img:
                img = img.convert("RGB")
                thumb = img.copy()
            thumb.thumbnail((300, 300), Image.Resampling.LANCZOS)

            quant = thumb.convert('P', palette=Image.ADAPTIVE, colors=32)
            counts = quant.getcolors(maxcolors=256)  # Limit for performance
            palette_list = quant.getpalette() or []

            entries = []
            if counts:
                for cnt, idx in counts:
                    base = idx * 3
                    if base + 2 < len(palette_list):
                        rgb = (palette_list[base], palette_list[base + 1], palette_list[base + 2])
                        entries.append((cnt, rgb))

            if self.sort_combo.currentText() == "Brightness":
                def bright(item):
                    r,g,b = item[1]
                    return 0.299*r + 0.587*g + 0.114*b
                entries.sort(reverse=True, key=bright)
            else:
                entries.sort(reverse=True, key=lambda t: t[0])

            # pick top distinct colors using simple euclidean distance threshold
            def dist2(a, b):
                return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2

            picked = []
            max_colors = self.num_colors_spin.value()
            total_count = sum(cnt for cnt,_ in entries)
            for cnt, rgb in entries:
                if not picked:
                    picked.append((rgb, cnt))
                else:
                    too_close = False
                    for prgb, _ in picked:
                        if dist2(prgb, rgb) < 2200:
                            too_close = True
                            break
                    if not too_close:
                        picked.append((rgb, cnt))
                if len(picked) >= max_colors:
                    break

            self.dominant_colors = [p[0] for p in picked]
            self.color_counts = picked
            
            # Close the thumbnail to free memory
            thumb.close()
            del thumb

            # clear previous widgets immediately to prevent conflicts
            while self.colors_layout.count():
                item = self.colors_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # choose number of columns for grid
            cols = min(6, self.num_colors_spin.value())
            # add swatches in grid positions
            for idx, rgb in enumerate(self.dominant_colors):
                hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                percent = ""
                if hasattr(self, 'color_counts') and idx < len(self.color_counts):
                    cnt = self.color_counts[idx][1]
                    total = sum(c for _,c in self.color_counts)
                    percent = f" ({cnt/total*100:.1f}%)"
                swatch = QWidget()
                swatch.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                swatch.setStyleSheet(f"background-color: {hex_code}; border: none;")
                swatch.setCursor(Qt.PointingHandCursor)
                swatch.setFocusPolicy(Qt.NoFocus)

                cont = QWidget()
                cont.setFocusPolicy(Qt.NoFocus)
                cl = QVBoxLayout(cont)
                cl.setContentsMargins(0, 0, 0, 0)
                cl.setSpacing(0)
                lbl = QLabel(hex_code + percent)
                lbl.setFocusPolicy(Qt.NoFocus)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("font-size: 12px; color: rgba(255,255,255,0.9); background: transparent;")
                cl.addWidget(swatch)
                cl.addWidget(lbl)
                # clicking either the swatch or label copies the hex to clipboard
                cont.mousePressEvent = lambda e, h=hex_code: QApplication.clipboard().setText(h)
                swatch.mousePressEvent = lambda e, h=hex_code: QApplication.clipboard().setText(h)
                lbl.mousePressEvent = lambda e, h=hex_code: QApplication.clipboard().setText(h)
                row = idx // cols
                col = idx % cols
                self.colors_layout.addWidget(cont, row, col)

            # set stretch for rows/cols
            for c in range(cols):
                self.colors_layout.setColumnStretch(c, 1)
            rows = (len(self.dominant_colors) + cols - 1) // cols
            for r in range(rows):
                self.colors_layout.setRowStretch(r, 1)
            
            # Update gradient preview
            self.update_gradient_preview()
        except Exception as e:
            # Log error instead of silent failure
            print(f"Palette extraction error: {e}")
            QMessageBox.warning(self, "Extraction Error", f"Failed to extract palette: {str(e)}")

    def update_gradient_preview(self):
        """Update the gradient preview bar with current palette colors"""
        if not self.dominant_colors:
            self.gradient_preview.setStyleSheet("background: #0B0F15; border-radius: 8px; border: 2px solid #141A22;")
            self.gradient_preview.setPixmap(QPixmap())
            return
        
        # Create gradient image
        grad_width = 500
        grad_height = 50
        gradient_img = Image.new("RGB", (grad_width, grad_height))
        
        # Fill gradient with palette colors
        pixels = gradient_img.load()
        for x in range(grad_width):
            # Calculate position between 0 and 1
            pos = x / grad_width
            
            # Find which color segment this is
            n = len(self.dominant_colors)
            segment_size = 1.0 / (n - 1) if n > 1 else 1.0
            segment_idx = int(pos / segment_size)
            segment_idx = min(segment_idx, n - 2)
            
            # Interpolate between adjacent colors
            local_pos = (pos - segment_idx * segment_size) / segment_size if n > 1 else 0
            
            if n == 1:
                r, g, b = self.dominant_colors[0]
            else:
                r1, g1, b1 = self.dominant_colors[segment_idx]
                r2, g2, b2 = self.dominant_colors[segment_idx + 1]
                r = int(r1 * (1 - local_pos) + r2 * local_pos)
                g = int(g1 * (1 - local_pos) + g2 * local_pos)
                b = int(b1 * (1 - local_pos) + b2 * local_pos)
            
            # Fill vertical column
            for y in range(grad_height):
                pixels[x, y] = (r, g, b)
        
        # Convert to QPixmap and display
        data = gradient_img.tobytes()
        qimg = QImage(data, grad_width, grad_height, grad_width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit label
        pixmap = pixmap.scaledToHeight(50, Qt.SmoothTransformation)
        self.gradient_preview.setPixmap(pixmap)

    def copy_all_hex(self):
        if self.dominant_colors:
            text = "\n".join([f"#{r:02x}{g:02x}{b:02x}" for r,g,b in self.dominant_colors])
            QApplication.clipboard().setText(text)

    def export_palette(self):
        if not self.dominant_colors: return
        file, _ = QFileDialog.getSaveFileName(self, "Save Palette", "", "PNG (*.png)")
        if file:
            pal = Image.new("RGB", (len(self.dominant_colors) * 160, 160))
            draw = ImageDraw.Draw(pal)
            for i, rgb in enumerate(self.dominant_colors):
                draw.rectangle([i * 160, 0, (i + 1) * 160, 160], fill=rgb)
            pal.save(file, optimize=True)

    def copy_gradient(self):
        if not self.dominant_colors: return
        # create CSS linear-gradient string
        stops = []
        n = len(self.dominant_colors)
        for i, rgb in enumerate(self.dominant_colors):
            hexc = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            stops.append(f"{hexc} {int(i/(n-1)*100) if n>1 else 0}%")
        css = f"background: linear-gradient(90deg, {', '.join(stops)});"
        QApplication.clipboard().setText(css)

    def export_css_vars(self):
        if not self.dominant_colors: return
        text = ""
        for i, rgb in enumerate(self.dominant_colors):
            hexc = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            text += f"--color{i+1}: {hexc};\n"
        QApplication.clipboard().setText(text)

    def export_json(self):
        if not self.dominant_colors: return
        arr = [f"#{r:02x}{g:02x}{b:02x}" for r,g,b in self.dominant_colors]
        file, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON (*.json)")
        if file:
            import json
            with open(file, 'w') as f:
                json.dump(arr, f)

    def export_scss(self):
        if not self.dominant_colors: return
        text = "$palette: (\n"
        for i, rgb in enumerate(self.dominant_colors):
            hexc = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            text += f"  color{i+1}: {hexc},\n"
        text += ");"
        QApplication.clipboard().setText(text)

    def update_theme_colors(self):
        """Update colors based on current theme"""
        window = self.window()
        if not hasattr(window, 'current_theme_colors'):
            return
        colors = window.current_theme_colors
        accent = colors.get('accent', '#00FFC6')
        self.palette_title.setStyleSheet(f"font-size: 19px; font-weight: bold; color: {accent}; margin: 15px 0;")


# ===================================================================
# VECTORIZATION TOOL - Image Trace (Adobe Illustrator style)
# ===================================================================

class HEXToolPage(CardPage):
    def __init__(self):
        super().__init__("HEX Tool")
        # allow this page to receive key events (Space randomizer)
        self.setFocusPolicy(Qt.StrongFocus)

        # ===================== INPUT =====================
        top = QHBoxLayout()
        self.hex_input = QLineEdit("#00ffc6")
        self.hex_input.setStyleSheet("font-size: 22px; padding: 12px;")
        top.addWidget(QLabel("HEX"))
        top.addWidget(self.hex_input)
        self.card_layout.addLayout(top)

        self.color_preview = QLabel()
        self.color_preview.setFixedHeight(30)
        self.color_preview.mousePressEvent = self.pick_color_from_preview
        self.card_layout.addWidget(self.color_preview)

        info = QHBoxLayout()
        self.rgb_label = QLabel("RGB: ")
        self.hsl_label = QLabel("HSL: ")
        info.addWidget(self.rgb_label)
        info.addWidget(self.hsl_label)
        self.card_layout.addLayout(info)

        controls = QHBoxLayout()
        self.harm_count_combo = NoWheelComboBox()
        self.harm_count_combo.addItems(["3", "4", "5", "6"])
        controls.addWidget(QLabel("Palette Size"))
        controls.addWidget(self.harm_count_combo)

        self.check_contrast_btn = QPushButton("Contrast Check")
        self.check_contrast_btn.setToolTip("Check contrast ratio of base color vs white/black")
        controls.addWidget(self.check_contrast_btn)

        self.dark_mode_btn = QPushButton("Dark Mode Variant")
        self.dark_mode_btn.setToolTip("Generate dark mode variant of base color")
        controls.addWidget(self.dark_mode_btn)

        self.gradient_btn = QPushButton("Gradient CSS")
        self.gradient_btn.setToolTip("Copy CSS gradient for current palette")
        controls.addWidget(self.gradient_btn)

        self.copy_css_btn = QPushButton("Copy CSS Snippet")
        self.copy_css_btn.setToolTip("Copy CSS variable list for palette")
        controls.addWidget(self.copy_css_btn)

        self.export_btn = QPushButton("Export Palette")
        self.export_btn.setToolTip("Export HEX values to text file")
        controls.addWidget(self.export_btn)

        self.export_png_btn = QPushButton("Export PNG")
        self.export_png_btn.setToolTip("Save current swatches as PNG image")
        controls.addWidget(self.export_png_btn)

        self.card_layout.addLayout(controls)

        # ===================== FULL WIDTH PALETTE =====================
        self.harmony_widget = QWidget()
        self.harmony_layout = QGridLayout(self.harmony_widget)
        # restore original tight layout so swatches fill available width
        self.harmony_layout.setSpacing(0)
        self.harmony_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.addWidget(self.harmony_widget, 1)

        # ===================== SIGNALS =====================
        self.hex_input.textChanged.connect(self.update_all)
        self.harm_count_combo.currentIndexChanged.connect(self.update_all)
        self.export_btn.clicked.connect(self.export_palette)
        self.export_png_btn.clicked.connect(self.export_palette_png)
        self.check_contrast_btn.clicked.connect(self.contrast_check)
        self.dark_mode_btn.clicked.connect(self.apply_dark_variant)
        self.gradient_btn.clicked.connect(self.copy_gradient_css)
        self.copy_css_btn.clicked.connect(self.copy_css_snippet)

        # track locked swatches
        self.locked = {}
        # track user-adjusted (but not locked) swatches
        self.edited = {}
        
        # Track if this is the initial load to set theme accent color
        self._initial_theme_load = True

        self.update_all()

    # ==========================================================
    # BASIC CONVERSIONS
    # ==========================================================
    def hex_to_rgb(self, hex_str):
        hex_str = hex_str.lstrip("#")
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

    def rgb_to_hsl(self, rgb):
        r, g, b = [x / 255.0 for x in rgb]
        return colorsys.rgb_to_hls(r, g, b)

    def hsl_to_hex(self, h, l, s):
        return self.rgb_to_hex(colorsys.hls_to_rgb(h, l, s))

    # ==========================================================
    # DESIGNER PALETTE LOGIC
    # ==========================================================
    def generate_best_palette(self, base_hex, count):
        """Generate professional color-theory based palette with better harmonies"""
        rgb = self.hex_to_rgb(base_hex)
        h, l, s = self.rgb_to_hsl(rgb)
        
        palette = []
        
        if count == 3:
            # 3-color: tones of the base color
            palette.append(self.hsl_to_hex(h, max(0.08, l*0.35), s))      # Dark
            palette.append(self.hsl_to_hex(h, l, s))                       # Base
            palette.append(self.hsl_to_hex(h, min(0.95, l+0.3), min(1.0, s*0.8))) # Light
            
        elif count == 4:
            # 4-color: base tones + true complementary
            palette.append(self.hsl_to_hex(h, max(0.1, l*0.3), s))
            palette.append(self.hsl_to_hex(h, l, s))
            palette.append(self.hsl_to_hex(h, min(0.95, l+0.3), min(1.0, s*0.8)))
            
            # Complementary with better saturation handling
            comp_h = (h + 0.5) % 1.0
            comp_l = 0.5 if l < 0.3 else (0.65 if l < 0.7 else l)
            palette.append(self.hsl_to_hex(comp_h, comp_l, min(1.0, s*1.15)))
            
        elif count == 5:
            # 5-color: Material Design-inspired with better colors
            palette.append(self.hsl_to_hex(h, min(0.95, l+0.3), min(0.6, s*0.6)))   # Very light
            palette.append(self.hsl_to_hex(h, l, s))                                   # Base
            palette.append(self.hsl_to_hex(h, max(0.15, l*0.45), min(1.0, s*1.1)))  # Dark
            
            # Complementary
            comp_h = (h + 0.5) % 1.0
            comp_l = 0.5 if l < 0.35 else (0.6 if l < 0.7 else l - 0.1)
            palette.append(self.hsl_to_hex(comp_h, comp_l, min(1.0, s*1.2)))
            
            # Analogous (30° away)
            ana_h = (h + 0.083) % 1.0
            palette.append(self.hsl_to_hex(ana_h, l, min(1.0, s*1.05)))
            
        else:
            # 6+ colors: comprehensive scheme
            # Monochromatic tones first
            palette.append(self.hsl_to_hex(h, min(0.98, l+0.35), min(0.5, s*0.6)))   # Lightest
            palette.append(self.hsl_to_hex(h, min(0.90, l+0.20), s))
            palette.append(self.hsl_to_hex(h, l, s))                                   # Base
            palette.append(self.hsl_to_hex(h, max(0.15, l*0.55), min(1.0, s*1.1)))   # Dark
            palette.append(self.hsl_to_hex(h, max(0.08, l*0.25), s))                  # Very dark
            
            # Complementary (true opposite)
            comp_h = (h + 0.5) % 1.0
            comp_l = 0.5 if l < 0.35 else (0.6 if l < 0.7 else l - 0.1)
            palette.append(self.hsl_to_hex(comp_h, comp_l, min(1.0, s*1.2)))
            
            if count >= 7:
                # Analogous warm (±30°)
                warm_h = (h + 0.083) % 1.0
                palette.append(self.hsl_to_hex(warm_h, l, min(1.0, s*1.1)))
            
            if count >= 8:
                # Analogous cool (-30°)
                cool_h = (h - 0.083) % 1.0
                palette.append(self.hsl_to_hex(cool_h, l, min(1.0, s*1.05)))
            
            if count >= 9:
                # Triadic (±120°)
                tri1_h = (h + 0.333) % 1.0
                palette.append(self.hsl_to_hex(tri1_h, l, s*0.95))
            
            if count >= 10:
                # Triadic 2 (±240°)
                tri2_h = (h + 0.667) % 1.0
                palette.append(self.hsl_to_hex(tri2_h, l, s*0.95))
        
        return palette[:count]


    def create_swatch(self, color, index):
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container.setStyleSheet(f"background-color: {color};")
        container.setFocusPolicy(Qt.NoFocus)

        layout = QVBoxLayout(container)
        # give a little breathing room so the overlay controls don't get clipped
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignCenter)

        # simple emoji icons (FontAwesome disabled)
        lock_btn = QPushButton("🔓")
        up_btn = QPushButton("▲")
        down_btn = QPushButton("▼")

        # if already locked, show locked icon and keep stored color
        if index in self.locked:
            lock_btn.setText("🔒")
            color = self.locked[index]
            container.setStyleSheet(f"background-color: {color};")

        # determine overlay color for this swatch
        try:
            qbase = QColor(color)
            qoverlay = qbase.darker(120)
            qoverlay.setAlpha(int(255 * 0.6))
            overlay_style = qoverlay.name(QColor.HexArgb)
            # choose contrasting icon color
            brightness = (qbase.red() + qbase.green() + qbase.blue()) / 3
            icon_color = "#000" if brightness > 128 else "#fff"
        except:
            overlay_style = "rgba(0,0,0,0.6)"
            icon_color = "#fff"

        btn_style = (
            f"color: {icon_color}; background-color: {overlay_style};"
            "border: none; border-radius: 4px;"
            "font-size: 14px; font-weight: bold;"
        )
        for b in (lock_btn, up_btn, down_btn):
            b.setFixedSize(34, 28)
            b.setStyleSheet(btn_style.replace('font-size: 14px;', 'font-size: 16px;'))
            b.setFocusPolicy(Qt.NoFocus)

        lock_btn.setToolTip("Lock / unlock color")
        up_btn.setToolTip("Lighten shade")
        down_btn.setToolTip("Darken shade")

        # overlay container (darkened version of swatch)
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(6)
        def update_overlay(col):
            try:
                q = QColor(col)
                q = q.darker(120)
                q.setAlpha(int(255 * 0.6))
                controls.setStyleSheet(f"background-color: {q.name(QColor.HexArgb)}; border-radius: 8px;")
            except:
                controls.setStyleSheet("background: rgba(0,0,0,0.6); border-radius: 8px;")
        update_overlay(color)
        controls_layout.addWidget(lock_btn)
        controls_layout.addWidget(up_btn)
        controls_layout.addWidget(down_btn)
        layout.addWidget(controls)

        # hide until hover
        controls.setVisible(False)
        def enterEvent(e):
            controls.setVisible(True)
            e.accept()
        def leaveEvent(e):
            controls.setVisible(False)
            e.accept()
        container.enterEvent = enterEvent
        container.leaveEvent = leaveEvent

        def _on_click_copy(e, h=color):
            self.setFocus()
            self.copy_to_clip(h)
        container.mousePressEvent = _on_click_copy

        def toggle_lock():
            nonlocal color
            if index in self.locked:
                # unlock but keep any edited color visible
                del self.locked[index]
                lock_btn.setText("🔓")
            else:
                # lock the current visible color (edited if present)
                cur = self.edited.get(index, color)
                self.locked[index] = cur
                lock_btn.setText("🔒")
        lock_btn.clicked.connect(toggle_lock)

        def adjust(amount):
            nonlocal color
            # adjust the currently visible color without implicitly locking
            current = self.locked.get(index, self.edited.get(index, color))
            rgb = self.hex_to_rgb(current)
            h, l, s = self.rgb_to_hsl(rgb)
            l = max(0, min(1, l + amount))
            new_color = self.hsl_to_hex(h, l, s)
            container.setStyleSheet(f"background-color: {new_color};")
            color = new_color
            # store as edited (temporary) unless this slot is locked
            if index in self.locked:
                self.locked[index] = new_color
            else:
                self.edited[index] = new_color
            update_overlay(new_color)
        up_btn.clicked.connect(lambda: adjust(0.05))
        down_btn.clicked.connect(lambda: adjust(-0.05))

        return container

    # ==========================================================
    # UPDATE UI
    # ==========================================================
    def update_all(self):
        hex_code = self.hex_input.text().strip()
        if not hex_code.startswith("#"):
            hex_code = "#" + hex_code
        if len(hex_code) != 7:
            return

        try:
            rgb = self.hex_to_rgb(hex_code)
            h, l, s = self.rgb_to_hsl(rgb)

            self.rgb_label.setText(f"RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}")
            self.hsl_label.setText(f"HSL: {int(h*360)}°, {int(s*100)}%, {int(l*100)}%")
            self.color_preview.setStyleSheet(f"background-color: {hex_code};")

            # Clear old widgets
            for i in reversed(range(self.harmony_layout.count())):
                w = self.harmony_layout.itemAt(i).widget()
                if w:
                    w.deleteLater()
            # clear any existing stretch settings so removed columns don't hold space
            for col in range(self.harmony_layout.columnCount()):
                self.harmony_layout.setColumnStretch(col, 0)
            self.harmony_layout.setRowStretch(0, 0)

            count = int(self.harm_count_combo.currentText())
            palette = self.generate_best_palette(hex_code, count)

            # respect locks (keep existing color if a swatch is locked)
            # also prune any locks beyond the new palette size
            self.locked = {k: v for k, v in self.locked.items() if k < count}
            # prune edited entries too and apply them (but they don't convert to locks)
            self.edited = {k: v for k, v in self.edited.items() if k < count}
            for i in range(len(palette)):
                if i in self.locked:
                    palette[i] = self.locked[i]
                elif i in self.edited:
                    palette[i] = self.edited[i]

            # Full width interactive swatches
            for i, color in enumerate(palette):
                sw = self.create_swatch(color, i)
                self.harmony_layout.addWidget(sw, 0, i)
            # make each column expand equally
            for col in range(count):
                self.harmony_layout.setColumnStretch(col, 1)
            self.harmony_layout.setRowStretch(0, 1)

        except:
            pass

    # ==========================================================
    # EXPORT
    # ==========================================================
    def export_palette(self):
        count = int(self.harm_count_combo.currentText())
        palette = self.generate_best_palette(self.hex_input.text(), count)
        # override with locked and edited visible colors
        for i in range(count):
            if i in self.locked:
                palette[i] = self.locked[i]
            elif i in self.edited:
                palette[i] = self.edited[i]

        path, _ = QFileDialog.getSaveFileName(self, "Save Palette", "", "Text Files (*.txt)")
        if path:
            with open(path, "w") as f:
                for c in palette:
                    f.write(c + "\n")

    def export_palette_png(self):
        count = int(self.harm_count_combo.currentText())
        base = self.hex_input.text().strip()
        if not base.startswith("#"):
            base = "#" + base
        # generate canonical palette then override with any locked/modified swatches
        palette = self.generate_best_palette(base, count)
        for i in range(count):
            if i in self.locked:
                palette[i] = self.locked[i]
            elif i in self.edited:
                palette[i] = self.edited[i]

        path, _ = QFileDialog.getSaveFileName(self, "Save Palette Image", "", "PNG (*.png)")
        if path:
            # construct horizontal strip using full hex strings
            width = 160 * count
            pal = Image.new("RGB", (width, 160))
            draw = ImageDraw.Draw(pal)
            for i, hexc in enumerate(palette):
                # PIL accepts hex strings like '#rrggbb'
                draw.rectangle([i*160, 0, (i+1)*160, 160], fill=hexc)
            pal.save(path, optimize=True)

    # ==========================================================
    # CLIPBOARD
    # ==========================================================
    def copy_to_clip(self, text):
        QApplication.clipboard().setText(text)

    # ==========================================================
    # COLOR PICKER
    # ==========================================================
    def pick_color_from_preview(self, event):
        color = QColorDialog.getColor(initial=QColor(self.hex_input.text()))
        if color.isValid():
            self.hex_input.setText(color.name())

    # helper used when randomising
    def _random_variation(self, hex_color):
        # slightly jitter hue/lighting around provided color
        rgb = self.hex_to_rgb(hex_color)
        h, l, s = self.rgb_to_hsl(rgb)
        h = (h + random.uniform(-0.08, 0.08)) % 1.0
        l = max(0, min(1, l + random.uniform(-0.1, 0.1)))
        s = max(0, min(1, s + random.uniform(-0.1, 0.1)))
        return self.hsl_to_hex(h, l, s)

    def contrast_check(self):
        base = self.hex_input.text()
        try:
            rgb = self.hex_to_rgb(base)
        except:
            return
        def lum(c):
            vals = []
            for x in c:
                v = x/255.0
                if v <= 0.03928:
                    vals.append(v/12.92)
                else:
                    vals.append(((v+0.055)/1.055) ** 2.4)
            return 0.2126*vals[0] + 0.7152*vals[1] + 0.0722*vals[2]
        l1 = lum(rgb)
        lwhite = lum((255,255,255))
        lblack = lum((0,0,0))
        def ratio(lA,lB):
            if lA>lB: return (lA+0.05)/(lB+0.05)
            else: return (lB+0.05)/(lA+0.05)
        rwhite = ratio(l1,lwhite)
        rblack = ratio(l1,lblack)
        msg = f"Contrast vs white: {rwhite:.1f}\nContrast vs black: {rblack:.1f}"
        QMessageBox.information(self, "Contrast Check", msg)

    def apply_dark_variant(self):
        base = self.hex_input.text()
        try:
            rgb = self.hex_to_rgb(base)
        except:
            return
        # simple dark mode: invert luminance
        h,l,s = self.rgb_to_hsl(rgb)
        new_l = 1 - l
        dark = self.hsl_to_hex(h, new_l, s)
        QApplication.clipboard().setText(dark)
        QMessageBox.information(self, "Dark Mode", f"Dark variant copied: {dark}")

    def copy_gradient_css(self):
        # build gradient from current swatches
        count = int(self.harm_count_combo.currentText())
        base = self.hex_input.text().strip()
        if not base.startswith("#"):
            base = "#" + base
        palette = self.generate_best_palette(base, count)
        for i in range(count):
            if i in self.locked:
                palette[i] = self.locked[i]
            elif i in self.edited:
                palette[i] = self.edited[i]
        if not palette: return
        stops = []
        n = len(palette)
        for idx, c in enumerate(palette):
            stops.append(f"{c} {int(idx/(n-1)*100) if n>1 else 0}%")
        css = f"background: linear-gradient(90deg, {', '.join(stops)});"
        QApplication.clipboard().setText(css)
        QMessageBox.information(self, "Copied", "Gradient CSS copied to clipboard!")

    def copy_css_snippet(self):
        count = int(self.harm_count_combo.currentText())
        base = self.hex_input.text().strip()
        if not base.startswith("#"):
            base = "#" + base
        palette = self.generate_best_palette(base, count)
        for i in range(count):
            if i in self.locked:
                palette[i] = self.locked[i]
            elif i in self.edited:
                palette[i] = self.edited[i]
        if not palette: return
        text = ""
        for i, c in enumerate(palette):
            text += f"--color{i+1}: {c};\n"
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Copied", "CSS variables copied to clipboard!")

    # ==========================================================
    # SPACE RANDOMIZER
    # ==========================================================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            # randomize palette based on the current hex_input but preserve locks
            count = int(self.harm_count_combo.currentText())
            base = self.hex_input.text().strip()
            if not base.startswith("#"):
                base = "#" + base
            # prune locks that are no longer relevant
            self.locked = {k: v for k, v in self.locked.items() if k < count}
            # generate a starting palette anchored to base color
            palette = self.generate_best_palette(base, count)
            # apply random variation to unlocked slots
            for i in range(count):
                if i in self.locked:
                    palette[i] = self.locked[i]
                else:
                    palette[i] = self._random_variation(palette[i])
            # clear any temporary edited adjustments for unlocked slots
            self.edited = {k: v for k, v in self.edited.items() if k in self.locked}
            # rebuild UI
            for i in reversed(range(self.harmony_layout.count())):
                w = self.harmony_layout.itemAt(i).widget()
                if w:
                    w.deleteLater()
            # clear previous stretch settings
            for col in range(self.harmony_layout.columnCount()):
                self.harmony_layout.setColumnStretch(col, 0)
            self.harmony_layout.setRowStretch(0, 0)
            for i, col in enumerate(palette):
                sw = self.create_swatch(col, i)
                self.harmony_layout.addWidget(sw, 0, i)
            for col in range(count):
                self.harmony_layout.setColumnStretch(col, 1)
            self.harmony_layout.setRowStretch(0, 1)
            # NOTE: Do NOT change hex_input - just randomize the palette colors
            # The base color remains unchanged, only palette swatches are randomized
            event.accept()
        else:
            super().keyPressEvent(event)

    def update_theme_colors(self):
        """Update colors based on current theme"""
        window = self.window()
        if not hasattr(window, 'current_theme_colors'):
            return
        
        # On initial load, set the hex input to the theme's accent color
        if self._initial_theme_load:
            colors = window.current_theme_colors
            accent = colors.get('accent', '#00FFC6')
            self.hex_input.setText(accent)
            self._initial_theme_load = False


# ===================================================================
# PIXEL ART MODE - Retro image converter
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
        self.settings = QSettings("PixelForge", "PixelForgeColorLab")
        self.setWindowTitle("PixelForge - Color Lab")
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
            "Palette Extractor",
            "HEX Tool",
        ]

        self.stack = QStackedWidget()
        self.pages = {
            "Home": self._build_home_page(),
            "Palette Extractor": PaletteExtractorPage(),
            "HEX Tool": HEXToolPage()
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self.sidebar_buttons = {}
        category_label = QLabel("Color Lab")
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

        self.home_title = QLabel("Color Lab Workspace")
        self.home_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FFC6;")
        subtitle = QLabel(
            "Extract palettes from images and generate designer-friendly HEX palettes, gradients, and CSS tokens."
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

        stats = QLabel("Included: Palette Extractor, HEX Tool")
        stats.setWordWrap(True)

        quick = QHBoxLayout()
        self.home_quick_buttons = []
        for name in self.tool_names:
            btn = QPushButton(name)
            btn.setMinimumHeight(44)
            btn.clicked.connect(lambda checked=False, n=name: self.switch_page(n))
            quick.addWidget(btn)
            self.home_quick_buttons.append(btn)
        quick.addStretch(1)

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
