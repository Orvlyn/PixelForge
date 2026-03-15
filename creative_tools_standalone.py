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

class PixelArtPage(CardPage):
    def __init__(self):
        super().__init__("Pixel Art Mode")
        self.input_file = ""
        self.original = None
        self.pixelated = None
        
        # Retro palettes
        self.retro_palettes = {
            "None (Original Colors)": None,
            "Game Boy (4 colors)": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
            "NES (54 colors)": ["#7C7C7C", "#0000FC", "#0000BC", "#4428BC", "#940084", "#A80020", "#A81000", "#881400",
                                "#503000", "#007800", "#006800", "#005800", "#004058", "#000000"],
            "SNES (16 colors)": ["#000000", "#1D2B53", "#7E2553", "#008751", "#AB5236", "#5F574F", "#C2C3C7",
                                 "#FFF1E8", "#FF004D", "#FFA300", "#FFEC27", "#00E436", "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"],
            "Sega Genesis": ["#000000", "#1C1C1C", "#383838", "#545454", "#707070", "#8C8C8C", "#A8A8A8", "#C4C4C4",
                             "#E0E0E0", "#FCFCFC"],
            "Commodore 64": ["#000000", "#FFFFFF", "#880000", "#AAFFEE", "#CC44CC", "#00CC55", "#0000AA", "#EEEE77",
                             "#DD8855", "#664400", "#FF7777", "#333333", "#777777", "#AAFF66", "#0088FF", "#BBBBBB"],
            "CGA (4 colors)": ["#000000", "#00AA00", "#AA0000", "#AAAA00", "#0000AA", "#AA00AA", "#0055AA", "#AAAAAA"],
            "Pico-8": ["#000000", "#1D2B53", "#7E2553", "#008751", "#AB5236", "#5F574F", "#C2C3C7", "#FFF1E8",
                       "#FF004D", "#FFA300", "#FFEC27", "#00E436", "#29ADFF", "#83769C", "#FF77A8", "#FFCCAA"]
        }
        
        # Layout
        main_layout = QHBoxLayout()
        
        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        
        # File input
        self.input_btn = QPushButton("📁 Load Image")
        self.input_btn.clicked.connect(self.load_image)
        left_layout.addWidget(self.input_btn)
        
        self.input_label = QLabel("No file loaded")
        self.input_label.setWordWrap(True)
        left_layout.addWidget(self.input_label)
        
        # Pixel Size
        pixel_group = QGroupBox("Pixel Size")
        pixel_layout = QVBoxLayout()
        self.pixel_size_combo = NoWheelComboBox()
        self.pixel_size_combo.addItems(["2px", "4px", "8px", "16px", "32px", "Custom"])
        self.pixel_size_combo.setCurrentText("8px")
        pixel_layout.addWidget(self.pixel_size_combo)
        
        self.custom_pixel_spin = QSpinBox()
        self.custom_pixel_spin.setRange(1, 128)
        self.custom_pixel_spin.setValue(8)
        self.custom_pixel_spin.setPrefix("Custom: ")
        self.custom_pixel_spin.setSuffix("px")
        self.custom_pixel_spin.setEnabled(False)
        pixel_layout.addWidget(self.custom_pixel_spin)
        pixel_group.setLayout(pixel_layout)
        left_layout.addWidget(pixel_group)
        
        # Color Reduction
        color_group = QGroupBox("Color Reduction")
        color_layout = QVBoxLayout()
        self.color_count_combo = NoWheelComboBox()
        self.color_count_combo.addItems(["4 colors", "8 colors", "16 colors", "32 colors", "64 colors", "Custom"])
        self.color_count_combo.setCurrentText("16 colors")
        color_layout.addWidget(self.color_count_combo)
        
        self.custom_color_spin = QSpinBox()
        self.custom_color_spin.setRange(2, 256)
        self.custom_color_spin.setValue(16)
        self.custom_color_spin.setPrefix("Colors: ")
        self.custom_color_spin.setEnabled(False)
        color_layout.addWidget(self.custom_color_spin)
        color_group.setLayout(color_layout)
        left_layout.addWidget(color_group)
        
        # Dithering
        dither_group = QGroupBox("Dithering")
        dither_layout = QVBoxLayout()
        self.dither_combo = NoWheelComboBox()
        self.dither_combo.addItems(["None (Clean)", "Floyd-Steinberg", "Ordered (Bayer)"])
        dither_layout.addWidget(self.dither_combo)
        dither_group.setLayout(dither_layout)
        left_layout.addWidget(dither_group)
        
        # Retro Palette
        palette_group = QGroupBox("Retro Palette")
        palette_layout = QVBoxLayout()
        self.palette_combo = NoWheelComboBox()
        self.palette_combo.addItems(list(self.retro_palettes.keys()))
        palette_layout.addWidget(self.palette_combo)
        palette_group.setLayout(palette_layout)
        left_layout.addWidget(palette_group)
        
        # Output Scaling
        scale_group = QGroupBox("Output Scaling")
        scale_layout = QVBoxLayout()
        self.scale_combo = NoWheelComboBox()
        self.scale_combo.addItems(["Keep Small", "2x Scale Up", "4x Scale Up", "8x Scale Up", "Original Size"])
        self.scale_combo.setCurrentText("4x Scale Up")
        scale_layout.addWidget(self.scale_combo)
        
        self.grid_overlay_check = QCheckBox("Show Grid Overlay")
        scale_layout.addWidget(self.grid_overlay_check)
        scale_group.setLayout(scale_layout)
        left_layout.addWidget(scale_group)
        
        # Action buttons
        self.preview_btn = QPushButton("🎨 Generate Preview")
        self.preview_btn.clicked.connect(self.generate_pixel_art)
        left_layout.addWidget(self.preview_btn)
        
        self.save_btn = QPushButton("💾 Save Pixel Art")
        self.save_btn.clicked.connect(self.save_pixel_art)
        self.save_btn.setEnabled(False)
        left_layout.addWidget(self.save_btn)
        
        left_layout.addStretch()
        left.setFixedWidth(320)
        
        # RIGHT: Preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.preview_label = QLabel("Load an image to begin")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 400)
        self.preview_label.setStyleSheet("border: 2px dashed #555; border-radius: 8px;")
        
        scroll = QScrollArea()
        scroll.setWidget(self.preview_label)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)
        
        main_layout.addWidget(left)
        main_layout.addWidget(right, 1)
        self.card_layout.addLayout(main_layout)
        
        # Signals
        self.pixel_size_combo.currentTextChanged.connect(self.toggle_custom_pixel)
        self.color_count_combo.currentTextChanged.connect(self.toggle_custom_color)
        
    def toggle_custom_pixel(self, text):
        self.custom_pixel_spin.setEnabled(text == "Custom")
        
    def toggle_custom_color(self, text):
        self.custom_color_spin.setEnabled(text == "Custom")
        
    def load_image(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("PixelArtPage_last_input_folder", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", last_folder, "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if path:
            self.input_file = path
            settings.setValue("PixelArtPage_last_input_folder", os.path.dirname(path))
            self.input_label.setText(os.path.basename(path))
            self.original = Image.open(path).convert("RGB")
            self.preview_label.setText("Image loaded. Click 'Generate Preview'")
            
    def get_pixel_size(self):
        text = self.pixel_size_combo.currentText()
        if text == "Custom":
            return self.custom_pixel_spin.value()
        return int(text.replace("px", ""))
        
    def get_color_count(self):
        text = self.color_count_combo.currentText()
        if text == "Custom":
            return self.custom_color_spin.value()
        return int(text.split()[0])
        
    def generate_pixel_art(self):
        if not self.original:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        pixel_size = self.get_pixel_size()
        color_count = self.get_color_count()
        dither_mode = self.dither_combo.currentText()
        palette_name = self.palette_combo.currentText()
        
        # Step 1: Downscale
        small_w = max(1, self.original.width // pixel_size)
        small_h = max(1, self.original.height // pixel_size)
        small = self.original.resize((small_w, small_h), Image.NEAREST)
        
        # Step 2: Apply retro palette if selected
        if self.retro_palettes[palette_name]:
            small = self.apply_retro_palette(small, self.retro_palettes[palette_name], dither_mode)
        else:
            # Step 3: Color quantization
            if dither_mode == "Floyd-Steinberg":
                small = small.quantize(colors=color_count, method=2, dither=1)
            elif dither_mode == "Ordered (Bayer)":
                small = self.apply_bayer_dithering(small, color_count)
            else:
                small = small.quantize(colors=color_count, method=2, dither=0)
            small = small.convert("RGB")
        
        # Step 4: Scale back up
        scale_text = self.scale_combo.currentText()
        if scale_text == "Keep Small":
            self.pixelated = small
        elif scale_text == "Original Size":
            self.pixelated = small.resize(self.original.size, Image.NEAREST)
        else:
            scale_factor = int(scale_text.split("x")[0])
            new_w = small.width * scale_factor
            new_h = small.height * scale_factor
            self.pixelated = small.resize((new_w, new_h), Image.NEAREST)
            
        # Step 5: Add grid overlay if enabled
        if self.grid_overlay_check.isChecked():
            self.pixelated = self.add_grid_overlay(self.pixelated, pixel_size)
            
        # Display preview
        self.display_preview()
        self.save_btn.setEnabled(True)
        
    def apply_retro_palette(self, img, palette_hex, dither_mode):
        """Map image to a specific retro palette"""
        # Convert hex palette to RGB
        palette_rgb = [self.hex_to_rgb(h) for h in palette_hex]
        
        arr = np.array(img)
        h, w, _ = arr.shape
        
        if dither_mode == "Floyd-Steinberg":
            # Floyd-Steinberg dithering
            arr = arr.astype(np.float32)
            for y in range(h):
                for x in range(w):
                    old_pixel = arr[y, x]
                    new_pixel = self.nearest_color(old_pixel, palette_rgb)
                    arr[y, x] = new_pixel
                    error = old_pixel - new_pixel
                    
                    if x + 1 < w:
                        arr[y, x + 1] += error * 7/16
                    if y + 1 < h:
                        if x > 0:
                            arr[y + 1, x - 1] += error * 3/16
                        arr[y + 1, x] += error * 5/16
                        if x + 1 < w:
                            arr[y + 1, x + 1] += error * 1/16
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            # No dithering - just map to nearest color
            for y in range(h):
                for x in range(w):
                    arr[y, x] = self.nearest_color(arr[y, x], palette_rgb)
                    
        return Image.fromarray(arr)
        
    def hex_to_rgb(self, hex_str):
        hex_str = hex_str.lstrip("#")
        return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)])
        
    def nearest_color(self, pixel, palette):
        """Find nearest color in palette"""
        distances = [np.sum((pixel - color)**2) for color in palette]
        return palette[np.argmin(distances)]
        
    def apply_bayer_dithering(self, img, colors):
        """Apply ordered Bayer dithering"""
        # 4x4 Bayer matrix
        bayer = np.array([
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5]
        ]) / 16.0 - 0.5
        
        arr = np.array(img).astype(np.float32)
        h, w, c = arr.shape
        
        for y in range(h):
            for x in range(w):
                threshold = bayer[y % 4, x % 4] * 32
                arr[y, x] += threshold
                
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        result = Image.fromarray(arr)
        return result.quantize(colors=colors, method=2, dither=0).convert("RGB")
        
    def add_grid_overlay(self, img, pixel_size):
        """Add grid lines to show pixel boundaries"""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        w, h = img.size
        
        # Determine grid spacing (after scaling)
        scale_text = self.scale_combo.currentText()
        if scale_text == "Keep Small":
            spacing = 1
        elif scale_text == "Original Size":
            spacing = img.width // (self.original.width // pixel_size)
        else:
            scale_factor = int(scale_text.split("x")[0])
            spacing = scale_factor
            
        # Draw vertical lines
        for x in range(0, w, spacing):
            draw.line([(x, 0), (x, h)], fill=(100, 100, 100), width=1)
            
        # Draw horizontal lines
        for y in range(0, h, spacing):
            draw.line([(0, y), (w, y)], fill=(100, 100, 100), width=1)
            
        return img
        
    def display_preview(self):
        if self.pixelated:
            # Limit display size for very large images
            display = self.pixelated.copy()
            max_display = 800
            if display.width > max_display or display.height > max_display:
                display.thumbnail((max_display, max_display), Image.NEAREST)
                
            qimg = QImage(display.tobytes(), display.width, display.height, 
                          display.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setStyleSheet("")
            
    def save_pixel_art(self):
        if not self.pixelated:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Pixel Art", "", 
                                               "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
        if path:
            # Optimize based on format
            if path.lower().endswith('.png'):
                self.pixelated.save(path, optimize=True)
            elif path.lower().endswith(('.jpg', '.jpeg')):
                self.pixelated.save(path, quality=95, optimize=True)
            else:
                self.pixelated.save(path)
            QMessageBox.information(self, "Saved", f"Pixel art saved to:\n{path}")


# ===================================================================
# POWER-OF-TWO CONVERTER + DDS EXPORT
# ===================================================================

class PowerOfTwoPage(CardPage):
    def __init__(self):
        super().__init__("Power-of-Two + DDS Export")
        self.input_file = ""
        self.original = None
        self.converted = None
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        
        # File input
        self.input_btn = QPushButton("📁 Load Image")
        self.input_btn.clicked.connect(self.load_image)
        left_layout.addWidget(self.input_btn)
        
        self.input_label = QLabel("No file loaded")
        self.input_label.setWordWrap(True)
        left_layout.addWidget(self.input_label)
        
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        left_layout.addWidget(self.info_label)
        
        # Conversion Mode
        mode_group = QGroupBox("Conversion Mode")
        mode_layout = QVBoxLayout()
        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItems([
            "Nearest Power of Two (Scale)",
            "Next Power of Two (Scale)",
            "Force Exact Size",
            "Expand Canvas (Pad)",
            "Nearest Square"
        ])
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)
        left_layout.addWidget(mode_group)
        
        # Exact size controls (for Force Exact Size mode)
        size_group = QGroupBox("Target Size")
        size_layout = QVBoxLayout()
        self.size_combo = NoWheelComboBox()
        sizes = ["128", "256", "512", "1024", "2048", "4096", "Custom"]
        self.size_combo.addItems(sizes)
        self.size_combo.setCurrentText("512")
        self.size_combo.setEnabled(False)
        size_layout.addWidget(self.size_combo)
        
        custom_layout = QHBoxLayout()
        self.custom_width_spin = QSpinBox()
        self.custom_width_spin.setRange(1, 8192)
        self.custom_width_spin.setValue(512)
        self.custom_width_spin.setPrefix("W: ")
        self.custom_height_spin = QSpinBox()
        self.custom_height_spin.setRange(1, 8192)
        self.custom_height_spin.setValue(512)
        self.custom_height_spin.setPrefix("H: ")
        custom_layout.addWidget(self.custom_width_spin)
        custom_layout.addWidget(self.custom_height_spin)
        size_layout.addLayout(custom_layout)
        size_group.setLayout(size_layout)
        left_layout.addWidget(size_group)
        
        # Padding options (for Expand Canvas mode)
        pad_group = QGroupBox("Padding Options")
        pad_layout = QVBoxLayout()
        self.pad_combo = NoWheelComboBox()
        self.pad_combo.addItems(["Transparent", "Black", "White", "Custom Color", "Edge Extend"])
        pad_layout.addWidget(self.pad_combo)
        
        self.pad_color_btn = QPushButton("Pick Padding Color")
        self.pad_color_btn.clicked.connect(self.pick_pad_color)
        self.pad_color_btn.setEnabled(False)
        pad_layout.addWidget(self.pad_color_btn)
        pad_group.setLayout(pad_layout)
        left_layout.addWidget(pad_group)
        
        # DDS Export Options
        dds_group = QGroupBox("DDS Export Options")
        dds_layout = QVBoxLayout()
        
        self.compression_combo = NoWheelComboBox()
        self.compression_combo.addItems(["DXT1 (No Alpha)", "DXT3 (Sharp Alpha)", "DXT5 (Smooth Alpha)", 
                                         "BC7 (Best Quality)", "Uncompressed"])
        dds_layout.addWidget(QLabel("Compression:"))
        dds_layout.addWidget(self.compression_combo)
        
        self.mipmap_check = QCheckBox("Generate Mipmaps")
        self.mipmap_check.setChecked(True)
        dds_layout.addWidget(self.mipmap_check)
        
        self.normalmap_check = QCheckBox("Normal Map Mode")
        dds_layout.addWidget(self.normalmap_check)
        
        self.flip_y_check = QCheckBox("Flip Y Channel (OpenGL)")
        self.flip_y_check.setEnabled(False)
        dds_layout.addWidget(self.flip_y_check)
        
        dds_group.setLayout(dds_layout)
        left_layout.addWidget(dds_group)
        
        # Action buttons
        self.convert_btn = QPushButton("🔄 Convert")
        self.convert_btn.clicked.connect(self.convert_image)
        left_layout.addWidget(self.convert_btn)
        
        save_layout = QHBoxLayout()
        self.save_png_btn = QPushButton("💾 Save PNG")
        self.save_png_btn.clicked.connect(self.save_png)
        self.save_png_btn.setEnabled(False)
        save_layout.addWidget(self.save_png_btn)
        
        self.save_dds_btn = QPushButton("💾 Save DDS")
        self.save_dds_btn.clicked.connect(self.save_dds)
        self.save_dds_btn.setEnabled(False)
        save_layout.addWidget(self.save_dds_btn)
        left_layout.addLayout(save_layout)
        
        left_layout.addStretch()
        left.setFixedWidth(340)
        
        # RIGHT: Preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.preview_label = QLabel("Load an image to begin")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 400)
        self.preview_label.setStyleSheet("border: 2px dashed #555; border-radius: 8px;")
        
        scroll = QScrollArea()
        scroll.setWidget(self.preview_label)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)
        
        main_layout.addWidget(left)
        main_layout.addWidget(right, 1)
        self.card_layout.addLayout(main_layout)
        
        # Signals
        self.mode_combo.currentTextChanged.connect(self.update_mode_controls)
        self.pad_combo.currentTextChanged.connect(self.update_pad_controls)
        self.normalmap_check.toggled.connect(self.flip_y_check.setEnabled)
        self.size_combo.currentTextChanged.connect(self.update_custom_size_fields)
        
        self.pad_color = (0, 0, 0)
        
    def update_mode_controls(self, mode):
        is_force_exact = mode == "Force Exact Size"
        self.size_combo.setEnabled(is_force_exact)
        self.custom_width_spin.setEnabled(is_force_exact and self.size_combo.currentText() == "Custom")
        self.custom_height_spin.setEnabled(is_force_exact and self.size_combo.currentText() == "Custom")
        
    def update_custom_size_fields(self, text):
        is_custom = text == "Custom"
        self.custom_width_spin.setEnabled(is_custom and self.mode_combo.currentText() == "Force Exact Size")
        self.custom_height_spin.setEnabled(is_custom and self.mode_combo.currentText() == "Force Exact Size")
        if not is_custom and text:
            size = int(text)
            self.custom_width_spin.setValue(size)
            self.custom_height_spin.setValue(size)
            
    def update_pad_controls(self, mode):
        self.pad_color_btn.setEnabled(mode == "Custom Color")
        
    def pick_pad_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.pad_color = (color.red(), color.green(), color.blue())
            
    def load_image(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("PowerOfTwoPage_last_input_folder", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", last_folder, "Images (*.png *.jpg *.jpeg *.bmp *.tga)")
        if path:
            self.input_file = path
            settings.setValue("PowerOfTwoPage_last_input_folder", os.path.dirname(path))
            self.input_label.setText(os.path.basename(path))
            self.original = Image.open(path)
            if self.original.mode not in ("RGB", "RGBA"):
                self.original = self.original.convert("RGBA")
            
            w, h = self.original.size
            self.info_label.setText(f"Current: {w}x{h}\nNearest PoT: {self.nearest_pot(w)}x{self.nearest_pot(h)}")
            
    def nearest_pot(self, n):
        """Find nearest power of two"""
        if n <= 0:
            return 1
        power = round(np.log2(n))
        return 2 ** power
        
    def next_pot(self, n):
        """Find next power of two"""
        if n <= 0:
            return 1
        power = np.ceil(np.log2(n))
        return int(2 ** power)
        
    def convert_image(self):
        if not self.original:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        mode = self.mode_combo.currentText()
        w, h = self.original.size
        
        if mode == "Nearest Power of Two (Scale)":
            new_w = self.nearest_pot(w)
            new_h = self.nearest_pot(h)
            self.converted = self.original.resize((new_w, new_h), Image.LANCZOS)
            
        elif mode == "Next Power of Two (Scale)":
            new_w = self.next_pot(w)
            new_h = self.next_pot(h)
            self.converted = self.original.resize((new_w, new_h), Image.LANCZOS)
            
        elif mode == "Force Exact Size":
            if self.size_combo.currentText() == "Custom":
                new_w = self.custom_width_spin.value()
                new_h = self.custom_height_spin.value()
            else:
                size = int(self.size_combo.currentText())
                new_w = new_h = size
            self.converted = self.original.resize((new_w, new_h), Image.LANCZOS)
            
        elif mode == "Expand Canvas (Pad)":
            new_w = self.next_pot(w)
            new_h = self.next_pot(h)
            self.converted = self.create_padded_image(self.original, new_w, new_h)
            
        elif mode == "Nearest Square":
            size = self.nearest_pot(max(w, h))
            self.converted = self.original.resize((size, size), Image.Resampling.LANCZOS)
            
        self.display_preview()
        self.save_png_btn.setEnabled(True)
        self.save_dds_btn.setEnabled(True)
        
    def create_padded_image(self, img, target_w, target_h):
        """Create padded image based on padding mode"""
        pad_mode = self.pad_combo.currentText()
        
        if pad_mode == "Transparent":
            new_img = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
        elif pad_mode == "Black":
            new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        elif pad_mode == "White":
            new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        elif pad_mode == "Custom Color":
            new_img = Image.new("RGB", (target_w, target_h), self.pad_color)
        elif pad_mode == "Edge Extend":
            # Create expanded image by extending edges
            new_img = img.resize((target_w, target_h), Image.NEAREST)
            new_img.paste(img, ((target_w - img.width) // 2, (target_h - img.height) // 2))
            return new_img
            
        # Center paste
        paste_x = (target_w - img.width) // 2
        paste_y = (target_h - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
        
    def display_preview(self):
        if self.converted:
            display = self.converted.copy()
            max_display = 800
            if display.width > max_display or display.height > max_display:
                display.thumbnail((max_display, max_display), Image.LANCZOS)
                
            if display.mode == "RGBA":
                # Show alpha as checkerboard
                checker = self.create_checkerboard(display.width, display.height)
                checker.paste(display, (0, 0), display)
                display = checker
                
            display = display.convert("RGB")
            qimg = QImage(display.tobytes(), display.width, display.height, 
                          display.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setStyleSheet("")
            
            w, h = self.converted.size
            self.info_label.setText(f"Converted to: {w}x{h}")
            
    def create_checkerboard(self, w, h, square_size=10):
        """Create checkerboard pattern for alpha preview"""
        checker = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(checker)
        for y in range(0, h, square_size):
            for x in range(0, w, square_size):
                if (x // square_size + y // square_size) % 2:
                    draw.rectangle([x, y, x + square_size, y + square_size], fill=(200, 200, 200))
        return checker
        
    def save_png(self):
        if not self.converted:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "", "PNG (*.png)")
        if path:
            self.converted.save(path, "PNG")
            QMessageBox.information(self, "Saved", f"Image saved to:\n{path}")
            
    def save_dds(self):
        if not self.converted:
            return
            
        # Note: DDS saving requires special library. For now, show message.
        QMessageBox.information(self, "DDS Export", 
                                "DDS export requires 'nvidia-texture-tools' or similar.\n\n"
                                "For now, save as PNG and convert using:\n"
                                "- NVIDIA Texture Tools\n"
                                "- Compressonator\n"
                                "- texconv.exe (DirectXTex)\n\n"
                                "Full DDS support coming soon!")


# ===================================================================
# IMAGE GRID COMPOSER - Create image grids/collages
# ===================================================================

class ImageGridPage(CardPage):
    def __init__(self):
        super().__init__("Image Grid Composer")
        self.images = []
        self.grid_result = None
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        
        # Add images
        self.add_btn = QPushButton("➕ Add Images")
        self.add_btn.clicked.connect(self.add_images)
        left_layout.addWidget(self.add_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear All")
        self.clear_btn.clicked.connect(self.clear_images)
        left_layout.addWidget(self.clear_btn)
        
        self.image_list = QLabel("No images added")
        self.image_list.setWordWrap(True)
        left_layout.addWidget(self.image_list)
        
        # Layout options
        layout_group = QGroupBox("Grid Layout")
        layout_layout = QVBoxLayout()
        
        self.layout_combo = NoWheelComboBox()
        self.layout_combo.addItems(["2x2", "3x3", "4x4", "2x3", "3x2", "4x2", "Auto Grid", "Custom"])
        layout_layout.addWidget(self.layout_combo)
        
        custom_layout = QHBoxLayout()
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(2)
        self.rows_spin.setPrefix("Rows: ")
        self.rows_spin.setEnabled(False)
        
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.setValue(2)
        self.cols_spin.setPrefix("Cols: ")
        self.cols_spin.setEnabled(False)
        
        custom_layout.addWidget(self.rows_spin)
        custom_layout.addWidget(self.cols_spin)
        layout_layout.addLayout(custom_layout)
        layout_group.setLayout(layout_layout)
        left_layout.addWidget(layout_group)
        
        # Fit mode
        fit_group = QGroupBox("Image Fit Mode")
        fit_layout = QVBoxLayout()
        self.fit_combo = NoWheelComboBox()
        self.fit_combo.addItems(["Crop to Fill", "Fit with Padding", "Stretch to Fill"])
        fit_layout.addWidget(self.fit_combo)
        fit_group.setLayout(fit_layout)
        left_layout.addWidget(fit_group)
        
        # Styling
        style_group = QGroupBox("Styling")
        style_layout = QVBoxLayout()
        
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Spacing:"))
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(0, 100)
        self.spacing_spin.setValue(10)
        self.spacing_spin.setSuffix("px")
        spacing_layout.addWidget(self.spacing_spin)
        style_layout.addLayout(spacing_layout)
        
        self.bg_color_btn = QPushButton("Background Color")
        self.bg_color_btn.clicked.connect(self.pick_bg_color)
        style_layout.addWidget(self.bg_color_btn)
        
        self.rounded_check = QCheckBox("Rounded Corners")
        style_layout.addWidget(self.rounded_check)
        
        self.shadow_check = QCheckBox("Drop Shadow")
        style_layout.addWidget(self.shadow_check)
        
        self.border_check = QCheckBox("Outer Border")
        style_layout.addWidget(self.border_check)
        
        self.filename_check = QCheckBox("Show Filenames")
        style_layout.addWidget(self.filename_check)
        
        style_group.setLayout(style_layout)
        left_layout.addWidget(style_group)
        
        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("Optional grid title")
        title_layout.addWidget(self.title_input)
        left_layout.addLayout(title_layout)
        
        # Action buttons
        self.generate_btn = QPushButton("🎨 Generate Grid")
        self.generate_btn.clicked.connect(self.generate_grid)
        left_layout.addWidget(self.generate_btn)
        
        self.save_btn = QPushButton("💾 Save Grid")
        self.save_btn.clicked.connect(self.save_grid)
        self.save_btn.setEnabled(False)
        left_layout.addWidget(self.save_btn)
        
        left_layout.addStretch()
        left.setFixedWidth(340)
        
        # RIGHT: Preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        self.preview_label = QLabel("Add images to begin")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(600, 400)
        self.preview_label.setStyleSheet("border: 2px dashed #555; border-radius: 8px;")
        
        scroll = QScrollArea()
        scroll.setWidget(self.preview_label)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)
        
        main_layout.addWidget(left)
        main_layout.addWidget(right, 1)
        self.card_layout.addLayout(main_layout)
        
        # Signals
        self.layout_combo.currentTextChanged.connect(self.toggle_custom_layout)
        
        self.bg_color = (255, 255, 255)
        
    def toggle_custom_layout(self, text):
        is_custom = text == "Custom"
        self.rows_spin.setEnabled(is_custom)
        self.cols_spin.setEnabled(is_custom)
        
    def add_images(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("ImageGridPage_last_input_folder", "")
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", last_folder, 
                                                 "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if paths:
            settings.setValue("ImageGridPage_last_input_folder", os.path.dirname(paths[0]))
            for path in paths:
                try:
                    img = Image.open(path)
                    self.images.append({"path": path, "image": img, "name": os.path.basename(path)})
                except:
                    pass
            self.update_image_list()
            
    def clear_images(self):
        self.images = []
        self.update_image_list()
        
    def update_image_list(self):
        if self.images:
            text = f"{len(self.images)} images loaded:\n"
            for img in self.images[:5]:
                text += f"• {img['name']}\n"
            if len(self.images) > 5:
                text += f"... and {len(self.images) - 5} more"
            self.image_list.setText(text)
        else:
            self.image_list.setText("No images added")
            
    def pick_bg_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.bg_color = (color.red(), color.green(), color.blue())
            
    def generate_grid(self):
        if not self.images:
            QMessageBox.warning(self, "No Images", "Please add images first.")
            return
            
        layout_text = self.layout_combo.currentText()
        
        if layout_text == "Auto Grid":
            # Calculate optimal grid
            count = len(self.images)
            cols = int(np.ceil(np.sqrt(count)))
            rows = int(np.ceil(count / cols))
        elif layout_text == "Custom":
            rows = self.rows_spin.value()
            cols = self.cols_spin.value()
        else:
            parts = layout_text.split("x")
            rows = int(parts[0])
            cols = int(parts[1])
            
        # Cell size (use first image as reference, or fixed size)
        cell_w = 400
        cell_h = 400
        
        spacing = self.spacing_spin.value()
        fit_mode = self.fit_combo.currentText()
        
        # Calculate grid dimensions
        grid_w = cols * cell_w + (cols + 1) * spacing
        grid_h = rows * cell_h + (rows + 1) * spacing
        
        # Add space for title
        title_height = 80 if self.title_input.text() else 0
        grid_h += title_height
        
        # Create base image
        grid_img = Image.new("RGB", (grid_w, grid_h), self.bg_color)
        draw = ImageDraw.Draw(grid_img)
        
        # Draw title
        if self.title_input.text():
            try:
                font = ImageFont.truetype("arial.ttf", 48)
            except:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), self.title_input.text(), font=font)
            text_w = bbox[2] - bbox[0]
            draw.text(((grid_w - text_w) // 2, 20), self.title_input.text(), fill=(0, 0, 0), font=font)
        
        # Place images
        for idx, img_data in enumerate(self.images):
            if idx >= rows * cols:
                break
                
            row = idx // cols
            col = idx % cols
            
            x = col * cell_w + (col + 1) * spacing
            y = row * cell_h + (row + 1) * spacing + title_height
            
            img = img_data["image"].copy()
            
            # Fit image to cell
            if fit_mode == "Crop to Fill":
                img = self.crop_to_fill(img, cell_w, cell_h)
            elif fit_mode == "Fit with Padding":
                img = self.fit_with_padding(img, cell_w, cell_h, self.bg_color)
            else:  # Stretch
                img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                
            # Apply effects
            if self.rounded_check.isChecked():
                img = self.round_corners(img, 20)
                
            if self.shadow_check.isChecked():
                # Simple shadow effect - paste on offset
                shadow = Image.new("RGBA", (cell_w + 10, cell_h + 10), (0, 0, 0, 100))
                grid_img.paste(shadow, (x + 5, y + 5), shadow)
                
            # Paste image
            if img.mode == "RGBA":
                grid_img.paste(img, (x, y), img)
            else:
                grid_img.paste(img, (x, y))
                
            # Draw filename
            if self.filename_check.isChecked():
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                name = img_data["name"][:30]
                draw.text((x + 5, y + cell_h - 25), name, fill=(255, 255, 255), font=font)
                
        # Add outer border
        if self.border_check.isChecked():
            draw.rectangle([0, 0, grid_w-1, grid_h-1], outline=(0, 0, 0), width=3)
            
        self.grid_result = grid_img
        self.display_preview()
        self.save_btn.setEnabled(True)
        
    def crop_to_fill(self, img, w, h):
        """Crop image to fill dimensions"""
        aspect = w / h
        img_aspect = img.width / img.height
        
        if img_aspect > aspect:
            # Image is wider
            new_h = img.height
            new_w = int(new_h * aspect)
            left = (img.width - new_w) // 2
            img = img.crop((left, 0, left + new_w, new_h))
        else:
            # Image is taller
            new_w = img.width
            new_h = int(new_w / aspect)
            top = (img.height - new_h) // 2
            img = img.crop((0, top, new_w, top + new_h))
            
        return img.resize((w, h), Image.Resampling.LANCZOS)
        
    def fit_with_padding(self, img, w, h, bg_color):
        """Fit image with padding"""
        img.thumbnail((w, h), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (w, h), bg_color)
        paste_x = (w - img.width) // 2
        paste_y = (h - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img
        
    def round_corners(self, img, radius):
        """Add rounded corners to image"""
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([0, 0, img.width, img.height], radius, fill=255)
        img.putalpha(mask)
        return img
        
    def display_preview(self):
        if self.grid_result:
            display = self.grid_result.copy()
            max_display = 900
            if display.width > max_display or display.height > max_display:
                display.thumbnail((max_display, max_display), Image.LANCZOS)
                
            display = display.convert("RGB")
            qimg = QImage(display.tobytes(), display.width, display.height, 
                          display.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setStyleSheet("")
            
    def save_grid(self):
        if not self.grid_result:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Grid", "", 
                                               "PNG (*.png);;JPEG (*.jpg)")
        if path:
            # Optimize based on format
            if path.lower().endswith('.png'):
                self.grid_result.save(path, optimize=True)
            elif path.lower().endswith(('.jpg', '.jpeg')):
                self.grid_result.save(path, quality=95, optimize=True)
            else:
                self.grid_result.save(path)
            QMessageBox.information(self, "Saved", f"Grid saved to:\n{path}")


# ===================================================================
# BATCH BORDER DESIGNER - Add borders to images
# ===================================================================

class BatchBorderPage(CardPage):
    def __init__(self):
        super().__init__("Batch Border Designer")
        self.input_files = []
        self.output_dir = ""
        self.preview_image = None
        
        # Layout
        split = QHBoxLayout()
        
        # LEFT: Controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        
        # File selection
        self.select_btn = QPushButton("📁 Select Images")
        self.select_btn.clicked.connect(self.select_images)
        left_layout.addWidget(self.select_btn)
        
        self.preview_btn = QPushButton("🖼️ Load Preview Image")
        self.preview_btn.clicked.connect(self.load_preview)
        left_layout.addWidget(self.preview_btn)
        
        self.output_btn = QPushButton("📂 Select Output Folder")
        self.output_btn.clicked.connect(self.select_output)
        left_layout.addWidget(self.output_btn)
        
        self.file_count_label = QLabel("No files selected")
        left_layout.addWidget(self.file_count_label)
        
        # Border Type
        type_group = QGroupBox("Border Type")
        type_layout = QVBoxLayout()
        self.border_type_combo = NoWheelComboBox()
        self.border_type_combo.addItems(["Solid Color", "Gradient", "Double Border", "Per-Side Custom"])
        self.border_type_combo.currentTextChanged.connect(self.update_preview)
        type_layout.addWidget(self.border_type_combo)
        type_group.setLayout(type_layout)
        left_layout.addWidget(type_group)
        
        # Thickness
        thickness_group = QGroupBox("Border Thickness")
        thickness_layout = QVBoxLayout()
        
        uniform_layout = QHBoxLayout()
        uniform_layout.addWidget(QLabel("Uniform:"))
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 500)
        self.thickness_spin.setValue(20)
        self.thickness_spin.setSuffix("px")
        self.thickness_spin.valueChanged.connect(self.update_preview)
        uniform_layout.addWidget(self.thickness_spin)
        thickness_layout.addLayout(uniform_layout)
        
        # Per-side controls
        self.top_spin = QSpinBox()
        self.top_spin.setRange(0, 500)
        self.top_spin.setValue(20)
        self.top_spin.setPrefix("Top: ")
        self.top_spin.setSuffix("px")
        self.top_spin.setEnabled(False)
        
        self.bottom_spin = QSpinBox()
        self.bottom_spin.setRange(0, 500)
        self.bottom_spin.setValue(20)
        self.bottom_spin.setPrefix("Bottom: ")
        self.bottom_spin.setSuffix("px")
        self.bottom_spin.setEnabled(False)
        
        self.left_spin = QSpinBox()
        self.left_spin.setRange(0, 500)
        self.left_spin.setValue(20)
        self.left_spin.setPrefix("Left: ")
        self.left_spin.setSuffix("px")
        self.left_spin.setEnabled(False)
        
        self.right_spin = QSpinBox()
        self.right_spin.setRange(0, 500)
        self.right_spin.setValue(20)
        self.right_spin.setPrefix("Right: ")
        self.right_spin.setSuffix("px")
        self.right_spin.setEnabled(False)
        
        thickness_layout.addWidget(self.top_spin)
        thickness_layout.addWidget(self.bottom_spin)
        thickness_layout.addWidget(self.left_spin)
        thickness_layout.addWidget(self.right_spin)
        thickness_group.setLayout(thickness_layout)
        left_layout.addWidget(thickness_group)
        
        # Colors
        color_group = QGroupBox("Colors")
        color_layout = QVBoxLayout()
        
        self.color1_btn = QPushButton("Primary Color")
        self.color1_btn.clicked.connect(lambda: self.pick_color(1))
        self.color1_btn.setStyleSheet("background-color: rgb(0,0,0); color: white;")
        color_layout.addWidget(self.color1_btn)
        
        self.color2_btn = QPushButton("Secondary Color (Gradient/Double)")
        self.color2_btn.clicked.connect(lambda: self.pick_color(2))
        self.color2_btn.setStyleSheet("background-color: rgb(255,255,255); color: black;")
        color_layout.addWidget(self.color2_btn)
        
        color_group.setLayout(color_layout)
        left_layout.addWidget(color_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        self.rounded_check = QCheckBox("Rounded Corners")
        options_layout.addWidget(self.rounded_check)
        
        self.inner_stroke_check = QCheckBox("Inner Stroke")
        options_layout.addWidget(self.inner_stroke_check)
        
        self.expand_canvas_check = QCheckBox("Expand Canvas (Don't Shrink Image)")
        self.expand_canvas_check.setChecked(True)
        options_layout.addWidget(self.expand_canvas_check)
        
        options_group.setLayout(options_layout)
        left_layout.addWidget(options_group)
        
        # Process button
        self.process_btn = QPushButton("🎨 Process Images")
        self.process_btn.clicked.connect(self.process_images)
        left_layout.addWidget(self.process_btn)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)
        
        left_layout.addStretch()
        left.setFixedWidth(340)
        
        # RIGHT: Preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        
        right_layout.addWidget(QLabel("Live Preview"))
        self.preview_label = QLabel("Load a preview image to see border effect")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 480)
        self.preview_label.setStyleSheet("border: 1px solid #333; border-radius: 8px; background: #0a0a0a;")
        right_layout.addWidget(self.preview_label, 1)
        
        split.addWidget(left)
        split.addWidget(right, 1)
        self.card_layout.addLayout(split)
        
        # Signals
        self.border_type_combo.currentTextChanged.connect(self.update_border_controls)
        
        self.color1 = (0, 0, 0)
        self.color2 = (255, 255, 255)
        
    def update_border_controls(self, text):
        is_per_side = text == "Per-Side Custom"
        self.top_spin.setEnabled(is_per_side)
        self.bottom_spin.setEnabled(is_per_side)
        self.left_spin.setEnabled(is_per_side)
        self.right_spin.setEnabled(is_per_side)
        self.thickness_spin.setEnabled(not is_per_side)
        
    def select_images(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("BatchBorderPage_last_input_folder", "")
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", last_folder, 
                                                 "Images (*.png *.jpg *.jpeg *.bmp)")
        if paths:
            settings.setValue("BatchBorderPage_last_input_folder", os.path.dirname(paths[0]))
            self.input_files = paths
            self.file_count_label.setText(f"{len(paths)} files selected")
            
    def select_output(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("BatchBorderPage_last_output_folder", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", last_folder)
        if folder:
            self.output_dir = folder
            settings.setValue("BatchBorderPage_last_output_folder", folder)

    def load_preview(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("BatchBorderPage_last_preview_folder", "")
        path, _ = QFileDialog.getOpenFileName(self, "Select Preview Image", last_folder, "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                settings.setValue("BatchBorderPage_last_preview_folder", os.path.dirname(path))
                self.preview_image = Image.open(path)
                if self.preview_image.mode not in ("RGB", "RGBA"):
                    self.preview_image = self.preview_image.convert("RGBA")
                self.update_preview()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load preview image:\n{e}")
    
    def update_preview(self):
        if not self.preview_image:
            return
        try:
            border_type = self.border_type_combo.currentText()
            expand = self.expand_canvas_check.isChecked()
            
            img = self.preview_image.copy()
            if border_type == "Solid Color":
                result = self.apply_solid_border(img, expand)
            elif border_type == "Gradient":
                result = self.apply_gradient_border(img, expand)
            elif border_type == "Double Border":
                result = self.apply_double_border(img, expand)
            else:
                result = self.apply_custom_border(img, expand)
            
            result.thumbnail((450, 450), Image.Resampling.LANCZOS)
            data = result.convert("RGB").tobytes()
            qimg = QImage(data, result.width, result.height, result.width * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.preview_label.setPixmap(pix)
        except Exception as e:
            self.preview_label.setText(f"Preview error: {e}")
    
    def pick_color(self, num):
        color = QColorDialog.getColor()
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            if num == 1:
                self.color1 = rgb
                self.color1_btn.setStyleSheet(f"background-color: rgb{rgb}; color: {'white' if sum(rgb) < 384 else 'black'};")
            else:
                self.color2 = rgb
                self.color2_btn.setStyleSheet(f"background-color: rgb{rgb}; color: {'white' if sum(rgb) < 384 else 'black'};")
            self.update_preview()
                
    def process_images(self):
        if not self.input_files:
            QMessageBox.warning(self, "No Files", "Please select images first.")
            return
            
        if not self.output_dir:
            QMessageBox.warning(self, "No Output", "Please select an output folder.")
            return
            
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.input_files))
        
        border_type = self.border_type_combo.currentText()
        expand = self.expand_canvas_check.isChecked()
        
        for idx, file_path in enumerate(self.input_files):
            try:
                with Image.open(file_path) as img:
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGBA")
                    else:
                        img = img.copy()  # Need a copy since we're in context manager
                    
                    # Apply border
                    if border_type == "Solid Color":
                        result = self.apply_solid_border(img, expand)
                    elif border_type == "Gradient":
                        result = self.apply_gradient_border(img, expand)
                    elif border_type == "Double Border":
                        result = self.apply_double_border(img, expand)
                    else:  # Per-Side Custom
                        result = self.apply_custom_border(img, expand)
                    
                    # Save with optimization
                    name = os.path.basename(file_path)
                    name_no_ext = os.path.splitext(name)[0]
                    output_path = os.path.join(self.output_dir, f"{name_no_ext}_bordered.png")
                    result.save(output_path, optimize=True)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
            self.progress.setValue(idx + 1)
            
        self.progress.setVisible(False)
        QMessageBox.information(self, "Complete", f"Processed {len(self.input_files)} images!")
        
    def apply_solid_border(self, img, expand):
        """Apply solid color border"""
        thickness = self.thickness_spin.value()
        
        if expand:
            new_w = img.width + thickness * 2
            new_h = img.height + thickness * 2
            new_img = Image.new("RGB", (new_w, new_h), self.color1)
            new_img.paste(img, (thickness, thickness))
            return new_img
        else:
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([i, i, img.width-1-i, img.height-1-i], outline=self.color1, width=1)
            return img
            
    def apply_gradient_border(self, img, expand):
        """Apply gradient border"""
        thickness = self.thickness_spin.value()
        
        # Create gradient
        if expand:
            new_w = img.width + thickness * 2
            new_h = img.height + thickness * 2
            new_img = Image.new("RGB", (new_w, new_h))
            
            # Simple gradient from color1 to color2
            for y in range(new_h):
                ratio = y / new_h
                r = int(self.color1[0] * (1-ratio) + self.color2[0] * ratio)
                g = int(self.color1[1] * (1-ratio) + self.color2[1] * ratio)
                b = int(self.color1[2] * (1-ratio) + self.color2[2] * ratio)
                draw = ImageDraw.Draw(new_img)
                draw.line([(0, y), (new_w, y)], fill=(r, g, b), width=1)
                
            new_img.paste(img, (thickness, thickness))
            return new_img
        else:
            return self.apply_solid_border(img, False)
            
    def apply_double_border(self, img, expand):
        """Apply double border"""
        thickness = self.thickness_spin.value()
        outer = thickness // 2
        inner = thickness - outer
        
        if expand:
            new_w = img.width + thickness * 2
            new_h = img.height + thickness * 2
            new_img = Image.new("RGB", (new_w, new_h), self.color1)
            
            # Inner rectangle
            draw = ImageDraw.Draw(new_img)
            draw.rectangle([outer, outer, new_w-outer-1, new_h-outer-1], fill=self.color2)
            
            new_img.paste(img, (thickness, thickness))
            return new_img
        else:
            return self.apply_solid_border(img, False)
            
    def apply_custom_border(self, img, expand):
        """Apply per-side custom border"""
        top = self.top_spin.value()
        bottom = self.bottom_spin.value()
        left = self.left_spin.value()
        right = self.right_spin.value()
        
        if expand:
            new_w = img.width + left + right
            new_h = img.height + top + bottom
            new_img = Image.new("RGB", (new_w, new_h), self.color1)
            new_img.paste(img, (left, top))
            return new_img
        else:
            return self.apply_solid_border(img, False)

class LightDirectionWidget(QWidget):
    directionChanged = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(150, 150)
        self._x = 0.35
        self._y = -0.25

    def get_direction(self):
        return self._x, self._y

    def set_direction(self, x, y):
        mag = math.sqrt(x * x + y * y)
        if mag > 1.0:
            x /= mag
            y /= mag
        self._x = x
        self._y = y
        self.directionChanged.emit(self._x, self._y)
        self.update()

    def _update_from_pos(self, pos):
        w = max(1, self.width())
        h = max(1, self.height())
        cx = w / 2.0
        cy = h / 2.0
        r = min(w, h) * 0.42
        x = (pos.x() - cx) / r
        y = (pos.y() - cy) / r
        self.set_direction(float(x), float(y))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._update_from_pos(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._update_from_pos(event.position())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        cx = rect.width() / 2.0
        cy = rect.height() / 2.0
        r = min(rect.width(), rect.height()) * 0.42

        painter.setPen(QPen(QColor(90, 90, 90), 2))
        painter.setBrush(QBrush(QColor(35, 35, 35)))
        painter.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        painter.setPen(QPen(QColor(120, 120, 120), 1, Qt.DashLine))
        painter.drawLine(int(cx - r), int(cy), int(cx + r), int(cy))
        painter.drawLine(int(cx), int(cy - r), int(cx), int(cy + r))

        px = cx + self._x * r
        py = cy + self._y * r
        painter.setPen(QPen(QColor(0, 255, 198), 2))
        painter.drawLine(int(cx), int(cy), int(px), int(py))
        painter.setBrush(QBrush(QColor(0, 255, 198)))
        painter.drawEllipse(int(px - 6), int(py - 6), 12, 12)


class TexturePreviewPage(CardPage):
    def __init__(self):
        super().__init__("Texture Preview")
        self.map_paths = {
            "base": "",
            "normal": "",
            "roughness": "",
            "height": "",
            "ao": "",
        }
        self.output_dir = ""
        self.preview_size = (720, 420)
        self._build_ui()

    def _build_ui(self):
        title = QLabel("Texture Preview Simulator + Material Pack Builder")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.card_layout.addWidget(title)

        layout = QHBoxLayout()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setFixedWidth(430)

        map_group = QGroupBox("Material Maps")
        map_layout = QGridLayout()

        self.base_label = QLabel("No base color selected")
        self.normal_label = QLabel("No normal map")
        self.rough_label = QLabel("No roughness map")
        self.height_label = QLabel("No height map")
        self.ao_label = QLabel("No AO map")

        self._add_map_row(map_layout, 0, "Base Color", "base", self.base_label)
        self._add_map_row(map_layout, 1, "Normal", "normal", self.normal_label)
        self._add_map_row(map_layout, 2, "Roughness", "roughness", self.rough_label)
        self._add_map_row(map_layout, 3, "Height", "height", self.height_label)
        self._add_map_row(map_layout, 4, "AO", "ao", self.ao_label)
        map_group.setLayout(map_layout)
        left_layout.addWidget(map_group)

        control_group = QGroupBox("Viewport Controls")
        control_layout = QVBoxLayout()

        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        self.view_combo = NoWheelComboBox()
        self.view_combo.addItems(["Plane", "Sphere", "Cube", "3D Card", "Cylinder"])
        self.view_combo.currentIndexChanged.connect(self.update_preview)
        view_row.addWidget(self.view_combo)
        control_layout.addLayout(view_row)

        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("Tiling (1x-10x):"))
        self.tiling_slider = NoWheelSlider(Qt.Horizontal)
        self.tiling_slider.setRange(1, 10)
        self.tiling_slider.setValue(1)
        self.tiling_slider.valueChanged.connect(self._on_slider_change)
        self.tiling_value = QLabel("1x")
        tile_row.addWidget(self.tiling_slider)
        tile_row.addWidget(self.tiling_value)
        control_layout.addLayout(tile_row)

        rough_row = QHBoxLayout()
        rough_row.addWidget(QLabel("Roughness:"))
        self.rough_slider = NoWheelSlider(Qt.Horizontal)
        self.rough_slider.setRange(0, 100)
        self.rough_slider.setValue(45)
        self.rough_slider.valueChanged.connect(self._on_slider_change)
        self.rough_value = QLabel("0.45")
        rough_row.addWidget(self.rough_slider)
        rough_row.addWidget(self.rough_value)
        control_layout.addLayout(rough_row)

        metal_row = QHBoxLayout()
        metal_row.addWidget(QLabel("Metallic:"))
        self.metal_slider = NoWheelSlider(Qt.Horizontal)
        self.metal_slider.setRange(0, 100)
        self.metal_slider.setValue(10)
        self.metal_slider.valueChanged.connect(self._on_slider_change)
        self.metal_value = QLabel("0.10")
        metal_row.addWidget(self.metal_slider)
        metal_row.addWidget(self.metal_value)
        control_layout.addLayout(metal_row)

        light_title = QLabel("Light Direction (drag):")
        control_layout.addWidget(light_title)
        self.light_widget = LightDirectionWidget()
        self.light_widget.directionChanged.connect(lambda _x, _y: self.update_preview())
        control_layout.addWidget(self.light_widget)

        self.refresh_btn = QPushButton("Refresh Preview")
        self.refresh_btn.clicked.connect(self.update_preview)
        control_layout.addWidget(self.refresh_btn)

        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        export_group = QGroupBox("Material Pack Builder")
        export_layout = QVBoxLayout()

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Material Name:"))
        self.material_name = QLineEdit()
        self.material_name.setPlaceholderText("MyMaterial")
        self.material_name.setText("NewMaterial")
        name_row.addWidget(self.material_name)
        export_layout.addLayout(name_row)

        self.output_btn = QPushButton("Select Export Folder")
        self.output_btn.clicked.connect(self.select_output_folder)
        export_layout.addWidget(self.output_btn)

        self.output_label = QLabel("No export folder selected")
        self.output_label.setWordWrap(True)
        export_layout.addWidget(self.output_label)

        self.zip_check = QCheckBox("Create zip archive")
        self.zip_check.setChecked(True)
        export_layout.addWidget(self.zip_check)

        self.export_btn = QPushButton("Export Material Pack")
        self.export_btn.clicked.connect(self.export_material_pack)
        export_layout.addWidget(self.export_btn)

        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)

        left_layout.addStretch()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Preview"))
        self.preview_label = QLabel("Load a Base Color texture to start")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(*self.preview_size)
        self.preview_label.setStyleSheet("border: 1px solid #333; border-radius: 8px;")
        right_layout.addWidget(self.preview_label, 1)

        self.diagnostic_label = QLabel("Ready")
        self.diagnostic_label.setWordWrap(True)
        right_layout.addWidget(self.diagnostic_label)

        layout.addWidget(left)
        layout.addWidget(right, 1)
        self.card_layout.addLayout(layout)
        self._on_slider_change()

    def _add_map_row(self, parent_layout, row, label_text, key, display_label):
        parent_layout.addWidget(QLabel(label_text), row, 0)
        btn = QPushButton("Load")
        btn.clicked.connect(lambda: self.load_map(key, display_label))
        parent_layout.addWidget(btn, row, 1)
        parent_layout.addWidget(display_label, row, 2)

    def _on_slider_change(self):
        self.tiling_value.setText(f"{self.tiling_slider.value()}x")
        self.rough_value.setText(f"{self.rough_slider.value()/100:.2f}")
        self.metal_value.setText(f"{self.metal_slider.value()/100:.2f}")
        self.update_preview()

    def load_map(self, key, display_label):
        path, _ = QFileDialog.getOpenFileName(self, "Select Map", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tga)")
        if not path:
            return
        self.map_paths[key] = path
        display_label.setText(os.path.basename(path))
        if key == "base":
            base_name = os.path.splitext(os.path.basename(path))[0]
            if self.material_name.text().strip() in ("", "NewMaterial"):
                self.material_name.setText(base_name)
        self.update_preview()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder:
            self.output_dir = folder
            self.output_label.setText(folder)

    def _load_image_arr(self, path, mode="RGB"):
        if not path or not os.path.exists(path):
            return None
        img = Image.open(path).convert(mode)
        return np.array(img).astype(np.float32) / 255.0

    def _sample_map(self, arr, u, v, tiling):
        if arr is None:
            return None
        h, w = arr.shape[:2]
        uu = np.mod(u * tiling, 1.0)
        vv = np.mod(v * tiling, 1.0)
        x = np.clip((uu * (w - 1)).astype(np.int32), 0, w - 1)
        y = np.clip((vv * (h - 1)).astype(np.int32), 0, h - 1)
        return arr[y, x]

    def _build_surface(self, view_name, width, height):
        ys, xs = np.mgrid[0:height, 0:width]
        x = (xs / max(1, width - 1)) * 2.0 - 1.0
        y = (ys / max(1, height - 1)) * 2.0 - 1.0

        mask = np.ones((height, width), dtype=bool)
        u = (x + 1.0) * 0.5
        v = (y + 1.0) * 0.5
        nx = np.zeros_like(x)
        ny = np.zeros_like(y)
        nz = np.ones_like(x)

        if view_name == "Sphere":
            r2 = x * x + y * y
            mask = r2 <= 1.0
            z = np.zeros_like(x)
            z[mask] = np.sqrt(np.clip(1.0 - r2[mask], 0.0, 1.0))
            nx, ny, nz = x.copy(), y.copy(), z
            u = 0.5 + np.arctan2(nx, np.maximum(nz, 1e-6)) / (2 * np.pi)
            v = 0.5 - np.arcsin(np.clip(ny, -1.0, 1.0)) / np.pi
        elif view_name == "Cylinder":
            mask = np.abs(x) <= 0.85
            xc = np.clip(x / 0.85, -1.0, 1.0)
            z = np.sqrt(np.clip(1.0 - xc * xc, 0.0, 1.0))
            nx, ny, nz = xc, np.zeros_like(x), z
            u = 0.5 + np.arctan2(xc, np.maximum(z, 1e-6)) / (2 * np.pi)
            v = (y + 1.0) * 0.5
        elif view_name == "Cube":
            ax = np.abs(x)
            ay = np.abs(y)
            major = np.maximum(ax, ay)
            front = major <= 0.58
            right = (x > 0.58) & (ax >= ay)
            left = (x < -0.58) & (ax >= ay)
            top = (y < -0.58) & (ay > ax)
            bottom = (y > 0.58) & (ay > ax)
            mask = front | right | left | top | bottom

            u = np.zeros_like(x)
            v = np.zeros_like(y)

            u[front] = (x[front] / 1.16) + 0.5
            v[front] = (y[front] / 1.16) + 0.5
            nx[front], ny[front], nz[front] = 0.0, 0.0, 1.0

            u[right] = (y[right] + 1.0) * 0.5
            v[right] = (1.0 - (x[right] - 0.58) / 0.42)
            nx[right], ny[right], nz[right] = 1.0, 0.0, 0.0

            u[left] = (y[left] + 1.0) * 0.5
            v[left] = (x[left] + 1.0) / 0.42
            nx[left], ny[left], nz[left] = -1.0, 0.0, 0.0

            u[top] = (x[top] + 1.0) * 0.5
            v[top] = (y[top] + 1.0) / 0.42
            nx[top], ny[top], nz[top] = 0.0, -1.0, 0.0

            u[bottom] = (x[bottom] + 1.0) * 0.5
            v[bottom] = (1.0 - (y[bottom] - 0.58) / 0.42)
            nx[bottom], ny[bottom], nz[bottom] = 0.0, 1.0, 0.0

            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
        elif view_name == "3D Card":
            rx = 0.82
            ry = 0.58
            corner = 0.12
            ax = np.abs(x)
            ay = np.abs(y)
            core = (ax <= rx - corner) & (ay <= ry)
            side = (ax <= rx) & (ay <= ry - corner)
            cx = np.clip(ax - (rx - corner), 0, None)
            cy = np.clip(ay - (ry - corner), 0, None)
            corner_mask = (cx * cx + cy * cy) <= (corner * corner)
            mask = core | side | corner_mask

            u = (x / (2 * rx)) + 0.5
            v = (y / (2 * ry)) + 0.5
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            nx = x * 0.30
            ny = y * 0.15
            nz = np.sqrt(np.clip(1.0 - np.minimum(0.95, nx * nx + ny * ny), 0.0, 1.0))

        n = np.stack([nx, ny, nz], axis=-1)
        n_len = np.linalg.norm(n, axis=-1, keepdims=True)
        n = n / np.maximum(n_len, 1e-6)
        return u, v, n, mask

    def update_preview(self):
        if not self.map_paths["base"]:
            self.preview_label.setText("Load a Base Color texture to start")
            self.diagnostic_label.setText("Base color is required for preview and export.")
            return

        try:
            base = self._load_image_arr(self.map_paths["base"], "RGB")
            normal_map = self._load_image_arr(self.map_paths["normal"], "RGB")
            rough_map = self._load_image_arr(self.map_paths["roughness"], "L")
            height_map = self._load_image_arr(self.map_paths["height"], "L")
            ao_map = self._load_image_arr(self.map_paths["ao"], "L")

            w, h = self.preview_size
            view_name = self.view_combo.currentText()
            tiling = self.tiling_slider.value()
            rough_scalar = self.rough_slider.value() / 100.0
            metal = self.metal_slider.value() / 100.0

            u, v, geo_n, mask = self._build_surface(view_name, w, h)
            base_col = self._sample_map(base, u, v, tiling)
            if base_col is None:
                return

            n = geo_n.copy()
            if normal_map is not None:
                n_tex = self._sample_map(normal_map, u, v, tiling)
                n_tex = (n_tex * 2.0) - 1.0
                n = n + np.stack([n_tex[..., 0], -n_tex[..., 1], n_tex[..., 2]], axis=-1) * 0.45

            if height_map is not None:
                h_tex = self._sample_map(height_map, u, v, tiling)
                grad_y, grad_x = np.gradient(h_tex)
                n[..., 0] -= grad_x * 0.35
                n[..., 1] -= grad_y * 0.35

            n_len = np.linalg.norm(n, axis=-1, keepdims=True)
            n = n / np.maximum(n_len, 1e-6)

            lx, ly = self.light_widget.get_direction()
            lz = math.sqrt(max(0.05, 1.0 - lx * lx - ly * ly))
            light = np.array([lx, ly, lz], dtype=np.float32)
            light /= np.linalg.norm(light)

            view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            half_v = light + view_dir
            half_v /= max(np.linalg.norm(half_v), 1e-6)

            diff = np.clip(np.sum(n * light[None, None, :], axis=-1), 0.0, 1.0)
            nh = np.clip(np.sum(n * half_v[None, None, :], axis=-1), 0.0, 1.0)

            if rough_map is not None:
                rough = self._sample_map(rough_map, u, v, tiling)
            else:
                rough = np.full((h, w), rough_scalar, dtype=np.float32)
            rough = np.clip(rough, 0.02, 1.0)

            if ao_map is not None:
                ao = np.clip(self._sample_map(ao_map, u, v, tiling), 0.0, 1.0)
            else:
                ao = np.ones((h, w), dtype=np.float32)

            shininess = 4.0 + (1.0 - rough) * 160.0
            spec = np.power(nh, shininess)
            spec_color = (0.04 * (1.0 - metal)) + base_col * metal

            lit = base_col * (0.12 + diff[..., None] * (1.0 - metal * 0.15))
            lit += spec[..., None] * spec_color
            lit *= ao[..., None]

            out = np.clip(lit, 0.0, 1.0)
            out[~mask] = 0.06

            out8 = (out * 255).astype(np.uint8)
            qimg = QImage(out8.data, out8.shape[1], out8.shape[0], out8.shape[1] * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg.copy())
            self.preview_label.setPixmap(pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            maps_loaded = sum(1 for k in self.map_paths if self.map_paths[k])
            self.diagnostic_label.setText(
                f"View: {view_name} | Maps loaded: {maps_loaded}/5 | Tiling: {tiling}x | Roughness: {rough_scalar:.2f} | Metallic: {metal:.2f}"
            )
        except Exception as e:
            self.diagnostic_label.setText(f"Preview error: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.map_paths["base"]:
            self.update_preview()

    def export_material_pack(self):
        if not self.map_paths["base"]:
            QMessageBox.warning(self, "Missing Base Color", "Please load a Base Color map first.")
            return

        if not self.output_dir:
            QMessageBox.warning(self, "No Output Folder", "Please select an export folder.")
            return

        material_name = self.material_name.text().strip()
        if not material_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a material name.")
            return

        safe_name = "".join(c for c in material_name if c.isalnum() or c in ("_", "-"))
        if not safe_name:
            safe_name = "Material"

        root = os.path.join(self.output_dir, safe_name)
        tex_dir = os.path.join(root, "Textures")
        os.makedirs(tex_dir, exist_ok=True)

        suffixes = {
            "base": "BaseColor",
            "normal": "Normal",
            "roughness": "Roughness",
            "height": "Height",
            "ao": "AO",
        }
        exported = []
        for key, suffix in suffixes.items():
            src = self.map_paths.get(key, "")
            if not src:
                continue
            try:
                mode = "RGB" if key in ("base", "normal") else "L"
                img = Image.open(src).convert(mode)
                out_path = os.path.join(tex_dir, f"{safe_name}_{suffix}.png")
                img.save(out_path, "PNG")
                exported.append(out_path)
            except Exception:
                pass

        info_path = os.path.join(root, "MaterialInfo.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"Material: {safe_name}\n")
            f.write("Naming Convention: <MaterialName>_<MapType>.png\n")
            f.write("Included Maps:\n")
            for path in exported:
                f.write(f"- {os.path.basename(path)}\n")

        zip_path = ""
        if self.zip_check.isChecked():
            zip_path = os.path.join(self.output_dir, f"{safe_name}.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for folder, _, files in os.walk(root):
                    for file in files:
                        full = os.path.join(folder, file)
                        rel = os.path.relpath(full, self.output_dir)
                        zf.write(full, rel)

        msg = f"Exported material pack to:\n{root}"
        if zip_path:
            msg += f"\n\nZip:\n{zip_path}"
        QMessageBox.information(self, "Export Complete", msg)


# ===================================================================
# URL IMAGE/VIDEO SCRAPER - Download images from websites
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
        self.settings = QSettings("PixelForge", "PixelForgeCreativeTools")
        self.setWindowTitle("PixelForge - Creative Tools")
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
            "Pixel Art Mode",
            "Power-of-Two",
            "Image Grid",
            "Border",
            "Texture Preview",
        ]

        self.stack = QStackedWidget()
        self.pages = {
            "Home": self._build_home_page(),
            "Pixel Art Mode": PixelArtPage(),
            "Power-of-Two": PowerOfTwoPage(),
            "Image Grid": ImageGridPage(),
            "Border": BatchBorderPage(),
            "Texture Preview": TexturePreviewPage()
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self.sidebar_buttons = {}
        category_label = QLabel("Creative Tools")
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

        self.home_title = QLabel("Creative Tools Workspace")
        self.home_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FFC6;")
        subtitle = QLabel(
            "Design-ready art and texture workflows for pixel content, texture packs, and presentation grids."
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
            "Included: Pixel Art Mode, Power-of-Two, Image Grid, Border, Texture Preview"
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
