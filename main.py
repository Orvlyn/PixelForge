import sys, os, random, math, zipfile, logging, gc, json
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

# ============================
# LOGGING SETUP
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'pixelforge.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================
# CONFIGURATION
# ============================
ICON_PATH = os.path.join(os.path.dirname(__file__), "pixelforge.ico")
if not os.path.exists(ICON_PATH):
    fallback_icon = os.path.join(os.path.dirname(__file__), "pixelforge.ico")
    if os.path.exists(fallback_icon):
        ICON_PATH = fallback_icon

APP_VERSION = "3.1.0"
UPDATE_CHECK_URL = "https://raw.githubusercontent.com/Orvlyn/PixelForge/main/version.json"
ICON_GITHUB_URL = "https://raw.githubusercontent.com/Orvlyn/PixelForge/main/pixelforge.ico"
ICON_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".pixelforge_icon_cache.ico")


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


from Tools import (
    BackgroundToolsPage,
    BatchBorderPage,
    BatchWatermarkPage,
    FolderAnalyzerPage,
    FormatConverterPage,
    HEXToolPage,
    ImageGridPage,
    ImageResizerPage,
    PaletteExtractorPage,
    PhotoEditingPage,
    PixelArtPage,
    PowerOfTwoPage,
    RenameToolPage,
    TexturePreviewPage,
    VectorizationPage,
)


class HomePage(CardPage):
    def __init__(self):
        super().__init__("Home")

        self.title = QLabel("PixelForge v3.1")
        self.title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffc6; margin-bottom: 12px;")
        self.card_layout.addWidget(self.title)
        # theme selector with color presets
        theme_h = QHBoxLayout()
        theme_h.addWidget(QLabel("Theme"))
        self.theme_combo = NoWheelComboBox()
        self.theme_combo.addItems([
            "Dark (Original)", "Light (Default)", "Ocean Blue", "Purple Dream", 
            "Sunset Fire", "Rose Blush", "Midnight Black",
            "Coral Reef", "Mint Fresh", "Indigo Night", "Amber Gold",
            "Crimson Bold", "Teal Harmony", "Cyberpunk Neon",
            "Dark Academia", "Blue Lagoon", "Synthwave",
            "Gruvbox", "Nord", "Dracula", "Monokai",
            "Solarized Dark", "Tokyo Night",
            # NEW THEMES
            "Cherry Blossom", "Neon City", "Deep Ocean", "Autumn Leaves",
            "Lime Zest", "Blood Moon", "Tropical Paradise",
            "Electric Storm", "Royal Velvet",
            "Crimson Sunset", "Seafoam Dream",
            "Arcade Glow", "Outrun Sunset", "Neon Circuit", "CRT Amber",
            "Terminal Mono", "Laserwave",
            "Emerald Forest", "Lavender Haze", "Copper Glow", "Desert Sand",
            "Steel Blue", "Bubblegum Pop", "Jade Garden",
            "Sunset Orange", "Midnight Purple", "Ocean Breeze"
        ])
        theme_h.addWidget(self.theme_combo)
        self.card_layout.addLayout(theme_h)

        # quick access buttons (balanced across all categories)
        self.quick_buttons = []
        quick_grid = QGridLayout()
        quick_grid.setHorizontalSpacing(8)
        quick_grid.setVerticalSpacing(8)
        quick_actions = [
            ("Photo Editing", "Photo Editing"),
            ("Image Resizer", "Image Resizer"),
            ("Watermark", "Watermark"),
            ("Background Tools", "Background Tools"),
            ("Pixel Art", "Pixel Art Mode"),
            ("Texture Preview", "Texture Preview"),
            ("Palette Extractor", "Palette Extractor"),
            ("Format Converter", "Format Converter")
        ]
        for idx, (label, target) in enumerate(quick_actions):
            btn = QPushButton(label)
            btn.setMinimumHeight(48)
            btn.clicked.connect(lambda _, n=target: self.window().switch_page(n))
            quick_grid.addWidget(btn, idx // 4, idx % 4)
            self.quick_buttons.append(btn)
        self.card_layout.addLayout(quick_grid)

        # connect theme control after window exists
        self.theme_combo.currentTextChanged.connect(lambda t: self.window().apply_theme(t))
        
        # Stats section
        stats_group = QGroupBox("📊 PixelForge Stats")
        stats_layout = QHBoxLayout()
        stat1 = QLabel("16 Production Tools")
        stat2 = QLabel("50+ Color Themes")
        stat3 = QLabel("75+ Photo Presets")
        stat4 = QLabel("6 Tool Categories")
        for stat in [stat1, stat2, stat3, stat4]:
            stat.setAlignment(Qt.AlignCenter)
            stat.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
            stats_layout.addWidget(stat)
        stats_group.setLayout(stats_layout)
        self.card_layout.addWidget(stats_group)
        
        # Toolkit overview section (all categories, not single-tool focused)
        overview_group = QGroupBox("🧭 Toolkit Overview")
        overview_layout = QVBoxLayout()
        overview_label = QLabel(
            "<b>Image Tools</b> — Resizer, Photo Editing, Watermark, Background Tools, Vectorization<br/>"
            "<b>Creative Tools</b> — Pixel Art, Power-of-Two, Image Grid, Batch Border, Texture Preview<br/>"
            "<b>Color Lab</b> — Palette Extractor and HEX Tool for palette/gradient/design workflows<br/>"
            "<b>Utilities</b> — Rename Tool, Folder Analyzer, Format Converter<br/><br/>"
            "<b>Photo Editing Highlights</b> — 75+ presets, LUTs, color wheels, clean non-graded looks, and tri-tone grading presets.<br/>"
            "<b>Workflow Highlights</b> — batch processing, real-time previews, drag & drop, and optimized export paths."
        )
        overview_label.setWordWrap(True)
        overview_layout.addWidget(overview_label)
        overview_group.setLayout(overview_layout)
        self.card_layout.addWidget(overview_group)

        # Suggested workflows section
        workflow_group = QGroupBox("🚀 Recommended Workflows")
        workflow_layout = QVBoxLayout()
        workflow_label = QLabel(
            "• <b>Social Content:</b> Image Resizer → Photo Editing → Watermark → Format Converter<br/>"
            "• <b>Brand Kit:</b> Palette Extractor / HEX Tool → Batch Border → Image Grid<br/>"
            "• <b>Game Assets:</b> Power-of-Two → Texture Preview → Format Converter<br/>"
            "• <b>Archive Cleanup:</b> Folder Analyzer → Rename Tool → Format Converter"
        )
        workflow_label.setWordWrap(True)
        workflow_layout.addWidget(workflow_label)
        workflow_group.setLayout(workflow_layout)
        self.card_layout.addWidget(workflow_group)

        self.card_layout.addSpacing(8)

        social_row = QHBoxLayout()
        self.update_btn = QPushButton("Check Updates")
        self.update_btn.clicked.connect(lambda: self.window().check_for_updates())
        self.follow_btn = QPushButton("𝕏 Follow @Orvlyn")
        self.follow_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://x.com/Orvlyn")))
        self.site_btn = QPushButton("Orvlyn.me")
        self.site_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://Orvlyn.me")))
        social_row.addWidget(self.update_btn)
        social_row.addWidget(self.follow_btn)
        social_row.addWidget(self.site_btn)
        self.card_layout.addLayout(social_row)

        self.tips_label = QLabel(
            "⌨️ <b>KEYBOARD SHORTCUTS:</b><br/>"
            "• In HEX Tool: Press <b>SPACE</b> to randomize palette<br/>"
            "• In any preview: <b>Mouse wheel</b> on zoom slider to zoom in/out<br/><br/>"
            "🎯 <b>HIGHLIGHTS:</b><br/>"
            "• 75+ professional photo presets (including no-grading and tri-tone collections)<br/>"
            "• 50 themes and a unified UI style across tool pages<br/>"
            "• Real-time preview with ultra-fast processing (10-50x speed boost!)<br/>"
            "• Batch-ready workflows for resizing, watermarking, conversion, and organization<br/>"
            "• Support for PNG, JPG, GIF, WEBP, BMP, TIFF, TGA & ICO workflows"
        )
        self.tips_label.setStyleSheet("font-size: 11px; margin-top: 10px;")
        self.card_layout.addWidget(self.tips_label)
        
        # Call update on initial theme
        self.update_theme_colors()
    
    def update_theme_colors(self):
        """Update colors based on current theme"""
        window = self.window()
        if not hasattr(window, 'current_theme_colors'):
            return
        
        colors = window.current_theme_colors
        accent = colors.get('accent', '#00FFC6')
        secondary_bg = colors.get('secondary_bg', '#0B0F15')
        primary_bg = colors.get('primary_bg', '#070A0E')
        
        # Update title color
        self.title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {accent}; margin-bottom: 12px;")
        
        # Update quick access buttons
        for btn in self.quick_buttons:
            btn.setStyleSheet(f"font-size: 16px; font-weight: bold; background: {secondary_bg}; color: {accent};")
        
        # Update social buttons
        self.update_btn.setStyleSheet(f"background: {secondary_bg}; color: {accent}; font-weight: bold;")
        self.follow_btn.setStyleSheet(f"background: {accent}; color: {primary_bg}; font-weight: bold;")
        self.site_btn.setStyleSheet(f"background: {secondary_bg}; color: {accent};")
        
        # Update tips label
        self.tips_label.setStyleSheet(f"font-size: 11px; color: {accent}; margin-top: 10px;")
        
        # Set combo box to current theme (block signals to prevent re-applying)
        if hasattr(window, 'current_theme_name'):
            self.theme_combo.blockSignals(True)
            index = self.theme_combo.findText(window.current_theme_name)
            if index >= 0:
                self.theme_combo.setCurrentIndex(index)
            self.theme_combo.blockSignals(False)

class RenameToolPage(CardPage):
    def __init__(self):
        super().__init__("Batch Rename")

        self.folder_btn = QPushButton("Select Folder")
        self.card_layout.addWidget(self.folder_btn)

        # rename options panel
        opts_layout = QGridLayout()
        self.pad_spin = QSpinBox()
        self.pad_spin.setRange(0, 10)
        self.pad_spin.setValue(0)
        opts_layout.addWidget(QLabel("Zero padding"), 0, 0)
        opts_layout.addWidget(self.pad_spin, 0, 1)

        self.replace_combo = NoWheelComboBox()
        self.replace_combo.addItems(["None","_","-"])
        opts_layout.addWidget(QLabel("Replace spaces with"), 1, 0)
        opts_layout.addWidget(self.replace_combo, 1, 1)

        self.remove_special = QCheckBox("Remove special chars")
        opts_layout.addWidget(self.remove_special, 2, 0, 1, 2)

        self.case_combo = NoWheelComboBox()
        self.case_combo.addItems(["none","lower","upper","title"])
        opts_layout.addWidget(QLabel("Change case"), 3, 0)
        opts_layout.addWidget(self.case_combo, 3, 1)

        self.regex_mode = QCheckBox("Regex mode")
        opts_layout.addWidget(self.regex_mode, 4, 0, 1, 2)

        self.preview_btn = QPushButton("Preview Rename")
        opts_layout.addWidget(self.preview_btn, 5, 0, 1, 2)

        self.card_layout.addLayout(opts_layout)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search files...")
        self.search_input.textChanged.connect(self.filter_table)
        self.card_layout.addWidget(QLabel("Search"))
        self.card_layout.addWidget(self.search_input)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Original Name", "New Name"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.card_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.reset_btn = QPushButton("Reset New Names")
        self.reset_btn.clicked.connect(self.reset_new_names)
        self.undo_btn = QPushButton("Undo Rename")
        self.undo_btn.setEnabled(False)
        btn_row.addWidget(self.undo_btn)
        self.apply_btn = QPushButton("Apply Rename")
        self.apply_btn.clicked.connect(self.apply_rename)
        btn_row.addWidget(self.apply_btn)
        self.card_layout.addLayout(btn_row)

        self.folder = ""
        self.last_renames = []
        self.folder_btn.clicked.connect(self.select_folder)
        self.preview_btn.clicked.connect(self.preview_rename)
        self.undo_btn.clicked.connect(self.undo_rename)

    def select_folder(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("RenameToolPage_last_folder", "")
        self.folder = QFileDialog.getExistingDirectory(self, "Select Folder", last_folder)
        if self.folder:
            settings.setValue("RenameToolPage_last_folder", self.folder)
            self.populate_table()

    def populate_table(self):
        self.table.setRowCount(0)
        files = sorted([f for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder, f))])
        self.table.setRowCount(len(files))
        for row, f in enumerate(files):
            self.table.setItem(row, 0, QTableWidgetItem(f))
            self.table.setItem(row, 1, QTableWidgetItem(f))

    def filter_table(self):
        search = self.search_input.text().lower()
        for row in range(self.table.rowCount()):
            orig = self.table.item(row, 0).text().lower()
            self.table.setRowHidden(row, bool(search) and search not in orig)

    def preview_rename(self):
        # generate new names based on options
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row): continue
            orig = self.table.item(row, 0).text()
            new = orig
            if self.regex_mode.isChecked():
                # could prompt for regex pattern replacement? skip for now
                pass
            else:
                # remove special
                if self.remove_special.isChecked():
                    import re
                    new = re.sub(r'[^\w\s\-]', '', new)
                # replace spaces
                rep = self.replace_combo.currentText()
                if rep != "None":
                    new = new.replace(' ', rep)
                # case
                case = self.case_combo.currentText()
                if case == 'lower': new = new.lower()
                elif case == 'upper': new = new.upper()
                elif case == 'title': new = new.title()
                # padding digits
                if self.pad_spin.value() > 0:
                    import re
                    match = re.search(r'(\d+)', new)
                    if match:
                        num = match.group(1)
                        new = new.replace(num, num.zfill(self.pad_spin.value()))
            self.table.item(row, 1).setText(new)

    def undo_rename(self):
        for new_path, old_path in reversed(self.last_renames):
            try:
                if os.path.exists(new_path):
                    os.rename(new_path, old_path)
            except Exception:
                pass
        self.last_renames = []
        self.undo_btn.setEnabled(False)
        self.populate_table()

    def reset_new_names(self):
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                orig = self.table.item(row, 0).text()
                self.table.item(row, 1).setText(orig)

    def apply_rename(self):
        if not self.folder: return
        self.last_renames = []
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row): continue
            old_name = self.table.item(row, 0).text()
            new_name = self.table.item(row, 1).text()
            if new_name and new_name != old_name:
                old_path = os.path.join(self.folder, old_name)
                new_path = os.path.join(self.folder, new_name)
                if os.path.exists(old_path):
                    try:
                        os.rename(old_path, new_path)
                        self.last_renames.append((new_path, old_path))
                    except Exception:
                        pass
        self.populate_table()
        if self.last_renames:
            self.undo_btn.setEnabled(True)


# ===================================================================
# MAIN WINDOW
# ===================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # load persisted theme
        self.settings = QSettings("PixelForge", "PixelForge")
        theme = self.settings.value("theme", "Dark (Original)")
        self.setWindowTitle("PixelForge")
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

        self.stack = QStackedWidget()

        self.pages = {
            "Home": HomePage(),
            "Image Resizer": ImageResizerPage(),
            "Photo Editing": PhotoEditingPage(),
            "Watermark": BatchWatermarkPage(),
            "Background Tools": BackgroundToolsPage(),
            "Palette Extractor": PaletteExtractorPage(),
            "Vectorization": VectorizationPage(),
            "HEX Tool": HEXToolPage(),
            "Pixel Art Mode": PixelArtPage(),
            "Power-of-Two": PowerOfTwoPage(),
            "Image Grid": ImageGridPage(),
            "Batch Border": BatchBorderPage(),
            "Texture Preview": TexturePreviewPage(),
            "Rename Tool": RenameToolPage(),
            "Folder Analyzer": FolderAnalyzerPage(),
            "Format Converter": FormatConverterPage()
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self.sidebar_buttons = {}
        self.submenus = {}

        # Build sidebar
        def add_category(name, items):
            cat_btn = QPushButton(name)
            cat_btn.setCheckable(True)
            self.sidebar_layout.addWidget(cat_btn)
            submenu = QWidget()
            submenu_layout = QVBoxLayout(submenu)
            submenu_layout.setContentsMargins(20, 0, 0, 0)
            submenu_layout.setSpacing(5)
            submenu.setVisible(False)
            self.sidebar_layout.addWidget(submenu)
            self.submenus[cat_btn] = submenu

            def toggle():
                for b, sm in self.submenus.items():
                    visible = (b == cat_btn and not sm.isVisible())
                    sm.setVisible(visible)
                    b.setChecked(visible)
            cat_btn.clicked.connect(toggle)

            for item_name in items:
                btn = QPushButton(item_name)
                btn.setCheckable(True)
                btn.clicked.connect(lambda checked, n=item_name: self.switch_page(n))
                submenu_layout.addWidget(btn)
                self.sidebar_buttons[item_name] = btn

        add_category("Welcome", ["Home"])
        add_category("Image Tools", ["Image Resizer", "Photo Editing", "Watermark", "Background Tools", "Vectorization"])
        add_category("Creative Tools", ["Pixel Art Mode", "Power-of-Two", "Image Grid", "Batch Border", "Texture Preview"])
        add_category("Color Lab", ["Palette Extractor", "HEX Tool"])
        add_category("Rename Tool", ["Rename Tool"])
        add_category("Utilities", ["Folder Analyzer", "Format Converter"])
        
        # default to Home page
        if "Home" in self.sidebar_buttons:
            self.switch_page("Home")
        else:
            if "Home" in self.pages:
                self.stack.setCurrentWidget(self.pages["Home"])

        footer_widget = QWidget()
        footer_layout = QHBoxLayout(footer_widget)
        footer_layout.setContentsMargins(0, 0, 0, 0)

        self.made_label = QLabel("Made by Orvlyn")
        self.made_label.setStyleSheet("font-size: 13px;")
        footer_layout.addWidget(self.made_label)
        footer_layout.addStretch()

        self.sidebar_layout.addStretch()
        self.sidebar_layout.addWidget(footer_widget)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(self.sidebar_layout)
        sidebar_widget.setFixedWidth(340)
        sidebar_widget.setObjectName("Sidebar")

        layout.addWidget(sidebar_widget)
        layout.addWidget(self.stack)

        # Enable drag and drop
        self.setAcceptDrops(True)

        # Apply theme after everything is built
        self.apply_theme(theme)

    def _version_tuple(self, value: str) -> tuple:
        parts = []
        for token in str(value or "").replace("-", ".").split("."):
            if token.isdigit():
                parts.append(int(token))
        return tuple(parts) if parts else (0,)

    def check_for_updates(self) -> None:
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

    def apply_theme(self, name):
        """Apply theme with full styling support"""
        themes = {
            "Dark (Original)": {
                "accent": "#00FFC6",
                "primary_bg": "#070A0E",
                "secondary_bg": "#0B0F15",
                "tertiary_bg": "#141A22",
                "text": "#E6EAF0",
                "css_base": "#00FFC6"
            },
            "Light (Default)": {
                "accent": "#0078d4",
                "primary_bg": "#f5f5f5",
                "secondary_bg": "#ffffff",
                "tertiary_bg": "#e8e8e8",
                "text": "#1a1a1a",
                "css_base": "#0078d4",
                "is_light": True
            },
            "Ocean Blue": {
                "accent": "#42a5f5",
                "primary_bg": "#0a1929",
                "secondary_bg": "#0d2136",
                "tertiary_bg": "#1565c0",
                "text": "#e3f2fd",
                "css_base": "#42a5f5"
            },
            "Purple Dream": {
                "accent": "#f4d35e",
                "primary_bg": "#120018",
                "secondary_bg": "#1f0030",
                "tertiary_bg": "#360052",
                "text": "#f5e9ff",
                "css_base": "#f4d35e"
            },
            "Sunset Fire": {
                "accent": "#ff9800",
                "primary_bg": "#1f0f00",
                "secondary_bg": "#331a00",
                "tertiary_bg": "#e65100",
                "text": "#fff3e0",
                "css_base": "#ff9800"
            },
            "Rose Blush": {
                "accent": "#da70d6",
                "primary_bg": "#1a0a1a",
                "secondary_bg": "#2d142d",
                "tertiary_bg": "#4d1f4d",
                "text": "#f0d9f0",
                "css_base": "#da70d6"
            },
            "Midnight Black": {
                "accent": "#78909c",
                "primary_bg": "#000000",
                "secondary_bg": "#121212",
                "tertiary_bg": "#212121",
                "text": "#b0bec5",
                "css_base": "#78909c"
            },
            "Coral Reef": {
                "accent": "#7b3f00",
                "primary_bg": "#0a0400",
                "secondary_bg": "#1a0d00",
                "tertiary_bg": "#2d1a00",
                "text": "#d4a574",
                "css_base": "#7b3f00"
            },
            "Mint Fresh": {
                "accent": "#ff8a65",
                "primary_bg": "#0a1a14",
                "secondary_bg": "#112921",
                "tertiary_bg": "#ff6e40",
                "text": "#ffe0b2",
                "css_base": "#ff8a65"
            },
            "Indigo Night": {
                "accent": "#708090",
                "primary_bg": "#0a0d0f",
                "secondary_bg": "#151b21",
                "tertiary_bg": "#2d3a45",
                "text": "#d0d8e0",
                "css_base": "#708090"
            },
            "Amber Gold": {
                "accent": "#d2b48c",
                "primary_bg": "#1a1410",
                "secondary_bg": "#2a2117",
                "tertiary_bg": "#3d3220",
                "text": "#f5e6d3",
                "css_base": "#d2b48c"
            },
            "Crimson Bold": {
                "accent": "#dc2626",
                "primary_bg": "#1a0a0a",
                "secondary_bg": "#2c1010",
                "tertiary_bg": "#7f1d1d",
                "text": "#fee2e2",
                "css_base": "#dc2626"
            },
            "Teal Harmony": {
                "accent": "#00f5d4",
                "primary_bg": "#001b1e",
                "secondary_bg": "#003039",
                "tertiary_bg": "#005b65",
                "text": "#c8fff4",
                "css_base": "#00f5d4"
            },
            "Cyberpunk Neon": {
                "accent": "#39ff14",
                "primary_bg": "#050a05",
                "secondary_bg": "#0b160b",
                "tertiary_bg": "#123312",
                "text": "#baffc9",
                "css_base": "#39ff14"
            },
            "Steel Blue": {
                "accent": "#4682b4",
                "primary_bg": "#0a0f14",
                "secondary_bg": "#14212d",
                "tertiary_bg": "#1f3847",
                "text": "#d1e3f0",
                "css_base": "#4682b4"
            },
            "Dark Academia": {
                "accent": "#c4a747",
                "primary_bg": "#0f0f0f",
                "secondary_bg": "#1d1d1d",
                "tertiary_bg": "#2d2416",
                "text": "#e8e8d8",
                "css_base": "#c4a747"
            },
            "Desert Sand": {
                "accent": "#edc9af",
                "primary_bg": "#1a140f",
                "secondary_bg": "#2d241f",
                "tertiary_bg": "#4d3d32",
                "text": "#f5e8dc",
                "css_base": "#edc9af"
            },
            "Blue Lagoon": {
                "accent": "#00d9ff",
                "primary_bg": "#001219",
                "secondary_bg": "#005f73",
                "tertiary_bg": "#0a9396",
                "text": "#94d2bd",
                "css_base": "#00d9ff"
            },
            "Synthwave": {
                "accent": "#ff4fd8",
                "primary_bg": "#0b0d2b",
                "secondary_bg": "#141a44",
                "tertiary_bg": "#2a0f6b",
                "text": "#f7d1ff",
                "css_base": "#ff4fd8"
            },
            "Bubblegum Pop": {
                "accent": "#ff6ec7",
                "primary_bg": "#1a0014",
                "secondary_bg": "#2d0024",
                "tertiary_bg": "#4d003d",
                "text": "#ffcce6",
                "css_base": "#ff6ec7"
            },
            "Gruvbox": {
                "accent": "#556b2f",
                "primary_bg": "#0d0d00",
                "secondary_bg": "#1a1a00",
                "tertiary_bg": "#2d2d00",
                "text": "#c4d96f",
                "css_base": "#556b2f"
            },
            "Nord": {
                "accent": "#0047ab",
                "primary_bg": "#0a0f1a",
                "secondary_bg": "#141f2d",
                "tertiary_bg": "#1f3347",
                "text": "#b3d9ff",
                "css_base": "#0047ab"
            },
            "Dracula": {
                "accent": "#8b0000",
                "primary_bg": "#1a0a0a",
                "secondary_bg": "#2d1414",
                "tertiary_bg": "#4d1f1f",
                "text": "#ffb3b3",
                "css_base": "#8b0000"
            },
            "Monokai": {
                "accent": "#f9d423",
                "primary_bg": "#1b1a14",
                "secondary_bg": "#2a281f",
                "tertiary_bg": "#3d3a2b",
                "text": "#fff3c4",
                "css_base": "#f9d423"
            },
            "Solarized Dark": {
                "accent": "#268bd2",
                "primary_bg": "#002b36",
                "secondary_bg": "#073642",
                "tertiary_bg": "#586e75",
                "text": "#fdf6e3",
                "css_base": "#268bd2"
            },
            "Tokyo Night": {
                "accent": "#ff8c00",
                "primary_bg": "#0d1424",
                "secondary_bg": "#152038",
                "tertiary_bg": "#1f2f52",
                "text": "#ffd8a6",
                "css_base": "#ff8c00"
            },
            # NEW UNIQUE THEMES
            "Neon City": {
                "accent": "#6a5acd",
                "primary_bg": "#0d0721",
                "secondary_bg": "#1a1038",
                "tertiary_bg": "#2d1f5a",
                "text": "#e6d9ff",
                "css_base": "#6a5acd"
            },
            "Deep Ocean": {
                "accent": "#40e0d0",
                "primary_bg": "#001a1a",
                "secondary_bg": "#003333",
                "tertiary_bg": "#004d4d",
                "text": "#b3f0e8",
                "css_base": "#40e0d0"
            },
            "Autumn Leaves": {
                "accent": "#a4512e",
                "primary_bg": "#1a0d07",
                "secondary_bg": "#2d140b",
                "tertiary_bg": "#4d2415",
                "text": "#f2c9b5",
                "css_base": "#a4512e"
            },
            "Lime Zest": {
                "accent": "#32cd32",
                "primary_bg": "#0a1400",
                "secondary_bg": "#142400",
                "tertiary_bg": "#1f3d00",
                "text": "#d4f4a8",
                "css_base": "#32cd32"
            },
            "Blood Moon": {
                "accent": "#ff0033",
                "primary_bg": "#1a0000",
                "secondary_bg": "#2d0000",
                "tertiary_bg": "#4d0000",
                "text": "#ffcccc",
                "css_base": "#ff0033"
            },
            "Tropical Paradise": {
                "accent": "#00ff9f",
                "primary_bg": "#001a11",
                "secondary_bg": "#002d1c",
                "tertiary_bg": "#004d2e",
                "text": "#ccffe6",
                "css_base": "#00ff9f"
            },
            "Cherry Blossom": {
                "accent": "#ffb7d5",
                "primary_bg": "#1a0810",
                "secondary_bg": "#2d1020",
                "tertiary_bg": "#4d1a35",
                "text": "#ffe0ed",
                "css_base": "#ffb7d5"
            },
            "Electric Storm": {
                "accent": "#bf00ff",
                "primary_bg": "#0f001a",
                "secondary_bg": "#1a002d",
                "tertiary_bg": "#2d004d",
                "text": "#e6ccff",
                "css_base": "#bf00ff"
            },
            "Royal Velvet": {
                "accent": "#6a0dad",
                "primary_bg": "#0d001a",
                "secondary_bg": "#1a002d",
                "tertiary_bg": "#2d004d",
                "text": "#d9b3ff",
                "css_base": "#6a0dad"
            },
            "Crimson Sunset": {
                "accent": "#e63946",
                "primary_bg": "#1a0608",
                "secondary_bg": "#2d0d10",
                "tertiary_bg": "#4d161a",
                "text": "#ffccd0",
                "css_base": "#e63946"
            },
            "Seafoam Dream": {
                "accent": "#7fffd4",
                "primary_bg": "#001a14",
                "secondary_bg": "#002d21",
                "tertiary_bg": "#004d38",
                "text": "#d9fff0",
                "css_base": "#7fffd4"
            },
            "Arcade Glow": {
                "accent": "#00e5ff",
                "primary_bg": "#0b0014",
                "secondary_bg": "#170028",
                "tertiary_bg": "#2a0047",
                "text": "#c9f7ff",
                "css_base": "#00e5ff"
            },
            "Outrun Sunset": {
                "accent": "#ff6b35",
                "primary_bg": "#1a0013",
                "secondary_bg": "#2d0023",
                "tertiary_bg": "#4d0040",
                "text": "#ffd1b8",
                "css_base": "#ff6b35"
            },
            "Neon Circuit": {
                "accent": "#7cff00",
                "primary_bg": "#0b1400",
                "secondary_bg": "#132300",
                "tertiary_bg": "#1e3d00",
                "text": "#e2ffc2",
                "css_base": "#7cff00"
            },
            "CRT Amber": {
                "accent": "#ffb000",
                "primary_bg": "#1a0f00",
                "secondary_bg": "#2d1a00",
                "tertiary_bg": "#4d2b00",
                "text": "#ffe2a8",
                "css_base": "#ffb000"
            },
            "Terminal Mono": {
                "accent": "#c0c0c0",
                "primary_bg": "#0d0d0d",
                "secondary_bg": "#1a1a1a",
                "tertiary_bg": "#2b2b2b",
                "text": "#f2f2f2",
                "css_base": "#c0c0c0"
            },
            "Laserwave": {
                "accent": "#ff00a8",
                "primary_bg": "#0b001a",
                "secondary_bg": "#160033",
                "tertiary_bg": "#2b0052",
                "text": "#ffd1f0",
                "css_base": "#ff00a8"
            },
            "Emerald Forest": {
                "accent": "#50c878",
                "primary_bg": "#0a1409",
                "secondary_bg": "#142414",
                "tertiary_bg": "#1f3d1f",
                "text": "#d4f4dd",
                "css_base": "#50c878"
            },
            "Lavender Haze": {
                "accent": "#e6b3ff",
                "primary_bg": "#14091a",
                "secondary_bg": "#24142d",
                "tertiary_bg": "#3d1f4d",
                "text": "#f5e6ff",
                "css_base": "#e6b3ff"
            },
            "Copper Glow": {
                "accent": "#d4894f",
                "primary_bg": "#1a0f07",
                "secondary_bg": "#2d1a0b",
                "tertiary_bg": "#4d2a15",
                "text": "#f4dcc9",
                "css_base": "#d4894f"
            },
            "Jade Garden": {
                "accent": "#00a86b",
                "primary_bg": "#001a0f",
                "secondary_bg": "#002d1a",
                "tertiary_bg": "#004d2a",
                "text": "#ccffe6",
                "css_base": "#00a86b"
            },
            "Sunset Orange": {
                "accent": "#ff6347",
                "primary_bg": "#1a0a05",
                "secondary_bg": "#2d140a",
                "tertiary_bg": "#4d2410",
                "text": "#ffd4cc",
                "css_base": "#ff6347"
            },
            "Midnight Purple": {
                "accent": "#9370db",
                "primary_bg": "#0f0a1a",
                "secondary_bg": "#1a142d",
                "tertiary_bg": "#2d1f4d",
                "text": "#e0d4ff",
                "css_base": "#9370db"
            },
            "Ocean Breeze": {
                "accent": "#48d1cc",
                "primary_bg": "#0a1a1a",
                "secondary_bg": "#142d2d",
                "tertiary_bg": "#1f4d4d",
                "text": "#d4f4f2",
                "css_base": "#48d1cc"
            }
        }
        
        theme = themes.get(name, themes["Dark (Original)"])
        
        # Store current theme colors for access by pages
        self.current_theme_name = name
        self.current_theme_colors = theme
        
        # Build comprehensive stylesheet
        accent = theme.get("accent", "#00FFC6")
        primary = theme.get("primary_bg", "#070A0E")
        secondary = theme.get("secondary_bg", "#0B0F15")
        tertiary = theme.get("tertiary_bg", "#141A22")
        text = theme.get("text", "#E6EAF0")
        is_light = theme.get("is_light", False)
        
        css = f"""
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
                color: {'#ffffff' if is_light else primary};
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
                QLineEdit, QComboBox, QSpinBox {{
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
                QSlider {{
                    background: {primary};
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
                    min-width: 120px;
                }}
                QTabBar::tab:selected {{
                    background: {accent};
                    color: {primary};
                    font-weight: bold;
                }}
                QTabBar::tab:hover {{
                    background: {tertiary};
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
                QComboBox::drop-down {{
                    border-left: 1px solid {tertiary};
                    width: 20px;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    border-bottom: 2px solid {accent};
                    border-right: 2px solid {accent};
                    width: 6px;
                    height: 6px;
                    margin-right: 5px;
                }}
                /* Scrollbar Styling - Match Theme Colors */
                QScrollBar:vertical {{
                    background: {secondary};
                    width: 12px;
                    border-radius: 6px;
                    margin: 0px;
                }}
                QScrollBar::handle:vertical {{
                    background: {accent};
                    border-radius: 6px;
                    min-height: 30px;
                }}
                QScrollBar::handle:vertical:hover {{
                    background: {tertiary};
                }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                    height: 0px;
                }}
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                    background: {secondary};
                }}
                QScrollBar:horizontal {{
                    background: {secondary};
                    height: 12px;
                    border-radius: 6px;
                    margin: 0px;
                }}
                QScrollBar::handle:horizontal {{
                    background: {accent};
                    border-radius: 6px;
                    min-width: 30px;
                }}
                QScrollBar::handle:horizontal:hover {{
                    background: {tertiary};
                }}
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                    width: 0px;
                }}
                QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                    background: {secondary};
                }}
            """
        self.setStyleSheet(css)
        
        # Update Made by Orvlyn label to match theme
        if hasattr(self, 'made_label'):
            self.made_label.setStyleSheet(f"font-size: 11px; color: {accent}; padding: 10px;")
        
        # Store current theme for reference by pages
        self.current_theme_name = name
        
        # Notify all pages of theme change
        for page in self.pages.values():
            if hasattr(page, 'update_theme_colors'):
                page.update_theme_colors()
        
        # Update title bar to match theme
        self.apply_titlebar_color(is_light)
        
        self.settings.setValue("theme", name)

    def switch_page(self, name):
        for btn in self.sidebar_buttons.values():
            btn.setChecked(False)
        self.sidebar_buttons[name].setChecked(True)
        self.stack.setCurrentWidget(self.pages[name])

    def dragEnterEvent(self, event):
        """Accept drag events with files"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        """Handle dropped files"""
        urls = event.mimeData().urls()
        if not urls:
            return
            
        # Get file paths
        files = [url.toLocalFile() for url in urls if url.isLocalFile()]
        if not files:
            return
            
        # Filter image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tga', '.ico']
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        
        if not image_files:
            QMessageBox.warning(self, "No Images", "No valid image files were dropped.")
            return
            
        # Show action dialog
        self.show_drop_action_dialog(image_files)
        
    def show_drop_action_dialog(self, files):
        """Show dialog asking what to do with dropped files"""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("What would you like to do?")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"Dropped {len(files)} image(s).\nWhat would you like to do?"))
        
        # Action buttons
        actions = [
            ("Resize", "Image Resizer"),
            ("Convert Format", "Format Converter"),
            ("Add Watermark", "Watermark"),
            ("Photo Editing", "Photo Editing"),
            ("Pixel Art", "Pixel Art Mode"),
            ("Power-of-Two", "Power-of-Two"),
            ("Add Border", "Batch Border"),
            ("Create Grid", "Image Grid"),
            ("Extract Palette", "Palette Extractor")
        ]
        
        for label, page_name in actions:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, pn=page_name, fs=files: self.handle_drop_action(pn, fs, dialog))
            layout.addWidget(btn)
            
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        layout.addWidget(cancel_btn)
        
        dialog.exec()
        
    def handle_drop_action(self, page_name, files, dialog):
        """Handle the selected action for dropped files"""
        dialog.accept()
        
        # Switch to the selected page
        self.switch_page(page_name)
        
        # Try to load the files into the page
        page = self.pages[page_name]
        
        # Different pages have different ways to load files
        if hasattr(page, 'input_file') and len(files) == 1:
            # Single file pages
            page.input_file = files[0]
            if hasattr(page, 'input_label'):
                page.input_label.setText(os.path.basename(files[0]))
            if hasattr(page, 'original'):
                try:
                    page.original = Image.open(files[0])
                    if hasattr(page, 'preview_label'):
                        page.preview_label.setText("Image loaded. Click a button to process.")
                except:
                    pass
                    
        elif hasattr(page, 'input_files'):
            # Batch processing pages
            page.input_files = files
            if hasattr(page, 'file_count_label'):
                page.file_count_label.setText(f"{len(files)} files selected")
            elif hasattr(page, 'update_image_list'):
                # Image Grid page
                page.images = []
                for f in files:
                    try:
                        img = Image.open(f)
                        page.images.append({"path": f, "image": img, "name": os.path.basename(f)})
                    except:
                        pass
                page.update_image_list()
                
        QMessageBox.information(self, "Files Loaded", 
                                f"Loaded {len(files)} file(s) into {page_name}.\nConfigure settings and process!")

    def apply_titlebar_color(self, is_light=False):
        """Apply title bar color to match theme on Windows 10/11"""
        if HAS_WINDOWS_TITLEBAR and sys.platform == 'win32':
            try:
                hwnd = int(self.winId())
                # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Windows 11) or 19 (Windows 10 build 19041+)
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                value = ctypes.c_int(0 if is_light else 1)  # 0 = light, 1 = dark
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_USE_IMMERSIVE_DARK_MODE,
                    ctypes.byref(value),
                    ctypes.sizeof(value)
                )
            except Exception:
                pass  # Silently fail on older Windows versions
    
    def open_social(self):
        QDesktopServices.openUrl(QUrl("https://x.com/Orvlyn"))


# Run App
if __name__ == "__main__":
    app = QApplication(sys.argv)

    qss = """
    QMainWindow { background: #070A0E; }
    QWidget { background: #070A0E; color: #E6EAF0; font: 13px 'Segoe UI'; }
    #Sidebar QPushButton {
        background: #0B0F15; border: none; padding: 12px; border-radius: 6px; text-align: left;
    }
    #Sidebar QPushButton:checked { background: #00FFC6; color: #070A0E; }
    #Sidebar QPushButton:hover { background: #141A22; }
    QWidget#Card {
        background: #0B0F15; border-radius: 12px; padding: 20px;
    }
    QPushButton { background: #0B0F15; border: 1px solid #141A22; border-radius: 8px; padding: 8px 12px; }
    QPushButton:hover { border: 1px solid #00FFC6; }
    QLineEdit, QComboBox, QSlider, QSpinBox { background: #0B0F15; border: 1px solid #141A22; border-radius: 6px; padding: 6px; color: #E6EAF0; }
    QProgressBar { background: #0B0F15; border-radius: 6px; text-align: center; }
    QProgressBar::chunk { background: #00FFC6; }
    QLabel { font: 13px 'Segoe UI'; }
    QScrollArea { border: none; background: #0B0F15; }
    QTableWidget { background: #0B0F15; gridline-color: #141A22; }
    QSlider::groove:horizontal { background: #141A22; height: 6px; border-radius: 3px; }
    QSlider::handle:horizontal { background: #00FFC6; width: 14px; margin: -4px 0; border-radius: 7px; }
    """
    app.setStyleSheet(qss)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())