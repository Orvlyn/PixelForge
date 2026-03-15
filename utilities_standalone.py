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

class DuplicateFinderPage(CardPage):
    def __init__(self):
        super().__init__("Duplicate Finder")
        self.folder_btn = QPushButton("Select Folder")
        self.start_btn = QPushButton("Scan")
        self.delete_btn = QPushButton("Delete Selected")
        self.result_area = QTableWidget()
        self.result_area.setColumnCount(3)
        self.result_area.setHorizontalHeaderLabels(["Hash","File","Size"])
        self.result_area.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        # add widgets to card layout
        self.card_layout.addWidget(self.folder_btn)
        self.card_layout.addWidget(self.start_btn)
        self.card_layout.addWidget(self.delete_btn)
        self.card_layout.addWidget(self.result_area)
        self.folder = ""
        self.folder_btn.clicked.connect(self.select_folder)
        self.start_btn.clicked.connect(self.scan_duplicates)
        self.delete_btn.clicked.connect(self.delete_selected)

    def select_folder(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("DuplicateFinderPage_last_folder", "")
        self.folder = QFileDialog.getExistingDirectory(self, "Select Folder", last_folder)
        if self.folder:
            settings.setValue("DuplicateFinderPage_last_folder", self.folder)
            self.scan_duplicates()

    def delete_selected(self):
        rows = sorted({i.row() for i in self.result_area.selectedIndexes()}, reverse=True)
        if not rows:
            return
        reply = QMessageBox.question(self, "Confirm Delete", f"Delete {len(rows)} selected files?", QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for r in rows:
            path = self.result_area.item(r,1).text()
            try:
                os.remove(path)
            except:
                pass
            self.result_area.removeRow(r)

    def scan_duplicates(self):
        if not self.folder: return
        import hashlib
        hashes = {}
        # walk recursively
        for root, dirs, files in os.walk(self.folder):
            for fname in files:
                if fname.lower().endswith((".png",".jpg",".jpeg",".gif")):
                    path = os.path.join(root, fname)
                    try:
                        h = hashlib.md5(open(path,'rb').read()).hexdigest()
                        hashes.setdefault(h, []).append(path)
                    except:
                        pass
        self.result_area.setRowCount(0)
        for h, flist in hashes.items():
            if len(flist) > 1:
                for f in flist:
                    row = self.result_area.rowCount()
                    self.result_area.insertRow(row)
                    self.result_area.setItem(row,0, QTableWidgetItem(h))
                    self.result_area.setItem(row,1, QTableWidgetItem(f))
                    self.result_area.setItem(row,2, QTableWidgetItem(str(os.path.getsize(f)//1024)+"KB"))

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

class FolderAnalyzerPage(CardPage):
    def __init__(self):
        super().__init__("Folder Analyzer")
        self.folder_btn = QPushButton("Select Folder")
        self.info_label = QPlainTextEdit()
        self.info_label.setReadOnly(True)
        self.card_layout.addWidget(self.folder_btn)
        self.card_layout.addWidget(self.info_label)
        self.folder_btn.clicked.connect(self.analyze)

    def analyze(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder =settings.value("FolderAnalyzerPage_last_folder", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", last_folder)
        if not folder: 
            return
        if folder:
            settings.setValue("FolderAnalyzerPage_last_folder", folder)
        total = 0
        size = 0
        largest = ("",0)
        formats = {}
        portrait = landscape = 0
        hashes = {}
        dimensions = {}
        import hashlib
        
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".png",".jpg",".jpeg",".gif",".webp",".bmp",".tiff")):
                    total += 1
                    path = os.path.join(root, f)
                    sz = os.path.getsize(path)
                    size += sz
                    if sz > largest[1]: largest = (path, sz)
                    fmt = os.path.splitext(f)[1].lower()
                    formats[fmt] = formats.get(fmt,0)+1
                    from PIL import Image
                    try:
                        img = Image.open(path)
                        w,h = img.size
                        dimensions[path] = (w, h)
                        if w >= h: landscape += 1
                        else: portrait += 1
                        img.close()
                    except:
                        pass
                    try:
                        hsh = hashlib.md5(open(path,'rb').read()).hexdigest()
                        hashes.setdefault(hsh, []).append(path)
                    except:
                        pass
        
        dup_count = sum(len(v) for v in hashes.values() if len(v)>1)
        dup_size = sum(largest[1] for h,paths in hashes.items() if len(paths)>1 for path in paths[1:] if os.path.exists(path))
        
        fmt_lines = "\n".join([f"  {k}: {v} files" for k,v in sorted(formats.items())])
        
        # Calculate average imageresolution
        avg_width = sum(w for w,h in dimensions.values()) // len(dimensions) if dimensions else 0
        avg_height = sum(h for w,h in dimensions.values()) // len(dimensions) if dimensions else 0
        
        # Find largest/smallest files
        sorted_files_by_size = sorted([(os.path.getsize(os.path.join(r,f)), os.path.join(r,f)) for r,ds,fs in os.walk(folder) for f in fs if f.lower().endswith((".png",".jpg",".jpeg",".gif",".webp",".bmp",".tiff"))], reverse=True)
        top_files = "\n".join([f"  {os.path.getsize(p)//1024:>5} KB - {os.path.basename(p)}" for sz,p in sorted_files_by_size[:5]])
        
        info = (
            f"╔{'═'*60}╗\n"
            f"║ FOLDER ANALYSIS REPORT{' '*(60-22)}║\n"
            f"╚{'═'*60}╝\n\n"
            f"📁 Folder:\n   {folder}\n\n"
            f"📊 OVERALL STATISTICS:\n"
            f"  Total images: {total}\n"
            f"  Total size: {size//1024//1024} MB\n"
            f"  Average file size: {(size//total//1024) if total else 0} KB\n"
            f"  Average resolution: {avg_width}x{avg_height} px\n\n"
            f"🖼️ LARGEST FILES:\n{top_files}\n\n"
            f"📐 ORIENTATION:\n"
            f"  Landscape: {landscape} ({landscape*100//total if total else 0}%)\n"
            f"  Portrait: {portrait} ({portrait*100//total if total else 0}%)\n\n"
            f"📄 FORMATS:\n{fmt_lines}\n\n"
            f"🔀 DUPLICATES:\n"
            f"  Duplicate sets: {sum(1 for h,paths in hashes.items() if len(paths)>1)}\n"
            f"  Duplicate images: {dup_count}\n"
            f"  Space wasted: {dup_size//1024//1024} MB\n\n"
            f"{'═'*62}\n"
        )
        self.info_label.setPlainText(info)
        self.info_label.setStyleSheet("font-family: monospace; font-size: 11px; color: #E6EAF0;")

class FormatConverterPage(CardPage):
    def __init__(self):
        super().__init__("Format Converter")
        self.input_files = []
        self.output_dir = ""

        # Input
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        self.add_files_btn = QPushButton("Add Files")
        self.add_folder_btn = QPushButton("Add Folder")
        self.clear_btn = QPushButton("Clear")
        input_layout.addWidget(self.add_files_btn)
        input_layout.addWidget(self.add_folder_btn)
        input_layout.addWidget(self.clear_btn)
        input_group.setLayout(input_layout)

        self.file_list = QPlainTextEdit()
        self.file_list.setReadOnly(True)
        self.file_list.setPlaceholderText("No files selected")

        # Output
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        self.output_btn = QPushButton("Select Output Folder")
        self.output_label = QLabel("No output folder selected")
        output_layout.addWidget(self.output_btn)
        output_layout.addWidget(self.output_label)
        output_group.setLayout(output_layout)

        # Format settings
        format_group = QGroupBox("Format Settings")
        format_layout = QVBoxLayout()
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.format_combo = NoWheelComboBox()
        self.format_combo.addItems(["PNG", "JPG", "WEBP", "BMP", "TIFF", "ICO"])
        fmt_row.addWidget(self.format_combo)
        format_layout.addLayout(fmt_row)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Quality:"))
        self.quality_slider = NoWheelSlider(Qt.Horizontal)
        self.quality_slider.setRange(10, 100)
        self.quality_slider.setValue(90)
        self.quality_label = QLabel("90")
        quality_row.addWidget(self.quality_slider)
        quality_row.addWidget(self.quality_label)
        format_layout.addLayout(quality_row)

        ico_row = QHBoxLayout()
        ico_row.addWidget(QLabel("ICO Sizes:"))
        self.ico_sizes = QLineEdit("16,32,48,64,128,256")
        ico_row.addWidget(self.ico_sizes)
        format_layout.addLayout(ico_row)

        format_group.setLayout(format_layout)

        # Progress and action
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.convert_btn = QPushButton("Convert")

        # Layout
        self.card_layout.addWidget(input_group)
        self.card_layout.addWidget(self.file_list)
        self.card_layout.addWidget(output_group)
        self.card_layout.addWidget(format_group)
        self.card_layout.addWidget(self.progress)
        self.card_layout.addWidget(self.convert_btn)

        # Connections
        self.add_files_btn.clicked.connect(self.add_files)
        self.add_folder_btn.clicked.connect(self.add_folder)
        self.clear_btn.clicked.connect(self.clear_files)
        self.output_btn.clicked.connect(self.select_output)
        self.convert_btn.clicked.connect(self.convert_files)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_label.setText(str(v)))

    def add_files(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("FormatConverterPage_last_input_folder", "")
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            last_folder,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.gif);;All Files (*.*)"
        )
        if files:
            settings.setValue("FormatConverterPage_last_input_folder", os.path.dirname(files[0]))
            self.input_files.extend(files)
            self.refresh_file_list()

    def add_folder(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("FormatConverterPage_last_folder_input", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", last_folder)
        if not folder:
            return
        settings.setValue("FormatConverterPage_last_folder_input", folder)
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif")):
                    self.input_files.append(os.path.join(root, f))
        self.refresh_file_list()

    def clear_files(self):
        self.input_files = []
        self.refresh_file_list()

    def refresh_file_list(self):
        if not self.input_files:
            self.file_list.setPlainText("")
            self.file_list.setPlaceholderText("No files selected")
            return
        self.file_list.setPlainText("\n".join(self.input_files))

    def select_output(self):
        settings = QSettings("PixelForge", "PixelForge")
        last_folder = settings.value("FormatConverterPage_last_output_folder", "")
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", last_folder)
        if folder:
            self.output_dir = folder
            settings.setValue("FormatConverterPage_last_output_folder", folder)
            self.output_label.setText(folder)

    def convert_files(self):
        if not self.input_files:
            QMessageBox.warning(self, "No Files", "Please add files to convert.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "No Output", "Please select an output folder.")
            return

        fmt = self.format_combo.currentText().lower()
        quality = self.quality_slider.value()
        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.input_files))
        self.progress.setValue(0)

        for idx, fpath in enumerate(self.input_files, start=1):
            try:
                img = Image.open(fpath)
                base = os.path.splitext(os.path.basename(fpath))[0]

                if fmt == "jpg":
                    out_path = os.path.join(self.output_dir, f"{base}.jpg")
                    img.convert("RGB").save(out_path, "JPEG", quality=quality, optimize=True)
                elif fmt == "webp":
                    out_path = os.path.join(self.output_dir, f"{base}.webp")
                    img.save(out_path, "WEBP", quality=quality, method=6)
                elif fmt == "ico":
                    out_path = os.path.join(self.output_dir, f"{base}.ico")
                    sizes = [int(s.strip()) for s in self.ico_sizes.text().split(",") if s.strip().isdigit()]
                    if not sizes:
                        sizes = [16, 32, 48, 64, 128, 256]
                    img_rgba = img.convert("RGBA")
                    img_rgba.save(out_path, format="ICO", sizes=[(s, s) for s in sizes])
                else:
                    # Generic format save with optimization for PNG
                    out_path = os.path.join(self.output_dir, f"{base}.{fmt}")
                    if fmt == "png":
                        img.save(out_path, optimize=True)
                    else:
                        img.save(out_path)
            except Exception as e:
                print(f"Convert failed for {fpath}: {e}")
            self.progress.setValue(idx)

        QMessageBox.information(self, "Done", "Conversion complete!")
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
        self.settings = QSettings("PixelForge", "PixelForgeUtilities")
        self.setWindowTitle("PixelForge - Utilities")
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

        self.tool_names = ["Rename Tool", "Folder Analyzer", "Format Converter"]

        self.stack = QStackedWidget()
        self.pages = {
            "Home": self._build_home_page(),
            "Rename Tool": RenameToolPage(),
            "Folder Analyzer": FolderAnalyzerPage(),
            "Format Converter": FormatConverterPage(),
        }
        for page in self.pages.values():
            self.stack.addWidget(page)

        self.sidebar_buttons = {}
        category_label = QLabel("Utilities")
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
        self.made_label.setStyleSheet("font-size: 11px; padding: 10px;")
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
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        self.home_title = QLabel("Utilities Workspace")
        self.home_title.setStyleSheet("font-size: 26px; font-weight: bold;")
        subtitle = QLabel("File cleanup and conversion toolkit.")
        subtitle.setWordWrap(True)

        theme_row = QHBoxLayout()
        theme_row.addWidget(QLabel("Theme"))
        self.home_theme_combo = QComboBox()
        self.home_theme_combo.addItems(list(self.THEME_PRESETS.keys()))
        self.home_theme_combo.currentTextChanged.connect(self.apply_theme)
        theme_row.addWidget(self.home_theme_combo)
        theme_row.addStretch(1)

        highlights = QLabel(
            "Includes Rename Tool, Folder Analyzer, and Format Converter."
        )
        highlights.setWordWrap(True)

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
        layout.addWidget(highlights)
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
            self.home_title.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {accent};")
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
