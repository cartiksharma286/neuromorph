#!/usr/bin/env python3
"""
DICOM Neuroimage Viewer - PyQt5 Application
Advanced neuroimaging viewer with ITK processing and VTK visualization
"""

import sys
import os
import numpy as np
from pathlib import Path
from threading import Thread
import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QTabWidget, QStatusBar, QMenuBar, QMenu,
    QProgressBar, QGroupBox, QGridLayout, QScrollArea, QCheckBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QFrame, QListWidget, QListWidgetItem
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import custom modules
from dicom_processor import DICOMProcessor
from dicom_downloader import DICOMDownloader
from vtk_visualizer import VTKNeuroimageVisualizer
import itk


class WorkerSignals(QObject):
    """Signals for background operations"""
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    progress = pyqtSignal(str)


class DICOMViewer(QMainWindow):
    """Main DICOM Neuroimage Viewer Application"""
    
    def __init__(self):
        """Initialize the application"""
        super().__init__()
        self.setWindowTitle("DICOM Neuroimage Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.processor = DICOMProcessor()
        self.downloader = DICOMDownloader()
        self.visualizer = VTKNeuroimageVisualizer()
        
        # UI state
        self.current_image = None
        self.current_slice = 0
        self.num_slices = 0
        self.image_array = None
        
        # Create UI
        self._create_ui()
        self._create_menu()
        self._create_status_bar()
        
        print("✓ DICOM Neuroimage Viewer initialized")
    
    def _create_ui(self):
        """Create user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel: Image display and controls
        left_layout = QVBoxLayout()
        
        # Tab widget for 2D and 3D
        self.tabs = QTabWidget()
        
        # 2D viewer tab
        self.tab_2d = QWidget()
        self.tab_2d_layout = QVBoxLayout()
        
        # Matplotlib figure for 2D display
        self.figure_2d = Figure(figsize=(6, 6), dpi=100)
        self.canvas_2d = FigureCanvas(self.figure_2d)
        self.tab_2d_layout.addWidget(self.canvas_2d)
        
        self.tab_2d.setLayout(self.tab_2d_layout)
        self.tabs.addTab(self.tab_2d, "2D Viewer")
        
        # 3D viewer tab (placeholder)
        self.tab_3d = QWidget()
        self.tab_3d_layout = QVBoxLayout()
        self.tab_3d_label = QLabel("3D visualization available when VTK is loaded")
        self.tab_3d_label.setAlignment(Qt.AlignCenter)
        self.tab_3d_layout.addWidget(self.tab_3d_label)
        self.tab_3d.setLayout(self.tab_3d_layout)
        self.tabs.addTab(self.tab_3d, "3D Viewer")
        
        left_layout.addWidget(self.tabs)
        
        # Slice navigation
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        nav_layout.addWidget(self.slice_slider)
        
        self.slice_label = QLabel("0/0")
        nav_layout.addWidget(self.slice_label)
        left_layout.addLayout(nav_layout)
        
        # Right panel: Controls and parameters
        right_layout = QVBoxLayout()
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        
        self.btn_load_file = QPushButton("Load DICOM File")
        self.btn_load_file.clicked.connect(self._on_load_file)
        file_layout.addWidget(self.btn_load_file)
        
        self.btn_load_series = QPushButton("Load DICOM Series")
        self.btn_load_series.clicked.connect(self._on_load_series)
        file_layout.addWidget(self.btn_load_series)
        
        self.btn_create_sample = QPushButton("Create Sample Brain")
        self.btn_create_sample.clicked.connect(self._on_create_sample)
        file_layout.addWidget(self.btn_create_sample)
        
        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)
        
        # Filters and processing
        filter_group = QGroupBox("Image Processing")
        filter_layout = QGridLayout()
        
        # Gaussian smoothing
        filter_layout.addWidget(QLabel("Gaussian Sigma:"), 0, 0)
        self.spin_gaussian = QDoubleSpinBox()
        self.spin_gaussian.setRange(0.1, 10.0)
        self.spin_gaussian.setValue(1.0)
        self.spin_gaussian.setSingleStep(0.1)
        filter_layout.addWidget(self.spin_gaussian, 0, 1)
        
        btn_gaussian = QPushButton("Apply Gaussian")
        btn_gaussian.clicked.connect(
            lambda: self._apply_filter('gaussian', {'sigma': self.spin_gaussian.value()})
        )
        filter_layout.addWidget(btn_gaussian, 0, 2)
        
        # Median filter
        filter_layout.addWidget(QLabel("Median Radius:"), 1, 0)
        self.spin_median = QSpinBox()
        self.spin_median.setRange(1, 10)
        self.spin_median.setValue(2)
        filter_layout.addWidget(self.spin_median, 1, 1)
        
        btn_median = QPushButton("Apply Median")
        btn_median.clicked.connect(
            lambda: self._apply_filter('median', {'radius': self.spin_median.value()})
        )
        filter_layout.addWidget(btn_median, 1, 2)
        
        # Bilateral filter
        filter_layout.addWidget(QLabel("Bilateral Domain σ:"), 2, 0)
        self.spin_bilateral_domain = QDoubleSpinBox()
        self.spin_bilateral_domain.setRange(0.5, 10.0)
        self.spin_bilateral_domain.setValue(2.0)
        filter_layout.addWidget(self.spin_bilateral_domain, 2, 1)
        
        filter_layout.addWidget(QLabel("Bilateral Range σ:"), 3, 0)
        self.spin_bilateral_range = QDoubleSpinBox()
        self.spin_bilateral_range.setRange(10, 100)
        self.spin_bilateral_range.setValue(50.0)
        filter_layout.addWidget(self.spin_bilateral_range, 3, 1)
        
        btn_bilateral = QPushButton("Apply Bilateral")
        btn_bilateral.clicked.connect(
            lambda: self._apply_filter('bilateral', {
                'domain_sigma': self.spin_bilateral_domain.value(),
                'range_sigma': self.spin_bilateral_range.value()
            })
        )
        filter_layout.addWidget(btn_bilateral, 3, 2)
        
        # Histogram equalization
        filter_layout.addWidget(QLabel("AHE Radius:"), 4, 0)
        self.spin_ahe = QSpinBox()
        self.spin_ahe.setRange(10, 100)
        self.spin_ahe.setValue(50)
        filter_layout.addWidget(self.spin_ahe, 4, 1)
        
        btn_ahe = QPushButton("Apply AHE")
        btn_ahe.clicked.connect(
            lambda: self._apply_filter('ahe', {'radius': self.spin_ahe.value()})
        )
        filter_layout.addWidget(btn_ahe, 4, 2)
        
        filter_group.setLayout(filter_layout)
        right_layout.addWidget(filter_group)
        
        # Thresholding
        threshold_group = QGroupBox("Thresholding")
        threshold_layout = QGridLayout()
        
        threshold_layout.addWidget(QLabel("Lower Threshold:"), 0, 0)
        self.spin_thresh_lower = QSpinBox()
        self.spin_thresh_lower.setRange(0, 255)
        self.spin_thresh_lower.setValue(0)
        threshold_layout.addWidget(self.spin_thresh_lower, 0, 1)
        
        threshold_layout.addWidget(QLabel("Upper Threshold:"), 1, 0)
        self.spin_thresh_upper = QSpinBox()
        self.spin_thresh_upper.setRange(0, 255)
        self.spin_thresh_upper.setValue(255)
        threshold_layout.addWidget(self.spin_thresh_upper, 1, 1)
        
        btn_threshold = QPushButton("Apply Threshold")
        btn_threshold.clicked.connect(
            lambda: self._apply_filter('threshold', {
                'lower': self.spin_thresh_lower.value(),
                'upper': self.spin_thresh_upper.value()
            })
        )
        threshold_layout.addWidget(btn_threshold, 2, 0, 1, 2)
        
        threshold_group.setLayout(threshold_layout)
        right_layout.addWidget(threshold_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        display_layout.addWidget(QLabel("Window/Level Controls:"))
        
        display_layout.addWidget(QLabel("Window Width:"))
        self.slider_window = QSlider(Qt.Horizontal)
        self.slider_window.setRange(1, 4095)
        self.slider_window.setValue(500)
        self.slider_window.valueChanged.connect(self._on_display_update)
        display_layout.addWidget(self.slider_window)
        
        display_layout.addWidget(QLabel("Window Level:"))
        self.slider_level = QSlider(Qt.Horizontal)
        self.slider_level.setRange(0, 4095)
        self.slider_level.setValue(2048)
        self.slider_level.valueChanged.connect(self._on_display_update)
        display_layout.addWidget(self.slider_level)
        
        display_group.setLayout(display_layout)
        right_layout.addWidget(display_group)
        
        # Reset and info
        btn_reset = QPushButton("Reset to Original")
        btn_reset.clicked.connect(self._on_reset)
        right_layout.addWidget(btn_reset)
        
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet("color: gray; font-size: 10px;")
        right_layout.addWidget(self.info_label)
        
        right_layout.addStretch()
        
        # Combine layouts
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        
        central_widget.setLayout(main_layout)
    
    def _create_menu(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        action_load_file = file_menu.addAction("Load DICOM File")
        action_load_file.triggered.connect(self._on_load_file)
        
        action_load_series = file_menu.addAction("Load DICOM Series")
        action_load_series.triggered.connect(self._on_load_series)
        
        file_menu.addSeparator()
        action_exit = file_menu.addAction("Exit")
        action_exit.triggered.connect(self.close)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        action_sample = tools_menu.addAction("Create Sample Brain")
        action_sample.triggered.connect(self._on_create_sample)
        
        tools_menu.addSeparator()
        action_repositories = tools_menu.addAction("DICOM Repositories")
        action_repositories.triggered.connect(self._on_show_repositories)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        action_about = help_menu.addAction("About")
        action_about.triggered.connect(self._on_about)
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")
    
    def _on_load_file(self):
        """Load single DICOM file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM File", "",
            "DICOM Files (*.dcm *.dicom);;All Files (*)"
        )
        if file_path:
            self._load_image(file_path)
    
    def _on_load_series(self):
        """Load DICOM series from directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select DICOM Series Directory"
        )
        if dir_path:
            if self.processor.load_dicom_series(dir_path):
                self._update_display()
                self.status.showMessage(f"Loaded DICOM series from {dir_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to load DICOM series")
    
    def _on_create_sample(self):
        """Create and load sample brain DICOM"""
        self.status.showMessage("Creating sample brain DICOM...")
        QApplication.processEvents()
        
        try:
            sample_dir = os.path.join(
                os.path.expanduser("~"),
                ".dicom_viewer",
                "sample_brain"
            )
            os.makedirs(sample_dir, exist_ok=True)
            
            dicom_files = self.downloader.create_sample_brain_dicom(
                sample_dir,
                size=(256, 256, 64)
            )
            
            if dicom_files:
                self.processor.load_dicom_series(sample_dir)
                self._update_display()
                self.status.showMessage("Sample brain DICOM created and loaded")
                QMessageBox.information(
                    self, "Success",
                    f"Created sample brain with {len(dicom_files)} slices"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to create sample brain")
        except Exception as e:
            print(f"Error creating sample: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to create sample:\n{str(e)}")
    
    def _load_image(self, file_path):
        """Load image from file"""
        self.status.showMessage(f"Loading {Path(file_path).name}...")
        QApplication.processEvents()
        
        if self.processor.load_dicom_file(file_path):
            self._update_display()
            self.status.showMessage(f"Loaded {Path(file_path).name}")
        else:
            QMessageBox.warning(self, "Error", "Failed to load DICOM file")
    
    def _update_display(self):
        """Update 2D and 3D displays"""
        self.image_array = self.processor.get_image_array()
        
        if self.image_array is not None:
            # Handle 2D and 3D images
            if self.image_array.ndim == 3:
                self.num_slices = self.image_array.shape[0]
                self.slice_slider.setMaximum(self.num_slices - 1)
                self.slice_slider.setValue(self.num_slices // 2)
                self.current_slice = self.num_slices // 2
            else:
                self.num_slices = 1
                self.slice_slider.setMaximum(0)
                self.current_slice = 0
            
            self._update_2d_display()
        
        # Update image info
        stats = self.processor.get_image_statistics()
        if stats:
            info_text = (
                f"Size: {self.processor.image.GetLargestPossibleRegion().GetSize()} | "
                f"Mean: {stats['mean']:.1f} | "
                f"Std: {stats['std']:.1f} | "
                f"Min: {stats['min']:.1f} | "
                f"Max: {stats['max']:.1f}"
            )
            self.info_label.setText(info_text)
    
    def _update_2d_display(self):
        """Update 2D slice display"""
        if self.image_array is None:
            return
        
        # Get current slice
        if self.image_array.ndim == 3:
            slice_data = self.image_array[self.current_slice, :, :]
        else:
            slice_data = self.image_array
        
        # Apply window/level
        window = self.slider_window.value()
        level = self.slider_level.value()
        
        lower = level - window / 2
        upper = level + window / 2
        
        display_data = np.clip(slice_data, lower, upper)
        display_data = ((display_data - lower) / (upper - lower) * 255).astype(np.uint8)
        
        # Display
        self.figure_2d.clear()
        ax = self.figure_2d.add_subplot(111)
        ax.imshow(display_data, cmap='gray', origin='upper')
        ax.set_title(f"Slice {self.current_slice + 1}/{self.num_slices}")
        ax.axis('off')
        self.canvas_2d.draw()
        
        self.slice_label.setText(f"{self.current_slice + 1}/{self.num_slices}")
    
    def _on_slice_changed(self, value):
        """Handle slice slider change"""
        self.current_slice = value
        self._update_2d_display()
    
    def _on_display_update(self):
        """Handle display parameter changes"""
        self._update_2d_display()
    
    def _apply_filter(self, filter_type, params):
        """Apply image processing filter"""
        self.status.showMessage(f"Applying {filter_type} filter...")
        QApplication.processEvents()
        
        try:
            if filter_type == 'gaussian':
                self.processor.apply_gaussian_smoothing(params['sigma'])
            elif filter_type == 'median':
                self.processor.apply_median_filter(params['radius'])
            elif filter_type == 'bilateral':
                self.processor.apply_bilateral_filter(
                    params['domain_sigma'],
                    params['range_sigma']
                )
            elif filter_type == 'ahe':
                self.processor.apply_adaptive_histogram_equalization(params['radius'])
            elif filter_type == 'threshold':
                self.processor.apply_threshold(params['lower'], params['upper'])
            
            self._update_display()
            self.status.showMessage("Filter applied successfully")
        except Exception as e:
            print(f"Filter error: {e}")
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to apply filter:\n{str(e)}")
    
    def _on_reset(self):
        """Reset to original image"""
        if self.processor.reset_to_original():
            self._update_display()
            self.status.showMessage("Reset to original image")
        else:
            QMessageBox.warning(self, "Error", "No original image to reset to")
    
    def _on_show_repositories(self):
        """Show available DICOM repositories"""
        self.downloader.list_repositories()
        self.downloader.get_sample_dicom_info()
        QMessageBox.information(
            self, "DICOM Repositories",
            "Available repositories and sources have been printed to console.\n\n"
            "Popular sources:\n"
            "- BRATS: Brain tumor datasets\n"
            "- Kaggle: Medical imaging datasets\n"
            "- NCBI: National Center for Biotechnology Information\n"
            "- NIH: National Institutes of Health imaging archive"
        )
    
    def _on_about(self):
        """Show about dialog"""
        QMessageBox.information(
            self, "About",
            "DICOM Neuroimage Viewer\n"
            "Advanced neuroimaging analysis tool\n\n"
            "Components:\n"
            "- ITK: Medical image processing\n"
            "- VTK: 3D visualization\n"
            "- PyQt5: User interface\n\n"
            "Features:\n"
            "- Load DICOM files and series\n"
            "- Advanced image filtering\n"
            "- Window/level adjustment\n"
            "- 2D and 3D visualization"
        )


def main():
    """Main entry point"""
    try:
        # Check for pydicom (optional, for synthetic DICOM creation)
        try:
            import pydicom
        except ImportError:
            print("⚠ pydicom not found. Run: pip3 install pydicom")
            print("  (This is needed to create sample DICOM files)")
        
        # Create and run application
        app = QApplication(sys.argv)
        viewer = DICOMViewer()
        viewer.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"✗ Application error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
