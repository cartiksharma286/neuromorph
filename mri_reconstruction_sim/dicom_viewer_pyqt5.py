#!/usr/bin/env python3
"""
DICOM Neuroimage Viewer with PyQt5 Frontend
Features:
- Load DICOM images from web or local files
- ITK/VTK image processing pipelines
- Interactive sliders for visualization control
- Real-time image filtering and segmentation
- 3D surface rendering
"""

import sys
import os
import urllib.request
import threading
import ssl
import warnings
from pathlib import Path

# Suppress SSL warnings
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore', message='.*urllib3.*')

# ITK and medical imaging
import itk
import numpy as np
import pydicom
from PIL import Image

# PyQt5
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QSpinBox, QPushButton, QComboBox, QProgressBar,
    QTabWidget, QAction, QFileDialog, QMessageBox, QInputDialog,
    QScrollArea, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon

# VTK
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Sample DICOM sources
SAMPLE_DICOM_URLS = {
    "Brain MRI (T1)": "https://example.com/brain_t1.dcm",  # Placeholder
    "Brain MRI (T2)": "https://example.com/brain_t2.dcm",  # Placeholder
    "Brain MRI (FLAIR)": "https://example.com/brain_flair.dcm",  # Placeholder
}

SAMPLE_DATASETS = {
    "Synthetic Brain": None,  # Will be generated
}


class DICOMLoader(QThread):
    """Background thread for loading DICOM images"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Load DICOM file
            if isinstance(self.file_path, str) and self.file_path.startswith('http'):
                # Download from web
                self.progress.emit(20)
                temp_file = "/tmp/dicom_temp.dcm"
                urllib.request.urlretrieve(self.file_path, temp_file)
                dcm_file = temp_file
                self.progress.emit(50)
            else:
                dcm_file = self.file_path
            
            # Load with pydicom
            dicom_data = pydicom.dcmread(dcm_file)
            self.progress.emit(70)
            
            # Extract pixel array
            pixel_array = dicom_data.pixel_array.astype(np.float32)
            
            # Normalize
            if pixel_array.max() > 0:
                pixel_array = (pixel_array / pixel_array.max()) * 255
            
            self.progress.emit(90)
            self.finished.emit((pixel_array, dicom_data))
            self.progress.emit(100)
            
        except Exception as e:
            self.error.emit(f"Error loading DICOM: {str(e)}")


class NeuroImageProcessor:
    """ITK-based image processing for neuroimages"""
    
    def __init__(self):
        self.image = None
        self.original_image = None
        self.ImageType = itk.Image[itk.F, 3]
    
    def load_pixel_array(self, pixel_array):
        """Convert numpy array to ITK image"""
        itk_image = itk.GetImageFromArray(pixel_array.astype(np.float32))
        self.original_image = itk_image
        self.image = itk_image
        return itk_image
    
    def apply_gaussian(self, sigma=1.0):
        """Apply Gaussian smoothing"""
        if self.image is None:
            return None
        
        gaussian = itk.SmoothingRecursiveGaussianImageFilter.New(
            Input=self.original_image,
            Sigma=sigma
        )
        gaussian.Update()
        self.image = gaussian.GetOutput()
        return self.image
    
    def apply_median(self, radius=1):
        """Apply median filter"""
        if self.image is None:
            return None
        
        median = itk.MedianImageFilter.New(
            Input=self.original_image,
            Radius=radius
        )
        median.Update()
        self.image = median.GetOutput()
        return self.image
    
    def apply_threshold(self, threshold=128):
        """Apply threshold filter"""
        if self.image is None:
            return None
        
        threshold_filter = itk.ThresholdImageFilter.New(Input=self.original_image)
        threshold_filter.SetLower(threshold)
        threshold_filter.SetUpper(255)
        threshold_filter.SetOutsideValue(0)
        threshold_filter.Update()
        self.image = threshold_filter.GetOutput()
        return self.image
    
    def get_slice(self, axis=2, index=0):
        """Extract 2D slice from 3D image"""
        if self.image is None:
            return None
        
        # Convert ITK to numpy
        array = itk.GetArrayFromImage(self.image)
        
        if array.ndim == 3:
            if axis == 0:
                return array[index, :, :]
            elif axis == 1:
                return array[:, index, :]
            else:  # axis == 2
                return array[:, :, index]
        else:
            return array
    
    def get_image_array(self):
        """Get current image as numpy array"""
        if self.image is None:
            return None
        return itk.GetArrayFromImage(self.image)


class VTK3DViewer(QWidget):
    """VTK-based 3D visualization widget"""
    
    def __init__(self):
        super().__init__()
        self.vtk_widget = QVTKRenderWindowInteractor()
        layout = QVBoxLayout()
        layout.addWidget(self.vtk_widget)
        self.setLayout(layout)
        
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.15)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
    
    def render_volume(self, vtk_image):
        """Render volume from VTK image"""
        self.renderer.RemoveAllViewProps()
        
        # Volume mapper
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(vtk_image)
        
        # Volume property
        volume_property = vtk.vtkVolumeProperty()
        volume_property.ShadeingOn()
        
        # Create volume
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(volume_property)
        
        self.renderer.AddActor(volume)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    def render_surface(self, vtk_image, threshold=128):
        """Render surface using marching cubes"""
        self.renderer.RemoveAllViewProps()
        
        # Contour filter
        contour = vtk.vtkContourFilter()
        contour.SetInputData(vtk_image)
        contour.SetValue(0, threshold)
        
        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        
        # Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.6, 0.4)
        
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()


class DICOMNeuroimageViewer(QMainWindow):
    """Main PyQt5 application for viewing DICOM neuroimages"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Neuroimage Viewer - PyQt5 + ITK + VTK")
        self.setGeometry(50, 50, 1400, 900)
        
        # Initialize processor
        self.processor = NeuroImageProcessor()
        self.current_image = None
        self.current_dicom = None
        self.current_slice = 0
        
        # Create UI
        self.create_menu()
        self.create_ui()
        
    def create_menu(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load DICOM...", self)
        load_action.triggered.connect(self.load_dicom_file)
        file_menu.addAction(load_action)
        
        load_web_action = QAction("Load from Web...", self)
        load_web_action.triggered.connect(self.load_dicom_web)
        file_menu.addAction(load_web_action)
        
        synthetic_action = QAction("Create Synthetic Image", self)
        synthetic_action.triggered.connect(self.create_synthetic_image)
        file_menu.addAction(synthetic_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_ui(self):
        """Create main UI"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        
        # Left panel - Controls
        left_panel = self.create_control_panel()
        left_panel.setMaximumWidth(300)
        
        # Right panel - Viewers
        right_panel = self.create_viewer_panel()
        
        main_layout.addWidget(left_panel, 0)
        main_layout.addWidget(right_panel, 1)
        
        main_widget.setLayout(main_layout)
    
    def create_control_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Slice control
        slice_group = QGroupBox("Slice Control")
        slice_layout = QVBoxLayout()
        
        slice_layout.addWidget(QLabel("Current Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.valueChanged.connect(self.update_slice)
        slice_layout.addWidget(self.slice_slider)
        
        self.slice_label = QLabel("Slice: 0/100")
        slice_layout.addWidget(self.slice_label)
        
        slice_group.setLayout(slice_layout)
        layout.addWidget(slice_group)
        
        # Image processing
        process_group = QGroupBox("Image Processing")
        process_layout = QVBoxLayout()
        
        # Gaussian
        process_layout.addWidget(QLabel("Gaussian Smoothing:"))
        self.gaussian_slider = QSlider(Qt.Horizontal)
        self.gaussian_slider.setMinimum(0)
        self.gaussian_slider.setMaximum(50)
        self.gaussian_slider.setValue(10)
        self.gaussian_slider.sliderMoved.connect(self.apply_gaussian)
        self.gaussian_slider.setTracking(False)
        process_layout.addWidget(self.gaussian_slider)
        
        self.gaussian_label = QLabel("Sigma: 1.0")
        process_layout.addWidget(self.gaussian_label)
        
        # Median filter
        process_layout.addWidget(QLabel("Median Filter Radius:"))
        self.median_slider = QSlider(Qt.Horizontal)
        self.median_slider.setMinimum(0)
        self.median_slider.setMaximum(10)
        self.median_slider.setValue(0)
        self.median_slider.sliderMoved.connect(self.apply_median)
        self.median_slider.setTracking(False)
        process_layout.addWidget(self.median_slider)
        
        self.median_label = QLabel("Radius: 0")
        process_layout.addWidget(self.median_label)
        
        # Threshold
        process_layout.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(128)
        self.threshold_slider.sliderMoved.connect(self.apply_threshold)
        self.threshold_slider.setTracking(False)
        process_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("Threshold: 128")
        process_layout.addWidget(self.threshold_label)
        
        # Reset button
        reset_btn = QPushButton("Reset Processing")
        reset_btn.clicked.connect(self.reset_processing)
        process_layout.addWidget(reset_btn)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        display_layout.addWidget(QLabel("Rendering Mode:"))
        self.render_combo = QComboBox()
        self.render_combo.addItems(["2D Slice", "3D Volume", "3D Surface"])
        self.render_combo.currentTextChanged.connect(self.change_render_mode)
        display_layout.addWidget(self.render_combo)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_viewer_panel(self):
        """Create right viewer panel with tabs"""
        self.tabs = QTabWidget()
        
        # 2D viewer
        self.image_2d = QLabel()
        self.image_2d.setStyleSheet("border: 1px solid gray; background-color: #2a2a2a")
        self.image_2d.setMinimumSize(600, 600)
        self.image_2d.setAlignment(Qt.AlignCenter)
        self.tabs.addTab(self.image_2d, "2D Viewer")
        
        # 3D VTK viewer (with fallback if initialization fails)
        try:
            self.vtk_viewer = VTK3DViewer()
            self.tabs.addTab(self.vtk_viewer, "3D Viewer")
        except Exception as e:
            print(f"Warning: Could not initialize VTK viewer: {e}")
            # Fallback: empty widget
            fallback = QLabel("3D Viewer not available on this system")
            fallback.setAlignment(Qt.AlignCenter)
            self.vtk_viewer = None
            self.tabs.addTab(fallback, "3D Viewer (Disabled)")
        
        return self.tabs
    
    def create_synthetic_image(self):
        """Create synthetic brain-like image"""
        print("Creating synthetic brain image...")
        
        # Create 3D Gaussian blobs for synthetic brain
        size = (128, 128, 64)
        array = np.zeros(size, dtype=np.float32)
        
        # Add multiple Gaussian blobs
        centers = [
            (64, 64, 32, 20),   # Left hemisphere (x, y, z, sigma)
            (64, 64, 32, 20),   # Right hemisphere (mirrored)
            (64, 64, 20, 10),   # Ventricles
        ]
        
        for x, y, z, sigma in centers:
            for i in range(size[0]):
                for j in range(size[1]):
                    for k in range(size[2]):
                        dx = i - x
                        dy = j - y
                        dz = k - z
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        array[i, j, k] += 200 * np.exp(-(dist**2) / (2 * sigma**2))
        
        array = np.clip(array, 0, 255).astype(np.float32)
        
        # Load into processor
        self.processor.load_pixel_array(array)
        self.current_image = array
        self.current_slice = size[2] // 2
        
        # Update UI
        self.slice_slider.setMaximum(size[2] - 1)
        self.slice_label.setText(f"Slice: {self.current_slice}/{size[2]-1}")
        
        self.update_display()
        QMessageBox.information(self, "Success", f"Created synthetic image: {size}")
    
    def load_dicom_file(self):
        """Load DICOM file from disk"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM File", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        
        if file_path:
            self.load_dicom_internal(file_path)
    
    def load_dicom_web(self):
        """Load DICOM from web URL"""
        url, ok = QInputDialog.getText(
            self, "Load from Web", "Enter DICOM URL:"
        )
        
        if ok and url:
            self.load_dicom_internal(url)
    
    def load_dicom_internal(self, source):
        """Internal method to load DICOM"""
        self.progress_bar.setVisible(True)
        
        loader = DICOMLoader(source)
        loader.progress.connect(self.progress_bar.setValue)
        loader.finished.connect(self.on_dicom_loaded)
        loader.error.connect(self.on_dicom_error)
        loader.start()
    
    def on_dicom_loaded(self, data):
        """Handle loaded DICOM data"""
        pixel_array, dicom_data = data
        
        self.current_image = pixel_array
        self.current_dicom = dicom_data
        
        # Load into processor
        self.processor.load_pixel_array(pixel_array)
        
        # Update slice control
        if pixel_array.ndim == 3:
            num_slices = pixel_array.shape[2]
        else:
            num_slices = 1
        
        self.slice_slider.setMaximum(max(0, num_slices - 1))
        self.current_slice = num_slices // 2 if num_slices > 1 else 0
        
        self.update_display()
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(
            self, "Success",
            f"Loaded DICOM image: {pixel_array.shape}\n"
            f"Min: {pixel_array.min():.1f}, Max: {pixel_array.max():.1f}"
        )
    
    def on_dicom_error(self, error):
        """Handle DICOM loading error"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", error)
    
    def update_slice(self, value):
        """Update current slice"""
        self.current_slice = value
        self.slice_label.setText(f"Slice: {self.current_slice}/{self.slice_slider.maximum()}")
        self.update_display()
    
    def apply_gaussian(self):
        """Apply Gaussian filter"""
        sigma = self.gaussian_slider.value() / 10.0
        self.gaussian_label.setText(f"Sigma: {sigma:.1f}")
        self.processor.apply_gaussian(sigma)
        self.update_display()
    
    def apply_median(self):
        """Apply median filter"""
        radius = self.median_slider.value()
        self.median_label.setText(f"Radius: {radius}")
        if radius > 0:
            self.processor.apply_median(radius)
        self.update_display()
    
    def apply_threshold(self):
        """Apply threshold"""
        threshold = self.threshold_slider.value()
        self.threshold_label.setText(f"Threshold: {threshold}")
        self.processor.apply_threshold(threshold)
        self.update_display()
    
    def reset_processing(self):
        """Reset to original image"""
        self.processor.image = self.processor.original_image
        self.gaussian_slider.setValue(10)
        self.median_slider.setValue(0)
        self.threshold_slider.setValue(128)
        self.update_display()
    
    def change_render_mode(self, mode):
        """Change rendering mode"""
        if self.current_image is not None:
            if mode == "2D Slice":
                self.tabs.setCurrentIndex(0)
                self.update_display()
            elif mode == "3D Volume":
                self.tabs.setCurrentIndex(1)
                self.render_3d_volume()
            elif mode == "3D Surface":
                self.tabs.setCurrentIndex(1)
                self.render_3d_surface()
    
    def update_display(self):
        """Update 2D display"""
        if self.current_image is not None:
            image_array = self.processor.get_image_array()
            
            if image_array.ndim == 3:
                slice_data = image_array[:, :, self.current_slice]
            else:
                slice_data = image_array
            
            # Normalize to 0-255
            if slice_data.max() > 0:
                slice_data = (slice_data / slice_data.max() * 255).astype(np.uint8)
            
            # Convert to RGB using PIL
            h, w = slice_data.shape
            grayscale_img = Image.fromarray(slice_data, mode='L')
            rgb_img = grayscale_img.convert('RGB')
            
            # Convert to QImage
            q_image = QImage(
                rgb_img.tobytes(), w, h,
                QImage.Format_RGB888
            )
            
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaledToWidth(600, Qt.SmoothTransformation)
            self.image_2d.setPixmap(scaled)
    
    def render_3d_volume(self):
        """Render 3D volume"""
        if self.current_image is None or self.vtk_viewer is None:
            QMessageBox.warning(self, "Not Available", "3D visualization not available on this system")
            return
        
        try:
            image_array = self.processor.get_image_array()
            
            # Convert to VTK
            itk_image = itk.GetImageFromArray(image_array.astype(np.float32))
            vtk_converter = itk.ImageToVTKImageFilter[type(itk_image)].New(Input=itk_image)
            vtk_converter.Update()
            vtk_image = vtk_converter.GetOutput()
            
            self.vtk_viewer.render_volume(vtk_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not render 3D volume: {str(e)}")
    
    def render_3d_surface(self):
        """Render 3D surface"""
        if self.current_image is None or self.vtk_viewer is None:
            QMessageBox.warning(self, "Not Available", "3D visualization not available on this system")
            return
        
        try:
            image_array = self.processor.get_image_array()
            
            # Convert to VTK
            itk_image = itk.GetImageFromArray(image_array.astype(np.float32))
            vtk_converter = itk.ImageToVTKImageFilter[type(itk_image)].New(Input=itk_image)
            vtk_converter.Update()
            vtk_image = vtk_converter.GetOutput()
            
            threshold = self.threshold_slider.value()
            self.vtk_viewer.render_surface(vtk_image, threshold)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not render 3D surface: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About DICOM Neuroimage Viewer",
            "DICOM Neuroimage Viewer v1.0\n\n"
            "Features:\n"
            "- Load DICOM images from disk or web\n"
            "- ITK-based image processing (Gaussian, Median, Threshold)\n"
            "- VTK 3D visualization (Volume & Surface rendering)\n"
            "- Real-time interactive sliders\n"
            "- PyQt5 user interface\n\n"
            "Libraries:\n"
            "- ITK 5.4.5 (Medical Image Processing)\n"
            "- VTK 9.6.0 (3D Visualization)\n"
            "- PyQt5 (GUI Framework)\n"
            "- pydicom (DICOM Loading)\n"
        )


def main():
    try:
        app = QApplication(sys.argv)
        viewer = DICOMNeuroimageViewer()
        viewer.show()
        print("✓ DICOM Neuroimage Viewer started successfully")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"✗ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
