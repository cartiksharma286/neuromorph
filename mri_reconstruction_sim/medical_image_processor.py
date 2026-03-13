#!/usr/bin/env python3
"""
Medical Image Processor - ITK and VTK Integration
Loads medical images using ITK and provides 3D visualization with VTK
"""

import itk
import numpy as np
from pathlib import Path
import sys
import os

# Set VTK to use headless rendering mode on macOS
os.environ['VTK_DEBUG_LEAKS'] = '0'

# Lazy import VTK to avoid display server issues
_vtk = None

def get_vtk():
    """Lazy load VTK module"""
    global _vtk
    if _vtk is None:
        import vtk
        # Configure VTK for headless rendering
        vtk.vtkObject.GlobalWarningDisplayOn()
        _vtk = vtk
    return _vtk

class MedicalImageProcessor:
    """Process medical images using ITK and VTK"""
    
    def __init__(self, image_path=None):
        """
        Initialize the medical image processor
        
        Args:
            image_path: Path to medical image file (NIfTI, DICOM, etc.)
        """
        self.image = None
        self.image_path = image_path
        self.voxel_data = None
        
    def load_image(self, file_path):
        """
        Load a medical image using ITK
        
        Args:
            file_path: Path to medical image file
            
        Returns:
            ITK image object
        """
        try:
            self.image = itk.imread(file_path)
            self.image_path = file_path
            print(f"✓ Image loaded: {file_path}")
            print(f"  Size: {self.image.GetLargestPossibleRegion().GetSize()}")
            print(f"  Spacing: {self.image.GetSpacing()}")
            print(f"  Pixel type: {self.image.GetPixelType()}")
            return self.image
        except Exception as e:
            print(f"✗ Failed to load image: {e}")
            return None
    
    def process_image(self, apply_gaussian=True, sigma=1.0):
        """
        Process the medical image with ITK filters
        
        Args:
            apply_gaussian: Apply Gaussian smoothing
            sigma: Sigma value for Gaussian filter
            
        Returns:
            Processed ITK image
        """
        if self.image is None:
            print("✗ No image loaded. Load an image first.")
            return None
        
        processed = self.image
        
        if apply_gaussian:
            # Apply Gaussian smoothing
            gaussian_filter = itk.SmoothingRecursiveGaussianImageFilter.New(
                Input=processed,
                Sigma=sigma
            )
            gaussian_filter.Update()
            processed = gaussian_filter.GetOutput()
            print(f"✓ Applied Gaussian smoothing (sigma={sigma})")
        
        return processed
    
    def create_3d_renderer(self, window_size=(800, 600)):
        """
        Create a VTK renderer for 3D visualization
        
        Args:
            window_size: Tuple of (width, height) for renderer window
            
        Returns:
            VTK renderer window
        """
        vtk = get_vtk()
        # Create renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.2)
        
        # Create render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(window_size[0], window_size[1])
        render_window.SetWindowName("MRI Medical Image Viewer")
        
        # Create interactor
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        
        print(f"✓ VTK renderer created ({window_size[0]}x{window_size[1]})")
        
        return renderer, render_window, interactor
    
    def add_volume_to_renderer(self, renderer, image_data):
        """
        Add volume rendering to the VTK renderer
        
        Args:
            renderer: VTK renderer
            image_data: VTK image data
        """
        vtk = get_vtk()
        # Create volume mapper
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(image_data)
        
        # Create volume and set mapper
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        
        # Add volume to renderer
        renderer.AddViewProp(volume)
        renderer.ResetCamera()
        
        print("✓ Volume rendering configured")
    
    def add_surface_to_renderer(self, renderer, image_data, threshold=100):
        """
        Add surface rendering to the VTK renderer using marching cubes
        
        Args:
            renderer: VTK renderer
            image_data: VTK image data
            threshold: Threshold value for surface extraction
        """
        vtk = get_vtk()
        # Contour filter (marching cubes)
        contour = vtk.vtkContourFilter()
        contour.SetInputData(image_data)
        contour.SetValue(0, threshold)
        
        # Mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        
        # Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.8, 0.6)
        
        # Add to renderer
        renderer.AddActor(actor)
        renderer.ResetCamera()
        
        print(f"✓ Surface rendering configured (threshold={threshold})")
    
    def display(self, render_window, interactor):
        """
        Display the VTK visualization window
        
        Args:
            render_window: VTK render window
            interactor: VTK interactor
        """
        interactor.Initialize()
        render_window.Render()
        print("✓ Visualization window opened (Press 'q' to close)")
        interactor.Start()


def demo_with_synthetic_data():
    """Demo using synthetic medical image data"""
    print("\n" + "="*50)
    print("Medical Image Processing Demo")
    print("="*50)
    
    # Create processor
    processor = MedicalImageProcessor()
    
    # Create synthetic medical image (3D Gaussian blob)
    size = (64, 64, 64)
    region = itk.ImageRegion(size)
    image = itk.Image.New(Dimension=3, DefaultPixelType=itk.F)
    image.SetRegions(region)
    image.Allocate()
    
    # Fill with synthetic data (Gaussian blob)
    size_array = image.GetBufferedRegion().GetSize()
    for z in range(size_array[2]):
        for y in range(size_array[1]):
            for x in range(size_array[0]):
                # Create Gaussian blob in center
                dx = x - size[0]/2
                dy = y - size[1]/2
                dz = z - size[2]/2
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                value = 255 * np.exp(-(distance**2) / (2 * 10**2))
                image.SetPixel([x, y, z], value)
    
    processor.image = image
    print("✓ Synthetic medical image created (64x64x64)")
    print(f"  Intensity range: 0-255")
    
    # Process the image
    processed = processor.process_image(apply_gaussian=True, sigma=1.5)
    
    # Create VTK visualization
    renderer, render_window, interactor = processor.create_3d_renderer((800, 600))
    
    # Convert ITK image to VTK format for visualization
    vtk_module = get_vtk()
    itk_image_to_vtk = itk.ImageToVTKImageFilter[type(processed)].New()
    itk_image_to_vtk.SetInput(processed)
    itk_image_to_vtk.Update()
    vtk_image = itk_image_to_vtk.GetOutput()
    
    # Add visualization
    processor.add_surface_to_renderer(renderer, vtk_image, threshold=128)
    
    # Display
    print("\nConfiguration complete. Ready to visualize.")
    
    return processor, renderer, render_window, interactor


if __name__ == "__main__":
    processor, renderer, render_window, interactor = demo_with_synthetic_data()
    
    # Uncomment to display:
    # processor.display(render_window, interactor)
    
    print("\n✓ Medical Image Processor ready for use")
    print("\nUsage:")
    print("  processor = MedicalImageProcessor()")
    print("  processor.load_image('path/to/scan.nii')")
    print("  processed = processor.process_image()")
