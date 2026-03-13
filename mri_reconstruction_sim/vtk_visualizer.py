#!/usr/bin/env python3
"""
VTK Visualizer for 3D neuroimage rendering
Provides volume and surface rendering capabilities
"""

import numpy as np
import itk
import os
import traceback

# Lazy import VTK to avoid display server issues on headless systems
_vtk = None

def get_vtk():
    """Lazy load VTK module"""
    global _vtk
    if _vtk is None:
        try:
            import vtk
            # Suppress VTK debug output
            os.environ['VTK_DEBUG_LEAKS'] = '0'
            _vtk = vtk
        except ImportError:
            print("✗ VTK not available")
            return None
    return _vtk


class VTKNeuroimageVisualizer:
    """3D visualization of neuroimage data using VTK"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.vtk = get_vtk()
        if self.vtk is None:
            print("✗ VTK initialization failed")
            return
        
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.volume_actor = None
        self.surface_actor = None
        self.outline_actor = None
        self.camera = None
        
    def create_renderer(self, window_size=(800, 600), background_color=(0.1, 0.1, 0.2)):
        """
        Create VTK renderer
        
        Args:
            window_size: Tuple of (width, height)
            background_color: RGB background color tuple
            
        Returns:
            Renderer object or None
        """
        if self.vtk is None:
            return None
        
        try:
            # Create renderer
            self.renderer = self.vtk.vtkRenderer()
            self.renderer.SetBackground(*background_color)
            
            # Create render window
            self.render_window = self.vtk.vtkRenderWindow()
            self.render_window.AddRenderer(self.renderer)
            self.render_window.SetSize(*window_size)
            self.render_window.SetWindowName("Neuroimage 3D Visualizer")
            
            # Create interactor
            self.interactor = self.vtk.vtkRenderWindowInteractor()
            self.interactor.SetRenderWindow(self.render_window)
            
            print(f"✓ VTK renderer created ({window_size[0]}×{window_size[1]})")
            return self.renderer
        except Exception as e:
            print(f"✗ Renderer creation failed: {e}")
            traceback.print_exc()
            return None
    
    def add_volume_rendering(self, image_data, opacity=0.7):
        """
        Add volume rendering to scene
        
        Args:
            image_data: VTK image data
            opacity: Volume opacity (0-1)
            
        Returns:
            Volume actor or None
        """
        if self.vtk is None or self.renderer is None:
            return None
        
        try:
            # Create volume mapper
            mapper = self.vtk.vtkGPUVolumeRayCastMapper()
            mapper.SetInputData(image_data)
            
            # Create color and opacity transfer functions
            color_func = self.vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)      # Black for 0
            color_func.AddRGBPoint(64.0, 1.0, 0.0, 0.0)     # Red
            color_func.AddRGBPoint(128.0, 1.0, 1.0, 0.0)    # Yellow
            color_func.AddRGBPoint(192.0, 1.0, 1.0, 1.0)    # White
            color_func.AddRGBPoint(255.0, 1.0, 1.0, 1.0)
            
            opacity_func = self.vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(0, 0.0)
            opacity_func.AddPoint(50, 0.1 * opacity)
            opacity_func.AddPoint(100, 0.3 * opacity)
            opacity_func.AddPoint(150, 0.6 * opacity)
            opacity_func.AddPoint(255, opacity)
            
            # Create volume properties
            volume_property = self.vtk.vtkVolumeProperty()
            volume_property.SetColor(color_func)
            volume_property.SetScalarOpacity(opacity_func)
            volume_property.ShadeOn()
            volume_property.SetAmbient(0.4)
            volume_property.SetDiffuse(0.6)
            volume_property.SetSpecular(0.2)
            
            # Create volume
            self.volume_actor = self.vtk.vtkVolume()
            self.volume_actor.SetMapper(mapper)
            self.volume_actor.SetProperty(volume_property)
            
            # Add to renderer
            self.renderer.AddVolume(self.volume_actor)
            self.renderer.ResetCamera()
            
            print(f"✓ Volume rendering added (opacity={opacity:.2f})")
            return self.volume_actor
        except Exception as e:
            print(f"✗ Volume rendering failed: {e}")
            traceback.print_exc()
            return None
    
    def add_surface_rendering(self, image_data, threshold=128, color=(1.0, 0.8, 0.6)):
        """
        Add surface rendering using marching cubes
        
        Args:
            image_data: VTK image data
            threshold: Threshold value for surface extraction
            color: RGB color tuple
            
        Returns:
            Surface actor or None
        """
        if self.vtk is None or self.renderer is None:
            return None
        
        try:
            # Contour filter (marching cubes)
            contour = self.vtk.vtkContourFilter()
            contour.SetInputData(image_data)
            contour.SetValue(0, threshold)
            contour.Update()
            
            # Smoothing for better appearance
            smoother = self.vtk.vtkPolyDataNormals()
            smoother.SetInputConnection(contour.GetOutputPort())
            smoother.SetFeatureAngle(60.0)
            smoother.Update()
            
            # Mapper
            mapper = self.vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(smoother.GetOutputPort())
            mapper.ScalarVisibilityOff()
            
            # Actor
            self.surface_actor = self.vtk.vtkActor()
            self.surface_actor.SetMapper(mapper)
            self.surface_actor.GetProperty().SetColor(*color)
            self.surface_actor.GetProperty().SetOpacity(0.9)
            self.surface_actor.GetProperty().EdgeVisibilityOff()
            
            # Add to renderer
            self.renderer.AddActor(self.surface_actor)
            self.renderer.ResetCamera()
            
            print(f"✓ Surface rendering added (threshold={threshold}, color={color})")
            return self.surface_actor
        except Exception as e:
            print(f"✗ Surface rendering failed: {e}")
            traceback.print_exc()
            return None
    
    def add_outline(self, image_data):
        """
        Add image outline to scene
        
        Args:
            image_data: VTK image data
            
        Returns:
            Outline actor or None
        """
        if self.vtk is None or self.renderer is None:
            return None
        
        try:
            outline = self.vtk.vtkOutlineFilter()
            outline.SetInputData(image_data)
            outline.Update()
            
            mapper = self.vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(outline.GetOutputPort())
            
            self.outline_actor = self.vtk.vtkActor()
            self.outline_actor.SetMapper(mapper)
            self.outline_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
            
            self.renderer.AddActor(self.outline_actor)
            
            print("✓ Outline added")
            return self.outline_actor
        except Exception as e:
            print(f"✗ Outline failed: {e}")
            return None
    
    def set_camera_position(self, position, focal_point=(0, 0, 0)):
        """
        Set camera position and orientation
        
        Args:
            position: Tuple (x, y, z) for camera position
            focal_point: Tuple (x, y, z) for camera focal point
        """
        if self.vtk is None or self.renderer is None:
            return
        
        try:
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(*position)
            camera.SetFocalPoint(*focal_point)
            camera.SetViewUp(0, 0, 1)
            self.renderer.ResetCamera()
        except Exception as e:
            print(f"✗ Camera setting failed: {e}")
    
    def update_volume_opacity(self, opacity):
        """
        Update volume actor opacity
        
        Args:
            opacity: Opacity value (0-1)
        """
        if self.volume_actor is not None and self.vtk is not None:
            try:
                prop = self.volume_actor.GetProperty()
                if prop is not None:
                    # Get current opacity function and update
                    opacity_func = self.vtk.vtkPiecewiseFunction()
                    opacity_func.AddPoint(0, 0.0)
                    opacity_func.AddPoint(50, 0.1 * opacity)
                    opacity_func.AddPoint(100, 0.3 * opacity)
                    opacity_func.AddPoint(150, 0.6 * opacity)
                    opacity_func.AddPoint(255, opacity)
                    prop.SetScalarOpacity(opacity_func)
                    
                    if self.render_window:
                        self.render_window.Render()
            except Exception as e:
                print(f"✗ Opacity update failed: {e}")
    
    def reset_view(self):
        """Reset camera to initial view"""
        if self.renderer is not None:
            try:
                self.renderer.ResetCamera()
                if self.render_window:
                    self.render_window.Render()
            except Exception as e:
                print(f"✗ Reset view failed: {e}")
    
    def render(self):
        """Render the scene"""
        if self.render_window is not None:
            try:
                self.render_window.Render()
            except Exception as e:
                print(f"✗ Render failed: {e}")
    
    def start_interactor(self):
        """Start interactive mode"""
        if self.interactor is not None:
            try:
                self.interactor.Initialize()
                self.render_window.Render()
                self.interactor.Start()
            except Exception as e:
                print(f"✗ Interactor start failed: {e}")
