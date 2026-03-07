import vtk
import numpy as np
from vtk.util import numpy_support

def create_synthetic_mri(width=256, height=256):
    """Generate a synthetic magnitude image."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    
    magnitude = np.zeros((height, width))
    magnitude[r < 0.9] = 50   # Outer boundary
    magnitude[r < 0.8] = 120  # Tissue
    magnitude[r < 0.6] = 180  # Inner structure
    magnitude[r < 0.3] = 100  # Core
    
    # Add noise
    noise = np.random.normal(0, 10, (height, width))
    magnitude = np.clip(magnitude + noise, 0, 255).astype(np.uint8)
    return magnitude

def simulate_heating(width=256, height=256, center=(128, 128), radius=30, max_temp_increase=0.0):
    """Simulate temperature map (baseline + heating profile)."""
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    
    # Gaussian heating profile
    dist_sq = (xx - center[0])**2 + (yy - center[1])**2
    sigma = radius / 2.0
    heating = max_temp_increase * np.exp(-dist_sq / (2 * sigma**2))
    
    # Base body temperature
    temperature = 37.0 + heating
    
    # Small noise on temperature measurements
    noise = np.random.normal(0, 0.5, (height, width))
    temperature += noise
    
    return temperature.astype(np.float32)

class MRThermometryApp:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height
        
        # 1. Initialize data arrays
        self.magnitude_np = create_synthetic_mri(width, height)
        self.temp_np = simulate_heating(width, height, max_temp_increase=0)
        
        # Retain references to flat arrays so we can update in-place
        self.temp_np_flat = self.temp_np.ravel()
        
        # 2. Create VTK Images
        self.magnitude_vtk = self._numpy_to_vtk_image(self.magnitude_np)
        
        # Use a persistent VTK array for temperature to update without reallocation
        self.temp_vtk_array = numpy_support.numpy_to_vtk(num_array=self.temp_np_flat, deep=False, array_type=vtk.VTK_FLOAT)
        self.temp_vtk = vtk.vtkImageData()
        self.temp_vtk.SetDimensions(width, height, 1)
        self.temp_vtk.GetPointData().SetScalars(self.temp_vtk_array)
        
        self.setup_pipeline()
        
    def _numpy_to_vtk_image(self, numpy_array):
        """Helper to convert static numpy 2d arrays to vtkImageData."""
        vtk_arr = numpy_support.numpy_to_vtk(num_array=numpy_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        img = vtk.vtkImageData()
        img.SetDimensions(numpy_array.shape[1], numpy_array.shape[0], 1)
        img.GetPointData().SetScalars(vtk_arr)
        return img

    def setup_pipeline(self):
        # -- Magnitude Image Pipeline (Grayscale) --
        lut_mag = vtk.vtkLookupTable()
        lut_mag.SetRange(0, 255)
        lut_mag.SetSaturationRange(0, 0)
        lut_mag.SetHueRange(0, 0)
        lut_mag.SetValueRange(0, 1)
        lut_mag.Build()
        
        map_mag = vtk.vtkImageMapToColors()
        map_mag.SetInputData(self.magnitude_vtk)
        map_mag.SetLookupTable(lut_mag)
        
        # -- Temperature Image Pipeline (Color overlay) --
        self.lut_temp = vtk.vtkLookupTable()
        self.lut_temp.SetRange(37, 80) # Body temp to thermal ablation scale
        
        # Use customized coloring: body temp (37) is transparent, heating goes blue -> red
        self.lut_temp.SetNumberOfTableValues(256)
        for i in range(256):
            val = 37.0 + (80.0 - 37.0) * (i / 255.0)
            # Create a color from blue to red
            rgb = [0.0, 0.0, 0.0]
            # using VTK math helper to create Hue from 0.66 (blue) to 0.0 (red)
            hue = 0.666 - 0.666 * (i / 255.0)
            vtk.vtkMath.HSVToRGB([hue, 1.0, 1.0], rgb)
            
            # Transparency: make normal temps transparent
            alpha = 0.0
            if val > 39.0:
                alpha = min(0.8, (val - 39.0) / 10.0)
                
            self.lut_temp.SetTableValue(i, rgb[0], rgb[1], rgb[2], alpha)
            
        self.lut_temp.Build()
        
        map_temp = vtk.vtkImageMapToColors()
        map_temp.SetInputData(self.temp_vtk)
        map_temp.SetLookupTable(self.lut_temp)
        map_temp.PassAlphaToOutputOn() # Critical for overlay blending
        
        # -- Image Blending --
        self.blend = vtk.vtkImageBlend()
        self.blend.AddInputConnection(map_mag.GetOutputPort())
        self.blend.AddInputConnection(map_temp.GetOutputPort())
        self.blend.SetOpacity(0, 1.0) # Base image fully opaque
        self.blend.SetOpacity(1, 1.0) # Overlay uses alphas from its LUT
        
        # -- Viewer Setup --
        self.viewer = vtk.vtkImageViewer2()
        self.viewer.SetInputConnection(self.blend.GetOutputPort())
        self.viewer.SetColorLevel(127.5)
        self.viewer.SetColorWindow(255.0)
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.viewer.SetupInteractor(self.interactor)
        
        self.setup_slider()
        
    def setup_slider(self):
        slider_rep = vtk.vtkSliderRepresentation2D()
        slider_rep.SetMinimumValue(0.0)
        slider_rep.SetMaximumValue(100.0)
        slider_rep.SetValue(0.0)
        slider_rep.SetTitleText("MRgFUS Heating Power %")
        
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint1Coordinate().SetValue(0.1, 0.1)
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        slider_rep.GetPoint2Coordinate().SetValue(0.9, 0.1)
        
        slider_rep.SetSliderLength(0.02)
        slider_rep.SetSliderWidth(0.03)
        slider_rep.SetEndCapLength(0.01)
        slider_rep.SetEndCapWidth(0.03)
        slider_rep.SetTubeWidth(0.005)
        
        self.slider_widget = vtk.vtkSliderWidget()
        self.slider_widget.SetInteractor(self.interactor)
        self.slider_widget.SetRepresentation(slider_rep)
        self.slider_widget.SetAnimationModeToAnimate()
        self.slider_widget.EnabledOn()
        
        # Register callback
        self.slider_widget.AddObserver("InteractionEvent", self.on_slider_change)
        
    def on_slider_change(self, obj, event):
        val = obj.GetRepresentation().GetValue()
        
        # Calculate max temperature based on slider (e.g. up to +45 C)
        max_t_increase = val * 0.45 
        
        # Re-calculate temperature 
        new_temp = simulate_heating(self.width, self.height, max_temp_increase=max_t_increase)
        
        # Update flat reference correctly
        np.copyto(self.temp_np_flat, new_temp.ravel())
        
        # Notify VTK that data has changed
        self.temp_vtk_array.Modified()
        self.temp_vtk.Modified()
        self.blend.Modified()
        
        self.viewer.Render()

    def run(self):
        # Render and start interaction
        self.viewer.Render()
        self.viewer.GetRenderer().ResetCamera()
        self.viewer.GetRenderWindow().SetWindowName("MR Guided Thermometry")
        self.viewer.Render()
        
        print("Starting MR Guided Thermometry App...")
        print("Adjust the slider to simulate MRgFUS heating and visualize PRF shifts mapped to temperature.")
        self.interactor.Start()

if __name__ == "__main__":
    app = MRThermometryApp(width=400, height=400)
    app.run()
