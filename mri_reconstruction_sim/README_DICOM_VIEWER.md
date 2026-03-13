# DICOM Neuroimage Viewer

Advanced neuroimaging visualization and processing application with PyQt5, ITK, and VTK integration.

## Features

### Image Processing (ITK)
- **Load DICOM files and series** from local storage
- **Create synthetic brain DICOM files** for testing
- **Advanced filtering algorithms**:
  - Gaussian smoothing
  - Median filtering
  - Bilateral filtering (edge-preserving)
  - Adaptive histogram equalization
  - Thresholding
  - Intensity normalization

### Visualization (VTK)
- **2D slice viewing** with window/level controls
- **3D volume and surface rendering** (when VTK is configured)
- **Multi-slice navigation** with slider controls
- **Image statistics** display (mean, std, min, max)

### User Interface (PyQt5)
- **Intuitive control panels** with sliders and spinboxes
- **Real-time parameter adjustment**
- **Organized filter groups**:
  - File Operations
  - Image Processing
  - Thresholding
  - Display Options
- **Menu bar** with shortcuts
- **Status bar** showing operation progress

## Installation

### Requirements
```bash
pip3 install PyQt5 itk vtk pydicom matplotlib
```

### Breakdown
- **PyQt5**: GUI framework
- **ITK** (Insight Toolkit): Medical image processing
- **VTK** (Visualization Toolkit): 3D visualization
- **pydicom**: DICOM file manipulation
- **matplotlib**: 2D image display

## Quick Start

### Run the Application
```bash
cd mri_reconstruction_sim
python3 dicom_neuro_viewer.py
```

### Create Sample Brain
1. Click "Create Sample Brain" button
2. Generates a synthetic 256×256×64 brain DICOM series
3. Automatically loads and displays the first slice

### Load Your Own DICOM Files
1. Click "Load DICOM File" to load a single file
2. Click "Load DICOM Series" to load all DICOM files from a directory

## File Structure

```
mri_reconstruction_sim/
├── dicom_neuro_viewer.py       # Main PyQt5 application
├── dicom_processor.py           # ITK-based DICOM processing
├── dicom_downloader.py          # DICOM acquisition and generation
├── vtk_visualizer.py            # VTK 3D visualization
└── README_DICOM_VIEWER.md       # This file
```

## Module Documentation

### dicom_processor.py
**DICOMProcessor** class handles all ITK-based image processing:

```python
from dicom_processor import DICOMProcessor

processor = DICOMProcessor()
processor.load_dicom_file('scan.dcm')
processor.apply_gaussian_smoothing(sigma=1.5)
processor.apply_median_filter(radius=2)
array = processor.get_image_array()
stats = processor.get_image_statistics()
```

**Methods**:
- `load_dicom_file(path)` - Load single DICOM
- `load_dicom_series(dir)` - Load all DICOM files in directory
- `apply_gaussian_smoothing(sigma)` - Gaussian blur
- `apply_median_filter(radius)` - Noise reduction
- `apply_bilateral_filter(domain_sigma, range_sigma)` - Edge-preserving smoothing
- `apply_adaptive_histogram_equalization(radius)` - Contrast enhancement
- `apply_threshold(lower, upper)` - Intensity thresholding
- `normalize_intensity()` - Scale to 0-255
- `reset_to_original()` - Undo all changes
- `get_image_array()` - Get as numpy array
- `get_image_statistics()` - Get mean, std, min, max
- `get_2d_slice(index)` - Extract single slice

### dicom_downloader.py
**DICOMDownloader** class manages DICOM file acquisition:

```python
from dicom_downloader import DICOMDownloader

downloader = DICOMDownloader()
sample_dir = downloader.download_sample_dataset('synthetic_brain')
files = downloader.create_sample_brain_dicom(sample_dir, size=(256, 256, 64))
```

**Methods**:
- `list_repositories()` - Show available data sources
- `download_sample_dataset(name)` - Prepare sample dataset
- `create_sample_brain_dicom(dir, size)` - Generate synthetic brain
- `download_from_url(url)` - Download from web
- `get_sample_dicom_info()` - List sample sources
- `list_cached_files()` - Show local cache

### vtk_visualizer.py
**VTKNeuroimageVisualizer** class handles 3D rendering:

```python
from vtk_visualizer import VTKNeuroimageVisualizer

viz = VTKNeuroimageVisualizer()
viz.create_renderer((800, 600))
viz.add_volume_rendering(vtk_image, opacity=0.7)
viz.add_surface_rendering(vtk_image, threshold=128)
viz.render()
```

**Methods**:
- `create_renderer(size, bg_color)` - Initialize renderer
- `add_volume_rendering(data, opacity)` - Volume rendering
- `add_surface_rendering(data, threshold, color)` - Surface extraction
- `add_outline(data)` - Show image bounds
- `set_camera_position(pos, focal_point)` - Camera control
- `update_volume_opacity(opacity)` - Real-time opacity
- `reset_view()` - Reset camera
- `render()` - Render current view
- `start_interactor()` - Interactive mode

## Filter Parameters Guide

### Gaussian Smoothing
- **Sigma**: 0.1-10.0
  - Higher = more blur
  - Good for noise reduction (1-2)

### Median Filter
- **Radius**: 1-10
  - Removes salt-and-pepper noise
  - Preserves edges better than Gaussian

### Bilateral Filter
- **Domain Sigma**: Spatial extent (1-5)
- **Range Sigma**: Intensity tolerance (10-100)
  - Smooths while preserving edges
  - Good for denoise with edge preservation

### Adaptive Histogram Equalization (AHE)
- **Radius**: 10-100
  - Higher = larger region for computation
  - Enhances local contrast

### Thresholding
- **Lower/Upper**: 0-255 (or full bit depth)
  - Keep only values in range
  - Good for segmentation

## Window/Level Controls

The 2D viewer includes DICOM standard window/level controls:

- **Window**: Controls visibility range width
  - Lower = narrower range (higher contrast)
  - Higher = wider range (more detail)
- **Level**: Centers the visibility window
  - Adjust to view specific tissue types

### Typical Window/Level Values
- Brain tissue: W=80, L=40
- Bone: W=1000, L=400
- Lungs: W=1500, L=-600

## Sourcing DICOM Data

### Public Repositories
1. **BRATS** - Brain tumor segmentation challenge
   - https://www.med.upenn.edu/cbica/brats2020/

2. **ISLES** - Ischemic stroke lesion segmentation
   - https://www.isles-challenge.org/

3. **Kaggle Brain MRI** - Various brain MRI datasets
   - https://www.kaggle.com/datasets/

4. **Open Science Brain**
   - https://openbraininitiative.net/

5. **NCBI/NIH** - National biomedical imaging
   - https://www.ncbi.nlm.nih.gov/

### Creating Synthetic Data
The app includes built-in synthetic DICOM generation for testing:
```bash
python3 dicom_neuro_viewer.py
# Click "Create Sample Brain" button
```

This creates a realistic 3D brain-like structure with:
- Ventricles (bright central region)
- Gray matter (medium intensity)
- Skull (outer boundary)
- Realistic noise

## Performance Notes

- **Large series**: Load in chunks if memory-limited
- **Real-time filters**: Some filters may be slow on large images
- **3D rendering**: Requires VTK with GPU support

## Troubleshooting

### VTK Import Hangs
On macOS with display server issues:
```python
# The app uses lazy VTK loading to handle this
# VTK only loads when needed
```

### pydicom Not Available
Install with:
```bash
pip3 install pydicom
```

### ITK Filter Errors
Check image type compatibility:
```python
# Most filters support float and integer types
# Use normalize_intensity() if needed
```

## Future Enhancements

- [ ] DICOM streaming from cloud storage
- [ ] Automated segmentation with deep learning
- [ ] Multi-modal image overlay
- [ ] 3D volume reconstruction animation
- [ ] Export processed images to formats
- [ ] Batch processing pipeline
- [ ] PACS integration
- [ ] Real-time 3D visualization

## References

### Documentation
- ITK: https://itk.org/
- VTK: https://vtk.org/
- DICOM: https://www.dicomstandard.org/
- PyQt5: https://pypi.org/project/PyQt5/

### Tutorials
- Medical Image Analysis with ITK: https://itk.org/ITKExamples/
- VTK Visualization: https://kitware.github.io/vtk-examples/
- DICOM with pydicom: https://pydicom.readthedocs.io/

## License

MIT License - See LICENSE file for details

## Authors

Created for the MRI Reconstruction Simulator project
Part of the Neuromorphic Computing Framework

---

For questions or issues, consult the main project documentation or contact the development team.
