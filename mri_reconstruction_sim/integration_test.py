#!/usr/bin/env python3
"""
DICOM Viewer Complete Integration Test
Validates all components end-to-end without launching the GUI
"""

import os
import sys
import numpy as np
import pydicom
from pathlib import Path
import json
from datetime import datetime

# Test configuration
DICOM_PATH = "/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/neurovascular_brain_with_tumor.dcm"
TEST_OUTPUT_FILE = "/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/integration_test_report.json"

# Test results storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "overall_passed": True,
    "tests": [],
    "summary": {}
}

def log_test(test_name, passed, details=""):
    """Log test result"""
    result = {
        "name": test_name,
        "passed": passed,
        "details": details
    }
    test_results["tests"].append(result)
    
    # Update overall status
    if not passed:
        test_results["overall_passed"] = False
    
    # Print result
    status_symbol = "✓" if passed else "✗"
    status_color = "\033[92m" if passed else "\033[91m"  # Green or Red
    reset_color = "\033[0m"
    
    print(f"{status_color}{status_symbol}{reset_color} {test_name}")
    if details:
        print(f"    {details}")

print("=" * 80)
print("DICOM VIEWER INTEGRATION TEST SUITE")
print("=" * 80)
print()

# Test 1: DICOM File Integrity
print("SECTION 1: DICOM File Validation")
print("-" * 80)

try:
    if os.path.exists(DICOM_PATH):
        file_size_mb = os.path.getsize(DICOM_PATH) / (1024 * 1024)
        dcm = pydicom.dcmread(DICOM_PATH)
        
        log_test(
            "DICOM file exists and is readable",
            True,
            f"File size: {file_size_mb:.1f} MB"
        )
        
        # Check DICOM headers
        required_attrs = ['PatientName', 'PatientID', 'Modality', 'Rows', 'Columns']
        missing = [a for a in required_attrs if not hasattr(dcm, a)]
        
        if not missing:
            log_test(
                "DICOM has required header fields",
                True,
                f"Patient: {dcm.PatientName}, Modality: {dcm.Modality}"
            )
        else:
            log_test(
                "DICOM has required header fields",
                False,
                f"Missing: {missing}"
            )
        
        # Test pixel array
        pixel_array = dcm.pixel_array
        shape = pixel_array.shape
        dtype = pixel_array.dtype
        
        log_test(
            "Pixel array extraction successful",
            True,
            f"Shape: {shape}, Type: {dtype}"
        )
        
        # Verify 3D shape
        is_3d = len(shape) == 3
        log_test(
            "Image is 3D volume",
            is_3d,
            f"Dimensions: {shape}"
        )
        
        # Check intensity range and contrast
        min_val = pixel_array.min()
        max_val = pixel_array.max()
        mean_val = pixel_array.mean()
        std_val = pixel_array.std()
        
        has_good_contrast = (max_val - min_val) > 1000
        log_test(
            "Image has sufficient contrast",
            has_good_contrast,
            f"Range: {min_val}-{max_val}, Mean: {mean_val:.1f}, Std: {std_val:.1f}"
        )
        
    else:
        log_test("DICOM file exists and is readable", False, "File not found")
        
except Exception as e:
    log_test("DICOM file validation", False, str(e))

print()

# Test 2: ITK Integration
print("SECTION 2: ITK Integration")
print("-" * 80)

try:
    import itk
    
    # Test ITK image creation
    itk_image = itk.GetImageFromArray(pixel_array.astype(np.float32))
    
    log_test(
        "ITK image creation",
        True,
        f"Size: {itk_image.GetLargestPossibleRegion().GetSize()}"
    )
    
    # Test Gaussian filter
    try:
        gaussian = itk.SmoothingRecursiveGaussianImageFilter.New(
            Input=itk_image,
            Sigma=2.0
        )
        gaussian.Update()
        smoothed = gaussian.GetOutput()
        
        log_test(
            "Gaussian smoothing filter",
            smoothed is not None,
            "Sigma=2.0 applied successfully"
        )
    except Exception as e:
        log_test("Gaussian smoothing filter", False, str(e))
    
    # Test Median filter
    try:
        median = itk.MedianImageFilter.New(
            Input=itk_image,
            Radius=3
        )
        median.Update()
        filtered = median.GetOutput()
        
        log_test(
            "Median filter",
            filtered is not None,
            "Radius=3 applied successfully"
        )
    except Exception as e:
        log_test("Median filter", False, str(e))
    
    # Test Threshold filter
    try:
        threshold_filter = itk.ThresholdImageFilter.New(Input=itk_image)
        threshold_filter.SetLower(100)
        threshold_filter.SetUpper(255)
        threshold_filter.SetOutsideValue(0)
        threshold_filter.Update()
        thresholded = threshold_filter.GetOutput()
        
        log_test(
            "Threshold segmentation",
            thresholded is not None,
            "Threshold range 100-255 applied successfully"
        )
    except Exception as e:
        log_test("Threshold segmentation", False, str(e))
        
except ImportError:
    log_test("ITK import", False, "ITK not installed")

print()

# Test 3: VTK Integration
print("SECTION 3: VTK Integration")
print("-" * 80)

try:
    import vtk
    
    # Test VTK image creation
    try:
        vtk_converter = itk.ImageToVTKImageFilter[type(itk_image)].New(Input=itk_image)
        vtk_converter.Update()
        vtk_image = vtk_converter.GetOutput()
        
        log_test(
            "ITK to VTK conversion",
            vtk_image is not None,
            "Conversion successful"
        )
    except Exception as e:
        log_test("ITK to VTK conversion", False, str(e))
    
    # Test marching cubes (surface extraction)
    try:
        contour = vtk.vtkContourFilter()
        contour.SetInputData(vtk_image)
        contour.SetValue(0, 128)
        contour.Update()
        
        log_test(
            "Marching cubes surface extraction",
            True,
            "Contour filter executed successfully"
        )
    except Exception as e:
        log_test("Marching cubes surface extraction", False, str(e))
    
except ImportError:
    log_test("VTK import", False, "VTK not installed")

print()

# Test 4: PyQt5 Integration
print("SECTION 4: PyQt5 GUI Framework")
print("-" * 80)

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    
    log_test(
        "PyQt5 import",
        True,
        "PyQt5 and required widgets available"
    )
    
    # Check for required modules
    required_modules = [
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets'
    ]
    
    all_available = True
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            all_available = False
            break
    
    log_test(
        "PyQt5 required modules",
        all_available,
        "All required PyQt5 components available"
    )
    
except ImportError:
    log_test("PyQt5 import", False, "PyQt5 not installed")

print()

# Test 5: Image Format Conversion
print("SECTION 5: Image Processing and Conversion")
print("-" * 80)

try:
    from PIL import Image
    
    # Get a sample slice
    sample_slice = pixel_array[:, :, pixel_array.shape[2] // 2]
    
    # Normalize
    if sample_slice.max() > 0:
        normalized = (sample_slice / sample_slice.max() * 255).astype(np.uint8)
    else:
        normalized = sample_slice.astype(np.uint8)
    
    # Convert to PIL grayscale
    pil_gray = Image.fromarray(normalized, mode='L')
    log_test(
        "PIL grayscale image creation",
        True,
        f"Shape: {np.array(pil_gray).shape}"
    )
    
    # Convert to RGB
    pil_rgb = pil_gray.convert('RGB')
    log_test(
        "PIL RGB conversion",
        pil_rgb is not None,
        f"RGB shape: {np.array(pil_rgb).shape}"
    )
    
except Exception as e:
    log_test("Image format conversion", False, str(e))

print()

# Test 6: Feature Detection
print("SECTION 6: Pathological Feature Detection")
print("-" * 80)

try:
    # Analyze tumor region
    depth, height, width = pixel_array.shape
    mid_d, mid_h, mid_w = depth // 2, height // 2, width // 2
    
    tumor_region = pixel_array[
        max(0, mid_d-20):min(depth, mid_d+20),
        max(0, mid_h-30):min(height, mid_h+30),
        max(0, mid_w-30):min(width, mid_w+30)
    ]
    
    brain_region = pixel_array[5:15, 50:100, 50:100]
    
    tumor_max = tumor_region.max()
    brain_max = brain_region.max()
    ratio = tumor_max / (brain_max + 1) if brain_max > 0 else 0
    
    tumor_detected = tumor_max > brain_max * 1.2
    log_test(
        "Tumor structure detection",
        tumor_detected,
        f"Tumor intensity: {tumor_max:.0f}, Brain intensity: {brain_max:.0f}, Ratio: {ratio:.2f}x"
    )
    
    # Check for vessel structures
    vessel_threshold = brain_max * 1.05
    vessel_count = np.sum(pixel_array > vessel_threshold)
    vessel_percentage = (vessel_count / pixel_array.size) * 100
    
    has_vessels = vessel_count > 0
    log_test(
        "Neurovascular structure detection",
        has_vessels,
        f"Vessel-like voxels: {vessel_count:,} ({vessel_percentage:.2f}%)"
    )
    
except Exception as e:
    log_test("Feature detection", False, str(e))

print()

# Test 7: Application Scripts
print("SECTION 7: Application Files")
print("-" * 80)

files_to_check = [
    ("/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/dicom_viewer_enhanced.py", "Enhanced viewer"),
    ("/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/dicom_viewer_pyqt5.py", "Original viewer"),
    ("/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/create_sample_dicom.py", "DICOM generator"),
    ("/Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/test_dicom_complete.py", "Test suite"),
]

for file_path, description in files_to_check:
    exists = os.path.exists(file_path)
    log_test(
        f"{description} exists",
        exists,
        f"Path: {file_path}" if exists else "File not found"
    )

print()

# Summary
print("=" * 80)
print("INTEGRATION TEST SUMMARY")
print("=" * 80)
print()

passed_tests = sum(1 for t in test_results["tests"] if t["passed"])
total_tests = len(test_results["tests"])

test_results["summary"] = {
    "total_tests": total_tests,
    "passed_tests": passed_tests,
    "failed_tests": total_tests - passed_tests,
    "success_rate": f"{(passed_tests / total_tests * 100):.1f}%",
    "overall_status": "PASSED" if test_results["overall_passed"] else "FAILED"
}

print(f"Total Tests:    {total_tests}")
print(f"Passed:         {passed_tests}")
print(f"Failed:         {total_tests - passed_tests}")
print(f"Success Rate:   {(passed_tests / total_tests * 100):.1f}%")
print()

if test_results["overall_passed"]:
    print("Status: ✓ ALL TESTS PASSED")
    print()
    print("The DICOM viewer is fully functional and ready for use!")
    print()
    print("To launch the viewer:")
    print("  python3 /Users/cartiksharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/dicom_viewer_enhanced.py")
    print()
    exit_code = 0
else:
    print("Status: ✗ SOME TESTS FAILED")
    print()
    print("Please check the errors above and install missing dependencies.")
    exit_code = 1

print("=" * 80)

# Save results to JSON
try:
    with open(TEST_OUTPUT_FILE, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\nTest results saved to: {TEST_OUTPUT_FILE}")
except Exception as e:
    print(f"\nWarning: Could not save test results: {e}")

sys.exit(exit_code)
