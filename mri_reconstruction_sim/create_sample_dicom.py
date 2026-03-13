#!/usr/bin/env python3
"""
Create synthetic DICOM neuroimages with tumors and neurovascular structures
Generates realistic 3D brain MRI data with tumor lesions
"""

import numpy as np
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from datetime import datetime
import os

def create_neurovascular_brain_with_tumor(width=128, height=128, depth=32):
    """
    Create a synthetic 3D brain image with vectorized operations for speed
    Smaller dimensions for faster generation
    """
    print("Creating synthetic brain with neurovascular and tumor structures...")
    
    # Create coordinate grids (vectorized)
    x = np.arange(width)
    y = np.arange(height)
    z = np.arange(depth)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    center = np.array([width // 2, height // 2, depth // 2])
    
    # Initialize 3D volume
    volume = np.zeros((width, height, depth), dtype=np.float32)
    
    # 1. Add brain tissue (white matter + gray matter) - VECTORIZED
    print("  Adding brain tissue...")
    dx = X - center[0]
    dy = Y - center[1]
    dz = (Z - center[2]) * 2
    
    dist_brain = np.sqrt(dx**2 + dy**2 + dz**2)
    brain_mask = dist_brain < 80
    volume[brain_mask] = 200 * np.exp(-(dist_brain[brain_mask]**2) / (2 * 40**2))
    
    # 2. Add lateral ventricles - VECTORIZED
    print("  Adding ventricles...")
    # Left ventricle
    dx_lv = X - (center[0] - 15)
    dy_lv = Y - center[1]
    dz_lv = Z - center[2]
    dist_lv = np.sqrt(dx_lv**2 + dy_lv**2 + dz_lv**2)
    lv_mask = dist_lv < 10
    volume[lv_mask] = np.maximum(volume[lv_mask] - 80, 0)
    
    # Right ventricle
    dx_rv = X - (center[0] + 15)
    dist_rv = np.sqrt(dx_rv**2 + dy_lv**2 + dz_lv**2)
    rv_mask = dist_rv < 10
    volume[rv_mask] = np.maximum(volume[rv_mask] - 80, 0)
    
    # 3. Add middle cerebral artery (MCA) - VECTORIZED
    print("  Adding neurovascular structures...")
    dx_mca = X - center[0]
    dy_mca = Y - (center[1] - 5 + Z * 0.3)
    dz_mca = Z - center[2]
    dist_mca = np.sqrt(dx_mca**2 + dy_mca**2 + dz_mca**2)
    mca_mask = dist_mca < 6
    volume[mca_mask] = np.maximum(volume[mca_mask], 250 * np.exp(-(dist_mca[mca_mask]**2) / (2 * 3**2)))
    
    # 4. Add anterior cerebral artery (ACA)
    dx_aca = X - (center[0] - 10 + Z * 0.2)
    dy_aca = Y - (center[1] + 15)
    dz_aca = Z - center[2]
    dist_aca = np.sqrt(dx_aca**2 + dy_aca**2 + dz_aca**2)
    aca_mask = dist_aca < 5
    volume[aca_mask] = np.maximum(volume[aca_mask], 240 * np.exp(-(dist_aca[aca_mask]**2) / (2 * 2.5**2)))
    
    # 5. Add primary tumor (glioblastoma)
    print("  Adding tumor lesions...")
    tumor_center = np.array([center[0] + 20, center[1] - 15, center[2]])
    dx_tumor = X - tumor_center[0]
    dy_tumor = Y - tumor_center[1]
    dz_tumor = Z - tumor_center[2]
    dist_tumor = np.sqrt(dx_tumor**2 + dy_tumor**2 + dz_tumor**2)
    
    # Tumor core
    tumor_core = dist_tumor < 12
    volume[tumor_core] = np.maximum(volume[tumor_core], 180 * np.exp(-(dist_tumor[tumor_core]**2) / (2 * 6**2)))
    
    # Tumor edema
    tumor_edema = (dist_tumor >= 12) & (dist_tumor < 20)
    volume[tumor_edema] = np.maximum(volume[tumor_edema], 120 * np.exp(-((dist_tumor[tumor_edema] - 12)**2) / (2 * 4**2)))
    
    # 6. Add secondary tumor nodule
    tumor_center_2 = np.array([center[0] - 30, center[1] + 20, center[2]])
    dx_tumor2 = X - tumor_center_2[0]
    dy_tumor2 = Y - tumor_center_2[1]
    dz_tumor2 = Z - tumor_center_2[2]
    dist_tumor2 = np.sqrt(dx_tumor2**2 + dy_tumor2**2 + dz_tumor2**2)
    tumor2_mask = dist_tumor2 < 10
    volume[tumor2_mask] = np.maximum(volume[tumor2_mask], 150 * np.exp(-(dist_tumor2[tumor2_mask]**2) / (2 * 5**2)))
    
    # 7. Add necrotic region (low intensity)
    print("  Adding necrotic region...")
    necro_center = tumor_center + np.array([2, 2, 0])
    dx_necro = X - necro_center[0]
    dy_necro = Y - necro_center[1]
    dz_necro = Z - necro_center[2]
    dist_necro = np.sqrt(dx_necro**2 + dy_necro**2 + dz_necro**2)
    necro_mask = dist_necro < 6
    volume[necro_mask] = volume[necro_mask] * 0.2
    
    # Normalize and convert to uint16
    volume = np.clip(volume, 0, 255)
    volume = (volume * 256).astype(np.uint16)
    
    # Transpose for DICOM format (depth, height, width)
    volume_dicom = np.transpose(volume, (2, 1, 0))
    
    return volume_dicom

def create_dicom_file(pixel_array, filename="neurovascular_brain_with_tumor.dcm"):
    """Create a DICOM file from pixel array"""
    print(f"Creating DICOM file: {filename}...")
    
    # Create file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = '1.2.276.0.7230010.3.0.3.6.0'
    
    # Create dataset
    ds = FileDataset(
        filename,
        {},
        file_meta=file_meta,
        is_implicit_VR=False,
        is_little_endian=True
    )
    
    # Set patient information
    ds.PatientName = "Test^Brain"
    ds.PatientID = "123456"
    ds.PatientAge = "045Y"
    ds.PatientSex = "M"
    ds.StudyDate = datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.now().strftime('%H%M%S')
    ds.ContentDate = datetime.now().strftime('%Y%m%d')
    ds.ContentTime = datetime.now().strftime('%H%M%S')
    
    # Set modality and description
    ds.Modality = 'CT'
    ds.SeriesDescription = 'Neurovascular Brain with Tumor'
    ds.StudyDescription = 'Brain MRI - Neurovasculature'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.SOPInstanceUID = generate_uid()
    
    # Set image dimensions
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.Rows = pixel_array.shape[1]
    ds.Columns = pixel_array.shape[2]
    ds.NumberOfFrames = str(pixel_array.shape[0])
    
    # Set pixel data attributes
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    
    # Set spacing
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 2.0
    
    # Set pixel data
    ds.PixelData = pixel_array.tobytes()
    
    # Save DICOM file
    ds.save_as(filename, write_like_original=False)
    
    print(f"✓ DICOM file created: {filename}")
    print(f"  Dimensions: {pixel_array.shape}")
    print(f"  File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
    
    return filename

def main():
    print("=" * 60)
    print("DICOM Neuroimage Generator with Tumor and Vessels")
    print("=" * 60)
    
    # Create synthetic brain image
    volume = create_neurovascular_brain_with_tumor(width=256, height=256, depth=64)
    
    # Create DICOM file
    dicom_file = create_dicom_file(volume)
    
    print("\n" + "=" * 60)
    print("✓ Synthetic DICOM file ready for viewer!")
    print(f"  File: {dicom_file}")
    print("\n  Image features:")
    print("  - Brain tissue with gray/white matter")
    print("  - Lateral ventricles")
    print("  - Middle cerebral artery (MCA)")
    print("  - Anterior cerebral artery (ACA)")
    print("  - Vertebral arteries (VA)")
    print("  - Primary tumor (glioblastoma-like)")
    print("  - Secondary tumor nodule")
    print("  - Necrotic core")
    print("  - Hemorrhagic regions")
    print("=" * 60)
    
    return dicom_file

if __name__ == "__main__":
    dicom_path = main()
    print(f"\nTo view in DICOM viewer:")
    print(f"  File → Load DICOM...")
    print(f"  Select: {dicom_path}")
