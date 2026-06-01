import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from skimage import measure
import trimesh

def create_dicom_slice(filename, pixel_array, instance_number, series_uid, study_uid):
    # Setup file metadata
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # MR Image Storage SOP Class
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = '1.2.3.4'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create dataset
    ds = FileDataset(filename, {}, file_meta=file_meta)
    ds.PatientName = "Mock^Subject"
    ds.PatientID = "MOCK999"
    ds.InstanceNumber = str(instance_number)
    
    # Set date & time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S')
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    # Set image grid attributes
    ds.Rows = pixel_array.shape[0]
    ds.Columns = pixel_array.shape[1]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    
    # Cast and convert pixel array to uint16 bytes
    pixel_array_u16 = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array_u16.tobytes()
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # Save with 128-byte preamble to be standard DICOM
    ds.preamble = b"\0" * 128
    ds.save_as(filename, write_like_original=False)

def generate_mock_dataset():
    # 1. Define output directory inside mersivity/mri/DICOM/mock_dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    subfolder = os.path.join(base_dir, 'mri', 'DICOM', 'mock_dataset')
    os.makedirs(subfolder, exist_ok=True)
    
    print(f"Creating mock dataset in subfolder: {subfolder}")

    # 2. Generate mock 3D cortical ellipsoid volume with high-frequency sulcal folds
    nx, ny, nz = 64, 64, 30
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Base head bounds: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
    a, b, c = 0.72, 0.82, 0.62
    r_sq = (X/a)**2 + (Y/b)**2 + (Z/c)**2
    
    # Add simulated sulci and gyri folds using sine/cosine combinations
    folds = 0.08 * np.sin(9 * X) * np.cos(9 * Y) * np.sin(9 * Z)
    volume_float = (1.0 - (r_sq + folds)) * 2000.0
    volume_float = np.clip(volume_float, 0, 4095)  # 12-bit max
    
    # Zero out background voxels
    volume = volume_float.astype(np.uint16)
    volume[r_sq > 0.98] = 0

    # 3. Generate Laser Scan STL Mesh via Marching Cubes
    # Isovalue at 500 captures the simulated scalp outer boundary
    verts, faces, _, _ = measure.marching_cubes(volume, level=500.0)
    
    # Export to STL file inside the same folder
    stl_path = os.path.join(subfolder, 'laser_scan.stl')
    laser_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    laser_mesh.export(stl_path)
    print(f"Laser scan STL mesh exported to: {stl_path}")

    # 4. Generate mock MRI DICOM slices
    series_uid = generate_uid()
    study_uid = generate_uid()
    for i in range(nz):
        slice_filename = os.path.join(subfolder, f"slice_{i+1:02d}.dcm")
        slice_pixel_array = volume[:, :, i]
        create_dicom_slice(slice_filename, slice_pixel_array, i + 1, series_uid, study_uid)
        
    print(f"Successfully generated {nz} mock MRI DICOM slices.")

if __name__ == "__main__":
    generate_mock_dataset()
