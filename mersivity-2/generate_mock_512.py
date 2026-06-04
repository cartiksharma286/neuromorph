import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from skimage import measure
import trimesh

def create_dicom_slice(filename, pixel_array, instance_number, series_uid, study_uid, series_number, series_description, rows=512, cols=512):
    """Creates a highly compliant high-resolution 512x512 DICOM slice."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # MR Image Storage SOP Class
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = '1.2.3.4'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta)
    ds.PatientName = "Atlas^Neurovascular^512"
    ds.PatientID = "ATLASVASC512"
    ds.InstanceNumber = str(instance_number)
    
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S')
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    
    ds.SeriesNumber = series_number
    ds.SeriesDescription = series_description
    
    # Set high-resolution dimensions (512 x 512)
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = [0.4, 0.4]  # high resolution pixel spacing
    ds.SliceThickness = 1.2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    
    # Cast pixel data to uint16
    pixel_array_u16 = np.clip(pixel_array, 0, 4095).astype(np.uint16)
    ds.PixelData = pixel_array_u16.tobytes()
    
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.preamble = b"\0" * 128
    
    ds.save_as(filename, write_like_original=False)

def compute_vascular_mask(X, Y, Z, segments):
    """Vectorized calculation of 3D vascular tree distance fields."""
    mask = np.zeros(X.shape, dtype=np.float64)
    coords = np.stack([X, Y, Z], axis=-1)
    
    for p0_raw, p1_raw, r in segments:
        p0 = np.array(p0_raw)
        p1 = np.array(p1_raw)
        v = p1 - p0
        v_len_sq = np.dot(v, v)
        if v_len_sq < 1e-8:
            continue
            
        diff = coords - p0
        t = np.sum(diff * v, axis=-1) / v_len_sq
        t = np.clip(t, 0.0, 1.0)
        
        closest = p0[None, None, None, :] + t[:, :, :, None] * v[None, None, None, :]
        dist = np.sqrt(np.sum((coords - closest)**2, axis=-1))
        
        # Soft-edged vascular tubes
        vessel_val = np.exp(-(dist**2) / (2 * (r**2)))
        mask = np.maximum(mask, vessel_val)
        
    return mask

def simulate_atlas_volume(nx=512, ny=512, nz=30, contrast_type="T1", segments=None):
    """
    Simulates a high-resolution 512x512x30 Atlas Brain structure + Vascular network.
    - contrast_type = "T1": CSF dark, WM bright.
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Head and Brain coordinates
    r_head = np.sqrt(X**2 + (Y/1.15)**2 + (Z/0.85)**2)
    
    # Rich high-resolution gyri and sulci fold perturbation
    folds = 0.06 * np.sin(14 * X) * np.cos(14 * Y) * np.sin(10 * Z) + 0.012 * np.sin(36 * X) * np.cos(36 * Y)
    r_brain = np.sqrt((X/0.74)**2 + (Y/0.84)**2 + (Z/0.64)**2) + folds
    r_wm = np.sqrt((X/0.54)**2 + (Y/0.64)**2 + (Z/0.46)**2) + 0.6 * folds
    
    ventricles = (X**2 / 0.13**2) + ((Y + 0.12 - 0.22 * np.abs(X))**2 / 0.14**2) + (Z**2 / 0.26**2)

    volume = np.zeros((nx, ny, nz), dtype=np.float64)

    # 1. Base Anatomical Signal Contrast
    if contrast_type == "T1":
        # Background air
        volume.fill(15.0)
        # Scalp soft tissue
        volume[r_head < 0.96] = 520.0
        # Skull bone (dark)
        volume[(r_head < 0.93) & (r_head > 0.85)] = 75.0
        # CSF spaces (dark T1)
        volume[r_head <= 0.85] = 140.0
        # Grey Matter Cortex
        volume[r_brain < 0.88] = 950.0
        # White Matter core
        volume[r_wm < 0.74] = 1450.0
        # Ventricles (CSF T1)
        volume[ventricles < 1.0] = 140.0
        
    # Mask out outer space
    volume[r_head >= 0.96] = 0.0

    # 2. Integrate Neurovascular Tree (Angiography / Willis Network)
    if segments is not None:
        vessel_mask = compute_vascular_mask(X, Y, Z, segments)
        volume = (1.0 - 0.25 * vessel_mask) * volume + 1800.0 * vessel_mask

    # 3. Scanner noise
    noise = np.random.normal(0, 18.0, size=volume.shape)
    volume += noise
    volume = np.clip(volume, 0, 4095)
    
    return volume

def generate_neurovascular_dataset_512():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    study_dir = os.path.join(base_dir, 'mri', 'DICOM', 'mock_dataset')
    os.makedirs(study_dir, exist_ok=True)
    
    # Erase existing slice DCM files and STL file in mock_dataset to make it clean
    for item in os.listdir(study_dir):
        item_path = os.path.join(study_dir, item)
        if item.endswith('.dcm') or item.endswith('.stl'):
            os.remove(item_path)
            print(f"Removed old file: {item}")
            
    # Generate unique Study Instance UID
    study_uid = generate_uid()
    print(f"Generating 512x512 Atlas Neurovascular study with StudyInstanceUID: {study_uid}")

    # Willis network segments
    segments = [
        # Basilar Artery (BA)
        ((0.0, -0.12, -0.6), (0.0, -0.12, -0.2), 0.035),
        # Left ICA
        ((0.14, -0.02, -0.6), (0.14, -0.02, -0.2), 0.035),
        # Right ICA
        ((-0.14, -0.02, -0.6), (-0.14, -0.02, -0.2), 0.035),
        
        # Circle of Willis hex connections
        ((0.14, -0.02, -0.2), (0.08, 0.12, -0.2), 0.03), # Left ICA to ACA
        ((-0.14, -0.02, -0.2), (-0.08, 0.12, -0.2), 0.03), # Right ICA to ACA
        ((0.08, 0.12, -0.2), (-0.08, 0.12, -0.2), 0.03), # Anterior Communicating
        ((0.14, -0.02, -0.2), (0.0, -0.12, -0.2), 0.03), # Left ICA to BA bifurcation
        ((-0.14, -0.02, -0.2), (0.0, -0.12, -0.2), 0.03), # Right ICA to BA bifurcation
        
        # Left MCA main stem
        ((0.14, -0.02, -0.2), (0.35, 0.0, -0.15), 0.03),
        # Right MCA main stem
        ((-0.14, -0.02, -0.2), (-0.35, 0.0, -0.15), 0.03),
        
        # Left MCA branches
        ((0.35, 0.0, -0.15), (0.55, 0.15, -0.05), 0.02),
        ((0.35, 0.0, -0.15), (0.52, -0.12, -0.05), 0.02),
        ((0.55, 0.15, -0.05), (0.65, 0.28, 0.15), 0.015),
        ((0.52, -0.12, -0.05), (0.62, -0.22, 0.15), 0.015),

        # Right MCA branches
        ((-0.35, 0.0, -0.15), (-0.55, 0.15, -0.05), 0.02),
        ((-0.35, 0.0, -0.15), (-0.52, -0.12, -0.05), 0.02),
        ((-0.55, 0.15, -0.05), (-0.65, 0.28, 0.15), 0.015),
        ((-0.52, -0.12, -0.05), (-0.62, -0.22, 0.15), 0.015),

        # Anterior Cerebral Arteries (ACA)
        ((0.08, 0.12, -0.2), (0.04, 0.35, 0.05), 0.025),
        ((-0.08, 0.12, -0.2), (-0.04, 0.35, 0.05), 0.025),
        ((0.04, 0.35, 0.05), (0.02, 0.55, 0.25), 0.02),
        ((-0.04, 0.35, 0.05), (-0.02, 0.55, 0.25), 0.02),
        
        # Posterior Cerebral Arteries (PCA)
        ((0.0, -0.12, -0.2), (0.18, -0.32, -0.15), 0.025),
        ((0.0, -0.12, -0.2), (-0.18, -0.32, -0.15), 0.025),
        ((0.18, -0.32, -0.15), (0.38, -0.52, 0.05), 0.02),
        ((-0.18, -0.32, -0.15), (-0.38, -0.52, 0.05), 0.02)
    ]

    print("\n--- Generating Volume (512 x 512 x 30) ---")
    t1_volume = simulate_atlas_volume(nx=512, ny=512, nz=30, contrast_type="T1", segments=segments)
    
    t1_series_uid = generate_uid()
    for i in range(30):
        filename = os.path.join(study_dir, f"slice_{i+1:02d}.dcm")
        slice_data = t1_volume[:, :, i]
        create_dicom_slice(
            filename=filename,
            pixel_array=slice_data,
            instance_number=i + 1,
            series_uid=t1_series_uid,
            study_uid=study_uid,
            series_number=1,
            series_description="Atlas Structural T1 Reference (512x512)"
        )
    print(f"512x512 DICOM slices written to: {study_dir}")

    # Generate laser scan STL mapping to the scalp boundary (512x512 space)
    print("Generating laser scan STL from structural volume...")
    verts, faces, _, _ = measure.marching_cubes(t1_volume, level=320.0)
    stl_path = os.path.join(study_dir, 'laser_scan.stl')
    laser_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    laser_mesh.export(stl_path)
    print(f"Laser scan STL mesh successfully exported to: {stl_path}")
    print("\nGeneration Complete!")

if __name__ == "__main__":
    generate_neurovascular_dataset_512()
