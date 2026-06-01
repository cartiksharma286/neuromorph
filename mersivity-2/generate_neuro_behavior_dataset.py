import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from skimage import measure
import trimesh

def create_dicom_slice(filename, pixel_array, instance_number, series_uid, study_uid, series_number, series_description, acquisition_number=1, rows=64, cols=64):
    """Creates a highly compliant DICOM slice for a given series and study."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = '1.2.3.4'
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta)
    ds.PatientName = "Cortex^Behavioral"
    ds.PatientID = "BEHAVIORAL001"
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
    ds.AcquisitionNumber = acquisition_number
    
    # Set dimensions
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
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

def simulate_brain_volume(nx=64, ny=64, nz=30, t_step=0, simulate_activation=False):
    """
    Simulates a rich 3D brain volume including:
    - Head/Scalp boundary
    - Skull layer (dark on T1)
    - CSF spaces
    - Butterfly-shaped lateral ventricles
    - White Matter core
    - Grey Matter (Cortex) with high-frequency Gyri and Sulci folds
    - Dynamic BOLD cortical behavioral activations (if simulate_activation=True)
    """
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 1. Coordinate distances for various brain structures
    # Outer head boundary
    r_head = np.sqrt(X**2 + (Y/1.1)**2 + (Z/0.8)**2)
    
    # Gyri & Sulci folds perturbation using sine/cosine combinations
    folds = 0.09 * np.sin(10 * X) * np.cos(10 * Y) * np.sin(10 * Z)
    
    # Brain boundary with folds integrated
    r_brain = np.sqrt((X/0.72)**2 + (Y/0.82)**2 + (Z/0.62)**2) + folds
    
    # Central White Matter boundary
    r_wm = np.sqrt((X/0.52)**2 + (Y/0.60)**2 + (Z/0.45)**2) + 0.6 * folds

    # Butterfly-shaped lateral ventricles (CSF filled)
    ventricles = (X**2 / 0.12**2) + ((Y + 0.15 - 0.25 * np.abs(X))**2 / 0.15**2) + (Z**2 / 0.28**2)

    # 2. Base anatomical intensity mapping (simulating T1-weighted contrast)
    volume = np.zeros((nx, ny, nz), dtype=np.float64)

    # CSF background / background air
    volume.fill(10.0)

    # Scalp tissue (outer shell of the head)
    volume[r_head < 0.96] = 500.0  # soft tissue T1 intensity
    
    # Skull bone (low signal on T1)
    volume[(r_head < 0.92) & (r_head > 0.84)] = 80.0
    
    # Cerebrospinal Fluid (CSF) space inside the skull
    volume[r_head <= 0.84] = 150.0

    # Brain tissue - Grey Matter / Cortex (intensity ~ 900)
    volume[r_brain < 0.88] = 900.0

    # Brain tissue - White Matter core (intensity ~ 1400, brighter on T1)
    volume[r_wm < 0.72] = 1400.0

    # Ventricles (CSF intensity ~ 150)
    volume[ventricles < 1.0] = 150.0
    
    # Mask out everything outside the head boundary
    volume[r_head >= 0.96] = 0.0

    # 3. Simulate Cortical Behavior / Dynamic Activation (fMRI/BOLD signal)
    if simulate_activation:
        # Define a bilateral motor cortex strip on the post/pre-central gyri (approx x=+-0.25, y=0.0, z=0.35)
        d_motor = np.exp(-((np.abs(X) - 0.26)**2 / 0.08**2) - (Y**2 / 0.10**2) - ((Z - 0.38)**2 / 0.12**2))
        
        # Activated states correspond to block design (e.g. active on odd timepoints, rest on even)
        # Block design: Rest (t=0,1), Task (t=2,3), Rest (t=4,5), Task (t=6,7), Rest (t=8,9)
        active_blocks = [2, 3, 6, 7]
        is_active = t_step in active_blocks
        
        if is_active:
            # Scale BOLD activation inside Grey Matter / Cortex only
            is_grey_matter = (r_brain < 0.88) & (r_wm >= 0.72) & (ventricles >= 1.0)
            
            # Simulated 14% BOLD intensity increase in primary motor strip
            activation = 0.14 * volume * d_motor * is_grey_matter
            volume += activation
            
            print(f"Time {t_step}: Motor Cortex Active (BOLD activation added)", flush=True)
        else:
            print(f"Time {t_step}: Baseline Rest State", flush=True)

    # 4. Add spatial gaussian-white noise for MRI scanner realism
    noise = np.random.normal(0, 25.0, size=volume.shape)
    volume += noise
    volume = np.clip(volume, 0, 4095)

    return volume

def generate_neuro_behavior_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    study_dir = os.path.join(base_dir, 'mri', 'DICOM', 'behavior_study')
    os.makedirs(study_dir, exist_ok=True)
    
    # Generate unique Study Instance UID for the entire study
    study_uid = generate_uid()
    print(f"Generating mock behavioral study with StudyInstanceUID: {study_uid}")

    # =========================================================================
    # SERIES 1: High-resolution anatomical T1 structural reference scan
    # =========================================================================
    t1_dir = os.path.join(study_dir, 'Series_01_T1_Structural')
    os.makedirs(t1_dir, exist_ok=True)
    t1_series_uid = generate_uid()
    
    print("\n--- Generating Series 1: T1 Structural reference scan ---")
    # Simulate high resolution structure (64 x 64 x 30)
    t1_volume = simulate_brain_volume(nx=64, ny=64, nz=30, t_step=0, simulate_activation=False)
    
    for i in range(30):
        filename = os.path.join(t1_dir, f"slice_{i+1:02d}.dcm")
        slice_data = t1_volume[:, :, i]
        create_dicom_slice(
            filename=filename,
            pixel_array=slice_data,
            instance_number=i + 1,
            series_uid=t1_series_uid,
            study_uid=study_uid,
            series_number=1,
            series_description="T1 Structural Reference"
        )
    print(f"Series 1 successfully written to: {t1_dir}")

    # Generate Laser Scan STL Mesh via Marching Cubes on structural T1 volume
    # Isovalue at 300 captures simulated scalp outer boundary
    print("Generating laser scan STL mapping from structural T1 volume...")
    verts, faces, _, _ = measure.marching_cubes(t1_volume, level=300.0)
    stl_path = os.path.join(study_dir, 'laser_scan.stl')
    laser_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    laser_mesh.export(stl_path)
    print(f"Laser scan STL mesh successfully exported to: {stl_path}")

    # =========================================================================
    # SERIES 2: 4D Functional MRI (fMRI) Cortical Behavior Series (10 Timepoints)
    # =========================================================================
    fmri_dir = os.path.join(study_dir, 'Series_02_fMRI_Behavior')
    os.makedirs(fmri_dir, exist_ok=True)
    fmri_series_uid = generate_uid()
    
    print("\n--- Generating Series 2: fMRI Cortical Behavior Time-Series (BOLD) ---")
    n_timepoints = 10
    total_slices_written = 0
    
    for t in range(n_timepoints):
        # Simulate active vs rest fMRI BOLD volume
        fmri_volume = simulate_brain_volume(nx=64, ny=64, nz=30, t_step=t, simulate_activation=True)
        
        for i in range(30):
            # Calculate instance number spanning across time slices
            instance_num = t * 30 + i + 1
            filename = os.path.join(fmri_dir, f"time_{t:02d}_slice_{i+1:02d}.dcm")
            slice_data = fmri_volume[:, :, i]
            create_dicom_slice(
                filename=filename,
                pixel_array=slice_data,
                instance_number=instance_num,
                series_uid=fmri_series_uid,
                study_uid=study_uid,
                series_number=2,
                series_description="fMRI Cortex Behavior - Motor Task",
                acquisition_number=t + 1
            )
            total_slices_written += 1
            
    print(f"Series 2 successfully written to: {fmri_dir} (Total {total_slices_written} slices across 10 temporal blocks)")
    print("\n=========================================================================")
    print("Mock behavioral neuroimaging DICOM study successfully generated.")
    print("=========================================================================")

if __name__ == "__main__":
    generate_neuro_behavior_dataset()
