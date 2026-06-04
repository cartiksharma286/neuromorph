import os
import datetime
import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import FileDataset, Dataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def convert_minc_to_dicom(minc_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading MINC file from: {minc_path}")
    img = nib.load(minc_path)
    data = img.get_fdata()
    
    # Get dimensions
    # Shape is (181, 217, 181)
    # We will slice along the third axis (axis 2) which yields 181 axial slices of shape (181, 217).
    nx, ny, nz = data.shape
    print(f"MINC Volume shape: {data.shape}")
    
    # Scale values to uint16 range (0 to 4095 for 12-bit stored DICOM)
    dmin, dmax = data.min(), data.max()
    if dmax - dmin > 1e-6:
        data_scaled = (data - dmin) / (dmax - dmin) * 4095.0
    else:
        data_scaled = np.zeros_like(data)
        
    data_scaled = np.clip(data_scaled, 0, 4095).astype(np.uint16)
    
    # Generate UIDs
    study_uid = generate_uid()
    series_uid = generate_uid()
    
    print(f"Writing {nz} DICOM slices to: {output_dir}")
    
    for i in range(nz):
        slice_data = data_scaled[:, :, i]
        filename = os.path.join(output_dir, f"slice_{i+1:03d}.dcm")
        
        # Setup file metadata
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # MR Image Storage SOP Class
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = '1.2.3.4'
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # Create dataset
        ds = FileDataset(filename, {}, file_meta=file_meta)
        ds.PatientName = "ICBM^MNI152^Normal"
        ds.PatientID = "ICBM152"
        ds.InstanceNumber = str(i + 1)
        
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        ds.ContentTime = dt.strftime('%H%M%S')
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        # Set image geometry attributes
        ds.Rows = slice_data.shape[0]     # 181
        ds.Columns = slice_data.shape[1]  # 217
        ds.PixelSpacing = [1.0, 1.0]      # 1mm voxel size
        ds.SliceThickness = 1.0           # 1mm slice thickness
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        
        # Pixel data
        ds.PixelData = slice_data.tobytes()
        
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.preamble = b"\0" * 128
        
        ds.save_as(filename, write_like_original=False)
        
    print(f"Successfully converted and saved {nz} DICOM images.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    minc_file = os.path.join(base_dir, 't1_icbm_normal_1mm_pn1_rf0.mnc.gz')
    output_folder = os.path.join(base_dir, 'converted_dicom')
    convert_minc_to_dicom(minc_file, output_folder)
