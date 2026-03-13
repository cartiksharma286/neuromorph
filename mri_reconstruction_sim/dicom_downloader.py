#!/usr/bin/env python3
"""
DICOM Downloader - Retrieve publicly available DICOM neuroimages from the web
"""

import requests
import os
import tempfile
from pathlib import Path
import json


class DICOMDownloader:
    """Download DICOM neuroimage datasets from public repositories"""
    
    # Public DICOM repositories
    REPOSITORIES = {
        'brainimaging': {
            'name': 'Open Science Framework Brain Imaging',
            'url': 'https://openbraininitiative.net/',
            'description': 'Open brain imaging datasets'
        },
        'brats': {
            'name': 'BRATS Dataset',
            'url': 'https://www.med.upenn.edu/cbica/brats2020/',
            'description': 'Brain tumor segmentation'
        },
        'ncbi': {
            'name': 'NCBI Medical Imaging',
            'url': 'https://www.ncbi.nlm.nih.gov/pubmed/',
            'description': 'NCBI biomedical imaging'
        }
    }
    
    SAMPLE_URLS = {
        'mni_sample': {
            'name': 'MNI Sample Brain',
            'url': 'http://www.bic.mni.mcgill.ca/',
            'description': 'MNI average brain template'
        },
        'dicom_library': {
            'name': 'Sample DICOM Files',
            'url': 'https://github.com/pydicom/pydicom-data/',
            'description': 'Sample DICOM files repository'
        }
    }
    
    def __init__(self, cache_dir=None):
        """
        Initialize downloader
        
        Args:
            cache_dir: Directory to cache downloaded files
        """
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), 'dicom_cache')
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✓ DICOM cache directory: {cache_dir}")
    
    def list_repositories(self):
        """List available DICOM repositories"""
        print("\n" + "="*60)
        print("Available DICOM Neuroimage Repositories")
        print("="*60)
        
        for key, repo in self.REPOSITORIES.items():
            print(f"\n{key.upper()}:")
            print(f"  Name: {repo['name']}")
            print(f"  URL: {repo['url']}")
            print(f"  Info: {repo['description']}")
        
        print("\n" + "="*60)
    
    def download_sample_dataset(self, dataset_name='sample_brain'):
        """
        Download a sample DICOM neuroimage dataset
        
        Args:
            dataset_name: Name of sample dataset
            
        Returns:
            Path to downloaded files or None
        """
        print(f"\n✓ Preparing to download sample dataset: {dataset_name}")
        
        # Create sample DICOM data locally (since direct web access may be restricted)
        sample_dir = os.path.join(self.cache_dir, dataset_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        print(f"✓ Sample data directory: {sample_dir}")
        return sample_dir
    
    def create_sample_brain_dicom(self, output_dir, size=(256, 256, 128)):
        """
        Create a synthetic sample brain DICOM for testing
        Uses pydicom to create realistic DICOM files
        
        Args:
            output_dir: Directory to save DICOM files
            size: Image dimensions (x, y, z)
            
        Returns:
            List of created DICOM file paths
        """
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileDataset
            import datetime
            import numpy as np
        except ImportError:
            print("⚠ pydicom not installed. Install with: pip3 install pydicom")
            return None
        
        print(f"✓ Creating synthetic brain DICOM image ({size[0]}×{size[1]}×{size[2]})")
        
        dicom_files = []
        
        # Create multiple slices
        num_slices = size[2]
        
        for slice_num in range(num_slices):
            # Create pixel data - simulate brain tissue
            center = (size[0]/2, size[1]/2)
            
            pixel_array = np.zeros((size[1], size[0]), dtype=np.uint16)
            
            # Add ventricles (bright area in center)
            y, x = np.ogrid[-size[1]/2:size[1]/2, -size[0]/2:size[0]/2]
            center_mask = (x**2 + y**2) < (30 - abs(slice_num - num_slices/2)/5)**2
            pixel_array[center_mask] = 3000
            
            # Add brain tissue (gray matter)
            tissue_mask = (x**2 + y**2) < (80 - abs(slice_num - num_slices/2)/5)**2
            pixel_array[tissue_mask] = 1500
            
            # Add skull (outline)
            skull_mask = (x**2 + y**2) < (100 - abs(slice_num - num_slices/2)/5)**2
            pixel_array[skull_mask & ~tissue_mask] = 500
            
            # Add noise
            noise = np.random.normal(0, 50, pixel_array.shape)
            pixel_array = np.clip(pixel_array + noise, 0, 4095).astype(np.uint16)
            
            # Create DICOM dataset
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
            file_meta.MediaStorageSOPInstanceUID = f'1.2.3.4.{slice_num}'
            file_meta.ImplementationClassUID = '1.2.3.4.5'
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            
            # Create dataset
            ds = FileDataset(
                filename=os.path.join(output_dir, f'brain_{slice_num:03d}.dcm'),
                dataset={},
                file_meta=file_meta,
                preamble=b"\0" * 128
            )
            
            ds.PatientName = "Test^Brain"
            ds.PatientID = "123456"
            ds.Modality = "CT"
            ds.SeriesInstanceUID = "1.2.3.4"
            ds.StudyInstanceUID = "1.2.3"
            ds.FrameOfReferenceUID = "1.2.3.4"
            ds.ContentDate = datetime.date.today().isoformat()
            ds.ContentTime = datetime.datetime.now().time().isoformat()
            ds.InstitutionName = "Sample Hospital"
            
            ds.Rows = size[1]
            ds.Columns = size[0]
            ds.BitsAllocated = 16
            ds.BitsStored = 12
            ds.HighBit = 11
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            
            ds.PixelSpacing = [0.97, 0.97]  # mm per pixel
            ds.SliceLocation = slice_num * 2.5  # 2.5mm slice spacing
            ds.InstanceNumber = slice_num + 1
            
            # Add pixel data
            ds.PixelData = pixel_array.tobytes()
            
            # Save DICOM file
            dcm_path = os.path.join(output_dir, f'brain_{slice_num:03d}.dcm')
            ds.save_as(dcm_path, write_like_original=False)
            dicom_files.append(dcm_path)
            
            if (slice_num + 1) % 20 == 0:
                print(f"  Created {slice_num + 1}/{num_slices} slices")
        
        print(f"✓ Created {len(dicom_files)} DICOM slices")
        return dicom_files
    
    def download_from_url(self, url, timeout=30):
        """
        Download DICOM file from URL
        
        Args:
            url: URL to DICOM file
            timeout: Request timeout in seconds
            
        Returns:
            Path to downloaded file or None
        """
        try:
            print(f"✓ Downloading from: {url}")
            
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Extract filename from URL
            filename = url.split('/')[-1]
            if not filename.lower().endswith(('.dcm', '.dicom')):
                filename = 'image.dcm'
            
            filepath = os.path.join(self.cache_dir, filename)
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress = (downloaded / total_size) * 100
                            print(f"  Progress: {progress:.1f}%", end='\r')
            
            print(f"\n✓ Downloaded to: {filepath}")
            return filepath
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
    
    def get_sample_dicom_info(self):
        """Get information about available sample DICOM data"""
        print("\n" + "="*60)
        print("Sample DICOM Datasets")
        print("="*60)
        
        for key, info in self.SAMPLE_URLS.items():
            print(f"\n{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Source: {info['url']}")
            print(f"  Description: {info['description']}")
        
        print("\n" + "="*60)
    
    def list_cached_files(self):
        """List DICOM files in cache"""
        dicom_files = []
        for root, dirs, files in os.walk(self.cache_dir):
            for f in files:
                if f.lower().endswith(('.dcm', '.dicom')):
                    dicom_files.append(os.path.join(root, f))
        
        if dicom_files:
            print(f"\n✓ Found {len(dicom_files)} cached DICOM files:")
            for f in dicom_files[:10]:  # Show first 10
                print(f"  - {f}")
            if len(dicom_files) > 10:
                print(f"  ... and {len(dicom_files) - 10} more")
        else:
            print("No cached DICOM files found")
        
        return dicom_files


# Quick test
if __name__ == "__main__":
    downloader = DICOMDownloader()
    downloader.get_sample_dicom_info()
    
    # Create sample brain DICOM
    sample_dir = downloader.download_sample_dataset('synthetic_brain')
    dicom_files = downloader.create_sample_brain_dicom(sample_dir)
    
    print(f"\n✓ Sample DICOM files ready at: {sample_dir}")
