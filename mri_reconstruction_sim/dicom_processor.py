#!/usr/bin/env python3
"""
DICOM Neuroimage Processor using ITK
Provides comprehensive DICOM loading, filtering, and image processing
"""

import itk
import numpy as np
from pathlib import Path
import os


class DICOMProcessor:
    """Process DICOM medical images using ITK"""
    
    def __init__(self):
        """Initialize DICOM processor"""
        self.image = None
        self.original_image = None
        self.image_path = None
        self.dicom_series_paths = []
        self.current_slice = 0
        
    def load_dicom_file(self, file_path):
        """
        Load a single DICOM file
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.image = itk.imread(file_path)
            self.original_image = itk.Image.New(self.image)
            self.image_path = file_path
            print(f"✓ DICOM file loaded: {file_path}")
            self._print_image_info()
            return True
        except Exception as e:
            print(f"✗ Failed to load DICOM: {e}")
            return False
    
    def load_dicom_series(self, dicom_dir):
        """
        Load a DICOM series from directory
        
        Args:
            dicom_dir: Directory containing DICOM series
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find DICOM files in directory
            series_reader = itk.ImageSeriesReader.New()
            
            # Get filenames
            from itk.support import path_to_string
            dicom_files = []
            for f in os.listdir(dicom_dir):
                if f.lower().endswith(('.dcm', '.dicom')):
                    dicom_files.append(os.path.join(dicom_dir, f))
            
            if not dicom_files:
                print(f"✗ No DICOM files found in {dicom_dir}")
                return False
            
            # Sort files
            dicom_files.sort()
            series_reader.SetFileNames(dicom_files)
            series_reader.Update()
            
            self.image = series_reader.GetOutput()
            self.original_image = itk.Image.New(self.image)
            self.dicom_series_paths = dicom_files
            print(f"✓ DICOM series loaded: {len(dicom_files)} files")
            self._print_image_info()
            return True
        except Exception as e:
            print(f"✗ Failed to load DICOM series: {e}")
            return False
    
    def _print_image_info(self):
        """Print image information"""
        if self.image:
            size = self.image.GetLargestPossibleRegion().GetSize()
            spacing = self.image.GetSpacing()
            print(f"  Dimensions: {size[0]} × {size[1]}", end="")
            if len(size) > 2:
                print(f" × {size[2]} (3D)")
            else:
                print(" (2D)")
            print(f"  Spacing: {spacing[0]:.2f} × {spacing[1]:.2f}", end="")
            if len(spacing) > 2:
                print(f" × {spacing[2]:.2f} mm")
            else:
                print(" mm")
    
    def apply_gaussian_smoothing(self, sigma=1.0):
        """
        Apply Gaussian smoothing filter
        
        Args:
            sigma: Gaussian kernel sigma value
            
        Returns:
            Filtered image
        """
        if self.image is None:
            return None
        
        try:
            gaussian = itk.SmoothingRecursiveGaussianImageFilter.New(
                Input=self.image,
                Sigma=sigma
            )
            gaussian.Update()
            self.image = gaussian.GetOutput()
            print(f"✓ Applied Gaussian smoothing (sigma={sigma})")
            return self.image
        except Exception as e:
            print(f"✗ Gaussian filter failed: {e}")
            return None
    
    def apply_median_filter(self, radius=2):
        """
        Apply median filter for noise reduction
        
        Args:
            radius: Kernel radius
            
        Returns:
            Filtered image
        """
        if self.image is None:
            return None
        
        try:
            radius_vec = itk.Index[3]()
            radius_vec[0] = radius
            radius_vec[1] = radius
            if self.image.GetImageDimension() > 2:
                radius_vec[2] = radius
            
            median = itk.MedianImageFilter.New(
                Input=self.image,
                Radius=radius
            )
            median.Update()
            self.image = median.GetOutput()
            print(f"✓ Applied median filter (radius={radius})")
            return self.image
        except Exception as e:
            print(f"✗ Median filter failed: {e}")
            return None
    
    def apply_threshold(self, lower=0, upper=255):
        """
        Apply threshold filter
        
        Args:
            lower: Lower threshold value
            upper: Upper threshold value
            
        Returns:
            Thresholded image
        """
        if self.image is None:
            return None
        
        try:
            threshold = itk.ThresholdImageFilter.New(Input=self.image)
            threshold.SetLower(lower)
            threshold.SetUpper(upper)
            threshold.Update()
            self.image = threshold.GetOutput()
            print(f"✓ Applied threshold ({lower}-{upper})")
            return self.image
        except Exception as e:
            print(f"✗ Threshold filter failed: {e}")
            return None
    
    def apply_bilateral_filter(self, domain_sigma=2.0, range_sigma=50.0):
        """
        Apply bilateral filter for edge-preserving smoothing
        
        Args:
            domain_sigma: Domain sigma (spatial)
            range_sigma: Range sigma (intensity)
            
        Returns:
            Filtered image
        """
        if self.image is None:
            return None
        
        try:
            bilateral = itk.BilateralImageFilter.New(
                Input=self.image,
                DomainSigma=domain_sigma,
                RangeSigma=range_sigma
            )
            bilateral.Update()
            self.image = bilateral.GetOutput()
            print(f"✓ Applied bilateral filter (domain={domain_sigma}, range={range_sigma})")
            return self.image
        except Exception as e:
            print(f"✗ Bilateral filter failed: {e}")
            return None
    
    def apply_adaptive_histogram_equalization(self, radius=50):
        """
        Apply adaptive histogram equalization for contrast enhancement
        
        Args:
            radius: Kernel radius
            
        Returns:
            Enhanced image
        """
        if self.image is None:
            return None
        
        try:
            ahe = itk.AdaptiveHistogramEqualizationImageFilter.New(
                Input=self.image,
                Radius=radius
            )
            ahe.Update()
            self.image = ahe.GetOutput()
            print(f"✓ Applied adaptive histogram equalization (radius={radius})")
            return self.image
        except Exception as e:
            print(f"✗ AHE filter failed: {e}")
            return None
    
    def normalize_intensity(self):
        """
        Normalize image intensity to 0-255 range
        
        Returns:
            Normalized image
        """
        if self.image is None:
            return None
        
        try:
            rescaler = itk.RescaleIntensityImageFilter.New(
                Input=self.image,
                OutputMinimum=0,
                OutputMaximum=255
            )
            rescaler.Update()
            self.image = rescaler.GetOutput()
            print(f"✓ Normalized intensity (0-255)")
            return self.image
        except Exception as e:
            print(f"✗ Normalization failed: {e}")
            return None
    
    def reset_to_original(self):
        """Reset to original image"""
        if self.original_image:
            self.image = itk.Image.New(self.original_image)
            print("✓ Reset to original image")
            return True
        return False
    
    def get_image_array(self):
        """
        Get image as numpy array
        
        Returns:
            Numpy array of image data
        """
        if self.image is None:
            return None
        
        try:
            # Convert ITK image to numpy array
            array = itk.array_view_from_image(self.image)
            return np.asarray(array)
        except Exception as e:
            print(f"✗ Failed to convert to array: {e}")
            return None
    
    def get_image_statistics(self):
        """
        Get image statistics
        
        Returns:
            Dictionary with statistics
        """
        if self.image is None:
            return None
        
        try:
            stats_filter = itk.StatisticsImageFilter.New(Input=self.image)
            stats_filter.Update()
            
            return {
                'mean': stats_filter.GetMean(),
                'std': stats_filter.GetSigma(),
                'min': stats_filter.GetMinimum(),
                'max': stats_filter.GetMaximum(),
                'count': stats_filter.GetSum()
            }
        except Exception as e:
            print(f"✗ Statistics failed: {e}")
            return None
    
    def get_2d_slice(self, slice_index=0):
        """
        Extract a 2D slice from 3D image
        
        Args:
            slice_index: Index of slice to extract
            
        Returns:
            2D numpy array
        """
        if self.image is None:
            return None
        
        try:
            array = self.get_image_array()
            if array is None:
                return None
            
            # Handle both 2D and 3D images
            if array.ndim == 3:
                return array[slice_index, :, :]
            else:
                return array
        except Exception as e:
            print(f"✗ Slice extraction failed: {e}")
            return None
