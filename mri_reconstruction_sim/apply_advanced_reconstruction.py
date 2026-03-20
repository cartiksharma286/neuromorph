#!/usr/bin/env python3
"""
Application: Statistical Optimal Control-Based MRI Reconstruction
=====================================================================

Applies advanced denoising pipeline to MRI images in the project
and generates high-quality noise-free reconstructions.

Author: NeuroPulse Reconstruction Engine
Date: March 19, 2026
"""

import numpy as np
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the advanced denoiser
from statistical_optimal_control_denoiser import (
    HybridSuperiorReconstructor,
    demonstrate_artifact_removal_pipeline
)


class MRIReconstructionPipeline:
    """
    End-to-end pipeline for applying advanced denoising to MRI images.
    """
    
    def __init__(self, output_dir='reconstruction_results_advanced'):
        """Initialize pipeline."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.reconstructor = HybridSuperiorReconstructor()
        
    def load_or_generate_test_image(self, mode='cardiac'):
        """
        Create synthetic MRI image with white speckle artifacts.
        
        Parameters
        ----------
        mode : str
            'cardiac', 'neurological', 'vascular'
            
        Returns
        -------
        np.ndarray : Synthetic noisy MRI image
        """
        np.random.seed(42)
        size = 256
        
        if mode == 'cardiac':
            # Synthetic cardiac image with anatomical detail
            y, x = np.ogrid[:size, :size]
            cy, cx = size // 2, size // 2
            
            # Left ventricle (darker ellipse)
            dist = np.sqrt(((x - cx + 20)**2 / 60**2 + (y - cy)**2 / 70**2))
            lv = np.exp(-dist**2 / 2) * 0.7
            
            # Right ventricle
            dist_rv = np.sqrt(((x - cx - 40)**2 / 50**2 + (y - cy - 30)**2 / 40**2))
            rv = np.exp(-dist_rv**2 / 2) * 0.6
            
            # Atria
            dist_at = np.sqrt(((x - cx)**2 / 80**2 + (y - cy - 80)**2 / 50**2))
            at = np.exp(-dist_at**2 / 2) * 0.5
            
            image = lv + rv + at
            
        elif mode == 'neurological':
            # Brain-like image
            y, x = np.ogrid[:size, :size]
            cy, cx = size // 2, size // 2
            
            # Brain outline
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            brain = np.exp(-(dist - 80)**2 / 400) * 0.8
            
            # Ventricles
            dist_vent = np.sqrt(((x - cx)**2 / 30**2 + (y - cy)**2 / 40**2))
            ventricles = np.exp(-dist_vent**2 / 2) * 0.4
            
            image = brain - ventricles * 0.3
            
        elif mode == 'vascular':
            # Vascular network
            y, x = np.ogrid[:size, :size]
            cy, cx = size // 2, size // 2
            
            # Main vessel (vertical)
            vessel_main = np.exp(-((x - cx)**2 / 8**2) - ((y - cy) / 100)**2) * 0.8
            
            # Branch vessels
            angle1 = 45 * np.pi / 180
            x_rot1 = (x - cx) * np.cos(angle1) - (y - cy) * np.sin(angle1)
            y_rot1 = (x - cx) * np.sin(angle1) + (y - cy) * np.cos(angle1)
            branch1 = np.exp(-(x_rot1**2 / 6**2) - (y_rot1**2 / 80**2)) * 0.5
            
            image = vessel_main + branch1
        else:
            # Generic circular phantom
            y, x = np.ogrid[:size, :size]
            cy, cx = size // 2, size // 2
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            image = np.exp(-(dist - 70)**2 / 300) * 0.8
        
        # Normalize
        image = image / np.max(image)
        
        # Add white speckle noise (sum of Gaussian and Laplacian components)
        gaussian_noise = np.random.normal(0, 0.08, image.shape)
        laplacian_noise = np.random.laplace(0, 0.05, image.shape)
        poisson_noise = np.random.poisson(0.3, image.shape) / 100.0
        
        noise = 0.6 * gaussian_noise + 0.3 * laplacian_noise + 0.1 * poisson_noise
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image, image
    
    def reconstruct_single_image(self, noisy_image, image_name):
        """
        Reconstruct a single image.
        
        Parameters
        ----------
        noisy_image : np.ndarray
            Input noisy image
        image_name : str
            Image identifier for output files
            
        Returns
        -------
        dict : Reconstruction results
        """
        print(f"\n[*] Reconstructing: {image_name}")
        result = demonstrate_artifact_removal_pipeline(noisy_image)
        
        # Save reconstructed image
        reconstructed = result['reconstructed']
        np.save(
            os.path.join(self.output_dir, f'{image_name}_reconstructed.npy'),
            reconstructed
        )
        
        # Clip to valid range
        reconstructed_clipped = np.clip(reconstructed, 0, 1)
        
        return result, reconstructed_clipped
    
    def visualize_reconstruction(self, original_noisy, ground_truth, reconstructed, 
                                artifact_map, image_name, metrics):
        """
        Create comprehensive visualization.
        
        Parameters
        ----------
        original_noisy : np.ndarray
            Original noisy image
        ground_truth : np.ndarray
            Ground truth clean image (if available)
        reconstructed : np.ndarray
            Reconstructed image
        artifact_map : np.ndarray
            Removed artifact map
        image_name : str
            Image identifier
        metrics : dict
            Quality metrics
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{image_name} - Advanced Statistical+Optimal Control Reconstruction', 
                    fontsize=14, fontweight='bold')
        
        # Original noisy
        axes[0, 0].imshow(original_noisy, cmap='gray')
        axes[0, 0].set_title(f"Original Noisy\nSNR: {metrics['original_snr_db']:.1f} dB")
        axes[0, 0].axis('off')
        
        # Ground truth (if available)
        if ground_truth is not None:
            axes[0, 1].imshow(ground_truth, cmap='gray')
            axes[0, 1].set_title("Ground Truth (Clean)")
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[0, 1].set_title("Ground Truth")
            axes[0, 1].axis('off')
        
        # Reconstructed
        axes[0, 2].imshow(reconstructed, cmap='gray')
        axes[0, 2].set_title(f"Reconstructed\nSNR: {metrics['final_snr_db']:.1f} dB")
        axes[0, 2].axis('off')
        
        # Artifact map (removed speckles)
        axes[1, 0].imshow(artifact_map, cmap='hot')
        axes[1, 0].set_title("Removed Artifacts\n(Speckle Map)")
        axes[1, 0].axis('off')
        
        # SNR improvement
        axes[1, 1].text(0.5, 0.7, f"SNR Improvement", ha='center', va='center', 
                       fontsize=12, fontweight='bold')
        axes[1, 1].text(0.5, 0.5, f"+{metrics['snr_improvement_db']:.2f} dB", 
                       ha='center', va='center', fontsize=20, fontweight='bold', color='green')
        axes[1, 1].text(0.5, 0.3, f"Noise Var: {metrics['noise_variance_est']:.4f}", 
                       ha='center', va='center', fontsize=10)
        axes[1, 1].axis('off')
        
        # Reconstruction stages comparison
        axes[1, 2].text(0.5, 0.9, "Pipeline Stages", ha='center', va='top', fontweight='bold')
        stages_text = """
1. Wiener (optimal MSE)
2. H-infinity (robust)
3. Anisotropic Diffusion
4. Kalman Smoothing
"""
        axes[1, 2].text(0.1, 0.7, stages_text, ha='left', va='top', fontfamily='monospace', fontsize=9)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f'{image_name}_reconstruction_visual.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print(f"    ✓ Saved visualization: {image_name}_reconstruction_visual.png")
    
    def run_full_pipeline(self):
        """Execute full reconstruction pipeline on multiple images."""
        print("\n" + "="*70)
        print("STATISTICAL OPTIMAL CONTROL MRI RECONSTRUCTION PIPELINE")
        print("="*70)
        
        results_summary = []
        
        # Process different imaging modes
        modes = ['cardiac', 'neurological', 'vascular']
        
        for mode in modes:
            print(f"\n[+] Processing {mode.upper()} imaging mode...")
            
            noisy_image, ground_truth = self.load_or_generate_test_image(mode)
            result, reconstructed_clipped = self.reconstruct_single_image(
                noisy_image, 
                f'{mode}_image'
            )
            
            # Visualize
            self.visualize_reconstruction(
                noisy_image,
                ground_truth,
                reconstructed_clipped,
                result['artifact_map'],
                mode,
                {
                    'original_snr_db': result['original_snr_db'],
                    'final_snr_db': result['final_snr_db'],
                    'snr_improvement_db': result['snr_improvement_db'],
                    'noise_variance_est': result['noise_variance_est']
                }
            )
            
            results_summary.append({
                'mode': mode,
                'metrics': {
                    'original_snr': result['original_snr_db'],
                    'final_snr': result['final_snr_db'],
                    'improvement': result['snr_improvement_db']
                }
            })
        
        # Print summary
        print("\n" + "="*70)
        print("RECONSTRUCTION PIPELINE SUMMARY")
        print("="*70)
        for item in results_summary:
            print(f"\n{item['mode'].upper()}:")
            print(f"  Original SNR:         {item['metrics']['original_snr']:.2f} dB")
            print(f"  Final SNR:            {item['metrics']['final_snr']:.2f} dB")
            print(f"  Improvement:          {item['metrics']['improvement']:.2f} dB")
        
        print("\n" + "="*70)
        print("All reconstructions complete. Results saved to:", self.output_dir)
        print("="*70 + "\n")
        
        return results_summary


if __name__ == '__main__':
    pipeline = MRIReconstructionPipeline()
    results = pipeline.run_full_pipeline()
