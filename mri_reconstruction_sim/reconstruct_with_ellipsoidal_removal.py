#!/usr/bin/env python3
"""
Ellipsoidal Bounding Box MRI Reconstruction with White Speckle Artifact Removal
=================================================================================

This script demonstrates the complete reconstruction pipeline with:
  1. K-space acquisition simulation
  2. Ellipsoidal bounding box detection (adaptive)
  3. White speckle noise artifact identification
  4. Artifact removal and image cleaning
  5. Quality metrics and visualization

Usage:
  python3 reconstruct_with_ellipsoidal_removal.py
"""

import numpy as np
import matplotlib.pyplot as plt
from simulator_core import MRIReconstructionSimulator
from ellipsoidal_artifact_removal import EllipsoidalArtifactRemover
import base64
import os


def reconstruct_brain_with_artifact_removal():
    """Demonstrates brain reconstruction with ellipsoidal artifact removal."""
    
    print("=" * 80)
    print("BRAIN MRI RECONSTRUCTION WITH ELLIPSOIDAL ARTIFACT REMOVAL")
    print("=" * 80)
    
    # Initialize simulator
    resolution = 256
    sim = MRIReconstructionSimulator(resolution=resolution)
    
    print(f"\n[1/6] Setting up phantom (resolution: {resolution}×{resolution})...")
    sim.setup_phantom(use_real_data=True, phantom_type='brain')
    
    print("[2/6] Generating RF coil sensitivities (8-coil array)...")
    sim.generate_coil_sensitivities(num_coils=8, coil_type='standard')
    
    print("[3/6] Acquiring k-space signal...")
    # Simulate a T1-weighted acquisition
    kspace, M_ref = sim.acquire_signal(
        sequence_type='SE',
        TR=500,
        TE=15,
        flip_angle=90,
        noise_level=0.02  # 2% Rician noise
    )
    print(f"   - K-space shape: {kspace[0].shape}")
    print(f"   - Number of coils: {len(kspace)}")
    
    print("[4/6] Reconstructing image (Sum of Squares)...")
    recon_standard, coil_imgs = sim.reconstruct_image(kspace, method='SoS')
    
    print("[5/6] Detecting and removing artifacts with ellipsoidal mask...")
    remover = EllipsoidalArtifactRemover(phantom_type='brain', resolution=resolution)
    
    # Adaptive ellipsoid fitting
    ellipse_params = remover.adaptive_ellipsoid_fit(recon_standard)
    print(f"   - Ellipsoid Center: ({ellipse_params['center_x']:.1f}, {ellipse_params['center_y']:.1f})")
    print(f"   - Semi-axes: a={ellipse_params['a']:.1f}, b={ellipse_params['b']:.1f}")
    print(f"   - Rotation angle: {np.degrees(ellipse_params['angle']):.1f}°")
    
    # Detect white speckles
    speckle_likelihood = remover.detect_white_speckles(recon_standard)
    
    # Remove artifacts
    recon_cleaned, artifact_stats = remover.remove_artifacts(
        recon_standard,
        use_ellipsoid=True,
        speckle_threshold=0.75
    )
    
    print(f"   - Artifacts detected and removed: {artifact_stats['artifacts_removed_count']}")
    print(f"   - Intensity preservation: {artifact_stats['intensity_preserved']*100:.1f}%")
    
    print("[6/6] Computing quality metrics...")
    metrics_standard = sim.compute_metrics(recon_standard, M_ref)
    metrics_cleaned = sim.compute_metrics(recon_cleaned, M_ref)
    
    print("\n" + "=" * 80)
    print("RECONSTRUCTION QUALITY COMPARISON")
    print("=" * 80)
    
    metrics_to_compare = ['SNR', 'contrast', 'sharpness']
    print(f"\n{'Metric':<15} {'Standard':<15} {'Artifact-Removed':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for metric in metrics_to_compare:
        val_std = metrics_standard.get(metric, 0)
        val_clean = metrics_cleaned.get(metric, 0)
        improvement = ((val_clean - val_std) / (val_std + 1e-9)) * 100
        
        print(f"{metric:<15} {val_std:<15.4f} {val_clean:<15.4f} {improvement:>+.1f}%")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    output_dir = 'reconstruction_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0f172a')
    
    im0 = axes[0].imshow(recon_standard, cmap='gray')
    axes[0].set_title('Standard Reconstruction', color='#38bdf8', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(speckle_likelihood, cmap='hot')
    axes[1].set_title('White Speckle Detection', color='#f43f5e', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(recon_cleaned, cmap='gray')
    axes[2].set_title('Artifact-Removed Reconstruction', color='#22c55e', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, facecolor='#0f172a', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {output_dir}/comparison.png")
    
    # Figure 2: Ellipsoidal mask
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')
    
    axes[0].imshow(recon_standard, cmap='gray')
    # Draw ellipse contour
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(
        (ellipse_params['center_x'], ellipse_params['center_y']),
        2*ellipse_params['a'], 2*ellipse_params['b'],
        angle=np.degrees(ellipse_params['angle']),
        fill=False, linewidth=2.5, color='#38bdf8', linestyle='--'
    )
    axes[0].add_patch(ellipse)
    axes[0].set_title('Ellipsoidal Bounding Box Fit', color='#38bdf8', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    mask = remover.create_ellipsoidal_mask()
    im = axes[1].imshow(mask, cmap='RdYlGn_r')
    axes[1].set_title('Artifact Removal Mask (Inside=1, Outside=0)', color='#22c55e', fontsize=13, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ellipsoid.png'), dpi=150, facecolor='#0f172a', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {output_dir}/ellipsoid.png")
    
    # Figure 3: Quality metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1a1f35')
    
    standard_vals = [metrics_standard.get(m, 0) for m in metrics_to_compare]
    cleaned_vals = [metrics_cleaned.get(m, 0) for m in metrics_to_compare]
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, standard_vals, width, label='Standard', 
                   color='#818cf8', alpha=0.8, edgecolor='#38bdf8', linewidth=1.5)
    bars2 = ax.bar(x + width/2, cleaned_vals, width, label='Artifact-Removed', 
                   color='#22c55e', alpha=0.8, edgecolor='#16a34a', linewidth=1.5)
    
    ax.set_xlabel('Metrics', color='#94a3b8', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', color='#94a3b8', fontsize=12, fontweight='bold')
    ax.set_title('Reconstruction Quality: Standard vs. Artifact-Removed', 
                color='#38bdf8', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_compare, color='#94a3b8')
    ax.legend(facecolor='#0f172a', edgecolor='#38bdf8', labelcolor='#94a3b8', fontsize=11)
    ax.grid(True, alpha=0.15, color='#38bdf8', linestyle='--')
    ax.tick_params(colors='#94a3b8')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', color='#e2e8f0', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'metrics.png'), dpi=150, facecolor='#0f172a', bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {output_dir}/metrics.png")
    
    print("\n" + "=" * 80)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 80)
    print(f"\n✓ All results saved to: {output_dir}/")
    print(f"\nSummary:")
    print(f"  - Original image max intensity: {artifact_stats['original_max']:.4f}")
    print(f"  - Cleaned image max intensity: {artifact_stats['final_max']:.4f}")
    print(f"  - Artifacts detected and suppressed: {artifact_stats['artifacts_removed_count']}")
    print(f"  - Anatomical detail preservation: {artifact_stats['intensity_preserved']*100:.1f}%")


def reconstruct_cardiac_with_artifact_removal():
    """Demonstrates cardiac reconstruction with ellipsoidal artifact removal."""
    
    print("\n\n" + "=" * 80)
    print("CARDIAC MRI RECONSTRUCTION WITH ELLIPSOIDAL ARTIFACT REMOVAL")
    print("=" * 80)
    
    resolution = 192
    sim = MRIReconstructionSimulator(resolution=resolution)
    
    print(f"\n[1/6] Setting up cardiac phantom (resolution: {resolution}×{resolution})...")
    sim.setup_phantom(use_real_data=True, phantom_type='cardiac')
    
    print("[2/6] Generating cardiothoracic coil array sensitivities...")
    sim.generate_coil_sensitivities(num_coils=8, coil_type='cardiothoracic_array')
    
    print("[3/6] Acquiring k-space signal (bSSFP sequence)...")
    kspace, M_ref = sim.acquire_signal(
        sequence_type='SSFP',
        TR=3.5,
        TE=1.75,
        flip_angle=50,
        noise_level=0.03
    )
    
    print("[4/6] Reconstructing cardiac image...")
    recon_standard, coil_imgs = sim.reconstruct_image(kspace, method='SoS')
    
    print("[5/6] Applying ellipsoidal artifact removal...")
    remover = EllipsoidalArtifactRemover(phantom_type='cardiac', resolution=resolution)
    ellipse_params = remover.adaptive_ellipsoid_fit(recon_standard)
    recon_cleaned, artifact_stats = remover.remove_artifacts(recon_standard, use_ellipsoid=True)
    
    print(f"   - Artifacts removed: {artifact_stats['artifacts_removed_count']}")
    print(f"   - Intensity preservation: {artifact_stats['intensity_preserved']*100:.1f}%")
    
    print("[6/6] Computing quality metrics...")
    metrics_standard = sim.compute_metrics(recon_standard, M_ref)
    metrics_cleaned = sim.compute_metrics(recon_cleaned, M_ref)
    
    print("\n" + "=" * 80)
    print("CARDIAC RECONSTRUCTION SUMMARY")
    print("=" * 80)
    print(f"\nStandard Reconstruction:")
    print(f"  - SNR: {metrics_standard.get('SNR', 0):.4f}")
    print(f"  - Contrast: {metrics_standard.get('contrast', 0):.4f}")
    print(f"\nArtifact-Removed Reconstruction:")
    print(f"  - SNR: {metrics_cleaned.get('SNR', 0):.4f}")
    print(f"  - Contrast: {metrics_cleaned.get('contrast', 0):.4f}")


if __name__ == '__main__':
    # Run brain reconstruction
    reconstruct_brain_with_artifact_removal()
    
    # Run cardiac reconstruction  
    reconstruct_cardiac_with_artifact_removal()
    
    print("\n✓ All reconstructions completed successfully!\n")
