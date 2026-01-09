#!/usr/bin/env python3
"""
Generate simulation images for the report and create PDF with embedded figures.
"""
import sys
sys.path.insert(0, '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim')

from simulator_core import MRIReconstructionSimulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/report_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_simulation_images():
    """Run simulations and save images for the report."""
    
    configs = [
        {'name': 'standard_se', 'coil': 'standard', 'seq': 'SE', 'tr': 2000, 'te': 100, 'ti': 500, 'fa': 90, 'label': 'Standard Coil + Spin Echo'},
        {'name': 'gemini_quantum', 'coil': 'gemini_14t', 'seq': 'QuantumEntangled', 'tr': 2000, 'te': 50, 'ti': 500, 'fa': 90, 'label': 'Gemini 14T + Quantum Entangled'},
        {'name': 'n25_zeropoint', 'coil': 'n25_array', 'seq': 'ZeroPointGradients', 'tr': 500, 'te': 20, 'ti': 500, 'fa': 60, 'label': 'N25 Array + Zero-Point Gradients'},
        {'name': 'lattice_dual_berry', 'coil': 'quantum_surface_lattice', 'seq': 'QuantumDualIntegral', 'tr': 3000, 'te': 30, 'ti': 500, 'fa': 45, 'label': 'Quantum Lattice + Dual Integral (Berry Phase)'},
        {'name': 'phased_congruence', 'coil': 'custom_phased_array', 'seq': 'QuantumStatisticalCongruence', 'tr': 1500, 'te': 80, 'ti': 500, 'fa': 90, 'label': 'Phased Array + Statistical Congruence'},
    ]
    
    for cfg in configs:
        print(f"Running simulation: {cfg['label']}...")
        
        sim = MRIReconstructionSimulator(resolution=128)
        sim.generate_brain_phantom()
        sim.generate_coil_sensitivities(num_coils=8, coil_type=cfg['coil'])
        
        kspace, M_ref = sim.acquire_signal(
            sequence_type=cfg['seq'],
            TR=cfg['tr'],
            TE=cfg['te'],
            TI=cfg['ti'],
            flip_angle=cfg['fa'],
            noise_level=0.02
        )
        
        recon_img, _ = sim.reconstruct_image(kspace, method='SoS')
        
        # K-space (acquired)
        avg_k = np.mean(np.abs(np.array(kspace)), axis=0)
        
        # Ideal K-space
        k_gt = np.fft.fftshift(np.fft.fft2(M_ref))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(cfg['label'], fontsize=14, fontweight='bold')
        
        # Ground Truth
        axes[0, 0].imshow(M_ref, cmap='gray', origin='lower')
        axes[0, 0].set_title('Ground Truth (Ideal M)')
        axes[0, 0].axis('off')
        
        # Reconstructed
        axes[0, 1].imshow(recon_img, cmap='gray', origin='lower')
        axes[0, 1].set_title('Reconstructed Image')
        axes[0, 1].axis('off')
        
        # Ideal K-Space
        axes[1, 0].imshow(np.log(np.abs(k_gt) + 1e-5), cmap='viridis', origin='lower')
        axes[1, 0].set_title('Ideal K-Space')
        axes[1, 0].axis('off')
        
        # Acquired K-Space
        axes[1, 1].imshow(np.log(avg_k + 1e-5), cmap='viridis', origin='lower')
        axes[1, 1].set_title('Acquired K-Space (with noise)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{cfg['name']}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {cfg['name']}.png")
    
    print("All simulation images generated.")

if __name__ == '__main__':
    generate_simulation_images()
