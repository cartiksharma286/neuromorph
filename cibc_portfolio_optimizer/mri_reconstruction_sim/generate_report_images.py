#!/usr/bin/env python3
"""
Generate simulation images for the report and create PDF with embedded figures.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from simulator_core import MRIReconstructionSimulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.getcwd(), 'report_images')
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
    
    print("Simulation images generated.")

def generate_circuit_images():
    """Generates schematic diagrams for all coil types."""
    coil_types = ['standard', 'solenoid', 'geodesic_chassis', 'quantum_surface_lattice', 'gemini_14t', 'n25_array', 'custom_phased_array']
    
    for c_type in coil_types:
        print(f"Generating circuit schematic for: {c_type}...")
        
        # Instantiate sim just to use its helper if needed, 
        # but we can also just recreate the plotting logic here for cleaner control over the file output.
        # Actually, let's just reuse the logic we know works but customized for the report (white background, etc).
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Report style: White background
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.axis('off')
        
        color_wire = 'black'
        color_comp = '#0066cc' # Blue components
        
        if c_type == 'standard':
            ax.set_title("Standard Birdcage Coil (High-Pass)", fontsize=16, fontweight='bold')
            t = np.linspace(0, 2*np.pi, 100)
            # Rings (Thicker lines for structure)
            ax.plot(np.cos(t), np.sin(t), color=color_wire, lw=3)
            ax.plot(0.7*np.cos(t), 0.7*np.sin(t), color=color_wire, lw=3)
            # Rungs
            for i in range(8):
                ang = 2*np.pi * i / 8
                x1, y1 = np.cos(ang), np.sin(ang)
                x2, y2 = 0.7*np.cos(ang), 0.7*np.sin(ang)
                # Wire
                ax.plot([x1, x2], [y1, y2], color=color_comp, lw=2, linestyle='--') 
                # Capacitor Symbol
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                dx, dy = (x2-x1)*0.1, (y2-y1)*0.1
                # Plate 1
                ax.plot([mid_x-dy, mid_x+dy], [mid_y+dx, mid_y-dx], color='red', lw=3)
                # Plate 2
                ax.plot([mid_x-dy*2, mid_x+dy*2], [mid_y+dx*2, mid_y-dx*2], color='red', lw=0) # Spacer hack
                ax.text(mid_x, mid_y, "||", color='red', ha='center', va='center', rotation=np.degrees(ang)+90, fontsize=14, fontweight='bold')
            
            ax.text(0, 0, "8-Rung Birdcage\n(High-Pass Mode)", ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        elif c_type == 'solenoid':
            ax.set_title("Solenoid Coil Circuit (High-Q)", fontsize=16, fontweight='bold')
            # Inductor (Zig-Zag)
            x_ind = np.linspace(1, 5, 200)
            y_ind = 2 + 0.3 * np.sin(20*x_ind)
            ax.plot(x_ind, y_ind, color=color_wire, lw=3, label='Inductor')
            ax.text(3, 2.6, "$L_{solenoid}$", color='black', ha='center', fontsize=14)
            
            # Connections
            ax.plot([1, 1], [2, 1], color=color_wire, lw=2)
            ax.plot([5, 5], [2, 1], color=color_wire, lw=2)
            
            # Tuning Cap (Variable)
            ax.plot([1, 2.8], [1, 1], color=color_wire, lw=2)
            ax.plot([3.2, 5], [1, 1], color=color_wire, lw=2)
            # Cap Plates
            ax.plot([2.8, 2.8], [0.8, 1.2], color=color_comp, lw=3)
            ax.plot([3.2, 3.2], [0.8, 1.2], color=color_comp, lw=3)
            # Variable Arrow
            ax.arrow(2.6, 0.7, 0.8, 0.6, head_width=0.1, color=color_comp)
            ax.text(3, 0.4, "$C_{tune}$ (Variable)", color=color_comp, ha='center', fontsize=12)

        elif c_type in ['geodesic_chassis', 'n25_array', 'custom_phased_array', 'gemini_14t']:
            title_map = {
                'geodesic_chassis': 'Geodesic Element (Golden Angle)', 
                'n25_array': 'N25 High-Density Array Element',
                'custom_phased_array': 'Phased Array Element (8ch)',
                'gemini_14t': 'Gemini 14T Ultra-High Field Element'
            }
            ax.set_title(title_map.get(c_type, c_type), fontsize=16, fontweight='bold')
            
            y_rail = 3
            # Input Loop Inductor (Zig Zag)
            x_l = np.linspace(0, 2, 100)
            y_l = y_rail + 0.2*np.sin(20*x_l)
            ax.plot(x_l, y_l, color=color_wire, lw=2)
            ax.text(1, y_rail+0.4, "$L_{loop}$", color='black', ha='center', fontsize=12)
            
            # Series Tune Cap
            ax.plot([2, 2.4], [y_rail, y_rail], color=color_wire, lw=2)
            ax.plot([2.4, 2.4], [y_rail-0.3, y_rail+0.3], color=color_comp, lw=3) # Plate 1
            ax.plot([2.6, 2.6], [y_rail-0.3, y_rail+0.3], color=color_comp, lw=3) # Plate 2
            ax.plot([2.6, 3], [y_rail, y_rail], color=color_wire, lw=2)
            ax.text(2.5, y_rail+0.5, "$C_{tune}$", color=color_comp, ha='center', fontsize=12)
            
            # Transmission Line
            ax.plot([3, 4], [y_rail, y_rail], color=color_wire, lw=2)
            
            # Parallel Match Cap
            ax.plot([4, 4], [y_rail, 1.5], color=color_wire, lw=2)
            ax.plot([3.8, 4.2], [1.5, 1.5], color=color_comp, lw=3)
            ax.plot([3.8, 4.2], [1.3, 1.3], color=color_comp, lw=3)
            ax.plot([4, 4], [1.3, 1], color=color_wire, lw=2)
            ax.text(4.4, 1.4, "$C_{match}$", color=color_comp, va='center', fontsize=12)
            
            # Ground Line
            ax.plot([0, 6], [1, 1], color='gray', lw=2, linestyle='--')
            ax.text(0.5, 0.8, "GND", color='gray', fontsize=10)
            
            # Ground Symbol at Match
            ax.plot([3.8, 4.2], [1, 1], color='gray', lw=2)
            ax.plot([3.9, 4.1], [0.9, 0.9], color='gray', lw=2)
            ax.plot([3.95, 4.05], [0.8, 0.8], color='gray', lw=2)
            
            # Output
            ax.plot([4, 6], [y_rail, y_rail], color=color_wire, lw=2)
            ax.arrow(6, y_rail, 0.5, 0, head_width=0.2, color='black')
            ax.text(6.6, y_rail, "Preamp (LNA)", ha='left', va='center', fontsize=12)
            
        elif c_type == 'quantum_surface_lattice':
            ax.set_title("Quantum Surface Lattice (Berry Phase)", fontsize=16, fontweight='bold')
            # Hexagonal Lattice
            for i in range(4):
                for j in range(4):
                    cx, cy = i*1.5, j*1.5 + (0.75 if i%2 else 0)
                    # Draw Hexagon
                    hex_ft = np.linspace(0, 2*np.pi, 7)
                    ax.plot(cx + 0.5*np.cos(hex_ft), cy + 0.5*np.sin(hex_ft), color='#0066cc', lw=1.5, alpha=0.7)
                    # Flux Vector
                    ax.arrow(cx, cy, 0, 0.3, head_width=0.1, color='red', alpha=0.5)
            
            ax.text(3, -0.5, r"Flux Quantum $\Phi_0$", color='red', ha='center', fontsize=12)
            ax.text(3, 5.5, "Berry Curvature Field $\Omega(k)$", ha='center', fontsize=12, color='darkblue')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"circuit_{c_type}.png"), dpi=150, bbox_inches='tight')
        plt.close()

def generate_all():
    generate_simulation_images()
    generate_circuit_images()

if __name__ == '__main__':
    generate_all()
