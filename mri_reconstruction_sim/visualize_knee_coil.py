"""
Visualization and Testing Suite for Knee Vascular Coil
Creates comprehensive visualizations of coil geometry, vascular anatomy,
and reconstruction results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
from knee_vascular_coil import KneeVascularCoil, KneeVascularReconstruction


def visualize_coil_geometry(coil: KneeVascularCoil, save_path: str = None):
    """Visualize 3D coil element arrangement."""
    fig = plt.figure(figsize=(15, 5))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    positions = coil.element_positions
    
    # Plot coil elements
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='blue', s=200, alpha=0.6, edgecolors='darkblue', linewidth=2)
    
    # Draw cylindrical surface
    theta = np.linspace(0, 2*np.pi, 100)
    z_cyl = np.linspace(-0.1, 0.1, 50)
    Theta, Z = np.meshgrid(theta, z_cyl)
    X = coil.coil_radius * np.cos(Theta)
    Y = coil.coil_radius * np.sin(Theta)
    
    ax1.plot_surface(X, Y, Z, alpha=0.1, color='cyan')
    
    # Label elements
    for i, pos in enumerate(positions):
        ax1.text(pos[0], pos[1], pos[2], f'{i+1}', fontsize=8, ha='center')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Coil Element Arrangement')
    ax1.view_init(elev=20, azim=45)
    
    # Top view
    ax2 = fig.add_subplot(132)
    ax2.scatter(positions[:, 0], positions[:, 1], c='blue', s=200, 
               alpha=0.6, edgecolors='darkblue', linewidth=2)
    
    # Draw coil elements as circles
    for i, pos in enumerate(positions):
        circle = Circle((pos[0], pos[1]), coil.element_size/2, 
                       fill=False, edgecolor='blue', linewidth=1.5, alpha=0.5)
        ax2.add_patch(circle)
        ax2.text(pos[0], pos[1], f'{i+1}', fontsize=8, ha='center', va='center')
    
    # Draw knee outline
    knee_circle = Circle((0, 0), 0.06, fill=False, edgecolor='red', 
                        linewidth=2, linestyle='--', label='Knee')
    ax2.add_patch(knee_circle)
    
    ax2.set_xlim(-0.15, 0.15)
    ax2.set_ylim(-0.15, 0.15)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (Axial)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Side view
    ax3 = fig.add_subplot(133)
    ax3.scatter(positions[:, 1], positions[:, 2], c='blue', s=200,
               alpha=0.6, edgecolors='darkblue', linewidth=2)
    
    for i, pos in enumerate(positions):
        ax3.text(pos[1], pos[2], f'{i+1}', fontsize=8, ha='center', va='center')
    
    ax3.set_xlim(-0.15, 0.15)
    ax3.set_ylim(-0.15, 0.15)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (Sagittal)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Coil geometry saved to: {save_path}")
    
    return fig


def visualize_vascular_anatomy(coil: KneeVascularCoil, save_path: str = None):
    """Visualize knee vascular anatomy in 3D."""
    fig = plt.figure(figsize=(16, 6))
    
    # 3D view
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot arteries
    colors_arteries = ['red', 'darkred', 'crimson', 'firebrick', 'indianred']
    for i, (name, vessel) in enumerate(coil.vascular_segments['arteries'].items()):
        path = vessel['path']
        color = colors_arteries[i % len(colors_arteries)]
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], 
                color=color, linewidth=vessel['diameter']*500, 
                label=name.replace('_', ' ').title(), alpha=0.8)
    
    # Plot veins
    for name, vessel in coil.vascular_segments['veins'].items():
        path = vessel['path']
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], 
                color='blue', linewidth=vessel['diameter']*500,
                label=name.replace('_', ' ').title(), alpha=0.8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Vascular Anatomy')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.view_init(elev=15, azim=120)
    
    # Sagittal view (side)
    ax2 = fig.add_subplot(132)
    
    for i, (name, vessel) in enumerate(coil.vascular_segments['arteries'].items()):
        path = vessel['path']
        color = colors_arteries[i % len(colors_arteries)]
        ax2.plot(path[:, 1], path[:, 2], 
                color=color, linewidth=vessel['diameter']*1000,
                label=name.replace('_', ' ').title(), alpha=0.8)
    
    for name, vessel in coil.vascular_segments['veins'].items():
        path = vessel['path']
        ax2.plot(path[:, 1], path[:, 2], 
                color='blue', linewidth=vessel['diameter']*1000,
                label=name.replace('_', ' ').title(), alpha=0.8)
    
    ax2.set_xlabel('Y - Anterior/Posterior (m)')
    ax2.set_ylabel('Z - Superior/Inferior (m)')
    ax2.set_title('Sagittal View')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Axial view (top)
    ax3 = fig.add_subplot(133)
    
    for i, (name, vessel) in enumerate(coil.vascular_segments['arteries'].items()):
        path = vessel['path']
        color = colors_arteries[i % len(colors_arteries)]
        ax3.plot(path[:, 0], path[:, 1], 
                color=color, linewidth=vessel['diameter']*1000,
                label=name.replace('_', ' ').title(), alpha=0.8)
    
    for name, vessel in coil.vascular_segments['veins'].items():
        path = vessel['path']
        ax3.plot(path[:, 0], path[:, 1], 
                color='blue', linewidth=vessel['diameter']*1000,
                label=name.replace('_', ' ').title(), alpha=0.8)
    
    ax3.set_xlabel('X - Medial/Lateral (m)')
    ax3.set_ylabel('Y - Anterior/Posterior (m)')
    ax3.set_title('Axial View')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Vascular anatomy saved to: {save_path}")
    
    return fig


def visualize_reconstruction_results(results: dict, save_path: str = None):
    """Visualize reconstruction results for different pulse sequences."""
    n_sequences = len(results)
    fig, axes = plt.subplots(2, n_sequences, figsize=(5*n_sequences, 10))
    
    if n_sequences == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (seq_name, result) in enumerate(results.items()):
        # Display 2D image
        ax_img = axes[0, idx]
        im = ax_img.imshow(result['image_2d'], cmap='gray', aspect='auto')
        ax_img.set_title(f'{seq_name}\nSNR: {result["snr"]:.1f}')
        ax_img.axis('off')
        plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        
        # Display k-space
        ax_k = axes[1, idx]
        k_space_2d = result['k_space'][result['k_space'].shape[0]//2, :, :]
        im_k = ax_k.imshow(np.log1p(k_space_2d), cmap='hot', aspect='auto')
        ax_k.set_title(f'K-Space (log scale)')
        ax_k.axis('off')
        plt.colorbar(im_k, ax=ax_k, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Reconstruction results saved to: {save_path}")
    
    return fig


def visualize_sensitivity_maps(coil: KneeVascularCoil, save_path: str = None):
    """Visualize coil sensitivity maps."""
    print("Calculating sensitivity maps (this may take a moment)...")
    
    # Use smaller grid for faster computation
    grid_size = 64
    sensitivity_maps = coil.calculate_sensitivity_map(grid_size)
    
    # Select central slice
    central_slice = grid_size // 2
    
    # Plot first 8 elements
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(8, coil.num_elements)):
        sens_2d = np.abs(sensitivity_maps[i, :, :, central_slice])
        
        im = axes[i].imshow(sens_2d, cmap='jet', aspect='auto')
        axes[i].set_title(f'Element {i+1}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Coil Sensitivity Maps (Central Slice)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sensitivity maps saved to: {save_path}")
    
    return fig


def visualize_snr_map(coil: KneeVascularCoil, save_path: str = None):
    """Visualize SNR distribution."""
    print("Calculating SNR map (this may take a moment)...")
    
    grid_size = 64
    snr_map = coil.calculate_snr_map(grid_size)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial view
    central_z = grid_size // 2
    im1 = axes[0].imshow(snr_map[:, :, central_z], cmap='hot', aspect='auto')
    axes[0].set_title('SNR Map - Axial View')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Sagittal view
    central_x = grid_size // 2
    im2 = axes[1].imshow(snr_map[central_x, :, :].T, cmap='hot', aspect='auto')
    axes[1].set_title('SNR Map - Sagittal View')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Coronal view
    central_y = grid_size // 2
    im3 = axes[2].imshow(snr_map[:, central_y, :].T, cmap='hot', aspect='auto')
    axes[2].set_title('SNR Map - Coronal View')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SNR map saved to: {save_path}")
    
    return fig


def create_comprehensive_report():
    """Create comprehensive visualization report."""
    print("\n" + "="*80)
    print("KNEE VASCULAR COIL - COMPREHENSIVE VISUALIZATION REPORT")
    print("="*80 + "\n")
    
    # Initialize coil
    print("[1/7] Initializing coil...")
    coil = KneeVascularCoil(num_elements=16, coil_radius=0.12)
    
    # Initialize reconstructor
    print("[2/7] Initializing reconstructor...")
    reconstructor = KneeVascularReconstruction(coil, matrix_size=128)
    
    # Visualize coil geometry
    print("[3/7] Visualizing coil geometry...")
    fig1 = visualize_coil_geometry(
        coil, 
        '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/knee_coil_geometry.png'
    )
    plt.close(fig1)
    
    # Visualize vascular anatomy
    print("[4/7] Visualizing vascular anatomy...")
    fig2 = visualize_vascular_anatomy(
        coil,
        '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/knee_vascular_anatomy.png'
    )
    plt.close(fig2)
    
    # Perform reconstructions
    print("[5/7] Performing pulse sequence reconstructions...")
    sequences = {
        'Proton Density': {
            'type': 'PD',
            'te': 15,
            'tr': 2000,
            'flip_angle': 90,
        },
        'TOF Angiography': {
            'type': 'TOF',
            'te': 3.5,
            'tr': 25,
            'flip_angle': 25,
        },
        'Phase Contrast': {
            'type': 'PC',
            'te': 5,
            'tr': 30,
            'flip_angle': 20,
            'venc': 0.5,
        },
    }
    
    results = {}
    for seq_name, seq_params in sequences.items():
        print(f"    - {seq_name}...")
        result = reconstructor.reconstruct_with_pulse_sequence(
            seq_params,
            acceleration=2,
            use_parallel_imaging=True
        )
        results[seq_name] = result
    
    # Visualize reconstruction results
    print("[6/7] Visualizing reconstruction results...")
    fig3 = visualize_reconstruction_results(
        results,
        '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/knee_reconstruction_results.png'
    )
    plt.close(fig3)
    
    # Visualize sensitivity maps
    print("[7/7] Visualizing sensitivity and SNR maps...")
    fig4 = visualize_sensitivity_maps(
        coil,
        '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/knee_sensitivity_maps.png'
    )
    plt.close(fig4)
    
    fig5 = visualize_snr_map(
        coil,
        '/Users/cartik_sharma/Downloads/neuromorph-main-10/mri_reconstruction_sim/knee_snr_map.png'
    )
    plt.close(fig5)
    
    # Print summary
    print("\n" + "="*80)
    print("REPORT SUMMARY")
    print("="*80)
    print(f"\nCoil Configuration:")
    print(f"  - Elements: {coil.num_elements}")
    print(f"  - Radius: {coil.coil_radius*100:.1f} cm")
    print(f"  - Frequency: {coil.frequency/1e6:.2f} MHz")
    
    print(f"\nVascular Structures:")
    total_vessels = sum(len(vessels) for vessels in coil.vascular_segments.values())
    print(f"  - Total vessels modeled: {total_vessels}")
    print(f"  - Arteries: {len(coil.vascular_segments['arteries'])}")
    print(f"  - Veins: {len(coil.vascular_segments['veins'])}")
    
    print(f"\nReconstruction Results:")
    for seq_name, result in results.items():
        print(f"  - {seq_name}:")
        print(f"      SNR: {result['snr']:.1f}")
        print(f"      Matrix: {result['image_2d'].shape}")
        print(f"      Acceleration: R={result['acceleration']}")
    
    print(f"\nGenerated Files:")
    print(f"  ✓ knee_coil_geometry.png")
    print(f"  ✓ knee_vascular_anatomy.png")
    print(f"  ✓ knee_reconstruction_results.png")
    print(f"  ✓ knee_sensitivity_maps.png")
    print(f"  ✓ knee_snr_map.png")
    print(f"  ✓ knee_vascular_coil_specs.json")
    
    print("\n" + "="*80)
    print("✓ Comprehensive report generation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    create_comprehensive_report()
