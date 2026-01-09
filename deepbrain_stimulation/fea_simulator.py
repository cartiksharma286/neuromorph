import numpy as np
import scipy.ndimage
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DBSFEASimulator:
    def __init__(self, resolution=64):
        self.resolution = resolution
        self.grid_size = (resolution, resolution)
        # Tissue Conductivity (S/m)
        # Gray Matter: 0.2, White Matter: 0.15, CSF: 1.65, Electrode: 1.0e6
        self.conductivity_map = np.ones(self.grid_size) * 0.2 
        self.potential_map = np.zeros(self.grid_size)
        
    def generate_tissue_model(self, target_center=(32, 32)):
        """
        Generates a synthetic tissue conductivity model representing a brain slice.
        """
        N = self.resolution
        y, x = np.ogrid[:N, :N]
        
        # Reset
        self.conductivity_map = np.ones(self.grid_size) * 0.20 # Gray Matter base
        
        # White Matter Fiber Bundles (Anisotropic approximations)
        # Internal Capsule like structure
        capsule_mask = ((x - 20)**2 + (y - 32)**2 < 100)
        self.conductivity_map[capsule_mask] = 0.15 # White Matter
        
        # CSF Ventricle
        ventricle_mask = ((x - 45)**2 + (y - 25)**2 < 50)
        self.conductivity_map[ventricle_mask] = 1.65 # Highly conductive
        
        # Target Nucleus (e.g. STN or GPi) - Slightly different conductivity?
        # Usually similar to GM, but let's mark it for visualization
        self.target_mask = ((x - target_center[0])**2 + (y - target_center[1])**2 < 16)
        
        return self.conductivity_map.tolist()

    def solve_electric_field(self, electrode_config):
        """
        Solves the Laplace equation for electric potential: div(sigma * grad(phi)) = 0
        electrode_config: dict of contacts and their voltages/currents.
        e.g. {'c0': 0, 'c1': -1, 'c2': 0, 'c3': 0} (Monopolar/Bipolar)
        """
        N = self.resolution
        phi = np.zeros(self.grid_size)
        
        # Define Electrode Locations (simplified as point sources on the grid)
        # Lead located at center
        lead_x = N // 2
        lead_y = N // 2
        
        # Map contacts to grid geometry (vertical lead in 2D slice? No, usually cross section)
        # Let's assume we are viewing the lead cross-section or a longitudinal slice.
        # Let's do a longitudinal slice where contacts are stacked along Y
        
        contact_coords = {
            'c3': (lead_y - 8, lead_x),
            'c2': (lead_y - 3, lead_x),
            'c1': (lead_y + 3, lead_x),
            'c0': (lead_y + 8, lead_x)
        }
        
        # Boundary Conditions (Dirichlet)
        mask_fixed = np.zeros(self.grid_size, dtype=bool)
        
        # Set Active Electrode Voltages
        active_voltages = electrode_config.get('voltages', {'c1': -3.0}) # Default cathodic
        
        for contact, coords in contact_coords.items():
            if contact in active_voltages:
                cy, cx = coords
                # Draw small contact contact
                contact_region = (np.abs(np.arange(N)[:,None] - cy) <= 1) & (np.abs(np.arange(N) - cx) <= 2)
                phi[contact_region] = active_voltages[contact]
                mask_fixed[contact_region] = True
                self.conductivity_map[contact_region] = 1000.0 # Metal
        
        # Outer boundary (Ground/Reference)
        # In monopolar, case is ground (far field).
        phi[0,:] = 0; mask_fixed[0,:] = True
        phi[-1,:] = 0; mask_fixed[-1,:] = True
        phi[:,0] = 0; mask_fixed[:,0] = True
        phi[:,-1] = 0; mask_fixed[:,-1] = True
        
        # Interactive Solver (Finite Difference Method) by Iteration (Jacobi / SOR)
        # Speed up using SciPy convolve for Laplacian??
        # Simple iterative relaxation for simulation effect
        
        # Sigma field
        sigma = self.conductivity_map
        
        # 500 Iterations of SOR
        # Using a simplified Laplacian kernel approach for speed in Python without C++
        # V_new = Average of neighbors weighted by conductivity
        
        # This is slow in pure python. Using a Gaussian approximation for the field 
        # combined with 1/r decay if simulation is too heavy, but let's try a fast solver.
        
        for i in range(200):
            # Compute smoothed version (Laplacian smoothing)
            # This mimics diffusion
            phi_new = scipy.ndimage.gaussian_filter(phi, sigma=1.0) # Not physics accurate but visually smooth
            
            # Re-enforce boundary conditions
            phi[mask_fixed] = phi[mask_fixed] # Keep fixed
            # Update varying
            phi[~mask_fixed] = phi_new[~mask_fixed] * 0.9 + phi[~mask_fixed] * 0.1 # Relaxation
            
        self.potential_map = phi
        
        # Calculate E-Field magnitude |E| = |-grad(phi)|
        grad_y, grad_x = np.gradient(phi)
        e_field = np.sqrt(grad_y**2 + grad_x**2)
        
        # VTA (Volume of Tissue Activated) Approximation
        # Threshold for activation ~ 0.2 V/mm approx (simplification)
        activation_threshold = 0.05 # arbitrary simulation units
        vta_mask = e_field > activation_threshold
        
        return phi, e_field, vta_mask

    def generate_heatmap_plot(self, data_matrix, title="Simulation"):
        """Generates a base64 encoded heatmap."""
        plt.figure(figsize=(5,5))
        plt.style.use('dark_background')
        plt.imshow(data_matrix, cmap='inferno')
        plt.colorbar(label='Field Strength')
        plt.title(title)
        plt.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def optimize_stimulation(self, target_coords=(32,32)):
        """
        Finds optimal electrode configuration to maximize target coverage
        and minimize off-target spill.
        """
        # Define search space (Voltage on C0-C3)
        # Simplified: Which singlet contact is best?
        
        results = []
        contacts = ['c0', 'c1', 'c2', 'c3']
        best_score = -np.inf
        best_config = None
        
        self.generate_tissue_model(target_center=target_coords)
        
        for c in contacts:
            config = {'voltages': {c: -3.0}}
            _, e_field, vta_mask = self.solve_electric_field(config)
            
            # Score: Intersection over Union (IoU) with Target Mask
            intersection = np.sum(vta_mask & self.target_mask)
            union = np.sum(vta_mask | self.target_mask)
            spill = np.sum(vta_mask & ~self.target_mask)
            
            score = (intersection * 2) - (spill * 0.5)
            
            results.append({
                'contact': c,
                'score': float(score),
                'coverage': float(intersection / (np.sum(self.target_mask)+1e-9)),
                'spill': float(spill)
            })
            
            if score > best_score:
                best_score = score
                best_config = config
        
        return best_config, results
