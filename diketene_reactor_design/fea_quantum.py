import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.sparse import lil_matrix, linalg
from sklearn.cluster import KMeans

class QuantumFEASolver:
    def __init__(self, diameter, height, grid_resolution=50):
        self.D = diameter
        self.H = height
        self.Nx = grid_resolution # Circumferential
        self.Ny = int(grid_resolution * (height / (np.pi * diameter))) # Axial scaled
        if self.Ny < 10: self.Ny = 10
        
        self.num_nodes = self.Nx * self.Ny
        self.system_matrix = None
        self.solution = None
        self.partitions = None

    def generate_mesh(self):
        """Generates a 2D unrolled mesh of the reactor surface."""
        x = np.linspace(0, np.pi * self.D, self.Nx)
        y = np.linspace(0, self.H, self.Ny)
        xv, yv = np.meshgrid(x, y)
        self.nodes = np.stack([xv.flatten(), yv.flatten()], axis=1)
        return self.nodes

    def assemble_matrix_quantum_partition(self, num_partitions=4):
        """
        Assembles a simplified thermal/stress Laplacian matrix and partitions it 
        simulating Quantum Qubit topology mapping.
        """
        N = self.num_nodes
        K = lil_matrix((N, N))
        
        # 5-point stencil (Finite Difference approximation on FEM nodes)
        for i in range(self.Ny):
            for j in range(self.Nx):
                idx = i * self.Nx + j
                K[idx, idx] = 4.0
                
                # Neighbors
                if j > 0: K[idx, idx - 1] = -1.0
                if j < self.Nx - 1: K[idx, idx + 1] = -1.0
                if i > 0: K[idx, idx - self.Nx] = -1.0
                if i < self.Ny - 1: K[idx, idx + self.Nx] = -1.0
                
                # Periodic BC for cylinder (left connects to right)
                if j == 0: K[idx, idx + self.Nx - 1] = -1.0
                if j == self.Nx - 1: K[idx, idx - self.Nx + 1] = -1.0

        self.system_matrix = K.tocsr()
        
        # Matrix Partitioning (Simulating Quantum Domain Decomposition)
        # We use KMeans on spatial coordinates to chunk the matrix
        kmeans = KMeans(n_clusters=num_partitions, n_init=10)
        self.partitions = kmeans.fit_predict(self.nodes)
        
        return self.partitions

    def solve_field(self):
        """
        Solves the system Kx = F using a simulated quantum linear solver step.
        (Here represented by standard sparse solve, but concept maps to HHL algorithm).
        """
        # Load Vector (Non-uniform heat/stress load)
        F = np.zeros(self.num_nodes)
        # Apply a Gaussian hot-spot (e.g., reaction runaway zone)
        center_idx = self.num_nodes // 2 + self.Nx // 4
        cx, cy = self.nodes[center_idx]
        
        for k in range(self.num_nodes):
            nx, ny = self.nodes[k]
            dist = (nx - cx)**2 + (ny - cy)**2
            F[k] = 1000.0 * np.exp(-dist / (0.1 * self.D)**2)
            
        # Solve
        # In a real QC app, this would be: x = QuantumInvert(K) @ F
        self.solution = linalg.spsolve(self.system_matrix, F)
        return self.solution

    def render_plots(self, output_base):
        """Generates plots for the partitioned grid and the FEA solution."""
        
        # Plot 1: Quantum Partitioning Grid
        plt.figure(figsize=(10, 6))
        plt.scatter(self.nodes[:, 0], self.nodes[:, 1], c=self.partitions, cmap='tab10', s=10, marker='s')
        plt.title(f"Quantum Matrix Partitioning (N={len(np.unique(self.partitions))})")
        plt.xlabel("Circumferential Position (m)")
        plt.ylabel("Axial Height (m)")
        plt.colorbar(label="QPU Partition ID")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_base}_partition.png")
        plt.close()
        
        # Plot 2: Field Solution (Heat/Stress)
        plt.figure(figsize=(10, 6))
        # Reshape for contour
        Z = self.solution.reshape(self.Ny, self.Nx)
        Extent = [0, np.pi * self.D, 0, self.H]
        
        plt.imshow(Z, origin='lower', extent=Extent, cmap='inferno', aspect='auto')
        plt.title("Finite Element Field Solution (Simulated Quantum Solver)")
        plt.xlabel("Circumferential Position (m)")
        plt.ylabel("Axial Height (m)")
        plt.colorbar(label="Field Intensity (Relative)")
        plt.savefig(f"{output_base}_solution.png")
        plt.close()

        return {
            "partition_plot": f"{output_base}_partition.png",
            "solution_plot": f"{output_base}_solution.png"
        }
