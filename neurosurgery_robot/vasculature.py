import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh

class VasculatureSpectralAnalyzer:
    """
    Finite Math Module for Vasculature Analysis.
    Computes spectral characteristics (eigenvalues/eigenvectors) of the vascular graph.
    """
    def __init__(self, num_nodes=50):
        self.num_nodes = num_nodes
        self.adj_matrix = np.zeros((num_nodes, num_nodes))
        self.generate_synthetic_vasculature()
        
    def generate_synthetic_vasculature(self):
        # Generate a random tree-like structure (mimicking vessels)
        # Simple algorithm: Attach new node to existing random node
        self.adj_matrix.fill(0)
        for i in range(1, self.num_nodes):
            parent = np.random.randint(0, i)
            self.adj_matrix[i, parent] = 1
            self.adj_matrix[parent, i] = 1 # Undirected graph for diffusion
            
            # Add some loops (anastomosis)
            if np.random.random() > 0.9:
                target = np.random.randint(0, i)
                self.adj_matrix[i, target] = 1
                self.adj_matrix[target, i] = 1

    def compute_spectrum(self):
        """
        Compute the spectrum of the Graph Laplacian.
        L = D - A
        Returns sorted eigenvalues.
        """
        # Graph Laplacian
        laplacian = csgraph.laplacian(self.adj_matrix, normed=False)
        
        # Eigen decomposition
        # We only need eigenvalues for spectral form
        vals = np.linalg.eigvalsh(laplacian)
        
        # Sort and take absolute (though they should be non-negative)
        vals = np.sort(np.abs(vals))
        return vals.tolist()

    def get_analysis(self):
        spectrum = self.compute_spectrum()
        # Characteristics
        # Fiedler value: Second smallest eigenvalue (Algebraic Connectivity)
        fiedler = spectrum[1] if len(spectrum) > 1 else 0
        spectral_radius = spectrum[-1] if len(spectrum) > 0 else 0
        
        return {
            "spectrum": spectrum, # The full spectral form
            "fiedler_value": fiedler,
            "spectral_radius": spectral_radius,
            "complexity_index": np.sum(spectrum) / self.num_nodes # Average degree approx
        }
    
    def update(self):
        # Dynamically modulate connectivity to simulate pulsatile flow or vaso-activity
        # Randomly flip an edge validity or weight (simplified)
        if np.random.random() > 0.8:
            n1 = np.random.randint(0, self.num_nodes)
            n2 = np.random.randint(0, self.num_nodes)
            if n1 != n2:
                # Toggle
                val = 1.0 - self.adj_matrix[n1, n2]
                self.adj_matrix[n1, n2] = val
                self.adj_matrix[n2, n1] = val
