
import math
import numpy as np
import networkx as nx

def generate_primes(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    primes = []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, limit + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
    return primes

class PrimeVortexField:
    """
    Implements a theoretical field based on Prime Number distributions and 
    Quantum Surface Integrals to optimize network topology.
    """
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        # Generate enough primes to map to qubits/edges
        self.primes = generate_primes(500) 
        self.prime_gaps = [self.primes[i+1] - self.primes[i] for i in range(len(self.primes)-1)]
        
    def calculate_surface_integral(self, graph, qubit_states):
        """
        Calculates a 'Quantum Surface Integral' over the network topology.
        This imagines the graph as a discretized manifold and calculates the
        flux of coherence across it.
        
        S = \oint \psi * \nabla \psi \cdot dA
        
        Here, we approximate this using discrete graph Laplacian and state coherence.
        """
        flux = 0.0
        
        # Get Adjacency matrix and Laplacian
        L = nx.normalized_laplacian_matrix(graph).todense()
        
        # State vector (amplitude)
        psi = np.array([q.amplitude for q in qubit_states])
        
        # Calculate 'divergence' or flux across the graph surface ( Laplacian * psi )
        # This represents the flow of information/entropy
        nabla_psi = np.dot(L, psi)
        
        # Integrate (sum) over the surface (nodes)
        # We weight it by the "Prime Potential" of each node to define the surface curvature
        for i in range(min(len(psi), len(self.primes))):
            # Prime Potential: P_i = 1 / ln(p_i) (Density of primes)
            p_i = self.primes[i] if i < len(self.primes) else i*2 + 1
            potential = 1.0 / math.log(p_i) if p_i > 1 else 1.0
            
            # Surface element contribution
            # Handle both matrix (2D) and array (1D) outputs from dot product
            val = nabla_psi[0, i] if len(nabla_psi.shape) > 1 else nabla_psi[i]
            flux += abs(val) * potential
            
        return flux

    def optimize_entanglement_distribution(self, entanglements):
        """
        Returns a map of optimized entanglement strengths based on Prime Gap Statistics.
        
        Hypothesis: Quantum critical systems often exhibit energy level spacings 
        that follow distributions similar to prime gaps (GUE/GOE statistics).
        We enforce this distribution to maximize 'quantum criticality' and plasticity.
        """
        optimized = {}
        sorted_keys = sorted(entanglements.keys())
        
        # Map prime gaps to edge weights
        # We normalize prime gaps to [0, 1] range to serve as weights
        max_gap = max(self.prime_gaps[:len(sorted_keys)]) if self.prime_gaps else 10
        
        for i, key in enumerate(sorted_keys):
            if i < len(self.prime_gaps):
                # Use the normalized gap as a 'natural' weight
                # Larger gap = "stable island" = Stronger Connection? 
                # Or Larger gap = "void" = Weaker Connection?
                # Let's say larger gap -> Stronger connection to bridge the void.
                gap_weight = self.prime_gaps[i] / max_gap
                
                # Smooth it: 0.1 + 0.9 * gap_weight
                weight = 0.2 + 0.8 * gap_weight
                optimized[key] = weight
            else:
                optimized[key] = entanglements[key]
                
        return optimized

    def calculate_repair_vector(self, current_topology, target_density):
        """
        Determines where to add edges to minimize Surface Tension (dementia).
        Uses a 'Prime Gradient' approach.
        """
        new_edges = []
        n = self.num_qubits
        
        # Identify "Low Prime Potential" areas -> nodes with low degree
        degrees = dict(current_topology.degree())
        
        # Sort nodes by degree (ascending) - these are the "damaged" areas
        weak_nodes = sorted(degrees, key=degrees.get)
        
        # Attempt to link weak nodes using Prime stepping
        # Connect node i to node (i + p) % n where p is a prime
        for i in weak_nodes[:5]: # Take top 5 weakest nodes
            for p in self.primes[:10]: # Try first 10 primes as step sizes
                target = (i + p) % n
                if target != i and not current_topology.has_edge(i, target):
                    # We found a prime-harmonic connection
                    new_edges.append((i, target))
                    if len(new_edges) >= 3: # Limit changes per step
                        break
            if len(new_edges) >= 5:
                break
                
        return new_edges
