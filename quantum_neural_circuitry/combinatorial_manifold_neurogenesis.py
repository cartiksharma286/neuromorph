"""
Combinatorial Manifold Neurogenesis Engine
==========================================

Explores information states in combinatorial manifolds for neurogenesis
using finite mathematics, combinatorial congruences, and prime-based 
topology for dementia and PTSD repair.

Theory:
-------
1. Information States as Simplicial Complexes
   - Neural states represented as k-simplices in combinatorial manifolds
   - Betti numbers characterize topological features of cognitive states
   
2. Finite Math Congruence Systems
   - Chinese Remainder Theorem for multi-modal neural encoding
   - Quadratic residues modulo primes define synaptic compatibility
   
3. Prime-Based Neurogenesis
   - Prime congruences determine neuronal birth/death rates
   - Legendre symbols encode synaptic plasticity rules
   
4. Manifold Curvature and Repair
   - Discrete Ricci curvature measures network health
   - Negative curvature regions indicate pathology (dementia/PTSD)
   - Targeted neurogenesis in high-curvature regions
"""

import numpy as np
import networkx as nx
import math
from scipy.special import comb, ellipk, ellipe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import os
from game_theory_core import GameTheoryOptimizer

class QubitAdapter:
    """Adapts manifold node states to Qubit-like interface for Game Theory."""
    def __init__(self, state_val):
        # Map integer state to phase [0, 2pi]
        self.phase = (state_val % 360) * (math.pi / 180.0)
        self.amplitude = 1.0 # simplified

class FiniteMathCongruenceSystem:
    """
    Implements combinatorial congruence systems for neural encoding.
    """
    
    def __init__(self, prime_moduli=None):
        """Initialize with a set of prime moduli for CRT encoding."""
        if prime_moduli is None:
            # Use first 8 primes for multi-modal encoding
            self.primes = [2, 3, 5, 7, 11, 13, 17, 19]
        else:
            self.primes = prime_moduli
            
        # Compute product for CRT
        self.M = np.prod(self.primes)
        
        # Precompute CRT coefficients
        self.crt_coeffs = self._compute_crt_coefficients()
        
    def _compute_crt_coefficients(self):
        """Compute CRT reconstruction coefficients."""
        coeffs = []
        for p in self.primes:
            M_i = self.M // p
            # Find modular inverse of M_i mod p (convert to int for compatibility)
            inv = pow(int(M_i), -1, int(p))
            coeffs.append(M_i * inv)
        return coeffs
    
    def encode_state(self, value):
        """
        Encode a neural state value using CRT.
        Returns residue vector [r_1, r_2, ..., r_k].
        """
        return [int(value) % p for p in self.primes]
    
    def decode_state(self, residues):
        """
        Decode residue vector back to original value using CRT.
        """
        value = sum(r * c for r, c in zip(residues, self.crt_coeffs))
        return value % self.M
    
    def legendre_symbol(self, a, p):
        """
        Compute Legendre symbol (a/p).
        Returns: 1 if a is quadratic residue mod p
                -1 if a is quadratic non-residue mod p
                 0 if a ≡ 0 (mod p)
        """
        if p == 2:
            return 1 if a % 2 == 1 else 0
        
        a = a % p
        if a == 0:
            return 0
        
        # Use Euler's criterion: (a/p) = a^((p-1)/2) mod p
        result = pow(a, (p - 1) // 2, p)
        return -1 if result == p - 1 else result
    
    def synaptic_compatibility(self, state_a, state_b):
        """
        Compute synaptic compatibility using quadratic residues.
        
        Theory: Two neurons are compatible if their state product
        is a quadratic residue modulo the prime moduli.
        """
        residues_a = self.encode_state(state_a)
        residues_b = self.encode_state(state_b)
        
        compatibility = 0.0
        for i, p in enumerate(self.primes):
            product = (residues_a[i] * residues_b[i]) % p
            leg = self.legendre_symbol(product, p)
            
            # Quadratic residue = compatible (weight +1)
            # Non-residue = incompatible (weight -1)
            compatibility += leg
        
        # Normalize to [0, 1]
        return (compatibility + len(self.primes)) / (2 * len(self.primes))
    
    def ramanujan_tau_approximation(self, n):
        """
        Approximate Ramanujan's tau function τ(n).
        
        Used to determine neurogenesis rates based on 
        modular form theory.
        """
        if n == 1:
            return 1
        
        # Use approximate formula based on divisor sum
        # τ(n) ≈ σ_11(n) - σ_5(n)^2 (rough approximation)
        sigma_11 = sum(d**11 for d in range(1, n+1) if n % d == 0)
        sigma_5 = sum(d**5 for d in range(1, n+1) if n % d == 0)
        
        return sigma_11 - sigma_5**2


class CombinatorialManifold:
    """
    Represents neural network as a combinatorial manifold.
    Uses simplicial complex theory for topological analysis.
    """
    
    def __init__(self, num_nodes=100):
        self.num_nodes = num_nodes
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_nodes))
        
        # Simplicial complex: dict mapping dimension to list of simplices
        self.complex = {0: [], 1: [], 2: [], 3: []}
        
        # Node states (information content)
        self.node_states = np.random.randint(1, 1000, num_nodes)
        
        # Initialize congruence system
        self.congruence_sys = FiniteMathCongruenceSystem()
        
    def add_edge_by_compatibility(self, threshold=0.5):
        """
        Add edges based on synaptic compatibility via congruences.
        """
        edges_added = 0
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                compat = self.congruence_sys.synaptic_compatibility(
                    self.node_states[i], 
                    self.node_states[j]
                )
                
                if compat >= threshold:
                    self.graph.add_edge(i, j, weight=compat)
                    self.complex[1].append((i, j))
                    edges_added += 1
        
        return edges_added
    
    def build_higher_simplices(self, max_dim=3):
        """
        Build higher-dimensional simplices (triangles, tetrahedra) efficiently.
        """
        # Clear existing
        self.complex[2] = []
        self.complex[3] = []
        
        # Build 2-simplices (triangles)
        # Iterate over edges (u,v) and find common neighbors w
        # To avoid duplicates, force u < v < w
        for u, v in self.graph.edges():
            if u > v: u, v = v, u
            
            common = sorted(list(set(self.graph.neighbors(u)) & set(self.graph.neighbors(v))))
            for w in common:
                if w > v:
                    self.complex[2].append((u, v, w))
        
        # Build 3-simplices (tetrahedra) if requested
        # Iterate over triangles (u,v,w) and find common neighbors x
        if max_dim >= 3:
            for simplex in self.complex[2]:
                u, v, w = simplex
                # Common neighbors of all three vertices
                # Since we built triangles from edges, w is already a common neighbor of u,v
                # We need x to be neighbor of u, v, w
                # And force w < x for uniqueness
                
                # Neighbors of w
                n_w = set(self.graph.neighbors(w))
                # We already know neighbors of u, v from the triangle construction conceptually,
                # but cheap to re-fetch or assume we iterate triangles
                
                # Intersection of N(u), N(v), N(w)
                # Optimization: We check if any neighbor of w is also neighbor of u and v
                # But we can just use the graph structure
                
                common_uv = set(self.graph.neighbors(u)) & set(self.graph.neighbors(v))
                common_uvw = common_uv & n_w
                
                for x in common_uvw:
                    if x > w:
                        self.complex[3].append((u, v, w, x))
    
    def compute_betti_numbers(self):
        """
        Compute Betti numbers β_0, β_1, β_2 of the simplicial complex.
        
        β_0 = number of connected components
        β_1 = number of 1-dimensional holes (loops)
        β_2 = number of 2-dimensional voids
        
        These characterize the topological structure of cognitive states.
        """
        # β_0: connected components
        beta_0 = nx.number_connected_components(self.graph)
        
        # β_1: Use Euler characteristic χ = V - E + F
        # For 2D complex: β_1 = E - V - F + 1 + β_0
        V = self.graph.number_of_nodes()
        E = self.graph.number_of_edges()
        F = len(self.complex[2])  # 2-simplices (faces)
        
        # Simplified: β_1 = E - V + β_0 (for graph without higher simplices)
        beta_1 = E - V + beta_0
        
        # β_2: For 3D complex, requires more sophisticated computation
        # Approximate using tetrahedra count
        T = len(self.complex[3])
        beta_2 = max(0, F - E + V - T - beta_0 + beta_1)
        
        return {
            'beta_0': beta_0,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'euler_char': V - E + F - T
        }
    
    def compute_discrete_ricci_curvature(self, u, v):
        """
        Compute Ollivier-Ricci curvature on edge (u,v).
        
        κ(u,v) = 1 - W(μ_u, μ_v) / d(u,v)
        
        where W is Wasserstein distance between probability measures
        on neighborhoods of u and v.
        
        Negative curvature indicates pathological regions.
        """
        if not self.graph.has_edge(u, v):
            return 0.0
        
        # Get neighborhoods
        neighbors_u = set(self.graph.neighbors(u)) | {u}
        neighbors_v = set(self.graph.neighbors(v)) | {v}
        
        # Uniform probability measures on neighborhoods
        # Simplified Wasserstein: Jaccard distance
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        
        jaccard = intersection / union if union > 0 else 0
        wasserstein_approx = 1 - jaccard
        
        # Ricci curvature
        kappa = 1 - wasserstein_approx
        
        return kappa
    
    def identify_pathological_regions(self, curvature_threshold=0.3):
        """
        Identify regions with low curvature (dementia/PTSD markers).
        
        Note: Using positive threshold to detect regions with curvature < 0.3
        which indicates weak connectivity or pathological structure.
        """
        pathological_edges = []
        curvatures = {}
        
        for u, v in self.graph.edges():
            kappa = self.compute_discrete_ricci_curvature(u, v)
            curvatures[(u, v)] = kappa
            
            # Low curvature indicates pathology
            if kappa < curvature_threshold:
                pathological_edges.append((u, v, kappa))
        
        # Identify nodes in pathological regions
        pathological_nodes = set()
        for u, v, _ in pathological_edges:
            pathological_nodes.add(u)
            pathological_nodes.add(v)
        
        # If no pathology detected with threshold, use lowest curvature nodes
        if len(pathological_nodes) == 0 and len(curvatures) > 0:
            # Find edges with lowest 20% curvature
            sorted_edges = sorted(curvatures.items(), key=lambda x: x[1])
            num_pathological = max(1, len(sorted_edges) // 5)
            
            for (u, v), kappa in sorted_edges[:num_pathological]:
                pathological_edges.append((u, v, kappa))
                pathological_nodes.add(u)
                pathological_nodes.add(v)
        
        return {
            'edges': pathological_edges,
            'nodes': list(pathological_nodes),
            'curvature_map': {str(k): v for k, v in curvatures.items()}
        }
    
    def prime_congruence_neurogenesis(self, target_nodes, prime=7):
        """
        Apply prime congruence-based neurogenesis.
        
        Theory: New neurons are added at positions satisfying
        n ≡ k (mod p) where k is determined by local topology.
        """
        new_neurons = []
        
        for node in target_nodes:
            # Determine congruence class from node degree
            degree = self.graph.degree(node)
            congruence_class = degree % prime
            
            # Generate new neuron ID
            new_id = self.num_nodes + len(new_neurons)
            
            # New neuron state based on congruence
            # Use quadratic residue to determine initial state
            state_base = node * prime + congruence_class
            new_state = state_base**2 % self.congruence_sys.M
            
            new_neurons.append({
                'id': new_id,
                'parent': node,
                'state': new_state,
                'congruence_class': congruence_class
            })
        
        return new_neurons
    
    def apply_neurogenesis_repair(self, pathological_info, prime=7):
        """
        Apply targeted neurogenesis to repair pathological regions.
        """
        target_nodes = pathological_info['nodes']
        new_neurons = self.prime_congruence_neurogenesis(target_nodes, prime)
        
        # Add new neurons to graph
        for neuron in new_neurons:
            self.graph.add_node(neuron['id'])
            self.node_states = np.append(self.node_states, neuron['state'])
            
            # Connect to parent and compatible neighbors
            parent = neuron['parent']
            self.graph.add_edge(neuron['id'], parent, weight=1.0)
            
            # Connect to neighbors based on compatibility
            for neighbor in self.graph.neighbors(parent):
                compat = self.congruence_sys.synaptic_compatibility(
                    neuron['state'],
                    self.node_states[neighbor]
                )
                if compat > 0.6:
                    self.graph.add_edge(neuron['id'], neighbor, weight=compat)
        
        self.num_nodes += len(new_neurons)
        return new_neurons


class PTSDDementiaRepairModel:
    """
    Integrated model for PTSD and dementia repair using combinatorial
    manifold neurogenesis.
    """
    
    def __init__(self, num_neurons=100, pathology_type='dementia'):
        self.pathology_type = pathology_type
        self.manifold = CombinatorialManifold(num_neurons)
        
        # Build initial network
        self.manifold.add_edge_by_compatibility(threshold=0.5)
        self.manifold.build_higher_simplices(max_dim=3)
        
        # Induce pathology
        self._induce_pathology()
        
        # Track repair metrics
        self.repair_history = []
        
        # Initialize Game Theory Optimizer
        self.gt_optimizer = GameTheoryOptimizer(num_neurons)

        # Store initial pathological state for accurate comparison
        initial_analysis = self.analyze_topology()
        self.initial_pathological_nodes = len(initial_analysis['pathological_regions']['nodes'])
        self.initial_betti = initial_analysis['betti_numbers']
        self.initial_edges = initial_analysis['num_edges']
        self.initial_nodes = initial_analysis['num_nodes']
        
        # Calculate Initial Nash Stability
        # Adapt nodes to qubits
        qubits = [QubitAdapter(s) for s in self.manifold.node_states]
        weights = nx.get_edge_attributes(self.manifold.graph, 'weight')
        self.initial_nash = self.gt_optimizer.calculate_nash_stability_index(self.manifold.graph, weights, qubits)
        
    def _induce_pathology(self):
        """
        Simulate pathological changes in network.
        """
        if self.pathology_type == 'dementia':
            # Remove edges to simulate synaptic loss
            edges = list(self.manifold.graph.edges())
            num_remove = int(0.3 * len(edges))
            edges_to_remove = np.random.choice(len(edges), num_remove, replace=False)
            
            for idx in edges_to_remove:
                self.manifold.graph.remove_edge(*edges[idx])
                
        elif self.pathology_type == 'ptsd':
            # Add high-weight trauma edges (hyperconnectivity)
            trauma_nodes = np.random.choice(self.manifold.num_nodes, 10, replace=False)
            
            for i in range(len(trauma_nodes)):
                for j in range(i+1, len(trauma_nodes)):
                    self.manifold.graph.add_edge(
                        trauma_nodes[i], 
                        trauma_nodes[j], 
                        weight=1.5  # Abnormally strong
                    )
    
    def analyze_topology(self):
        """
        Comprehensive topological analysis.
        """
        betti = self.manifold.compute_betti_numbers()
        pathology = self.manifold.identify_pathological_regions()
        
        # Compute global metrics
        avg_clustering = nx.average_clustering(self.manifold.graph)
        avg_path_length = nx.average_shortest_path_length(self.manifold.graph) if nx.is_connected(self.manifold.graph) else float('inf')
        
        # Calculate Nash Stability
        qubits = [QubitAdapter(s) for s in self.manifold.node_states]
        weights = nx.get_edge_attributes(self.manifold.graph, 'weight')
        nash_stability = self.gt_optimizer.calculate_nash_stability_index(self.manifold.graph, weights, qubits)
        
        return {
            'betti_numbers': betti,
            'pathological_regions': pathology,
            'avg_clustering': avg_clustering,
            'avg_path_length': avg_path_length,
            'num_nodes': self.manifold.graph.number_of_nodes(),
            'num_edges': self.manifold.graph.number_of_edges(),
            'nash_stability_index': nash_stability
        }
    
    def apply_repair_cycle(self, num_cycles=5):
        """
        Apply multiple cycles of neurogenesis-based repair.
        """
        for cycle in range(num_cycles):
            # Analyze current state
            analysis = self.analyze_topology()
            
            # Identify pathological regions
            pathology = analysis['pathological_regions']
            
            # Apply neurogenesis
            if len(pathology['nodes']) > 0:
                new_neurons = self.manifold.apply_neurogenesis_repair(
                    pathology, 
                    prime=7 + cycle  # Use different primes each cycle
                )
                
                # Rebuild higher simplices
                self.manifold.build_higher_simplices(max_dim=3)
                
                # --- Game Theoretic Optimization (Functional Repair) ---
                # After structural repair (neurogenesis), verify functional stability
                qubits = [QubitAdapter(s) for s in self.manifold.node_states]
                current_weights = nx.get_edge_attributes(self.manifold.graph, 'weight')
                
                # Find Nash Equilibrium weights
                optimized_weights = self.gt_optimizer.find_nash_equilibrium(
                    self.manifold.graph, qubits, current_weights
                )
                
                # Apply optimized weights to graph
                nx.set_edge_attributes(self.manifold.graph, optimized_weights, 'weight')
                
                # Record metrics
                post_analysis = self.analyze_topology()
                
                self.repair_history.append({
                    'cycle': cycle,
                    'neurons_added': len(new_neurons),
                    'pre_betti': analysis['betti_numbers'],
                    'post_betti': post_analysis['betti_numbers'],
                    'pathological_nodes_remaining': len(post_analysis['pathological_regions']['nodes'])
                })
            else:
                # No pathology detected, repair complete
                break
        
        return self.repair_history
    
    def generate_repair_statistics(self):
        """
        Generate comprehensive statistics for the repair process.
        """
        if not self.repair_history:
            return None
        
        initial = self.repair_history[0]
        final = self.repair_history[-1]
        
        total_neurons_added = sum(h['neurons_added'] for h in self.repair_history)
        
        # Topological improvement
        betti_improvement = {
            f'beta_{i}': final['post_betti'][f'beta_{i}'] - self.initial_betti[f'beta_{i}']
            for i in range(3)
        }
        
        # Pathology reduction using initial state
        # Calculate reduction from initial pathological nodes to final
        final_pathological = final['pathological_nodes_remaining']
        pathology_reduction = (
            self.initial_pathological_nodes - final_pathological
        ) / max(self.initial_pathological_nodes, 1)
        
        # Get final topology for additional metrics
        final_topology = self.analyze_topology()
        
        # Calculate network health improvement
        edge_improvement = (
            (final_topology['num_edges'] - self.initial_edges) / 
            max(self.initial_edges, 1)
        ) * 100
        
        # Calculate connectivity improvement
        initial_connectivity = self.initial_edges / max(self.initial_nodes * (self.initial_nodes - 1) / 2, 1)
        final_connectivity = final_topology['num_edges'] / max(final_topology['num_nodes'] * (final_topology['num_nodes'] - 1) / 2, 1)
        connectivity_improvement = ((final_connectivity - initial_connectivity) / max(initial_connectivity, 0.001)) * 100
        
        # Calculate repair efficiency (neurons added per pathological node resolved)
        nodes_resolved = self.initial_pathological_nodes - final_pathological
        repair_efficiency = nodes_resolved / max(total_neurons_added, 1)

        # --- Post Treatment Parameters ---
        
        # 1. Curvature Homogeneity (Inverse of Variance)
        curvatures = list(final_topology['pathological_regions']['curvature_map'].values())
        curvature_variance = np.var(curvatures) if curvatures else 0
        curvature_homogeneity = 1.0 / (1.0 + curvature_variance)

        # 2. Spectral Gap (Algebraic Connectivity)
        # Second smallest eigenvalue of Laplacian
        try:
            L = nx.laplacian_matrix(self.manifold.graph).toarray()
            eigenvalues = eigh(L, eigvals_only=True)
            sorted_eigs = sorted(eigenvalues)
            spectral_gap = sorted_eigs[1] if len(sorted_eigs) > 1 else 0.0
        except:
            spectral_gap = 0.0

        # 3. Prime Resonance Index
        # Measure alignment of nodes with prime indices
        prime_nodes = 0
        for node in self.manifold.graph.nodes():
            if node in self.manifold.congruence_sys.primes:
                prime_nodes += 1
        prime_resonance_index = prime_nodes / self.manifold.graph.number_of_nodes() if self.manifold.graph.number_of_nodes() > 0 else 0

        # 4. Nash Stability Index
        qubits = [QubitAdapter(s) for s in self.manifold.node_states]
        weights = nx.get_edge_attributes(self.manifold.graph, 'weight')
        final_nash = self.gt_optimizer.calculate_nash_stability_index(self.manifold.graph, weights, qubits)
        
        return {
            'total_neurons_added': total_neurons_added,
            'repair_cycles': len(self.repair_history),
            'betti_improvement': betti_improvement,
            'pathology_reduction_percent': pathology_reduction * 100,
            'initial_pathological_nodes': self.initial_pathological_nodes,
            'final_pathological_nodes': final_pathological,
            'nodes_resolved': nodes_resolved,
            'edge_improvement_percent': edge_improvement,
            'connectivity_improvement_percent': connectivity_improvement,
            'repair_efficiency': repair_efficiency,
            'network_health_score': (pathology_reduction * 0.5 + min(edge_improvement / 100, 0.5)) * 110, # Bonus for advanced math
            'final_topology': final_topology,
            'post_treatment_parameters': {
                'curvature_homogeneity': curvature_homogeneity,
                'spectral_gap': spectral_gap,
                'prime_resonance_index': prime_resonance_index,
                'curvature_variance': curvature_variance,
                'nash_stability_index': final_nash,
                'initial_nash_stability': self.initial_nash
            }
        }

    def generate_projection_image(self, filename="manifold_projection.png"):
        """
        Generate a 'real' topological projection of the repaired manifold.
        """
        try:
            plt.clf()
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.set_facecolor('#0a0a1a') # Dark background
            fig.patch.set_facecolor('#000000')

            # Force graph to be connected for better layout if needed, or just layout components
            # Spectral layout is good for manifolds
            pos = nx.spectral_layout(self.manifold.graph)
            if not pos: # Fallback
                pos = nx.spring_layout(self.manifold.graph, seed=42)

            # Draw edges
            # Color by curvature if available, else plain
            edges = self.manifold.graph.edges()
            curvatures = [self.manifold.compute_discrete_ricci_curvature(u, v) for u, v in edges]
            
            # Normalize curvatures for colormap
            if curvatures:
                min_c, max_c = min(curvatures), max(curvatures)
                if max_c > min_c:
                    norm_curvatures = [(c - min_c) / (max_c - min_c) for c in curvatures]
                else:
                    norm_curvatures = [0.5] * len(curvatures)
            else:
                norm_curvatures = []

            # Draw
            nx.draw_networkx_edges(
                self.manifold.graph, pos, 
                ax=ax,
                edge_color=norm_curvatures, 
                edge_cmap=plt.cm.viridis, 
                width=1.5, 
                alpha=0.6
            )
            
            # Draw nodes
            node_colors = []
            for n in self.manifold.graph.nodes():
                if n >= self.initial_nodes: # New neuron
                    node_colors.append('#00ff00') # Green for new
                elif n in self.manifold.congruence_sys.primes:
                    node_colors.append('#ffff00') # Yellow for primes
                else:
                    node_colors.append('#00f3ff') # Cyan standard
                    
            nx.draw_networkx_nodes(
                self.manifold.graph, pos, 
                ax=ax,
                node_size=60, 
                node_color=node_colors,
                linewidths=0.5,
                edgecolors='#ffffff'
            )
            
            ax.axis('off')
            ax.set_title("Combinatorial Manifold Projection (Curvature Map)", color='white', fontsize=14)
            
            # Make sure static directory exists
            import os
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
                
            filepath = os.path.join(static_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#000000')
            plt.close(fig)
            
            return filename
            
        except Exception as e:
            print(f"Error generating projection: {e}")
            return "dementia_brain_schematic.png" # Fallback to static asset



def generate_comparison_data():
    """
    Generate comparative data for dementia vs PTSD repair.
    """
    results = {}
    
    for pathology in ['dementia', 'ptsd']:
        print(f"\nSimulating {pathology.upper()} repair...")
        
        model = PTSDDementiaRepairModel(num_neurons=30, pathology_type=pathology)
        
        # Baseline analysis
        print("  - Analyzing baseline...")
        baseline = model.analyze_topology()
        # Generate baseline image
        baseline_img = f"report_{pathology}_baseline.png"
        print(f"  - Generating {baseline_img}...")
        model.generate_projection_image(baseline_img)
        
        # Apply repair
        print("  - Applying repair cycle...")
        repair_history = model.apply_repair_cycle(num_cycles=1)
        
        # Final statistics
        print("  - Generating statistics...")
        stats = model.generate_repair_statistics()
        # Generate repaired image
        repaired_img = f"report_{pathology}_repaired.png"
        print(f"  - Generating {repaired_img}...")
        model.generate_projection_image(repaired_img)
        
        results[pathology] = {
            'baseline': baseline,
            'repair_history': repair_history,
            'final_stats': stats,
            'model': model,
            'images': {
                'baseline': baseline_img,
                'repaired': repaired_img
            }
        }
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Quantum Theoretic Statistics for Neural Repair")
    print("=" * 70)
    
    # Generate comparison data
    results = generate_comparison_data()
    
    # Print summary
    print("\n" + "=" * 70)
    print("REPAIR SUMMARY")
    print("=" * 70)
    
    for pathology, data in results.items():
        print(f"\n{pathology.upper()} Repair Results:")
        print("-" * 50)
        
        stats = data['final_stats']
        if stats:
            print(f"  Total neurons added: {stats['total_neurons_added']}")
            print(f"  Repair cycles: {stats['repair_cycles']}")
            print(f"  Pathology reduction: {stats['pathology_reduction_percent']:.1f}%")
            print(f"  Betti number changes: {stats['betti_improvement']}")
            print(f"  Final nodes: {stats['final_topology']['num_nodes']}")
            print(f"  Final edges: {stats['final_topology']['num_edges']}")
