
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

    def calculate_hebbian_prime_factor(self, u, v, qubits, base_strength):
        """
        Calculates the Hebbian Amplification factor modulated by the Prime Surface Integral.
        
        Formula:
        W_new = W_old + \alpha * (Coherence_{uv}) * \Phi_{surface}(u,v)
        
        Where \Phi_{surface} is the local contribution to the surface flux.
        """
        # 1. Standard Hebbian Coherence (Fire together, wire together)
        # We use phase alignment as the firing correlation
        phase_diff = abs(qubits[u].phase - qubits[v].phase)
        coherence = math.cos(phase_diff) # 1.0 if aligned, -1.0 if anti-aligned
        
        # 2. Prime Surface Modulation
        # Nodes with higher 'Prime Potential' (lower index primes) have rigorous structural requirements
        # We use the Prime Number Theorem density approximation: ln(N)
        p_u = self.primes[u] if u < len(self.primes) else u*2+1
        p_v = self.primes[v] if v < len(self.primes) else v*2+1
        
        prime_density = 1.0 / (math.log(p_u) * math.log(p_v))
        
        # 3. Combined Amplification Factor
        # If coherence is positive, we amplify based on prime density
        # Highly coherent nodes in "dense" prime regions get massive boosts
        amplification = coherence * prime_density * 2.5 # Tuning constant
        
        # Sigmoid squash to keep it stable
        factor = 1.0 / (1.0 + math.exp(-amplification))
        
        # Apply to base strength
        # We allow a mix: 70% retention of old weight, 30% Hebbian update
        new_strength = (0.7 * base_strength) + (0.3 * factor)
        return max(0.1, min(1.0, new_strength))
        
    def apply_dual_prime_encoding(self, entanglements, qubits):
        """
        Applies 'Dual Encoding' to the network state.
        
        Encoding 1 (Magnitude): Structural connectivity is aligned to Prime Gaps (Energy Level Spacing).
        Encoding 2 (Phase): Functional coherence is aligned to Prime Modulo classes (e.g., p mod 4).
                            This creates distinct 'cognitive channels' or functional columns.
        """
        updates = {}
        
        # 1. Structural Encoding (Prime Gaps) - already implemented in optimize_entanglement
        structural_map = self.optimize_entanglement_distribution(entanglements)
        
        # 2. Functional Encoding (Phase / Modulo)
        # We classify nodes based on their prime mapping p mod 4
        # Class 1: p = 1 mod 4 (Gaussian Primes - Real)
        # Class 3: p = 3 mod 4 (Gaussian Primes - Imaginary dominant)
        
        for (u, v), base_weight in structural_map.items():
            p_u = self.primes[u] if u < len(self.primes) else u*2+1
            p_v = self.primes[v] if v < len(self.primes) else v*2+1
            
            # Check Modulo Class alignment
            mod_u = p_u % 4
            mod_v = p_v % 4
            
            dual_factor = 1.0
            
            if mod_u == mod_v:
                # Same Functional Class -> Boost Connection (Intra-columnar)
                dual_factor = 1.25 
            else:
                # Different Class -> Inhibit slightly (Inter-columnar inhibition)
                dual_factor = 0.8
                
            # Apply to weight
            new_weight = base_weight * dual_factor
            
            # Use Hebbian modulation as fine-tuning
            hebbian_val = self.calculate_hebbian_prime_factor(u, v, qubits, new_weight)
            
            # Default to base weight if hebbian calculation fails or returns None
            val = hebbian_val if hebbian_val is not None else new_weight
            
            updates[(u, v)] = max(0.0, min(1.0, val))
            
        return updates
            
    def optimize_for_hyper_criticality(self, entanglements):
        """
        Optimizes network for 'Super-Criticality' (Cognitive Enhancement).
        
        Hypothesis: Healthy brains operate at Criticality (Phase transition edge).
        Enhanced brains operate in a 'Super-Resonant' regime defined by Twin Primes.
        
        Twin Primes (p, p+2) represent the tightest possible non-touching correlation.
        We boost connections between nodes indexed by Twin Primes to create 
        hyper-efficient information highways.
        """
        optimized = entanglements.copy()
        
        # Identify Twin Prime Pairs in our range
        twin_primes = []
        for i in range(len(self.primes)-1):
            if self.primes[i+1] - self.primes[i] == 2:
                # Map prime index back to node index? 
                # Our basic mapping is Node K <-> K-th Prime
                if i+1 < self.num_qubits:
                    twin_primes.append((i, i+1))
                    
        # Apply "Super-Hub" Boosts
        for (u, v) in twin_primes:
            # Check if edge exists or needs creating
            pass # We rely on existing edges mostly, but can suggest new ones
            
        # Iterate all existing edges to apply Hyper-Resonance
        for (u, v) in optimized.keys():
            # Check if u or v are part of Twin Prime sets
            p_u = self.primes[u] if u < len(self.primes) else 0
            p_v = self.primes[v] if v < len(self.primes) else 0
            
            is_twin_u = any(abs(p_u - p) == 2 for p in self.primes[:self.num_qubits])
            is_twin_v = any(abs(p_v - p) == 2 for p in self.primes[:self.num_qubits])
            
            # Boost factor
            boost = 1.1 # Mild baseline boost for general coherence
            
            if is_twin_u and is_twin_v:
                # Twin-to-Twin connection: The "Hyper-Link"
                boost = 1.5
            elif is_twin_u or is_twin_v:
                boost = 1.25
                
            optimized[(u, v)] = min(1.5, optimized[(u, v)] * boost)
            
        return optimized

    def apply_elliptic_phi_resonance(self, entanglements):
        """
        Enhances connectivity using Elliptic Integrals and Golden Ratio (Phi) resonances.
        
        Theory:
        The Golden Ratio (Phi) represents optimal packing and minimal interference.
        Elliptic Integrals K(k) describe the period of a pendulum and fields in toroidal topology.
        Combining them creates 'Phi-Resonant' standing waves that minimize energy loss.
        """
        import scipy.special as sp
        phi = (1 + math.sqrt(5)) / 2
        golden_angle = 2 * math.pi * (1 - 1/phi)
        
        updates = {}
        for (u, v), w in entanglements.items():
            # Calculate a 'resonance' based on node indices and Phi
            # Using discrete golden angle steps
            delta = abs(u - v)
            
            # Map delta to a modulus k for the elliptic integral
            # We want k in [0, 1). 
            # Use k = |sin(delta * golden_angle)|
            k = abs(math.sin(delta * golden_angle))
            
            # Complete Elliptic Integral of the First Kind K(k^2) in scipy notation usually m=k^2
            # ellipk(m)
            # Handle potential edge cases where k approaches 1
            if k > 0.999: k = 0.999
            
            ellip_val = sp.ellipk(k * k)
            
            # Cap for stability
            if ellip_val > 6.0: ellip_val = 6.0
            
            # Resonance factor
            # Elliptic integral represents the 'period' of the connection
            # We normalize it relative to a baseline of pi/2 (1.57)
            resonance = ellip_val / 1.57
            
            # Apply resonance boost
            # w' = w + alpha * log(resonance)
            # This favors connections with specific Phi-harmonic distances
            boost = 0.1 * math.log(resonance) if resonance > 1.0 else 0
            
            new_w = w + boost
            
            updates[(u, v)] = min(1.0, max(0.0, new_w))
            
        return updates

    def apply_ramanujan_congruence(self, entanglements):
        """
        Applies 'Quantum Statistical Congruence' to the network.
        
        Theory:
        Ramanujan's congruences for the partition function p(n) and the Tau function
        suggest that quantum modular forms govern the density of states.
        We strictly enforce the 'Dimension 24' constraint (from Dedekind Eta function)
        to filter out 'noise' synapses that do not align with the modular symmetry.
        """
        filtered_updates = {}
        
        # Modular discriminant modulus (Bosonic string theory / Leech Lattice analogy)
        MODULUS = 24 
        
        for (u, v), w in entanglements.items():
            # Get prime indices
            p_u = self.primes[u] if u < len(self.primes) else u*2+1
            p_v = self.primes[v] if v < len(self.primes) else v*2+1
            
            # Calculate the modular residue of the connection
            # We treat the sum of primes as the energy state index
            energy_index = p_u + p_v
            
            residue = energy_index % MODULUS
            
            # Ramanujan-style Statistical Congruence Boost
            # Connections satisfying specific modular congruences are 'preferred' vacuum states
            if residue == 0:
                # Perfect cancellation of modular noise -> Super-Conducting channel
                # "Ramanujan Prime Force"
                boost = 1.618 # Phi again, or just a strong constant
            elif residue in [1, 5, 7, 11, 13, 17, 19, 23]:
                # Coprime to 24 (Unit group) -> Stable transmission
                boost = 1.1
            else:
                # Divisors of 24 -> Potential resonance interference (damping)
                boost = 0.95 
            
            # Apply the Congruence Factor
            new_w = w * boost
            
            # Non-linear activation "Beyond Hebbian"
            # w_{new} = tanh(w_{old} * boost) to saturate cleanly
            new_w = math.tanh(new_w * 1.5)
            
            filtered_updates[(u, v)] = min(1.0, max(0.0, new_w))
            
        return filtered_updates

    def get_continued_fraction(self, x, depth=10):
        """
        Computes the continued fraction representation of x up to a given depth.
        Returns a list of coefficients [a0, a1, ..., an].
        """
        coeffs = []
        for _ in range(depth):
            a = int(x)
            coeffs.append(a)
            rem = x - a
            if abs(rem) < 1e-9:
                break
            x = 1.0 / rem
        return coeffs

    def evaluate_continued_fraction(self, coeffs):
        """
        Evaluates a continued fraction [a0, a1, ...] back to a float.
        """
        val = 0.0
        for a in reversed(coeffs[1:]):
            val = 1.0 / (a + val)
        return coeffs[0] + val

    def apply_continued_fraction_stabilization(self, entanglements, target_ratio=None):
        """
        Optimizes network weights using Continued Fraction approximations towards a 'Noble Number'.
        
        Theory:
        By adjusting weights so their ratios or values approximate noble numbers (like Phi),
        we maximize stability against perturbative noise (KAM Theorem).
        This is crucial for DBS where we want to lock the brain into a stable, non-epileptic attractor.
        """
        if target_ratio is None:
            # Default to Golden Ratio (Most Noble Number) = (1 + sqrt(5))/2
            # Use conjugate 1/Phi approx 0.618 for weights < 1
            target_ratio = (math.sqrt(5) - 1) / 2 
            
        updates = {}
        
        # Get target continued fraction expansion
        target_coeffs = self.get_continued_fraction(target_ratio)
        
        for (u, v), w in entanglements.items():
            # We want the weight w to be a 'convergent' of the noble number
            # derived from local topology indices to allow diversity
            
            # Use node indices to determine 'depth' of approximation required
            # Deeper in the network (higher index) = More stable needed = Deeper approximation
            depth = (u + v) % 5 + 3 
            
            local_coeffs = target_coeffs[:depth]
            
            # Calculate the rational approximation (the convergent)
            convergent = self.evaluate_continued_fraction(local_coeffs)
            
            # Move weight towards this convergent
            # w_new = w + beta * (convergent - w)
            # This 'pulls' the weight into a KAM stability island
            
            # Check if current weight is already "chaotic" (far from any rational convergent)
            # For this metaphor, we just nudge it
            drift = convergent - w
            new_w = w + (0.15 * drift) # Nudge factor
            
            updates[(u, v)] = min(1.0, max(0.01, new_w))
            
        return updates
