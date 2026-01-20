#!/usr/bin/env python3
"""
Quantum Game-Theoretic Pulse Sequence Optimizer
Combines quantum computing, combinatorial game theory, and geodesic optimization
for optimal MRI pulse sequence design
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import json

class QuantumGameTheoreticOptimizer:
    """
    Innovative pulse sequence optimizer using:
    - Quantum game theory (Nash equilibria in Hilbert space)
    - Combinatorial optimization (sequence selection)
    - Geodesic optimization (Riemannian manifold of pulse parameters)
    """
    
    def __init__(self, num_players=3, num_strategies=5):
        """
        Initialize quantum game-theoretic optimizer
        
        Players represent competing objectives:
        - Player 1: SNR maximization
        - Player 2: Scan time minimization
        - Player 3: Contrast optimization
        
        Strategies represent pulse sequence parameters
        """
        self.num_players = num_players
        self.num_strategies = num_strategies
        self.payoff_matrices = self._initialize_payoff_matrices()
        
    def _initialize_payoff_matrices(self):
        """Initialize payoff matrices for quantum game"""
        matrices = []
        for player in range(self.num_players):
            # Payoff matrix for each player
            matrix = np.random.randn(self.num_strategies, self.num_strategies)
            # Make symmetric for cooperative game
            matrix = (matrix + matrix.T) / 2
            matrices.append(matrix)
        return matrices
    
    def quantum_nash_equilibrium(self, entanglement_param=0.5):
        """
        Find Nash equilibrium in quantum game with entanglement
        
        Uses quantum entanglement to allow correlated strategies
        
        Derivation:
        |ψ⟩ = √(1-γ)|00⟩ + √γ|11⟩  (entangled state)
        
        Payoff for player i:
        E_i = ⟨ψ|U_i ⊗ U_j|ψ⟩
        
        where U_i are unitary strategy operators
        """
        γ = entanglement_param
        
        # Quantum state coefficients
        c00 = np.sqrt(1 - γ)
        c11 = np.sqrt(γ)
        
        # Initialize strategies (probability distributions)
        strategies = [np.ones(self.num_strategies) / self.num_strategies 
                     for _ in range(self.num_players)]
        
        # Iterative best response
        for iteration in range(100):
            converged = True
            
            for player in range(self.num_players):
                # Calculate expected payoff for each strategy
                expected_payoffs = np.zeros(self.num_strategies)
                
                for strategy_idx in range(self.num_strategies):
                    # Quantum expected payoff with entanglement
                    payoff = 0
                    for other_strategy in range(self.num_strategies):
                        # Classical term
                        classical_term = (c00**2) * self.payoff_matrices[player][strategy_idx, other_strategy]
                        
                        # Quantum interference term (from entanglement)
                        quantum_term = (c11**2) * self.payoff_matrices[player][strategy_idx, strategy_idx]
                        
                        # Cross term (quantum correlation)
                        cross_term = 2 * c00 * c11 * np.sqrt(np.abs(
                            self.payoff_matrices[player][strategy_idx, other_strategy] * 
                            self.payoff_matrices[player][strategy_idx, strategy_idx]
                        ))
                        
                        payoff += (classical_term + quantum_term + cross_term) * \
                                 strategies[(player + 1) % self.num_players][other_strategy]
                    
                    expected_payoffs[strategy_idx] = payoff
                
                # Best response: choose strategy with highest expected payoff
                new_strategy = np.zeros(self.num_strategies)
                best_idx = np.argmax(expected_payoffs)
                new_strategy[best_idx] = 1.0
                
                # Check convergence
                if np.linalg.norm(new_strategy - strategies[player]) > 1e-6:
                    converged = False
                
                # Smooth update (avoid oscillations)
                strategies[player] = 0.9 * strategies[player] + 0.1 * new_strategy
            
            if converged:
                break
        
        return strategies
    
    def geodesic_optimization(self, initial_params, target_params):
        """
        Optimize pulse sequence parameters along geodesic on Riemannian manifold
        
        Manifold: Space of valid pulse sequences
        Metric: Fisher information metric
        
        Geodesic equation:
        d²x^μ/dt² + Γ^μ_αβ (dx^α/dt)(dx^β/dt) = 0
        
        where Γ^μ_αβ are Christoffel symbols
        """
        # Parameter space: [TR, TE, flip_angle, phase_encoding_steps]
        dim = len(initial_params)
        
        # Fisher information metric (defines Riemannian geometry)
        def fisher_metric(params):
            """
            Fisher information metric tensor
            g_μν = E[∂log p/∂θ^μ · ∂log p/∂θ^ν]
            """
            # Simplified metric (diagonal for computational efficiency)
            # In practice, would compute from signal model
            metric = np.diag([
                1.0 / (params[0]**2 + 1),  # TR
                1.0 / (params[1]**2 + 1),  # TE
                1.0 / (np.sin(params[2])**2 + 0.1),  # flip angle
                1.0 / (params[3] + 1)  # phase encoding
            ])
            return metric
        
        # Christoffel symbols (connection coefficients)
        def christoffel_symbols(params):
            """
            Γ^μ_αβ = (1/2) g^μλ (∂g_λβ/∂x^α + ∂g_λα/∂x^β - ∂g_αβ/∂x^λ)
            """
            eps = 1e-6
            g = fisher_metric(params)
            g_inv = np.linalg.inv(g)
            
            gamma = np.zeros((dim, dim, dim))
            
            for mu in range(dim):
                for alpha in range(dim):
                    for beta in range(dim):
                        # Numerical derivatives of metric
                        params_plus = params.copy()
                        params_plus[alpha] += eps
                        g_plus = fisher_metric(params_plus)
                        
                        params_minus = params.copy()
                        params_minus[alpha] -= eps
                        g_minus = fisher_metric(params_minus)
                        
                        dg_dalpha = (g_plus - g_minus) / (2 * eps)
                        
                        # Simplified calculation (assuming diagonal metric)
                        gamma[mu, alpha, beta] = 0.5 * g_inv[mu, mu] * dg_dalpha[mu, beta]
            
            return gamma
        
        # Geodesic shooting: integrate geodesic equation
        def geodesic_energy(velocity):
            """Energy functional to minimize"""
            # Integrate along geodesic
            num_steps = 50
            t = np.linspace(0, 1, num_steps)
            dt = t[1] - t[0]
            
            position = initial_params.copy()
            vel = velocity.copy()
            
            energy = 0
            
            for step in range(num_steps - 1):
                # Geodesic equation: acceleration = -Γ(v,v)
                gamma = christoffel_symbols(position)
                acceleration = np.zeros(dim)
                
                for mu in range(dim):
                    for alpha in range(dim):
                        for beta in range(dim):
                            acceleration[mu] -= gamma[mu, alpha, beta] * vel[alpha] * vel[beta]
                
                # Update velocity and position
                vel += acceleration * dt
                position += vel * dt
                
                # Accumulate energy
                g = fisher_metric(position)
                energy += np.dot(vel, np.dot(g, vel)) * dt
            
            # Penalty for not reaching target
            distance_penalty = 100 * np.linalg.norm(position - target_params)**2
            
            return energy + distance_penalty
        
        # Optimize initial velocity
        initial_velocity = (target_params - initial_params) / 10
        result = minimize(geodesic_energy, initial_velocity, method='BFGS')
        
        # Integrate optimal geodesic
        optimal_velocity = result.x
        num_steps = 50
        t = np.linspace(0, 1, num_steps)
        dt = t[1] - t[0]
        
        geodesic_path = [initial_params.copy()]
        position = initial_params.copy()
        vel = optimal_velocity.copy()
        
        for step in range(num_steps - 1):
            gamma = christoffel_symbols(position)
            acceleration = np.zeros(dim)
            
            for mu in range(dim):
                for alpha in range(dim):
                    for beta in range(dim):
                        acceleration[mu] -= gamma[mu, alpha, beta] * vel[alpha] * vel[beta]
            
            vel += acceleration * dt
            position += vel * dt
            geodesic_path.append(position.copy())
        
        return np.array(geodesic_path)
    
    def combinatorial_sequence_selection(self, available_sequences, constraints):
        """
        Select optimal combination of pulse sequences using combinatorial game theory
        
        This is a multi-objective optimization problem:
        - Maximize diagnostic information
        - Minimize total scan time
        - Satisfy clinical constraints
        
        Uses Shapley value to fairly distribute "value" among sequences
        """
        import math
        num_sequences = len(available_sequences)
        
        # Characteristic function: value of each coalition
        def coalition_value(coalition_mask):
            """
            Value of a coalition of sequences
            V(S) = diagnostic_value(S) - time_penalty(S)
            """
            selected = [available_sequences[i] for i, m in enumerate(coalition_mask) if m]
            
            if not selected:
                return 0
            
            # Diagnostic value (synergy between sequences)
            diagnostic_value = 0
            for seq in selected:
                diagnostic_value += seq.get('diagnostic_value', 1.0)
            
            # Synergy bonus for complementary sequences
            if len(selected) > 1:
                # T1 + T2 synergy
                has_t1 = any('T1' in seq.get('contrast', '') for seq in selected)
                has_t2 = any('T2' in seq.get('contrast', '') for seq in selected)
                if has_t1 and has_t2:
                    diagnostic_value *= 1.3
                
                # Anatomical + vascular synergy
                has_anatomical = any(seq.get('type') == 'anatomical' for seq in selected)
                has_vascular = any(seq.get('type') == 'vascular' for seq in selected)
                if has_anatomical and has_vascular:
                    diagnostic_value *= 1.2
            
            # Time penalty
            total_time = sum(seq.get('scan_time', 5) for seq in selected)
            time_penalty = total_time / constraints.get('max_time', 30)
            
            return diagnostic_value - time_penalty
        
        # Calculate Shapley values
        shapley_values = np.zeros(num_sequences)
        
        for i in range(num_sequences):
            # Marginal contribution of sequence i to all possible coalitions
            for coalition_int in range(2**num_sequences):
                coalition_mask = [(coalition_int >> j) & 1 for j in range(num_sequences)]
                
                if coalition_mask[i]:
                    continue  # Skip coalitions already containing i
                
                # Value without i
                value_without = coalition_value(coalition_mask)
                
                # Value with i
                coalition_mask[i] = 1
                value_with = coalition_value(coalition_mask)
                
                # Marginal contribution
                marginal = value_with - value_without
                
                # Weight by coalition size
                coalition_size = sum(coalition_mask) - 1
                weight = 1.0 / (num_sequences * math.comb(num_sequences - 1, coalition_size))
                
                shapley_values[i] += weight * marginal
        
        # Select sequences with highest Shapley values
        num_to_select = constraints.get('max_sequences', 3)
        selected_indices = np.argsort(shapley_values)[-num_to_select:]
        
        return selected_indices, shapley_values
    
    def generate_optimal_protocol(self):
        """
        Generate optimal knee MRI protocol using quantum game theory
        """
        # Available pulse sequences
        sequences = [
            {'name': 'PD-FSE', 'type': 'anatomical', 'contrast': 'PD', 
             'TR': 2500, 'TE': 25, 'scan_time': 4, 'diagnostic_value': 1.0},
            {'name': 'T2-FSE', 'type': 'anatomical', 'contrast': 'T2', 
             'TR': 4000, 'TE': 100, 'scan_time': 5, 'diagnostic_value': 1.0},
            {'name': 'T1-SE', 'type': 'anatomical', 'contrast': 'T1', 
             'TR': 600, 'TE': 15, 'scan_time': 3, 'diagnostic_value': 0.8},
            {'name': 'TOF-MRA', 'type': 'vascular', 'contrast': 'Flow', 
             'TR': 25, 'TE': 3.5, 'scan_time': 6, 'diagnostic_value': 1.2},
            {'name': 'PC-Flow', 'type': 'vascular', 'contrast': 'Velocity', 
             'TR': 40, 'TE': 8, 'scan_time': 5, 'diagnostic_value': 1.1},
            {'name': '3D-GRE', 'type': 'anatomical', 'contrast': 'T2*', 
             'TR': 30, 'TE': 5, 'scan_time': 7, 'diagnostic_value': 0.9},
            {'name': 'STIR', 'type': 'anatomical', 'contrast': 'T1-IR', 
             'TR': 5000, 'TE': 30, 'scan_time': 6, 'diagnostic_value': 0.85}
        ]
        
        # Constraints
        constraints = {
            'max_time': 20,  # minutes
            'max_sequences': 4
        }
        
        # Find quantum Nash equilibrium
        print("Computing quantum Nash equilibrium...")
        nash_strategies = self.quantum_nash_equilibrium(entanglement_param=0.7)
        
        # Combinatorial sequence selection
        print("Optimizing sequence combination...")
        selected_indices, shapley_values = self.combinatorial_sequence_selection(
            sequences, constraints
        )
        
        # Geodesic optimization for selected sequences
        print("Optimizing parameters via geodesic...")
        optimized_sequences = []
        
        for idx in selected_indices:
            seq = sequences[idx].copy()
            
            # Initial parameters
            initial = np.array([seq['TR'], seq['TE'], 30.0, 256.0])
            
            # Target parameters (optimized for knee)
            if seq['type'] == 'vascular':
                target = np.array([seq['TR'] * 0.9, seq['TE'] * 0.95, 25.0, 256.0])
            else:
                target = np.array([seq['TR'] * 1.1, seq['TE'], 35.0, 256.0])
            
            # Compute geodesic
            geodesic = self.geodesic_optimization(initial, target)
            
            # Use final point on geodesic
            optimal_params = geodesic[-1]
            seq['TR_optimized'] = optimal_params[0]
            seq['TE_optimized'] = optimal_params[1]
            seq['flip_angle'] = optimal_params[2]
            seq['shapley_value'] = shapley_values[idx]
            
            optimized_sequences.append(seq)
        
        return {
            'sequences': optimized_sequences,
            'nash_equilibrium': nash_strategies,
            'shapley_values': shapley_values.tolist(),
            'total_scan_time': sum(s['scan_time'] for s in optimized_sequences),
            'optimization_method': 'Quantum Game Theory + Geodesic Optimization'
        }

def generate_innovative_sequence_report():
    """Generate comprehensive report on quantum game-theoretic optimization"""
    
    optimizer = QuantumGameTheoreticOptimizer()
    protocol = optimizer.generate_optimal_protocol()
    
    # Save protocol
    import os
    # Save protocol
    base_dir = os.path.dirname(os.path.abspath(__file__))
    protocol_path = os.path.join(base_dir, 'quantum_game_protocol.json')
    with open(protocol_path, 'w') as f:
        json.dump(protocol, f, indent=2)
    
    print(f"\nOptimal Protocol Generated:")
    print(f"Total scan time: {protocol['total_scan_time']} minutes")
    print(f"\nSelected Sequences:")
    for seq in protocol['sequences']:
        print(f"  - {seq['name']}: Shapley value = {seq['shapley_value']:.3f}")
    
    return protocol

if __name__ == '__main__':
    print("=" * 80)
    print("QUANTUM GAME-THEORETIC PULSE SEQUENCE OPTIMIZER")
    print("=" * 80)
    print("\nInnovative Features:")
    print("  • Quantum Nash equilibrium with entanglement")
    print("  • Combinatorial game theory (Shapley values)")
    print("  • Geodesic optimization on Riemannian manifold")
    print("  • Multi-objective optimization")
    print("\nGenerating optimal protocol...")
    print()
    
    protocol = generate_innovative_sequence_report()
    
    print("\n" + "=" * 80)
    print("✓ OPTIMIZATION COMPLETE")
    print("=" * 80)
