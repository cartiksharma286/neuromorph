
import math
import numpy as np
import networkx as nx

class GameTheoryOptimizer:
    """
    Implements Game Theoretic models for Neural Circuitry optimization.
    
    Paradigm:
    - Synapses/Neurons are 'Players' in a non-cooperative game.
    - Strategies: Adjust synaptic weight (recruitment).
    - Payoff: Maximize Information Transfer (Coherence) - Metabolic Cost.
    - Equilibrium: Nash Equilibrium where no synapse can improve its gain locally.
    """
    
    def __init__(self, num_qubits, metabolic_cost_factor=0.2):
        self.num_qubits = num_qubits
        self.cost_factor = metabolic_cost_factor
        
    def calculate_payoff(self, weight, coherence, neighbor_activity):
        """
        Calculates the payoff for a single synapse.
        
        Payoff = (Weight * Coherence) + (Synergy with Neighbors) - (Metabolic Cost)
        
        Metabolic Cost is modeled as non-linear (convex): Cost ~ w^2
        """
        # Income: Information flow
        income = weight * coherence
        
        # Synergy: Bonus if neighbors are also active (Hebbian support)
        synergy = 0.1 * neighbor_activity * weight
        
        # Cost: Quadratic cost to maintain strong weights
        cost = self.cost_factor * (weight ** 2)
        
        return income + synergy - cost
        
    def find_nash_equilibrium(self, topology, qubits, initial_entanglements, iterations=20):
        """
        Iterative Best Response Dynamics to find Nash Equilibrium.
        """
        current_weights = initial_entanglements.copy()
        
        for it in range(iterations):
            max_delta = 0.0
            new_weights = current_weights.copy()
            
            # Iterate over all "players" (edges)
            for (u, v) in topology.edges():
                # 1. Determine local environment (coherence)
                # Coherence based on qubit phase alignment
                phase_diff = abs(qubits[u].phase - qubits[v].phase)
                coherence = (math.cos(phase_diff) + 1) / 2 # Normalize to [0, 1]
                
                # 2. Estimate Neighbor Activity (Mean weight of adjacent edges)
                # This represents the "Field"
                u_neighbors = [current_weights.get(tuple(sorted((u, n))), 0) for n in topology.neighbors(u) if n != v]
                v_neighbors = [current_weights.get(tuple(sorted((v, n))), 0) for n in topology.neighbors(v) if n != u]
                all_neighbors = u_neighbors + v_neighbors
                avg_neighbor = sum(all_neighbors) / len(all_neighbors) if all_neighbors else 0
                
                # 3. Best Response Strategy
                # To maximize P = w*C +0.1*N*w - alpha*w^2
                # dP/dw = C + 0.1*N - 2*alpha*w = 0
                # w* = (C + 0.1*N) / (2 * alpha)
                
                optimal_w = (coherence + 0.1 * avg_neighbor) / (2 * self.cost_factor)
                
                # Constrain weight to [0, 1]
                optimal_w = max(0.0, min(1.0, optimal_w))
                
                # Update with inertia (learning rate) for stability
                current_w = current_weights.get((u, v), 0.1)
                updated_w = 0.5 * current_w + 0.5 * optimal_w
                
                new_weights[(u, v)] = updated_w
                
                delta = abs(updated_w - current_w)
                if delta > max_delta:
                    max_delta = delta
            
            current_weights = new_weights
            
            # Check convergence
            if max_delta < 0.001:
                break
                
        return current_weights

    def calculate_nash_stability_index(self, topology, weights, qubits):
        """
        Calculates a global stability metric.
        How strictly does the network adhere to the Nash Equilibrium?
        Lower "Frustration" = Higher Stability.
        """
        total_frustration = 0.0
        count = 0
        
        for (u, v), w in weights.items():
             phase_diff = abs(qubits[u].phase - qubits[v].phase)
             coherence = (math.cos(phase_diff) + 1) / 2
             
             # Calculate ideal Nash weight locally assuming simplified neighbor field
             # Simplified w* = Coherence / 2*alpha (ignoring neighbors for base stability metric)
             ideal_w = coherence / (2 * self.cost_factor)
             ideal_w = max(0.0, min(1.0, ideal_w))
             
             frustration = (w - ideal_w) ** 2
             total_frustration += frustration
             count += 1
             
        if count == 0: return 0.0
        return 1.0 - math.sqrt(total_frustration / count) # Higher is better (1.0 = Perfect Nash)


class CombinatorialGameOptimizer:
    """
    Implements Combinatorial Game Theory (CGT) for Neural Circuitry.
    
    Paradigm:
    - Each neuron is a 'Nim-pile' whose size is defined by its excitation and connectivity.
    - Neural optimization is a game of Nim where the goal is to move the network 
      towards a P-position (Zero Game), representing a stabilized, coherent state.
    - We use Surreal Number mappings for precise synaptic weight balancing.
    """
    
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        
    def calculate_nim_sum(self, states):
        """Calculates the XOR sum (Nim-sum) of given integer states."""
        res = 0
        for s in states:
            res ^= int(s)
        return res
        
    def calculate_grundy_value(self, node_id, topology, weights, qubits):
        """
        Calculates the G-value (nim-value) of a node.
        Determined by the 'heap size' of its neighborhood.
        """
        # A node's value is derived from its excitation and the sum of its connectivity
        local_weight_sum = sum(weights.get(tuple(sorted((node_id, n))), 0) for n in topology.neighbors(node_id))
        
        # Quantize excitation (0-1) to an integer (0-15) for nim-play
        excitation_val = int(qubits[node_id].excitation_prob * 15)
        
        # Combined nim-value
        return int(local_weight_sum * 10) ^ excitation_val

    def find_p_position_weights(self, topology, qubits, initial_weights):
        """
        Adjusts weights to minimize the global Nim-sum of the network.
        This stabilizes the 'informational game' played by the neurons.
        """
        current_weights = initial_weights.copy()
        nodes = list(topology.nodes())
        
        # 1. Calculate current Grundy values
        grundy_values = [self.calculate_grundy_value(n, topology, current_weights, qubits) for n in nodes]
        global_nim_sum = self.calculate_nim_sum(grundy_values)
        
        if global_nim_sum == 0:
            return current_weights # Already in P-position
            
        # 2. Strategic adjustment (Iterative "Nim-move" simulation)
        # We try to nudge weights so that the nim-sum moves towards zero
        for (u, v) in topology.edges():
            g_u = self.calculate_grundy_value(u, topology, current_weights, qubits)
            g_v = self.calculate_grundy_value(v, topology, current_weights, qubits)
            
            # If reducing this connection's 'contribution' helps zero the nim-sum
            # we apply a "Surreal Nudge"
            target_g_u = g_u ^ global_nim_sum
            
            if target_g_u < g_u:
                # We need to reduce the contribution of node U
                # Adjust weight (u, v) downwards
                current_weights[(u, v)] *= 0.8
            else:
                # Nudge weight based on Surreal value (dyadic rationals)
                # This ensures weights occupy "stable" numerical positions
                current_weights[(u, v)] = self.apply_surreal_quantization(current_weights[(u, v)])
                
        return current_weights

    def apply_surreal_quantization(self, weight):
        """
        Maps a weight to the nearest dyadic rational (Surreal Number approximation).
        Stable positions are 1/2, 1/4, 3/4, 1/8, etc.
        """
        # Precision level 2^-4 (1/16)
        precision = 16
        quantized = round(weight * precision) / precision
        return max(0.01, min(1.0, quantized))

    def calculate_game_stability(self, topology, weights, qubits):
        """
        Returns the inverse of the global Nim-sum.
        Stability = 1 / (1 + NimSum)
        """
        grundy_values = [self.calculate_grundy_value(n, topology, weights, qubits) for n in topology.nodes()]
        nsum = self.calculate_nim_sum(grundy_values)
        return 1.0 / (1.0 + nsum)
