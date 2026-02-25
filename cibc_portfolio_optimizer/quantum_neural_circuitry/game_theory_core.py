
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

