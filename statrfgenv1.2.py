import numpy as np
from itertools import combinations
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RFCoil:
    """Represents an RF coil element"""
    id: int
    frequency: float  # MHz
    impedance: complex
    quality_factor: float
    
class AdaptiveRFCoilCombinatorics:
    """Generates RF coil schematics with adaptive combinatorics"""
    
    def __init__(self, num_elements: int = 8):
        self.num_elements = num_elements
        self.coils = self._initialize_coils()
        self.coupling_matrix = self._compute_coupling_matrix()
    
    def _initialize_coils(self) -> List[RFCoil]:
        """Initialize RF coil elements"""
        coils = []
        for i in range(self.num_elements):
            coil = RFCoil(
                id=i,
                frequency=63.86 + (i * 0.5),  # MRI frequency range
                impedance=50 + 1j * (i * 5),
                quality_factor=100 + (i * 10)
            )
            coils.append(coil)
        return coils
    
    def _compute_coupling_matrix(self) -> np.ndarray:
        """Compute mutual coupling matrix"""
        matrix = np.zeros((self.num_elements, self.num_elements))
        for i in range(self.num_elements):
            for j in range(self.num_elements):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    distance = abs(i - j)
                    matrix[i, j] = np.exp(-distance / 2.0)
        return matrix
    
    def generate_optimal_combinations(self, max_size: int = 4) -> List[Tuple]:
        """Generate adaptive coil combinations"""
        best_combinations = []
        
        for size in range(2, max_size + 1):
            for combo in combinations(range(self.num_elements), size):
                score = self._evaluate_combination(combo)
                best_combinations.append((combo, score))
        
        return sorted(best_combinations, key=lambda x: x[1], reverse=True)[:5]
    
    def _evaluate_combination(self, combo: Tuple) -> float:
        """Evaluate combination quality"""
        coupling_sum = np.sum(self.coupling_matrix[np.ix_(combo, combo)])
        q_factor_avg = np.mean([self.coils[i].quality_factor for i in combo])
        return q_factor_avg / coupling_sum
    
    def print_schematic(self, combination: Tuple) -> None:
        """Print ASCII schematic"""
        print(f"\nRF Coil Schematic - Elements: {combination}")
        print("=" * 50)
        for idx in combination:
            coil = self.coils[idx]
            print(f"Coil {idx}: {coil.frequency:.2f}MHz | Z={coil.impedance} | Q={coil.quality_factor}")

# Example usage
if __name__ == "__main__":
    combinator = AdaptiveRFCoilCombinatorics(num_elements=8)
    optimal = combinator.generate_optimal_combinations(max_size=4)
    
    print("Top 5 Optimal RF Coil Combinations:")
    for i, (combo, score) in enumerate(optimal, 1):
        print(f"\n{i}. Combination {combo} - Score: {score:.4f}")
        combinator.print_schematic(combo)