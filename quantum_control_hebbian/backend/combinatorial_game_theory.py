"""
Combinatorial Game Theory Module
Implements Nim, Sprague-Grundy theory, Nash equilibria, and surreal numbers
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from functools import reduce
import operator


class NimGame:
    """
    Nim game implementation with Sprague-Grundy theory
    """
    
    def __init__(self, heaps: List[int]):
        """
        Args:
            heaps: List of heap sizes
        """
        self.heaps = heaps
    
    def grundy_value(self) -> int:
        """
        Compute Grundy number (nim-value) of position
        For Nim, this is simply XOR of all heap sizes
        """
        return reduce(operator.xor, self.heaps, 0)
    
    def is_winning_position(self) -> bool:
        """Check if current position is winning (Grundy != 0)"""
        return self.grundy_value() != 0
    
    def find_winning_move(self) -> Tuple[int, int]:
        """
        Find a winning move if one exists
        Returns: (heap_index, new_heap_size) or (-1, -1) if no winning move
        """
        if not self.is_winning_position():
            return (-1, -1)
        
        target_grundy = 0
        current_grundy = self.grundy_value()
        
        for i, heap_size in enumerate(self.heaps):
            # Try removing from this heap
            for new_size in range(heap_size):
                # Compute new Grundy value
                new_heaps = self.heaps.copy()
                new_heaps[i] = new_size
                new_grundy = reduce(operator.xor, new_heaps, 0)
                
                if new_grundy == target_grundy:
                    return (i, new_size)
        
        return (-1, -1)
    
    def all_moves(self) -> List[Tuple[int, int]]:
        """Generate all possible moves"""
        moves = []
        for i, heap_size in enumerate(self.heaps):
            for new_size in range(heap_size):
                moves.append((i, new_size))
        return moves
    
    def make_move(self, heap_index: int, new_size: int) -> 'NimGame':
        """Create new game state after move"""
        new_heaps = self.heaps.copy()
        new_heaps[heap_index] = new_size
        return NimGame(new_heaps)
    
    def mex(self, values: Set[int]) -> int:
        """
        Minimal excludant: smallest non-negative integer not in set
        """
        i = 0
        while i in values:
            i += 1
        return i


class SpragueGrundyGame:
    """
    General impartial game with Sprague-Grundy theory
    """
    
    def __init__(self):
        self.grundy_cache = {}
    
    def successors(self, position: tuple) -> List[tuple]:
        """
        Get all successor positions from current position
        Must be implemented by subclass
        """
        raise NotImplementedError
    
    def is_terminal(self, position: tuple) -> bool:
        """Check if position is terminal (no moves available)"""
        return len(self.successors(position)) == 0
    
    def grundy(self, position: tuple) -> int:
        """
        Compute Grundy number using mex of successor Grundy values
        """
        if position in self.grundy_cache:
            return self.grundy_cache[position]
        
        if self.is_terminal(position):
            self.grundy_cache[position] = 0
            return 0
        
        # Compute Grundy values of all successors
        successor_grundy = set()
        for succ in self.successors(position):
            successor_grundy.add(self.grundy(succ))
        
        # mex of successor Grundy values
        g = self.mex(successor_grundy)
        self.grundy_cache[position] = g
        return g
    
    def mex(self, values: Set[int]) -> int:
        """Minimal excludant"""
        i = 0
        while i in values:
            i += 1
        return i


class NashEquilibrium:
    """
    Nash equilibrium computation for finite games
    """
    
    def __init__(self, payoff_matrix: np.ndarray):
        """
        Args:
            payoff_matrix: n x m x 2 array where [i,j,0] is player 1's payoff
                          and [i,j,1] is player 2's payoff
        """
        self.payoff_matrix = payoff_matrix
        self.n_strategies_p1 = payoff_matrix.shape[0]
        self.n_strategies_p2 = payoff_matrix.shape[1]
    
    def find_pure_nash(self) -> List[Tuple[int, int]]:
        """
        Find all pure strategy Nash equilibria
        Returns: List of (strategy1, strategy2) tuples
        """
        equilibria = []
        
        for i in range(self.n_strategies_p1):
            for j in range(self.n_strategies_p2):
                # Check if (i, j) is Nash equilibrium
                is_nash = True
                
                # Player 1 cannot improve by deviating
                current_payoff_p1 = self.payoff_matrix[i, j, 0]
                for i_prime in range(self.n_strategies_p1):
                    if self.payoff_matrix[i_prime, j, 0] > current_payoff_p1:
                        is_nash = False
                        break
                
                if not is_nash:
                    continue
                
                # Player 2 cannot improve by deviating
                current_payoff_p2 = self.payoff_matrix[i, j, 1]
                for j_prime in range(self.n_strategies_p2):
                    if self.payoff_matrix[i, j_prime, 1] > current_payoff_p2:
                        is_nash = False
                        break
                
                if is_nash:
                    equilibria.append((i, j))
        
        return equilibria
    
    def find_mixed_nash_2x2(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find mixed strategy Nash equilibrium for 2x2 game
        Returns: (p1_strategy, p2_strategy) as probability distributions
        """
        if self.n_strategies_p1 != 2 or self.n_strategies_p2 != 2:
            raise ValueError("This method only works for 2x2 games")
        
        # Extract payoffs
        a11, a12 = self.payoff_matrix[0, 0, 0], self.payoff_matrix[0, 1, 0]
        a21, a22 = self.payoff_matrix[1, 0, 0], self.payoff_matrix[1, 1, 0]
        b11, b12 = self.payoff_matrix[0, 0, 1], self.payoff_matrix[0, 1, 1]
        b21, b22 = self.payoff_matrix[1, 0, 1], self.payoff_matrix[1, 1, 1]
        
        # Player 2's mixed strategy makes player 1 indifferent
        # a11 * q + a12 * (1-q) = a21 * q + a22 * (1-q)
        # Solve for q
        denom_p2 = (a11 - a12 - a21 + a22)
        if abs(denom_p2) > 1e-10:
            q = (a22 - a12) / denom_p2
            q = np.clip(q, 0, 1)
        else:
            q = 0.5
        
        # Player 1's mixed strategy makes player 2 indifferent
        # b11 * p + b21 * (1-p) = b12 * p + b22 * (1-p)
        # Solve for p
        denom_p1 = (b11 - b21 - b12 + b22)
        if abs(denom_p1) > 1e-10:
            p = (b22 - b21) / denom_p1
            p = np.clip(p, 0, 1)
        else:
            p = 0.5
        
        p1_strategy = np.array([p, 1 - p])
        p2_strategy = np.array([q, 1 - q])
        
        return p1_strategy, p2_strategy
    
    def expected_payoff(self, p1_strategy: np.ndarray, p2_strategy: np.ndarray) -> Tuple[float, float]:
        """
        Compute expected payoffs for both players given mixed strategies
        """
        payoff_p1 = 0
        payoff_p2 = 0
        
        for i in range(self.n_strategies_p1):
            for j in range(self.n_strategies_p2):
                prob = p1_strategy[i] * p2_strategy[j]
                payoff_p1 += prob * self.payoff_matrix[i, j, 0]
                payoff_p2 += prob * self.payoff_matrix[i, j, 1]
        
        return payoff_p1, payoff_p2


class SurrealNumber:
    """
    Conway's surreal numbers for continuous game values
    Simplified implementation using dyadic rationals
    """
    
    def __init__(self, left: Set['SurrealNumber'] = None, right: Set['SurrealNumber'] = None):
        """
        Create surreal number {L|R}
        """
        self.left = left if left is not None else set()
        self.right = right if right is not None else set()
        
        # Verify validity: all L < all R
        for l in self.left:
            for r in self.right:
                if not l < r:
                    raise ValueError("Invalid surreal number: L must be < R")
    
    def __lt__(self, other: 'SurrealNumber') -> bool:
        """
        x < y iff no x_R <= y and no y_L >= x
        """
        # Check no x_R <= y
        for x_r in self.right:
            if not (x_r < other or x_r == other):
                return False
        
        # Check no y_L >= x
        for y_l in other.left:
            if not (y_l < self or y_l == self):
                return False
        
        return True
    
    def __eq__(self, other: 'SurrealNumber') -> bool:
        """x == y iff not (x < y) and not (y < x)"""
        return not (self < other) and not (other < self)
    
    def __le__(self, other: 'SurrealNumber') -> bool:
        return self < other or self == other
    
    def to_float(self) -> float:
        """
        Convert to float (approximate for dyadic rationals)
        """
        if not self.left and not self.right:
            return 0.0
        
        if not self.left:
            # {|R} is less than all R
            return min(r.to_float() for r in self.right) - 1
        
        if not self.right:
            # {L|} is greater than all L
            return max(l.to_float() for l in self.left) + 1
        
        # {L|R} is between max(L) and min(R)
        max_left = max(l.to_float() for l in self.left)
        min_right = min(r.to_float() for r in self.right)
        return (max_left + min_right) / 2
    
    @staticmethod
    def zero() -> 'SurrealNumber':
        """Create surreal number 0 = {|}"""
        return SurrealNumber(set(), set())
    
    @staticmethod
    def one() -> 'SurrealNumber':
        """Create surreal number 1 = {0|}"""
        zero = SurrealNumber.zero()
        return SurrealNumber({zero}, set())
    
    @staticmethod
    def from_integer(n: int) -> 'SurrealNumber':
        """Create surreal number from integer"""
        if n == 0:
            return SurrealNumber.zero()
        elif n > 0:
            prev = SurrealNumber.from_integer(n - 1)
            return SurrealNumber({prev}, set())
        else:  # n < 0
            prev = SurrealNumber.from_integer(n + 1)
            return SurrealNumber(set(), {prev})


class TBIRecoveryGame:
    """
    Two-player game: Damage vs. Plasticity
    Models TBI recovery as a strategic game
    """
    
    def __init__(self, network_size: int = 100):
        self.network_size = network_size
        self.damage_strategies = ['high', 'medium', 'low']
        self.plasticity_strategies = ['aggressive', 'moderate', 'conservative']
        
        # Build payoff matrix
        self.payoff_matrix = self._build_payoff_matrix()
    
    def _build_payoff_matrix(self) -> np.ndarray:
        """
        Build payoff matrix for TBI recovery game
        Payoffs represent network performance (higher is better for plasticity)
        """
        # 3x3 game: damage strategies x plasticity strategies
        payoffs = np.zeros((3, 3, 2))
        
        # High damage
        payoffs[0, 0] = [-5, 3]   # High damage, aggressive plasticity
        payoffs[0, 1] = [-3, 2]   # High damage, moderate plasticity
        payoffs[0, 2] = [0, -2]   # High damage, conservative plasticity
        
        # Medium damage
        payoffs[1, 0] = [-2, 4]   # Medium damage, aggressive plasticity
        payoffs[1, 1] = [-1, 3]   # Medium damage, moderate plasticity
        payoffs[1, 2] = [1, 1]    # Medium damage, conservative plasticity
        
        # Low damage
        payoffs[2, 0] = [0, 2]    # Low damage, aggressive plasticity
        payoffs[2, 1] = [2, 3]    # Low damage, moderate plasticity
        payoffs[2, 2] = [3, 2]    # Low damage, conservative plasticity
        
        return payoffs
    
    def find_equilibrium(self) -> Dict:
        """Find Nash equilibrium for TBI recovery game"""
        nash = NashEquilibrium(self.payoff_matrix)
        pure_nash = nash.find_pure_nash()
        
        result = {
            'pure_equilibria': [],
            'payoff_matrix': self.payoff_matrix.tolist()
        }
        
        for i, j in pure_nash:
            result['pure_equilibria'].append({
                'damage_strategy': self.damage_strategies[i],
                'plasticity_strategy': self.plasticity_strategies[j],
                'payoffs': self.payoff_matrix[i, j].tolist()
            })
        
        return result
    
    def optimal_plasticity_strategy(self, damage_level: str) -> str:
        """
        Find optimal plasticity response to given damage level
        """
        damage_idx = self.damage_strategies.index(damage_level)
        
        # Find best response for plasticity (maximize payoff[1])
        best_plasticity_idx = 0
        best_payoff = self.payoff_matrix[damage_idx, 0, 1]
        
        for j in range(len(self.plasticity_strategies)):
            payoff = self.payoff_matrix[damage_idx, j, 1]
            if payoff > best_payoff:
                best_payoff = payoff
                best_plasticity_idx = j
        
        return self.plasticity_strategies[best_plasticity_idx]


# Example usage and testing
if __name__ == "__main__":
    print("=== Combinatorial Game Theory Module Test ===\n")
    
    # Test 1: Nim game
    print("1. Nim Game:")
    nim = NimGame([3, 5, 7])
    print(f"  Heaps: {nim.heaps}")
    print(f"  Grundy value: {nim.grundy_value()}")
    print(f"  Winning position: {nim.is_winning_position()}")
    
    if nim.is_winning_position():
        heap_idx, new_size = nim.find_winning_move()
        print(f"  Winning move: Remove from heap {heap_idx} to size {new_size}")
        print(f"  Verification: {nim.heaps[0]} ⊕ {nim.heaps[1]} ⊕ {nim.heaps[2]} = {nim.grundy_value()}\n")
    
    # Test 2: Nash equilibrium (Prisoner's Dilemma)
    print("2. Nash Equilibrium (Prisoner's Dilemma):")
    # Payoffs: (cooperate, cooperate) = (-1, -1)
    #          (cooperate, defect) = (-3, 0)
    #          (defect, cooperate) = (0, -3)
    #          (defect, defect) = (-2, -2)
    payoff_pd = np.array([
        [[-1, -1], [-3, 0]],
        [[0, -3], [-2, -2]]
    ])
    
    nash_pd = NashEquilibrium(payoff_pd)
    pure_nash = nash_pd.find_pure_nash()
    print(f"  Pure Nash equilibria: {pure_nash}")
    print(f"  (Both defect is the Nash equilibrium)\n")
    
    # Test 3: TBI Recovery Game
    print("3. TBI Recovery Game:")
    tbi_game = TBIRecoveryGame()
    equilibrium = tbi_game.find_equilibrium()
    
    print(f"  Pure Nash equilibria:")
    for eq in equilibrium['pure_equilibria']:
        print(f"    {eq['damage_strategy']} damage + {eq['plasticity_strategy']} plasticity")
        print(f"    Payoffs: {eq['payoffs']}")
    
    print(f"\n  Optimal plasticity responses:")
    for damage in ['high', 'medium', 'low']:
        optimal = tbi_game.optimal_plasticity_strategy(damage)
        print(f"    {damage} damage → {optimal} plasticity")
    
    print("\n✓ All tests completed successfully")
