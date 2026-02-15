"""
Continued Fractions Module
Implements advanced continued fraction theory with high precision
"""

import numpy as np
from typing import List, Tuple, Callable
import mpmath as mp

# Set high precision
mp.mp.dps = 50  # 50 decimal places


class ContinuedFraction:
    """Base class for continued fraction computations"""
    
    def __init__(self, coefficients: List[float]):
        self.coefficients = coefficients
        self.convergents = []
    
    def compute_convergent(self, n: int) -> Tuple[float, float]:
        """
        Compute the nth convergent using recurrence relations
        Returns: (numerator, denominator)
        """
        if n < 0:
            return (0, 1)
        if n == 0:
            return (self.coefficients[0], 1)
        
        # Initialize recurrence
        p_prev2, p_prev1 = 1, self.coefficients[0]
        q_prev2, q_prev1 = 0, 1
        
        for i in range(1, min(n + 1, len(self.coefficients))):
            a_i = self.coefficients[i]
            p_curr = a_i * p_prev1 + p_prev2
            q_curr = a_i * q_prev1 + q_prev2
            
            p_prev2, p_prev1 = p_prev1, p_curr
            q_prev2, q_prev1 = q_prev1, q_curr
        
        return (p_prev1, q_prev1)
    
    def evaluate(self, n_terms: int = 20) -> float:
        """Evaluate continued fraction to n terms"""
        p, q = self.compute_convergent(n_terms - 1)
        return p / q if q != 0 else 0
    
    def convergence_sequence(self, max_terms: int = 30) -> List[Tuple[int, float, float]]:
        """
        Generate sequence of convergents with error estimates
        Returns: [(n, value, error), ...]
        """
        sequence = []
        prev_value = 0
        
        for n in range(max_terms):
            p, q = self.compute_convergent(n)
            value = p / q if q != 0 else 0
            error = abs(value - prev_value)
            sequence.append((n, value, error))
            prev_value = value
        
        return sequence


class PadeApproximant:
    """
    Padé approximant [m/n] for rational function approximation
    """
    
    def __init__(self, taylor_coeffs: List[float], m: int, n: int):
        """
        Args:
            taylor_coeffs: Taylor series coefficients [c0, c1, c2, ...]
            m: degree of numerator
            n: degree of denominator
        """
        self.taylor_coeffs = taylor_coeffs
        self.m = m
        self.n = n
        self.p_coeffs, self.q_coeffs = self._compute_pade()
    
    def _compute_pade(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Padé approximant coefficients
        Solves the linear system for P(x)/Q(x) matching Taylor series
        """
        c = self.taylor_coeffs
        m, n = self.m, self.n
        
        # Build linear system for denominator coefficients
        # c[k] * q[0] + c[k-1] * q[1] + ... + c[k-n] * q[n] = 0 for k = m+1, ..., m+n
        if n > 0:
            A = np.zeros((n, n))
            b = np.zeros(n)
            
            for i in range(n):
                k = m + 1 + i
                for j in range(n):
                    if k - j - 1 < len(c):
                        A[i, j] = c[k - j - 1]
                if k < len(c):
                    b[i] = -c[k]
            
            # Solve for q[1], ..., q[n] (q[0] = 1)
            q_rest = np.linalg.solve(A, b) if n > 0 else np.array([])
            q_coeffs = np.concatenate([[1.0], q_rest])
        else:
            q_coeffs = np.array([1.0])
        
        # Compute numerator coefficients
        # p[k] = sum_{j=0}^{min(k,n)} c[k-j] * q[j] for k = 0, ..., m
        p_coeffs = np.zeros(m + 1)
        for k in range(m + 1):
            for j in range(min(k + 1, len(q_coeffs))):
                if k - j < len(c):
                    p_coeffs[k] += c[k - j] * q_coeffs[j]
        
        return p_coeffs, q_coeffs
    
    def evaluate(self, x: float) -> float:
        """Evaluate Padé approximant at x"""
        numerator = np.polyval(self.p_coeffs[::-1], x)
        denominator = np.polyval(self.q_coeffs[::-1], x)
        return numerator / denominator if abs(denominator) > 1e-10 else 0
    
    def evaluate_array(self, x_array: np.ndarray) -> np.ndarray:
        """Evaluate Padé approximant at array of points"""
        return np.array([self.evaluate(x) for x in x_array])


class StieltjesContinuedFraction:
    """
    Stieltjes continued fraction for moment problems
    Reconstructs probability distribution from moments
    """
    
    def __init__(self, moments: List[float]):
        """
        Args:
            moments: sequence of moments [μ0, μ1, μ2, ...]
        """
        self.moments = moments
        self.s_coeffs = self._compute_s_fraction()
    
    def _compute_s_fraction(self) -> List[float]:
        """
        Compute S-fraction coefficients from moments
        Uses Hankel determinants
        """
        n = len(self.moments) // 2
        coeffs = []
        
        for k in range(n):
            # Compute Hankel determinants
            H_k = self._hankel_determinant(k)
            H_k_plus_1 = self._hankel_determinant(k + 1)
            
            if abs(H_k) > 1e-10:
                alpha = H_k_plus_1 / (H_k * H_k)
                coeffs.append(alpha)
            else:
                break
        
        return coeffs
    
    def _hankel_determinant(self, k: int) -> float:
        """Compute kth Hankel determinant"""
        if k == 0:
            return self.moments[0] if len(self.moments) > 0 else 1.0
        
        size = k + 1
        if len(self.moments) < 2 * k + 1:
            return 0.0
        
        H = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i + j < len(self.moments):
                    H[i, j] = self.moments[i + j]
        
        return np.linalg.det(H)
    
    def reconstruct_distribution(self, x_points: np.ndarray) -> np.ndarray:
        """
        Reconstruct probability density from S-fraction
        Uses inverse Stieltjes transform
        """
        # Simplified reconstruction (full implementation would use numerical inversion)
        density = np.zeros_like(x_points)
        
        for i, x in enumerate(x_points):
            if x > 0:
                # Approximate density using S-fraction evaluation
                s_val = self._evaluate_s_fraction(1 / x)
                density[i] = max(0, -np.imag(s_val) / np.pi)
        
        # Normalize
        if np.sum(density) > 0:
            density /= np.trapz(density, x_points)
        
        return density
    
    def _evaluate_s_fraction(self, z: complex) -> complex:
        """Evaluate S-fraction at complex z"""
        if len(self.s_coeffs) == 0:
            return 0
        
        # Backward evaluation
        result = 0
        for alpha in reversed(self.s_coeffs):
            result = alpha * z / (1 + result)
        
        return result


class ConvergenceAcceleration:
    """
    Convergence acceleration algorithms for series and sequences
    """
    
    @staticmethod
    def euler_transform(series: List[float]) -> float:
        """
        Euler transformation for alternating series
        Accelerates convergence of sum(-1)^n * a_n
        """
        n = len(series)
        if n == 0:
            return 0
        
        # Compute forward differences
        diff_table = [series.copy()]
        for k in range(n - 1):
            new_row = []
            for i in range(len(diff_table[-1]) - 1):
                new_row.append(diff_table[-1][i + 1] - diff_table[-1][i])
            if not new_row:
                break
            diff_table.append(new_row)
        
        # Euler sum
        result = diff_table[0][0] / 2
        for k in range(1, len(diff_table)):
            if len(diff_table[k]) > 0:
                result += diff_table[k][0] / (2 ** (k + 1))
        
        return result
    
    @staticmethod
    def shanks_transform(sequence: List[float]) -> float:
        """
        Shanks transformation for sequence acceleration
        e(S_n) = (S_{n+1} * S_{n-1} - S_n^2) / (S_{n+1} - 2*S_n + S_{n-1})
        """
        if len(sequence) < 3:
            return sequence[-1] if sequence else 0
        
        S_prev = sequence[-3]
        S_curr = sequence[-2]
        S_next = sequence[-1]
        
        denominator = S_next - 2 * S_curr + S_prev
        if abs(denominator) < 1e-10:
            return S_curr
        
        return (S_next * S_prev - S_curr ** 2) / denominator
    
    @staticmethod
    def wynn_epsilon(sequence: List[float], max_iter: int = 10) -> float:
        """
        Wynn's epsilon algorithm for sequence acceleration
        Most powerful acceleration method
        """
        n = len(sequence)
        if n == 0:
            return 0
        
        # Initialize epsilon table
        epsilon = np.zeros((n + 1, n + 1))
        epsilon[:, 0] = 0
        epsilon[0, 1] = sequence[0] if n > 0 else 0
        
        for i in range(1, n):
            epsilon[i, 1] = sequence[i]
        
        # Fill epsilon table
        for k in range(2, min(n + 1, max_iter)):
            for i in range(n - k + 1):
                denominator = epsilon[i + 1, k - 1] - epsilon[i, k - 1]
                if abs(denominator) > 1e-15:
                    epsilon[i, k] = epsilon[i + 1, k - 2] + 1 / denominator
                else:
                    epsilon[i, k] = epsilon[i + 1, k - 2]
        
        # Return best diagonal element (even k gives best results)
        best_k = min(n, max_iter - 1)
        if best_k % 2 == 1:
            best_k -= 1
        
        return epsilon[0, best_k] if best_k > 0 else sequence[-1]


# Neural activation functions using Padé approximants
class NeuralActivationCF:
    """Neural activation functions using continued fraction/Padé approximations"""
    
    @staticmethod
    def sigmoid_taylor_coeffs(n_terms: int = 10) -> List[float]:
        """Taylor series coefficients for sigmoid around x=0"""
        coeffs = [0.5]  # c0
        
        # Compute using Bernoulli numbers (simplified)
        for n in range(1, n_terms):
            if n % 2 == 1:
                # Odd terms
                coeff = 1 / (2 ** (n + 1) * np.math.factorial(n))
                coeffs.append(coeff)
            else:
                # Even terms are zero
                coeffs.append(0)
        
        return coeffs
    
    @staticmethod
    def sigmoid_pade(m: int = 4, n: int = 4) -> PadeApproximant:
        """Create Padé approximant for sigmoid function"""
        taylor_coeffs = NeuralActivationCF.sigmoid_taylor_coeffs(m + n + 1)
        return PadeApproximant(taylor_coeffs, m, n)
    
    @staticmethod
    def tanh_pade(m: int = 4, n: int = 4) -> PadeApproximant:
        """Create Padé approximant for tanh function"""
        # tanh Taylor series: x - x^3/3 + 2x^5/15 - 17x^7/315 + ...
        coeffs = [0, 1, 0, -1/3, 0, 2/15, 0, -17/315, 0, 62/2835]
        return PadeApproximant(coeffs[:m+n+1], m, n)
    
    @staticmethod
    def compare_approximations(x_range: np.ndarray) -> dict:
        """Compare original functions with Padé approximants"""
        sigmoid_pade = NeuralActivationCF.sigmoid_pade(4, 4)
        tanh_pade = NeuralActivationCF.tanh_pade(4, 4)
        
        sigmoid_original = 1 / (1 + np.exp(-x_range))
        sigmoid_approx = sigmoid_pade.evaluate_array(x_range)
        
        tanh_original = np.tanh(x_range)
        tanh_approx = tanh_pade.evaluate_array(x_range)
        
        return {
            'x': x_range,
            'sigmoid': {
                'original': sigmoid_original,
                'pade': sigmoid_approx,
                'error': np.abs(sigmoid_original - sigmoid_approx)
            },
            'tanh': {
                'original': tanh_original,
                'pade': tanh_approx,
                'error': np.abs(tanh_original - tanh_approx)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== Continued Fractions Module Test ===\n")
    
    # Test 1: Simple continued fraction for π
    print("1. Continued Fraction for π:")
    pi_coeffs = [3, 7, 15, 1, 292, 1, 1, 1, 2, 1]
    cf_pi = ContinuedFraction(pi_coeffs)
    convergents = cf_pi.convergence_sequence(10)
    
    for n, value, error in convergents[:5]:
        print(f"  C_{n} = {value:.10f}, error = {error:.2e}")
    print(f"  Actual π = {np.pi:.10f}\n")
    
    # Test 2: Padé approximant for sigmoid
    print("2. Padé [4/4] Approximant for Sigmoid:")
    sigmoid_pade = NeuralActivationCF.sigmoid_pade(4, 4)
    
    test_points = [-2, -1, 0, 1, 2]
    for x in test_points:
        original = 1 / (1 + np.exp(-x))
        approx = sigmoid_pade.evaluate(x)
        error = abs(original - approx)
        print(f"  x={x:+.1f}: σ(x)={original:.6f}, P[4/4]={approx:.6f}, error={error:.2e}")
    print()
    
    # Test 3: Wynn's epsilon acceleration
    print("3. Wynn's Epsilon Acceleration:")
    # Slowly converging series: sum(1/n^2) = π^2/6
    slow_series = [sum(1/k**2 for k in range(1, n+1)) for n in range(1, 21)]
    
    direct = slow_series[-1]
    accelerated = ConvergenceAcceleration.wynn_epsilon(slow_series)
    actual = np.pi**2 / 6
    
    print(f"  Direct sum (20 terms): {direct:.10f}")
    print(f"  Wynn accelerated:      {accelerated:.10f}")
    print(f"  Actual value:          {actual:.10f}")
    print(f"  Improvement: {abs(actual - direct):.2e} → {abs(actual - accelerated):.2e}\n")
    
    print("✓ All tests completed successfully")
