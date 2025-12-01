#!/usr/bin/env python3
"""
sequence_optimizer.py
Advanced VFA optimization using scipy for hyperpolarized imaging

Sunnybrook Research Institute
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, List, Dict, Optional


class VFAOptimizer:
    """Variable Flip Angle optimization for hyperpolarized MRI"""
    
    def __init__(self, num_frames: int, t1: float, tr: float):
        """
        Initialize VFA optimizer
        
        Args:
            num_frames: Number of dynamic frames
            t1: T1 relaxation time (seconds)
            tr: Repetition time (milliseconds)
        """
        self.num_frames = num_frames
        self.t1 = t1
        self.tr = tr / 1000  # Convert to seconds
        
    def constant_signal_vfa(self) -> np.ndarray:
        """
        Calculate constant signal approach flip angles
        
        Reference: Nagashima K. MRM 2008
        """
        flip_angles = np.zeros(self.num_frames)
        
        for n in range(self.num_frames):
            # θ_n = arctan(1/sqrt(N-n))
            flip_angles[n] = np.arctan(1 / np.sqrt(self.num_frames - n))
        
        return np.degrees(flip_angles)
    
    def max_snr_vfa(self) -> np.ndarray:
        """
        Calculate maximum SNR flip angles
        
        Reference: Larson PEZ et al. MRM 2013
        """
        flip_angles = np.zeros(self.num_frames)
        e1 = np.exp(-self.tr / self.t1)
        
        for n in range(self.num_frames):
            # cos(θ_n) = sqrt((N-n)/(N-n+1)) * e^(-TR/T1)
            ratio = np.sqrt((self.num_frames - n) / (self.num_frames - n + 1))
            cos_theta = ratio * e1
            cos_theta = np.clip(cos_theta, 0, 1)  # Ensure valid range
            flip_angles[n] = np.arccos(cos_theta)
        
        return np.degrees(flip_angles)
    
    def simulate_signal(self, flip_angles_deg: np.ndarray, 
                       initial_magnetization: float = 1.0) -> np.ndarray:
        """
        Simulate signal evolution with given flip angle schedule
        
        Args:
            flip_angles_deg: Flip angles in degrees
            initial_magnetization: Initial magnetization (normalized)
            
        Returns:
            Signal array for each frame
        """
        flip_angles = np.radians(flip_angles_deg)
        signals = np.zeros(len(flip_angles))
        magnetization = initial_magnetization
        
        for i, flip_angle in enumerate(flip_angles):
            # Signal acquired
            signals[i] = magnetization * np.sin(flip_angle)
            
            # Remaining magnetization
            magnetization *= np.cos(flip_angle)
            
            # T1 decay (no recovery for hyperpolarized)
            magnetization *= np.exp(-self.tr / self.t1)
        
        return signals
    
    def calculate_total_snr(self, flip_angles_deg: np.ndarray) -> float:
        """Calculate total SNR for a flip angle schedule"""
        signals = self.simulate_signal(flip_angles_deg)
        return np.sqrt(np.sum(signals ** 2))
    
    def optimize_custom(self, 
                       min_angle: float = 1.0,
                       max_angle: float = 90.0,
                       method: str = 'differential_evolution') -> Tuple[np.ndarray, float]:
        """
        Custom optimization of flip angle schedule
        
        Args:
            min_angle: Minimum flip angle (degrees)
            max_angle: Maximum flip angle (degrees)
            method: Optimization method ('differential_evolution' or 'SLSQP')
            
        Returns:
            Tuple of (optimal_flip_angles, total_snr)
        """
        bounds = [(min_angle, max_angle) for _ in range(self.num_frames)]
        
        def objective(flip_angles):
            # Negative SNR because we minimize
            return -self.calculate_total_snr(flip_angles)
        
        if method == 'differential_evolution':
            result = differential_evolution(objective, bounds, seed=42, maxiter=500)
        else:
            # Use sequential optimization
            initial_guess = self.constant_signal_vfa()
            result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds)
        
        optimal_angles = result.x
        total_snr = -result.fun
        
        return optimal_angles, total_snr
    
    def optimize_with_constraints(self,
                                  max_angle: float = 90.0,
                                  monotonic: bool = False,
                                  smoothness_weight: float = 0.0) -> np.ndarray:
        """
        Optimize with additional constraints
        
        Args:
            max_angle: Maximum flip angle
            monotonic: Require monotonically increasing flip angles
            smoothness_weight: Weight for smoothness penalty (0 = no penalty)
            
        Returns:
            Optimized flip angles
        """
        bounds = [(1.0, max_angle) for _ in range(self.num_frames)]
        
        def objective(flip_angles):
            snr = -self.calculate_total_snr(flip_angles)
            
            # Add smoothness penalty
            if smoothness_weight > 0:
                differences = np.diff(flip_angles)
                smoothness_penalty = smoothness_weight * np.sum(differences ** 2)
                snr += smoothness_penalty
            
            return snr
        
        # Constraints
        constraints = []
        
        if monotonic:
            # Flip angles must increase
            for i in range(self.num_frames - 1):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i+1] - x[i]  # x[i+1] >= x[i]
                })
        
        initial_guess = self.constant_signal_vfa()
        
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def compare_strategies(self) -> Dict[str, Dict]:
        """
        Compare different VFA strategies
        
        Returns:
            Dictionary with results for each strategy
        """
        results = {}
        
        # Constant Signal
        cs_angles = self.constant_signal_vfa()
        cs_signals = self.simulate_signal(cs_angles)
        results['constant_signal'] = {
            'flip_angles': cs_angles,
            'signals': cs_signals,
            'total_snr': self.calculate_total_snr(cs_angles),
            'description': 'Constant Signal Approach (Nagashima)'
        }
        
        # Max SNR
        ms_angles = self.max_snr_vfa()
        ms_signals = self.simulate_signal(ms_angles)
        results['max_snr'] = {
            'flip_angles': ms_angles,
            'signals': ms_signals,
            'total_snr': self.calculate_total_snr(ms_angles),
            'description': 'Maximum SNR (Larson)'
        }
        
        # Optimized
        opt_angles, opt_snr = self.optimize_custom(method='differential_evolution')
        opt_signals = self.simulate_signal(opt_angles)
        results['optimized'] = {
            'flip_angles': opt_angles,
            'signals': opt_signals,
            'total_snr': opt_snr,
            'description': 'Numerically Optimized'
        }
        
        return results


def main():
    """Example usage"""
    import matplotlib.pyplot as plt
    
    # Example: C-13 Pyruvate imaging
    num_frames = 20
    t1 = 43  # seconds
    tr = 100  # ms
    
    optimizer = VFAOptimizer(num_frames, t1, tr)
    
    # Compare strategies
    results = optimizer.compare_strategies()
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, data in results.items():
        time_points = np.arange(num_frames) * tr / 1000
        ax1.plot(time_points, data['flip_angles'], 'o-', label=f"{name} (SNR={data['total_snr']:.3f})")
        ax2.plot(time_points, data['signals'], 'o-', label=name)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Flip Angle (degrees)')
    ax1.set_title('VFA Schedules')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Normalized Signal')
    ax2.set_title('Signal Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vfa_comparison.png', dpi=300)
    plt.show()
    
    print("\nVFA Strategy Comparison:")
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Total SNR: {data['total_snr']:.4f}")
        print(f"  First 5 flip angles: {data['flip_angles'][:5]}")


if __name__ == '__main__':
    main()
