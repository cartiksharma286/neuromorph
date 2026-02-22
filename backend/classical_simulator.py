"""
Classical Fallback Simulator for Quantum Portfolio Optimization
Used when CUDA-Q is not available
"""

import numpy as np
from scipy.optimize import minimize


class ClassicalPortfolioSimulator:
    """Classical simulator that mimics quantum optimization behavior"""
    
    def __init__(self, num_assets):
        self.num_assets = num_assets
        self.optimization_history = []
    
    def optimize(self, expected_returns, covariance_matrix, dividend_yields,
                 risk_aversion, dividend_weight, max_iterations=100):
        """
        Classical mean-variance optimization
        Simulates quantum optimization results
        """
        self.optimization_history = []
        
        def objective(weights):
            """Portfolio objective function"""
            # Combined returns
            combined_returns = (1 - dividend_weight) * expected_returns + dividend_weight * dividend_yields
            
            # Portfolio return
            portfolio_return = np.dot(weights, combined_returns)
            
            # Portfolio risk
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            # Objective: maximize return - risk_aversion * variance
            obj_value = -(portfolio_return - risk_aversion * portfolio_variance)
            
            # Record history
            self.optimization_history.append({
                'iteration': len(self.optimization_history),
                'energy': float(obj_value),
                'params': weights.tolist()
            })
            
            return obj_value
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = tuple((0.0, 0.15) for _ in range(self.num_assets))  # 0-15% per position
        
        # Initial guess
        initial_weights = np.ones(self.num_assets) / self.num_assets
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations}
        )
        
        return result.x, result.success, result.fun
    
    def sample(self, weights):
        """Simulate quantum sampling"""
        # Return weights as probability distribution
        return {format(i, f'0{self.num_assets}b'): weights[i] for i in range(len(weights))}


def create_spin_operator_classical(hamiltonian_str):
    """Classical representation of spin operator"""
    return {'hamiltonian': hamiltonian_str}


def observe_classical(weights, hamiltonian, num_assets):
    """Classical observation simulation"""
    # Return a mock expectation value
    return type('obj', (object,), {'expectation': lambda: np.random.random()})()
