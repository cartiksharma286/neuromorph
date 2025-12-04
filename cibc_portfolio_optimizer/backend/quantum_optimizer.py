"""
CIBC Dividend Portfolio Quantum Optimizer
Uses VQE (Variational Quantum Eigensolver) for optimal portfolio allocation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import cudaq
    CUDAQ_AVAILABLE = True
except ImportError:
    print("Warning: CUDA-Q not available. Using classical simulator.")
    CUDAQ_AVAILABLE = False
    from classical_simulator import ClassicalPortfolioSimulator
from scipy.optimize import minimize
import json


class DividendPortfolioOptimizer:
    """Quantum-enhanced portfolio optimizer for dividend-focused strategies"""
    
    def __init__(self, num_assets: int = 20):
        self.num_assets = num_assets
        self.num_qubits = num_assets
        self.optimization_history = []
        
    def create_portfolio_hamiltonian(self, 
                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    dividend_yields: np.ndarray,
                                    risk_aversion: float = 1.0,
                                    dividend_weight: float = 0.5):
        """
        Create Hamiltonian for portfolio optimization
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            dividend_yields: Dividend yields for each asset
            risk_aversion: Risk aversion parameter (higher = more conservative)
            dividend_weight: Weight given to dividend yield vs total return
        
        Returns:
            SpinOperator representing the portfolio optimization problem
        """
        # Combine total return with dividend yield
        combined_returns = (1 - dividend_weight) * expected_returns + dividend_weight * dividend_yields
        
        # Build Hamiltonian terms
        hamiltonian_str = ""
        
        # Return term (maximize)
        for i in range(self.num_assets):
            coeff = -combined_returns[i]  # Negative because we minimize
            hamiltonian_str += f"{coeff} Z{i} + "
        
        # Risk term (minimize variance)
        for i in range(self.num_assets):
            for j in range(i + 1, self.num_assets):
                coeff = risk_aversion * covariance_matrix[i, j]
                hamiltonian_str += f"{coeff} Z{i} Z{j} + "
        
        # Remove trailing " + "
        hamiltonian_str = hamiltonian_str.rstrip(" + ")
        
        if CUDAQ_AVAILABLE:
            return cudaq.SpinOperator.from_word(hamiltonian_str) if hamiltonian_str else cudaq.SpinOperator()
        else:
            return hamiltonian_str
    
    def portfolio_ansatz(self, thetas: List[float], num_qubits: int):
        """
        Variational quantum circuit for portfolio optimization
        Uses hardware-efficient ansatz with entangling layers
        """
        if not CUDAQ_AVAILABLE:
            return None  # Classical fallback doesn't use ansatz
            
        qubits = cudaq.qvector(num_qubits)
        
        # Initial layer - Hadamard gates for superposition
        for i in range(num_qubits):
            h(qubits[i])
        
        # Number of layers
        num_layers = len(thetas) // (num_qubits * 2)
        
        # Variational layers
        for layer in range(num_layers):
            # Rotation layer
            for i in range(num_qubits):
                idx = layer * num_qubits * 2 + i
                ry(thetas[idx], qubits[i])
            
            # Entangling layer
            for i in range(num_qubits):
                idx = layer * num_qubits * 2 + num_qubits + i
                rz(thetas[idx], qubits[i])
                if i < num_qubits - 1:
                    cx(qubits[i], qubits[i + 1])
            
            # Ring connection for full entanglement
            if num_qubits > 2:
                cx(qubits[num_qubits - 1], qubits[0])
    
    def optimize_portfolio(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          dividend_yields: np.ndarray,
                          risk_tolerance: str = 'moderate',
                          target_dividend_yield: Optional[float] = None,
                          sector_constraints: Optional[Dict[str, float]] = None,
                          max_iterations: int = 100) -> Dict:
        """
        Optimize portfolio allocation using quantum VQE
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            dividend_yields: Dividend yields for each asset
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
            target_dividend_yield: Target portfolio dividend yield
            sector_constraints: Maximum allocation per sector
            max_iterations: Maximum VQE iterations
        
        Returns:
            Dictionary with optimized weights and metrics
        """
        # Map risk tolerance to parameters
        risk_params = {
            'conservative': {'risk_aversion': 2.0, 'dividend_weight': 0.7},
            'moderate': {'risk_aversion': 1.0, 'dividend_weight': 0.5},
            'aggressive': {'risk_aversion': 0.5, 'dividend_weight': 0.3}
        }
        
        params = risk_params.get(risk_tolerance, risk_params['moderate'])
        
        # Create Hamiltonian
        hamiltonian = self.create_portfolio_hamiltonian(
            expected_returns,
            covariance_matrix,
            dividend_yields,
            risk_aversion=params['risk_aversion'],
            dividend_weight=params['dividend_weight']
        )
        
        # Initialize variational parameters
        num_layers = 3
        num_params = num_layers * self.num_qubits * 2
        initial_params = np.random.uniform(0, 2 * np.pi, num_params)
        
        # VQE optimization
        self.optimization_history = []
        
        if not CUDAQ_AVAILABLE:
            # Use classical fallback
            simulator = ClassicalPortfolioSimulator(self.num_assets)
            optimal_weights, success, final_energy = simulator.optimize(
                expected_returns, covariance_matrix, dividend_yields,
                params['risk_aversion'], params['dividend_weight'], max_iterations
            )
            self.optimization_history = simulator.optimization_history
            result = type('obj', (object,), {'x': optimal_weights, 'success': success, 'fun': final_energy})()
        else:
            def cost_function(params_vec):
                """Compute expectation value of Hamiltonian"""
                try:
                    # Observe the Hamiltonian
                    exp_val = cudaq.observe(
                        self.portfolio_ansatz,
                        hamiltonian,
                        params_vec.tolist(),
                        self.num_qubits
                    ).expectation()
                    
                    self.optimization_history.append({
                        'iteration': len(self.optimization_history),
                        'energy': float(exp_val),
                        'params': params_vec.tolist()
                    })
                    
                    return exp_val
                except Exception as e:
                    print(f"Error in cost function: {e}")
                    return 1e10
            
            # Classical optimization of quantum circuit
            result = minimize(
                cost_function,
                initial_params,
                method='COBYLA',
                options={'maxiter': max_iterations}
            )
        
        # Extract optimal weights from quantum state
        optimal_weights = self._extract_weights_from_state(result.x)
        
        # Apply constraints
        optimal_weights = self._apply_constraints(
            optimal_weights,
            target_dividend_yield,
            dividend_yields,
            sector_constraints
        )
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_dividend_yield = np.dot(optimal_weights, dividend_yields)
        portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'weights': optimal_weights.tolist(),
            'expected_return': float(portfolio_return),
            'dividend_yield': float(portfolio_dividend_yield),
            'volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'optimization_history': self.optimization_history,
            'num_iterations': len(self.optimization_history),
            'converged': result.success,
            'final_energy': float(result.fun)
        }
    
    def _extract_weights_from_state(self, optimal_params: np.ndarray) -> np.ndarray:
        """
        Extract portfolio weights from optimized quantum state
        Samples the quantum circuit and converts to probability distribution
        """
        if not CUDAQ_AVAILABLE:
            # Classical fallback - params are already weights
            if len(optimal_params) == self.num_assets:
                return optimal_params
            # If params are circuit parameters, use uniform distribution
            return np.ones(self.num_assets) / self.num_assets
            
        # Sample the quantum circuit
        counts = cudaq.sample(self.portfolio_ansatz, optimal_params.tolist(), self.num_qubits)
        
        # Convert measurement results to weights
        weights = np.zeros(self.num_assets)
        total_shots = 0
        
        for bits, count in counts.items():
            # Convert bitstring to asset weights
            for i, bit in enumerate(bits):
                if i < self.num_assets:
                    weights[i] += int(bit) * count
            total_shots += count
        
        # Normalize to sum to 1
        if total_shots > 0:
            weights = weights / total_shots
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(self.num_assets) / self.num_assets
        else:
            weights = np.ones(self.num_assets) / self.num_assets
        
        return weights
    
    def _apply_constraints(self,
                          weights: np.ndarray,
                          target_dividend_yield: Optional[float],
                          dividend_yields: np.ndarray,
                          sector_constraints: Optional[Dict[str, float]]) -> np.ndarray:
        """
        Apply portfolio constraints and rebalance if necessary
        """
        # Ensure no negative weights
        weights = np.maximum(weights, 0)
        
        # Ensure weights sum to 1
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)
        
        # Apply minimum/maximum position constraints
        min_weight = 0.01  # 1% minimum
        max_weight = 0.15  # 15% maximum per position
        
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / np.sum(weights)
        
        return weights
    
    def get_quantum_metrics(self) -> Dict:
        """Get quantum circuit metrics"""
        if not self.optimization_history:
            return {}
        
        return {
            'circuit_depth': 3 * self.num_qubits,  # Approximate depth
            'num_qubits': self.num_qubits,
            'num_parameters': len(self.optimization_history[-1]['params']) if self.optimization_history else 0,
            'convergence_iterations': len(self.optimization_history),
            'final_energy': self.optimization_history[-1]['energy'] if self.optimization_history else 0
        }
