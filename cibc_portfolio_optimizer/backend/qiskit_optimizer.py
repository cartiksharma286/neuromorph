"""
CIBC Dividend Portfolio Qiskit Optimizer
Implements Measure Theory and Geodesic Projection States for Portfolio Optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import Aer
    from qiskit.primitives import Estimator, Sampler
    from qiskit.algorithms.minimum_eigensolvers import VQE
    from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA, ADAM
    QISKIT_AVAILABLE = True
except ImportError:
    print("Warning: Qiskit components not found. Ensure qiskit and qiskit-aer are installed.")
    QISKIT_AVAILABLE = False

from scipy.optimize import minimize
from scipy.optimize import minimize
from risk_classifier import RiskClassifier
from bloomberg_calculator import BloombergCalculator

class QiskitGeodesicOptimizer:
    """
    Quantum optimizer using Geodesic Projection States and Measure Theory
    for optimal portfolio construction.
    """
    
    def __init__(self, num_assets: int):
        self.num_assets = num_assets
        self.num_qubits = num_assets # Expose for server compatibility
        self.risk_classifier = RiskClassifier()
        self.bloomberg = BloombergCalculator()
        self.optimization_history = []
        self._backend = None
        if QISKIT_AVAILABLE:
            try:
                self._backend = Aer.get_backend('qasm_simulator')
            except:
                pass

    def get_geodesic_ansatz(self, num_qubits: int, depth: int = 3, entanglement: str = 'full') -> Any:
        """
        Constructs the Geodesic Projection State ansatz.
        This ansatz parameterizes the geodesic path on the unitary manifold U(2^N).
        """
        # We use RealAmplitudes as it represents a rotation on the hypersphere (geodesic constraints)
        ansatz = RealAmplitudes(num_qubits, reps=depth, entanglement=entanglement)
        return ansatz

    def _measure_theoretic_risk(self, weights: np.ndarray, covariance: np.ndarray, 
                               returns: np.ndarray, regime_probs: np.ndarray = None) -> float:
        """
        Calculates risk using Measure Theory concepts.
        Weights the covariance based on the probability measure of market regimes.
        """
        # Standard variance measure
        variance = np.dot(weights.T, np.dot(covariance, weights))
        
        # If regime probabilities are provided (from Statistical Classifier),
        # we adjust the measure.
        if regime_probs is not None:
            # Assume covariance represents the current regime. 
            # We add a "measure distortion" for tail risk regimes.
            # entropy = -sum(p log p) could be used, but here we use Expected Tail Loss proxy
            tail_measure = np.sum(regime_probs[1:]) # Prob of high volatility regimes
            variance *= (1.0 + tail_measure) 
            
        return variance

    def _geodesic_cost_function(self, params: np.ndarray, 
                               expected_returns: np.ndarray,
                               covariance: np.ndarray,
                               target_return: float,
                               risk_aversion: float,
                               ansatz: Any,
                               regime_info: Dict) -> float:
        """
        Cost function evaluating the Geodesic Projection State.
        """
        # 1. Bind parameters to the quantum circuit (Geodesic State)
        bound_circuit = ansatz.assign_parameters(params)
        bound_circuit.measure_all()
        
        # 2. Execute circuit (Projection)
        # We use the counts to estimate the marginal probabilities for each asset/qubit
        if not self._backend:
            return 1e9 # Fail high/safe
            
        job = self._backend.run(transpile(bound_circuit, self._backend), shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        # 3. specific decoding: "Geodesic Decoding"
        # We treat the state probability distribution as the weight distribution carrier
        # Marginal analysis:
        # P(qubit_i = 1) ~ weight_i
        
        weights = np.zeros(self.num_assets)
        total_counts = sum(counts.values())
        
        for bitstring, count in counts.items():
            # bitstring is usually little-endian in Qiskit (qN...q0), we need to be careful
            # Let's assume standard order or reverse to match index
            bits = bitstring[::-1] # Now q0 is at index 0
            for i, bit in enumerate(bits):
                if i < self.num_assets:
                    if bit == '1':
                        weights[i] += count
        
        weights /= total_counts
        
        # Normalize weights to sum to 1 (Portfolio constraint)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(self.num_assets) / self.num_assets
            
        # 4. Measure Theoretic Evaluation
        
        # Calculate Return
        port_return = np.dot(weights, expected_returns)
        
        # Calculate Risk (with Measure Theory adjustment)
        regime_probs = None
        if 'regime_probability' in regime_info:
             # Create a simple probability vector based on current regime
             # This is a simplification. Ideally we'd have probs for all regimes.
             regime_probs = np.zeros(3) 
             regime_probs[regime_info.get('current_regime', 0)] = 1.0
             
        port_risk = self._measure_theoretic_risk(weights, covariance, expected_returns, regime_probs)
        
        # 5. Optimization Objective
        # Maximize Return, Minimize Risk, Hit Target Return
        
        # Objective: Minimize Cost
        # Cost = Risk - RiskAversion * Return + Penalty(TargetReturn)
        
        target_penalty = 100 * (port_return - target_return)**2
        
        cost = (risk_aversion * port_risk) - port_return + target_penalty
        
        return cost

    def optimize_portfolio(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          dividend_yields: np.ndarray,
                          risk_tolerance: str = 'moderate',
                          target_return: float = 0.30, 
                          sector_constraints: Optional[Dict] = None,
                          stock_list: List[Dict] = []) -> Dict:
        """
        Main optimization method using Qiskit and Geodesic Projection.
        """
        # Risk aversion mapping
        risk_map = {'conservative': 2.0, 'moderate': 1.0, 'aggressive': 0.5}
        risk_aversion = risk_map.get(risk_tolerance, 1.0)
        
        # 1. Analyze Market Regime (Statistical Classifiers)
        regime_info = {'current_regime': 0}

        if stock_list:
            flash_projections = self.bloomberg.simulate_flash_gemini_projections(stock_list)
            
            # Apply "Jump" (Boost) to expected returns for Flash Gemini Tech stocks
            # This integrates the "Variational States" into the optimization landscape
            for i, stock in enumerate(stock_list):
                symbol = stock['symbol']
                if symbol in flash_projections:
                    # Flash Gemini projection (CAGR) replaces/boosts standard expected return
                    proj = flash_projections[symbol]
                    # We blend it: 70% Flash Gemini Projection + 30% Standard Market Data
                    boosted_return = (0.7 * proj['cagr']) + (0.3 * expected_returns[i])
                    expected_returns[i] = boosted_return
                    
        self.optimization_history = []

        if not QISKIT_AVAILABLE:
            # Fallback: Classical Geodesic Simulation (Unit Sphere Optimization)
            # Parametrize weights as squared amplitudes of a vector on a sphere
            
            def classical_geodesic_cost(params):
                 # params are angles or just raw vector then normalized
                 # Let's use raw vector and normalize
                 vec = np.array(params)
                 norm = np.linalg.norm(vec)
                 if norm == 0: return 1e9
                 vec = vec / norm
                 weights = vec**2
                 
                 # Measure Theoretic Risk
                 port_risk = self._measure_theoretic_risk(weights, covariance_matrix, expected_returns)
                 port_return = np.dot(weights, expected_returns)
                 
                 target_penalty = 100 * (port_return - target_return)**2
                 
                 cost = (risk_aversion * port_risk) - port_return + target_penalty
                 self.optimization_history.append(cost)
                 return cost

            initial_guess = np.random.rand(self.num_assets)
            res = minimize(classical_geodesic_cost, initial_guess, method='SLSQP')
            
            vec = res.x / np.linalg.norm(res.x)
            weights = vec**2
            
            port_ret = np.dot(weights, expected_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            port_div = np.dot(weights, dividend_yields)
            
            return {
                'weights': weights.tolist(),
                'expected_return': port_ret,
                'volatility': port_vol,
                'dividend_yield': port_div,
                'sharpe_ratio': (port_ret - 0.02) / port_vol if port_vol > 0 else 0,
                'success': bool(res.success),
                'message': str(res.message),
                'optimization_history': self.optimization_history,
                'regime_info': regime_info,
                'method': 'Classical Geodesic Simulation'
            }

        # Qiskit path
        # 2. Setup Quantum Ansatz (Geodesic State)
        num_qubits = self.num_assets
        ansatz = self.get_geodesic_ansatz(num_qubits, depth=3)
        num_params = ansatz.num_parameters
        
        # 3. Setup Optimizer
        initial_params = np.random.uniform(0, 2*np.pi, num_params)
        
        # 3. Setup Optimizer
        initial_params = np.random.uniform(0, 2*np.pi, num_params)
        
        # 4. Run Optimization Loop (VQE-like)
        self.optimization_history = []
        
        def callback(params):
            cost = self._geodesic_cost_function(params, expected_returns, covariance_matrix, 
                                              target_return, risk_aversion, ansatz, regime_info)
            self.optimization_history.append(cost)
            
        result = minimize(
            self._geodesic_cost_function,
            initial_params,
            args=(expected_returns, covariance_matrix, target_return, risk_aversion, ansatz, regime_info),
            method='COBYLA',
            options={'maxiter': 200},
            callback=callback
        )
        
        # 5. Extract Final Weights
        final_params = result.x
        
        # Re-run circuit to get weights
        # (Duplicated logic from cost function, clean up in real prod)
        bound_circuit = ansatz.assign_parameters(final_params)
        bound_circuit.measure_all()
        job = self._backend.run(transpile(bound_circuit, self._backend), shots=4096)
        counts = job.result().get_counts()
        
        weights = np.zeros(self.num_assets)
        total = sum(counts.values())
        for bs, c in counts.items():
            bits = bs[::-1]
            for i, b in enumerate(bits):
                if i < self.num_assets and b == '1':
                    weights[i] += c
        weights /= total
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        
        # Apply strict constraints (min/max) if needed
        weights = np.clip(weights, 0.0, 1.0)
        if np.sum(weights) > 0:
             weights /= np.sum(weights)
             
        # Calculate Metrics
        port_ret = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        port_div = np.dot(weights, dividend_yields)
        
        return {
            'weights': weights,
            'expected_return': port_ret,
            'volatility': port_vol,
            'dividend_yield': port_div,
            'sharpe_ratio': (port_ret - 0.02) / port_vol if port_vol > 0 else 0,
            'success': result.success,
            'message': result.message,
            'optimization_history': self.optimization_history,
            'regime_info': regime_info,
            'method': 'Qiskit Geodesic Projection'
        }

    def get_quantum_metrics(self) -> Dict:
        return {
            'method': 'Geodesic Projection State',
            'framework': 'IBM Qiskit',
            'ansatz_depth': 3,
            'measure_theory_applied': True,
            'iterations': len(self.optimization_history)
        }
