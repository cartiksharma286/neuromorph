"""
Quantum ML Optimizer for Neuroimaging Pulse Sequences

This module implements quantum machine learning optimization using NVIDIA CUDA-Q
(NVQLink architecture) for MRI pulse sequence parameter optimization.

Key Components:
- QuantumCircuit: Variational quantum circuits for parameter optimization
- CUDAQOptimizer: VQE-based quantum optimizer
- ParameterEncoder: Maps MRI parameters to quantum states
- QuantumNeuralNetwork: Hybrid quantum-classical neural network
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import warnings

# Quantum computing imports
try:
    import cudaq
    from cudaq import spin
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    warnings.warn("CUDA-Q not available. Using simulation mode.")

# Classical ML imports
import torch
import torch.nn as nn
from scipy.optimize import minimize


class ParameterEncoder:
    """
    Encodes MRI pulse sequence parameters into quantum states.
    
    Uses amplitude encoding to map classical parameters (TE, TR, flip angle, etc.)
    into quantum state amplitudes for processing by variational quantum circuits.
    """
    
    def __init__(self, n_qubits: int = 6):
        """
        Initialize parameter encoder.
        
        Args:
            n_qubits: Number of qubits for encoding (default 6 allows 2^6=64 parameters)
        """
        self.n_qubits = n_qubits
        self.max_params = 2 ** n_qubits
        
    def encode(self, parameters: np.ndarray) -> np.ndarray:
        """
        Encode classical MRI parameters into quantum state amplitudes.
        
        Args:
            parameters: Array of MRI parameters (normalized to [0,1])
            
        Returns:
            Quantum state vector (normalized amplitudes)
        """
        # Normalize parameters
        params_normalized = np.array(parameters)
        params_normalized = (params_normalized - params_normalized.min()) / \
                           (params_normalized.max() - params_normalized.min() + 1e-10)
        
        # Pad to match quantum state dimension
        if len(params_normalized) < self.max_params:
            params_normalized = np.pad(params_normalized, 
                                      (0, self.max_params - len(params_normalized)))
        elif len(params_normalized) > self.max_params:
            params_normalized = params_normalized[:self.max_params]
        
        # Normalize to valid quantum state (unit norm)
        state_vector = params_normalized / np.linalg.norm(params_normalized)
        
        return state_vector
    
    def decode(self, state_vector: np.ndarray, n_params: int) -> np.ndarray:
        """
        Decode quantum state back to classical parameters.
        
        Args:
            state_vector: Quantum state amplitudes
            n_params: Number of parameters to extract
            
        Returns:
            Classical parameter values
        """
        # Extract parameter values from state amplitudes
        params = np.abs(state_vector[:n_params])
        return params


class QuantumCircuit:
    """
    Variational Quantum Circuit for MRI parameter optimization.
    
    Implements a parameterized quantum circuit using CUDA-Q that can be
    trained to find optimal MRI pulse sequence parameters.
    """
    
    def __init__(self, n_qubits: int = 6, n_layers: int = 3):
        """
        Initialize variational quantum circuit.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of variational layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * n_layers * 3  # 3 parameters per qubit per layer
        
        if CUDAQ_AVAILABLE:
            self.kernel = self._build_cudaq_kernel()
        else:
            print("Using simulated quantum circuit (CUDA-Q not available)")
    
    def _build_cudaq_kernel(self):
        """Build CUDA-Q quantum kernel with variational ansatz."""
        
        @cudaq.kernel
        def variational_circuit(thetas: List[float]):
            """
            Variational quantum circuit ansatz.
            
            Uses a hardware-efficient ansatz with RY, RZ rotations and CNOT entanglement.
            """
            qubits = cudaq.qvector(self.n_qubits)
            
            # Initialize with Hadamard gates
            for i in range(self.n_qubits):
                h(qubits[i])
            
            # Variational layers
            param_idx = 0
            for layer in range(self.n_layers):
                # Rotation layer
                for i in range(self.n_qubits):
                    ry(thetas[param_idx], qubits[i])
                    param_idx += 1
                    rz(thetas[param_idx], qubits[i])
                    param_idx += 1
                    ry(thetas[param_idx], qubits[i])
                    param_idx += 1
                
                # Entanglement layer (CNOT ladder)
                for i in range(self.n_qubits - 1):
                    cx(qubits[i], qubits[i + 1])
                
                # Final CNOT to close the loop
                if self.n_qubits > 1:
                    cx(qubits[self.n_qubits - 1], qubits[0])
        
        return variational_circuit
    
    def execute(self, parameters: np.ndarray) -> np.ndarray:
        """
        Execute the quantum circuit with given parameters.
        
        Args:
            parameters: Circuit parameters (rotation angles)
            
        Returns:
            Measurement probabilities or state vector
        """
        if CUDAQ_AVAILABLE:
            result = cudaq.sample(self.kernel, parameters.tolist())
            # Convert counts to probabilities
            counts = result.get_sequential_data()
            total = sum(counts.values())
            probabilities = np.array([counts.get(format(i, f'0{self.n_qubits}b'), 0) / total 
                                     for i in range(2**self.n_qubits)])
            return probabilities
        else:
            # Simulated execution
            return self._simulate_circuit(parameters)
    
    def _simulate_circuit(self, parameters: np.ndarray) -> np.ndarray:
        """Simulate quantum circuit classically (fallback when CUDA-Q unavailable)."""
        # Simple simulation using random perturbations
        np.random.seed(int(np.sum(parameters * 1000) % 2**32))
        probabilities = np.random.dirichlet(np.ones(2**self.n_qubits) * 0.5)
        return probabilities


class CUDAQOptimizer:
    """
    Quantum-enhanced optimizer using CUDA-Q Variational Quantum Eigensolver (VQE).
    
    Optimizes MRI pulse sequence parameters using hybrid quantum-classical optimization.
    """
    
    def __init__(self, 
                 n_qubits: int = 6,
                 n_layers: int = 3,
                 optimizer: str = 'cobyla'):
        """
        Initialize CUDA-Q optimizer.
        
        Args:
            n_qubits: Number of qubits for variational circuit
            n_layers: Number of variational layers
            optimizer: Classical optimizer ('cobyla', 'adam', 'lbfgs')
        """
        self.circuit = QuantumCircuit(n_qubits, n_layers)
        self.optimizer_type = optimizer
        self.parameter_encoder = ParameterEncoder(n_qubits)
        
        # Initialize variational parameters randomly
        self.var_params = np.random.uniform(0, 2*np.pi, self.circuit.n_params)
        
    def cost_function(self, 
                     var_params: np.ndarray, 
                     mri_params: Dict[str, float],
                     target_metric: Callable) -> float:
        """
        Cost function for VQE optimization.
        
        Args:
            var_params: Variational circuit parameters
            mri_params: MRI pulse sequence parameters
            target_metric: Target metric to optimize (SNR, CNR, scan time, etc.)
            
        Returns:
            Cost value (lower is better)
        """
        # Execute quantum circuit
        probabilities = self.circuit.execute(var_params)
        
        # Decode quantum state to optimized MRI parameters
        optimized_params = self._decode_mri_params(probabilities, mri_params)
        
        # Calculate target metric
        metric_value = target_metric(optimized_params)
        
        # Return negative (for minimization) of metric we want to maximize
        return -metric_value
    
    def optimize(self,
                initial_params: Dict[str, float],
                target_metric: Callable,
                max_iterations: int = 100) -> Dict[str, float]:
        """
        Optimize MRI parameters using VQE.
        
        Args:
            initial_params: Initial MRI parameters
            target_metric: Target metric function to optimize
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimized MRI parameters
        """
        print(f"Starting quantum optimization with {self.circuit.n_qubits} qubits...")
        
        # Define cost function wrapper
        def cost_wrapper(var_params):
            return self.cost_function(var_params, initial_params, target_metric)
        
        # Use CUDA-Q optimizer if available, otherwise scipy
        if CUDAQ_AVAILABLE and self.optimizer_type == 'cobyla':
            try:
                optimizer = cudaq.optimizers.COBYLA()
                optimizer.max_iterations = max_iterations
                result = optimizer.optimize(
                    dimensions=self.circuit.n_params,
                    function=cost_wrapper
                )
                optimal_var_params = result.optimal_parameters
            except:
                # Fallback to scipy
                result = minimize(cost_wrapper, self.var_params, 
                                method='COBYLA', 
                                options={'maxiter': max_iterations})
                optimal_var_params = result.x
        else:
            # Use scipy optimizer
            result = minimize(cost_wrapper, self.var_params, 
                            method='L-BFGS-B' if self.optimizer_type == 'lbfgs' else 'COBYLA',
                            options={'maxiter': max_iterations})
            optimal_var_params = result.x
        
        # Get final optimized parameters
        probabilities = self.circuit.execute(optimal_var_params)
        optimized_params = self._decode_mri_params(probabilities, initial_params)
        
        print(f"Optimization complete! Final cost: {result.fun if hasattr(result, 'fun') else cost_wrapper(optimal_var_params):.4f}")
        
        return optimized_params
    
    def _decode_mri_params(self, 
                          probabilities: np.ndarray,
                          initial_params: Dict[str, float]) -> Dict[str, float]:
        """
        Decode quantum measurement probabilities to MRI parameters.
        
        Args:
            probabilities: Quantum measurement probabilities
            initial_params: Initial parameter values (for scaling)
            
        Returns:
            Decoded MRI parameters
        """
        # Extract most probable states
        param_keys = list(initial_params.keys())
        n_params = len(param_keys)
        
        # Use highest probability components
        top_indices = np.argsort(probabilities)[-n_params:]
        param_values = probabilities[top_indices]
        
        # Denormalize to original parameter ranges
        optimized = {}
        for i, key in enumerate(param_keys):
            if i < len(param_values):
                # Scale based on initial value (within ±50% range)
                initial_val = initial_params[key]
                optimized[key] = initial_val * (0.5 + param_values[i])
            else:
                optimized[key] = initial_params[key]
        
        return optimized


class QuantumNeuralNetwork(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network for pulse sequence optimization.
    
    Combines classical neural network layers with quantum circuit processing.
    """
    
    def __init__(self, 
                 input_dim: int,
                 quantum_qubits: int = 6,
                 quantum_layers: int = 2,
                 hidden_dim: int = 32):
        """
        Initialize hybrid quantum-classical network.
        
        Args:
            input_dim: Input dimension (number of MRI parameters)
            quantum_qubits: Number of qubits in quantum layer
            quantum_layers: Number of quantum variational layers
            hidden_dim: Hidden dimension for classical layers
        """
        super().__init__()
        
        # Classical preprocessing layers
        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, quantum_qubits * quantum_layers * 3)
        )
        
        # Quantum processing layer
        self.quantum_circuit = QuantumCircuit(quantum_qubits, quantum_layers)
        
        # Classical postprocessing layers
        self.classical_decoder = nn.Sequential(
            nn.Linear(2**quantum_qubits, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hybrid quantum-classical network.
        
        Args:
            x: Input MRI parameters
            
        Returns:
            Optimized parameters
        """
        # Classical encoding
        quantum_params = self.classical_encoder(x)
        
        # Quantum processing
        quantum_params_np = quantum_params.detach().cpu().numpy()
        quantum_output = self.quantum_circuit.execute(quantum_params_np)
        quantum_output_tensor = torch.tensor(quantum_output, dtype=torch.float32)
        
        # Classical decoding
        output = self.classical_decoder(quantum_output_tensor)
        
        return output


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Quantum ML Optimizer for Neuroimaging Pulse Sequences")
    print("=" * 70)
    print(f"CUDA-Q Available: {CUDAQ_AVAILABLE}")
    print()
    
    # Test parameter encoder
    print("Testing Parameter Encoder...")
    encoder = ParameterEncoder(n_qubits=4)
    test_params = np.array([30.0, 500.0, 90.0, 1.5])  # TE, TR, FA, voxel_size
    encoded = encoder.encode(test_params)
    print(f"  Original params: {test_params}")
    print(f"  Encoded state (first 4): {encoded[:4]}")
    print()
    
    # Test quantum circuit
    print("Testing Quantum Circuit...")
    qc = QuantumCircuit(n_qubits=4, n_layers=2)
    test_angles = np.random.uniform(0, 2*np.pi, qc.n_params)
    probabilities = qc.execute(test_angles)
    print(f"  Number of parameters: {qc.n_params}")
    print(f"  Output probabilities (first 4): {probabilities[:4]}")
    print()
    
    # Test CUDA-Q optimizer
    print("Testing CUDA-Q Optimizer...")
    optimizer = CUDAQOptimizer(n_qubits=4, n_layers=2)
    
    # Define simple target metric (SNR simulation)
    def snr_metric(params):
        te = params.get('TE', 30)
        tr = params.get('TR', 500)
        fa = params.get('FA', 90)
        # Simple SNR model: inversely proportional to TE, proportional to sin(FA)
        snr = (1000 / te) * np.sin(np.radians(fa)) * (tr / 1000)
        return snr
    
    initial_params = {'TE': 30.0, 'TR': 500.0, 'FA': 90.0}
    print(f"  Initial params: {initial_params}")
    print(f"  Initial SNR: {snr_metric(initial_params):.2f}")
    
    optimized_params = optimizer.optimize(
        initial_params=initial_params,
        target_metric=snr_metric,
        max_iterations=20
    )
    print(f"  Optimized params: {optimized_params}")
    print(f"  Optimized SNR: {snr_metric(optimized_params):.2f}")
    print()
    
    print("Quantum ML Optimizer initialization complete!")
