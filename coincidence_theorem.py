import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

class MRLinacArcOptimizer:
    def __init__(self, tumor_profile, num_arc_segments=4):
        """
        Initialize the optimizer with tumor data from MR Simulator.
        
        :param tumor_profile: Normalized 1D array representing tumor density/thickness 
                              at different arc angles (simulated MR data).
        :param num_arc_segments: Number of discrete control points in the VMAT arc.
        """
        self.tumor_profile = tumor_profile
        self.num_qubits = num_arc_segments
        self.estimator = Estimator()
        
        # Define the Hamiltonian observable for the optimization
        # In this simplified model, we want to maximize the expectation value 
        # (minimize energy) where the tumor is densest.
        obs_list = [("Z" * i + "X" + "Z" * (self.num_qubits - 1 - i), weight) 
                    for i, weight in enumerate(tumor_profile)]
        self.hamiltonian = SparsePauliOp.from_list(obs_list)

    def variational_circuit(self, params):
        """
        Quantum Circuit representing the VMAT machine state.
        Each qubit represents an arc segment; rotation represents MLC aperture opening.
        """
        qc = QuantumCircuit(self.num_qubits)
        
        # Entangle arc segments (Mechanical constraints: adjacent leaves are correlated)
        for i in range(self.num_qubits - 1):
            qc.cnot(i, i+1)
            
        # Parameterized rotations (The "Action" to be optimized: Leaf positions/Dose rate)
        for i in range(self.num_qubits):
            qc.ry(params[i], i)
            qc.rz(params[i + self.num_qubits], i)
            
        return qc

    def coincidence_loss_function(self, params):
        """
        Calculates the 'Coincidence Loss'. 
        Goal: Minimize mismatch between Beam Intensity (Quantum State) and Tumor Profile.
        
        This relies on the Coincidence Theorem principle: 
        Finding a fixed point where Machine State M(x) coincides with Target State T(x).
        """
        qc = self.variational_circuit(params)
        
        # Calculate Expectation Value <Psi|H|Psi>
        # A lower value indicates better alignment (coincidence) with the tumor profile 
        # based on our Hamiltonian construction.
        job = self.estimator.run(qc, self.hamiltonian)
        expectation_value = job.result().values[0]
        
        # We add a penalty for "smoothness" (Gantry cannot jerk suddenly)
        smoothness_penalty = np.sum(np.diff(params)**2) * 0.5
        
        return np.real(expectation_value) + smoothness_penalty

    def optimize_arc(self):
        """
        Run the classical optimizer to update quantum circuit parameters.
        """
        print("Initializing Quantum-Classical Optimization Loop...")
        
        # Random initial leaf positions/gantry speeds
        init_params = np.random.rand(self.num_qubits * 2) * 2 * np.pi
        
        # Use COBYLA or similar classical optimizer
        result = minimize(self.coincidence_loss_function, init_params, 
                          method='COBYLA', tol=1e-4)
        
        return result

# --- SIMULATION WORKFLOW ---

# 1. Simulate MR Simulator Data (Tumor density seen from 4 cardinal angles)
#    [Angle 0, Angle 90, Angle 180, Angle 270]
mr_tumor_signal = np.array([0.1, 0.8, 0.2, 0.9]) # Tumor is "thicker" at 90 and 270 degrees

# 2. Instantiate Optimizer
optimizer = MRLinacArcOptimizer(mr_tumor_signal, num_arc_segments=4)

# 3. Run Optimization
result = optimizer.optimize_arc()

# 4. Decode Results for Elekta Machine
optimized_params = result.x
mlc_openings = np.cos(optimized_params[:4])**2 # Convert quantum rotation to probability/opening

print("\n--- OPTIMIZATION RESULTS ---")
print(f"Convergence Success: {result.success}")
print(f"Final Coincidence Loss: {result.fun:.4f}")
print("\nOptimal Arc Segments (MLC Aperture Weights):")
angles = [0, 90, 180, 270]
for angle, weight in zip(angles, mlc_openings):
    print(f"Gantry {angle}°: MLC Opening {weight:.2%}")

print("\nInterpretation:")
print("Higher percentages indicate the algorithm found a 'Coincidence' point")
print("requiring higher dose delivery at that angle to match tumor density.")
