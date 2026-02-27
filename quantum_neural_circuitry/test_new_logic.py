
import unittest
import numpy as np
from game_theory_core import CombinatorialGameOptimizer
from generative_quantum_core import UncertaintyPrincipleManifest
from server import DementiaTreatmentModel # QuantumState is not exported but brain.qubits contains them

class TestQuantumEnhancements(unittest.TestCase):
    def setUp(self):
        self.num_qubits = 10
        self.cgt = CombinatorialGameOptimizer(self.num_qubits)
        self.uncertainty = UncertaintyPrincipleManifest(self.num_qubits)
        self.brain = DementiaTreatmentModel(self.num_qubits)

    def test_cgt_stability(self):
        # Initial stability
        initial_status = self.brain.analyze_status()
        initial_stability = initial_status.nim_game_stability
        
        # Apply treatment
        self.brain.apply_treatment('combinatorial_game', 0.8)
        
        # New stability
        new_status = self.brain.analyze_status()
        new_stability = new_status.nim_game_stability
        print(f"CGT Stability: {initial_stability} -> {new_stability}")
        self.assertGreaterEqual(new_stability, initial_stability)

    def test_uncertainty_compliance(self):
        # Initial compliance
        initial_status = self.brain.analyze_status()
        initial_compliance = initial_status.uncertainty_bound_compliance
        
        # Apply treatment
        self.brain.apply_treatment('uncertainty_manifest', 0.8)
        
        # New compliance
        new_status = self.brain.analyze_status()
        new_compliance = new_status.uncertainty_bound_compliance
        print(f"Uncertainty Compliance: {initial_compliance} -> {new_compliance}")
        self.assertGreaterEqual(new_compliance, initial_compliance)

    def test_server_api_uncertainty(self):
        density = self.uncertainty.get_phase_space_density(self.brain.qubits)
        self.assertEqual(len(density), 10)
        self.assertEqual(len(density[0]), 10)

if __name__ == '__main__':
    unittest.main()
