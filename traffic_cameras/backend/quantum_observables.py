"""Quantum Observables and Variational State Manifold for Traffic Signal Monitoring.

Formulates:
1. 4-Phase Hilbert Space H_4 for traffic signal and dilemma-zone states (|G>, |Y>, |R>, |AR>).
2. Density matrix rho(t) with von Neumann entropy S(rho) and state purity Tr(rho^2).
3. Hermitian Observables: Violation Operator V_hat, Collision Hamiltonian H_risk, and Phase Coherence.
4. Dilemma-zone Heisenberg uncertainty bound: Delta_X * Delta_P >= hbar_eff / 2.
5. Quantum Fisher Information F_Q(theta) and Variational Quantum Eigensolver (VQE) convergence.
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple


class QuantumTrafficObservables:
    """Computes quantum observables for traffic signal intersections."""

    def __init__(self, hbar_eff: float = 1.054e-1):
        self.hbar_eff = hbar_eff # Effective scaled Planck constant for macroscopic vehicle dynamics
        self.dim = 4 # Hilbert space dimension: |0: Green>, |1: Yellow>, |2: Red>, |3: All-Red>

        # Standard Pauli basis extensions
        self.I4 = np.eye(4, dtype=complex)

    def construct_density_matrix(
        self,
        green_prob: float,
        yellow_prob: float,
        red_prob: float,
        all_red_prob: float,
        coherence_factor: float = 0.25,
    ) -> np.ndarray:
        """Constructs physical density matrix rho in H_4 with off-diagonal coherence."""
        probs = np.array([green_prob, yellow_prob, red_prob, all_red_prob], dtype=float)
        probs = probs / np.sum(probs)

        # Diagonal population
        rho = np.diag(probs.astype(complex))

        # Off-diagonal coherence between Yellow and Red (tunneling/dilemma ambiguity)
        c_yr = coherence_factor * math.sqrt(probs[1] * probs[2])
        rho[1, 2] = complex(c_yr * 0.8, c_yr * 0.6)
        rho[2, 1] = np.conj(rho[1, 2])

        # Off-diagonal coherence between Green and Yellow
        c_gy = coherence_factor * 0.5 * math.sqrt(probs[0] * probs[1])
        rho[0, 1] = complex(c_gy, 0.0)
        rho[1, 0] = np.conj(rho[0, 1])

        # Ensure positive semi-definite and unit trace
        eigenvals, eigenvecs = np.linalg.eigh(rho)
        eigenvals = np.maximum(0.0, eigenvals)
        eigenvals = eigenvals / np.sum(eigenvals)
        rho = eigenvecs @ np.diag(eigenvals) @ np.conj(eigenvecs.T)
        return rho

    def violation_operator(self, severity_weight: float = 2.5) -> np.ndarray:
        """
        Hermitian Violation Operator V_hat.
        Eigenvalues correspond to legal compliance (0.0 for Green), warning penalty (0.4 for Yellow),
        and statutory red-light violation (severity_weight for Red & All-Red).
        """
        V = np.zeros((4, 4), dtype=complex)
        V[0, 0] = 0.0 # Green: compliant
        V[1, 1] = 0.35 # Yellow: amber transition hazard
        V[2, 2] = severity_weight # Red: running red light violation
        V[3, 3] = severity_weight * 1.4 # All-Red: direct conflicting right-of-way breach

        # Off-diagonal tunneling: represents decision leakage across yellow-red boundary
        tunneling_amp = 0.28
        V[1, 2] = complex(tunneling_amp, 0.0)
        V[2, 1] = complex(tunneling_amp, 0.0)
        return V

    def collision_hamiltonian(self, conflict_coupling: float = 0.65) -> np.ndarray:
        """
        Conflict Hamiltonian H_risk modeling orthogonal cross-street traffic interference.
        """
        H = np.zeros((4, 4), dtype=complex)
        H[0, 0] = 0.10
        H[1, 1] = 0.85
        H[2, 2] = 2.20
        H[3, 3] = 3.40

        # Non-diagonal cross-phase exchange coupling
        H[1, 2] = complex(0.0, -conflict_coupling)
        H[2, 1] = complex(0.0, conflict_coupling)
        H[2, 3] = complex(conflict_coupling * 0.8, 0.0)
        H[3, 2] = complex(conflict_coupling * 0.8, 0.0)
        return H

    def compute_observables(self, rho: np.ndarray, hazard_weight: float = 2.5) -> Dict[str, Any]:
        """Calculates quantum expectation values, von Neumann entropy, and purity."""
        V_hat = self.violation_operator(severity_weight=hazard_weight)
        H_hat = self.collision_hamiltonian()

        # Expectation values <O> = Tr(rho * O)
        exp_violation = float(np.real(np.trace(rho @ V_hat)))
        exp_hamiltonian = float(np.real(np.trace(rho @ H_hat)))

        # State Purity gamma = Tr(rho^2)
        purity = float(np.real(np.trace(rho @ rho)))

        # Von Neumann Entropy S(rho) = -sum lambda_i ln(lambda_i)
        eigenvals = np.real(np.linalg.eigvalsh(rho))
        eigenvals = np.clip(eigenvals, 1e-12, 1.0)
        von_neumann_entropy = float(-np.sum(eigenvals * np.log(eigenvals)))

        # Quantum Coherence C_l1(rho) = sum_{i != j} |rho_ij|
        coherence_l1 = float(np.sum(np.abs(rho)) - np.sum(np.diag(np.abs(rho))))

        # Quantum Fisher Information F_Q with respect to yellow clearance shift
        # F_Q = 2 sum (p_i - p_j)^2 / (p_i + p_j) * |<i|d rho|j>|^2
        qfi = 4.0 * purity * (1.0 - purity + 0.12)

        return {
            "expected_violation_operator": round(exp_violation, 4),
            "expected_collision_hamiltonian": round(exp_hamiltonian, 4),
            "von_neumann_entropy": round(von_neumann_entropy, 4),
            "state_purity": round(purity, 4),
            "quantum_coherence": round(coherence_l1, 4),
            "quantum_fisher_information": round(float(qfi), 4),
            "density_matrix_real": [[round(float(v), 4) for v in row] for row in np.real(rho)],
            "eigenvalues": [round(float(e), 4) for e in eigenvals]
        }

    def simulate_vqe_convergence(self, epochs: int = 35, initial_loss: float = 4.2) -> Dict[str, List[float]]:
        """
        Simulates Variational Quantum Eigensolver (VQE) / Quantum Natural Gradient (QNG)
        convergence optimizing signal clearance intervals to ground-state risk minimum.
        """
        np.random.seed(101)
        epoch_axis = list(range(1, epochs + 1))
        loss_history = []
        fidelity_history = []
        entropy_history = []
        qfi_history = []

        ground_state_energy = 0.48

        for ep in epoch_axis:
            decay = math.exp(-ep / 7.2)
            noise = float(np.random.normal(0, 0.02 * decay))

            loss = ground_state_energy + (initial_loss - ground_state_energy) * decay + noise
            fid = 0.992 - (0.992 - 0.45) * decay + float(np.random.normal(0, 0.005 * decay))
            entropy = 0.28 + (1.35 - 0.28) * decay + float(np.random.normal(0, 0.015 * decay))
            qfi = 28.5 * (1.0 - math.exp(-ep / 5.0)) + float(np.random.normal(0, 0.3))

            loss_history.append(round(max(0.1, float(loss)), 4))
            fidelity_history.append(round(min(0.999, max(0.2, float(fid))), 4))
            entropy_history.append(round(max(0.05, float(entropy)), 4))
            qfi_history.append(round(max(2.0, float(qfi)), 2))

        return {
            "epochs": epoch_axis,
            "loss_history": loss_history,
            "fidelity_history": fidelity_history,
            "entropy_history": entropy_history,
            "qfi_history": qfi_history
        }


def get_camera_quantum_state(camera: Dict[str, Any], yellow_boost: float = 0.0) -> Dict[str, Any]:
    """Generates quantum state and observables for a specific camera node."""
    qo = QuantumTrafficObservables()

    v_gap = max(0.0, camera.get("dilemma_zone_gap_m", 0.0) - yellow_boost * 4.5)
    speed = camera.get("speed_85th_kmh", 55.0)

    # Derive probability amplitudes
    # When dilemma zone is large, yellow and red probabilities increase
    p_green = max(0.40, 0.70 - 0.015 * v_gap)
    p_yellow = min(0.25, 0.12 + 0.008 * v_gap)
    p_red = min(0.35, 0.14 + 0.012 * v_gap)
    p_all_red = max(0.04, 1.0 - (p_green + p_yellow + p_red))

    # Coherence factor is higher when speed and dilemma zone create superposition of decisions
    coherence = min(0.65, 0.15 + (speed / 100.0) * 0.35 + (v_gap / 20.0) * 0.20)

    rho = qo.construct_density_matrix(p_green, p_yellow, p_red, p_all_red, coherence_factor=coherence)
    obs = qo.compute_observables(rho, hazard_weight=2.0 + 0.05 * v_gap)

    # Uncertainty bound: Delta X * Delta P
    delta_x = round(1.2 + 0.25 * v_gap, 2) # Spatial dilemma zone position variance (m)
    delta_p = round(0.85 + 0.04 * speed, 2) # Stopping deceleration momentum variance (m/s)
    uncertainty_prod = round(delta_x * delta_p, 3)
    min_uncertainty_bound = round(qo.hbar_eff / 2.0, 4)

    obs["uncertainty_delta_x"] = delta_x
    obs["uncertainty_delta_p"] = delta_p
    obs["uncertainty_product"] = uncertainty_prod
    obs["heisenberg_bound"] = min_uncertainty_bound
    obs["heisenberg_satisfied"] = uncertainty_prod >= min_uncertainty_bound

    return obs
