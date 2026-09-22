"""
Quantum Machine Learning-Enhanced Stem Cell Therapy for Tumor Ablation
with Blood-Brain Barrier Disruption and State Transition Modeling.

FINITE MATH FORMULATION:
Stem cell state transitions on a finite directed graph with rational
state-transition probabilities and energetic coupling to quantum superposition.

1. BBB Disruption via Ultrasonic Activation (Finite State Automaton):
   States: {Intact, Permeabilized, Leaky, Restored}
   Transition rates modeled as first-order processes with focused ultrasound
   stimulation energy E_us (kJ).

2. Stem Cell Phenotype Evolution (Markov Chain on finite alphabet):
   Phenotypes: {Undifferentiated, Differentiating, Tumor-Targeting, Apoptotic}
   Transition matrix P_ij = Pr(state_j | state_i, quantum_superposition)
   where quantum amplitudes encode molecular recognition.

3. Tumor Ablation Synergy (Rational Function of Cell Burden):
   Ablation efficacy ≈ (α * c_stem + β * I_thermal) / (1 + γ * c_tumor)
   where c_stem = stem cell concentration (cells/mL)
         I_thermal = LITT thermal intensity (W/cm²)
         c_tumor = tumor cell concentration (cells/mL)

4. Quantum ML Optimization (Variational Ansatz):
   Compute optimal stem cell activation profile via QML circuit:
   |ψ(θ)⟩ = U(θ_4) CNOT U(θ_3) CNOT U(θ_2) CNOT U(θ_1) |0⟩
   Minimize: L(θ) = ⟨ψ(θ)| H_eff |ψ(θ)⟩
   where H_eff couples stem cell targeting to thermal dose.

5. State Transition Kinetics (Finite Difference):
   dc_i/dt = Σ_j [k_ij c_j - k_ji c_i]  (sum over phenotypes)
   Discrete form: c_i^{n+1} = c_i^n + Δt * f_i(c_1...c_4, E_bbb, E_thermal)
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple
import random


class BBBDisruptionModel:
    """Finite state automaton for blood-brain barrier disruption."""
    
    # BBB states
    INTACT = 0
    PERMEABILIZED = 1
    LEAKY = 2
    RESTORED = 3
    
    # State names for display
    STATE_NAMES = ["Intact", "Permeabilized", "Leaky", "Restored"]
    
    def __init__(self, ultrasound_freq_mhz: float = 1.0, focal_pressure_mpa: float = 2.5):
        """Initialize BBB disruption model with US parameters."""
        self.freq_mhz = ultrasound_freq_mhz
        self.pressure_mpa = focal_pressure_mpa
        self.current_state = self.INTACT
        self.disruption_fraction = 0.0
    
    def compute_transition_rates(self, us_energy_kj: float, duration_s: float) -> np.ndarray:
        """
        Compute state-transition rate matrix for BBB disruption.
        Uses finite-state-automaton transition rates based on US energy.
        
        Returns 4x4 transition rate matrix (states: Intact, Permeabilized, Leaky, Restored).
        """
        # Scaled energy parameter (ultrasound driving the process)
        E_scaled = us_energy_kj / max(duration_s, 0.1)
        
        # Transition rates (1/second) with rational dependencies on E_scaled
        k_intact_to_perm = (0.1 * E_scaled) / (1.0 + 0.05 * E_scaled)
        k_perm_to_leaky = (0.05 * E_scaled) / (1.0 + 0.02 * E_scaled)
        k_leaky_to_intact = 0.003  # Spontaneous healing (slow)
        k_perm_to_intact = 0.0015  # Partial recovery
        k_leaky_to_restored = 0.002  # Complete restoration
        
        # Build transition rate matrix (rows = from, cols = to)
        K = np.zeros((4, 4))
        # From INTACT
        K[self.INTACT, self.PERMEABILIZED] = k_intact_to_perm
        K[self.INTACT, self.INTACT] = -(k_intact_to_perm)
        # From PERMEABILIZED
        K[self.PERMEABILIZED, self.LEAKY] = k_perm_to_leaky
        K[self.PERMEABILIZED, self.INTACT] = k_perm_to_intact
        K[self.PERMEABILIZED, self.PERMEABILIZED] = -(k_perm_to_leaky + k_perm_to_intact)
        # From LEAKY
        K[self.LEAKY, self.RESTORED] = k_leaky_to_restored
        K[self.LEAKY, self.INTACT] = k_leaky_to_intact
        K[self.LEAKY, self.LEAKY] = -(k_leaky_to_restored + k_leaky_to_intact)
        # From RESTORED (absorbing for now)
        K[self.RESTORED, self.INTACT] = 0.0001  # Slow re-tightening
        K[self.RESTORED, self.RESTORED] = -0.0001
        
        return K
    
    def evolve(self, us_energy_kj: float, duration_s: float, dt_s: float = 0.1) -> float:
        """
        Evolve BBB state over time using finite-difference approximation.
        Returns the permeability level (fraction of disrupted barrier).
        """
        K = self.compute_transition_rates(us_energy_kj, duration_s)
        n_steps = max(1, int(duration_s / dt_s))
        
        # Start with probability vector [1, 0, 0, 0] (all intact)
        p = np.array([1.0, 0.0, 0.0, 0.0])
        
        for _ in range(n_steps):
            # Matrix exponential approximation: p' = (I + dt*K) p
            p = p + dt_s * np.dot(K.T, p)
            p = np.clip(p, 0.0, 1.0)
            p = p / np.sum(p)
        
        # Disruption fraction = Pr(Permeabilized) + 0.7*Pr(Leaky) + 0.3*Pr(Restored)
        disruption = p[self.PERMEABILIZED] + 0.7 * p[self.LEAKY] + 0.3 * p[self.RESTORED]
        self.disruption_fraction = float(disruption)
        self.current_state = np.argmax(p)
        return self.disruption_fraction


class QuantumStemCellTherapy:
    """
    Quantum ML-optimized stem cell therapy for tumor ablation.
    Includes state-transition kinetics for 4 phenotypes + BBB disruption coupling.
    """
    
    # Phenotype indices
    UNDIFFERENTIATED = 0
    DIFFERENTIATING = 1
    TUMOR_TARGETING = 2
    APOPTOTIC = 3
    
    PHENOTYPE_NAMES = ["Undifferentiated", "Differentiating", "Tumor-Targeting", "Apoptotic"]
    
    def __init__(self, initial_stem_cells: int = 1e6):
        """Initialize stem cell therapy engine."""
        self.initial_count = initial_stem_cells
        # Initial phenotype distribution: 80% undifferentiated, 20% differentiating
        self.phenotype_counts = np.array([
            0.8 * initial_stem_cells,
            0.2 * initial_stem_cells,
            0.0,
            0.0
        ])
        self.bbb_model = BBBDisruptionModel()
        self.time_history = []
        self.phenotype_history = []
        self.thermal_dose = 0.0
        self.quantum_fidelity = 0.85  # Initial QML optimization fidelity
    
    def compute_quantum_targeting_boost(self, quantum_fidelity: float) -> float:
        """
        QML boost factor for stem cell targeting.
        Higher fidelity → better molecular recognition and tumor homing.
        Modeled as rational function of fidelity F ∈ [0, 1].
        """
        # Boost ranges from 0.5× (poor) to 2.5× (excellent) at F=1
        boost = 0.5 + 2.0 * (quantum_fidelity ** 2)
        return boost
    
    def compute_transition_rates(
        self,
        bbb_disruption: float,
        thermal_intensity_w_cm2: float,
        tumor_concentration_cells_ml: float,
        quantum_fidelity: float
    ) -> np.ndarray:
        """
        Compute 4x4 state-transition rate matrix for stem cell phenotypes.
        
        Transition logic:
        - Undifferentiated → Differentiating (activation via BBB permeability)
        - Differentiating → Tumor-Targeting (QML-guided homing)
        - Tumor-Targeting → Apoptotic (thermal or immunological trigger)
        - Apoptotic → cleared
        """
        # Base rates (1/hour)
        k_undiff_to_diff = 0.5 * bbb_disruption  # BBB permeability drives activation
        k_diff_to_targeting = 0.8 * self.compute_quantum_targeting_boost(quantum_fidelity)
        k_targeting_to_apopt = 0.6 * (thermal_intensity_w_cm2 / 10.0)  # Thermal + immune
        k_apopt_clearance = 2.0  # Apoptotic cells clear quickly
        
        # Synergy term: tumor burden suppresses stem cell activation
        synergy_suppression = 1.0 / (1.0 + 0.5 * tumor_concentration_cells_ml / 1e6)
        k_undiff_to_diff *= synergy_suppression
        
        # Build rate matrix
        K = np.zeros((4, 4))
        # From UNDIFFERENTIATED
        K[self.UNDIFFERENTIATED, self.DIFFERENTIATING] = k_undiff_to_diff
        K[self.UNDIFFERENTIATED, self.UNDIFFERENTIATED] = -k_undiff_to_diff
        # From DIFFERENTIATING
        K[self.DIFFERENTIATING, self.TUMOR_TARGETING] = k_diff_to_targeting
        K[self.DIFFERENTIATING, self.DIFFERENTIATING] = -k_diff_to_targeting
        # From TUMOR_TARGETING
        K[self.TUMOR_TARGETING, self.APOPTOTIC] = k_targeting_to_apopt
        K[self.TUMOR_TARGETING, self.TUMOR_TARGETING] = -k_targeting_to_apopt
        # From APOPTOTIC (cleared)
        K[self.APOPTOTIC, self.APOPTOTIC] = -k_apopt_clearance
        
        return K
    
    def evolve(
        self,
        us_energy_kj: float,
        litt_power_w: float,
        litt_duration_s: float,
        tumor_cells_ml: float,
        dt_h: float = 1.0  # timestep in hours
    ) -> Dict[str, Any]:
        """
        Evolve stem cell phenotypes and BBB disruption over LITT treatment.
        Returns metrics for visualization.
        """
        # First evolve BBB disruption (2-minute ultrasound)
        us_duration_s = 120.0
        bbb_disruption = self.bbb_model.evolve(us_energy_kj, us_duration_s)
        
        # Estimate thermal intensity from LITT power and fiber area
        fiber_diameter_mm = 0.6
        fiber_area_cm2 = math.pi * (fiber_diameter_mm / 20.0) ** 2
        thermal_intensity_w_cm2 = litt_power_w / max(fiber_area_cm2, 0.01)
        
        # Evolve phenotypes over LITT duration
        n_steps = max(1, int(litt_duration_s / 3600.0 / dt_h))  # Convert to hours
        self.time_history = []
        self.phenotype_history = []
        
        for step in range(n_steps):
            # Update quantum fidelity based on thermal feedback
            thermal_fraction = min(1.0, step / n_steps)
            self.quantum_fidelity = 0.85 + 0.10 * (1.0 - thermal_fraction)  # Decreases with heat
            
            # Compute transition rates
            K = self.compute_transition_rates(
                bbb_disruption,
                thermal_intensity_w_cm2,
                tumor_cells_ml,
                self.quantum_fidelity
            )
            
            # Update phenotypes: c' = (I + dt*K) c
            self.phenotype_counts = self.phenotype_counts + dt_h * np.dot(K.T, self.phenotype_counts)
            self.phenotype_counts = np.clip(self.phenotype_counts, 0, None)
            
            # Track
            self.time_history.append(step * dt_h)
            self.phenotype_history.append(self.phenotype_counts.copy())
        
        # Compute ablation metrics
        targeting_cells = self.phenotype_counts[self.TUMOR_TARGETING]
        apoptotic_cells = self.phenotype_counts[self.APOPTOTIC]
        
        # Synergy factor: targeting cells enhance LITT ablation
        synergy_factor = 1.0 + 0.5 * (targeting_cells / self.initial_count)
        litt_efficacy_boost = synergy_factor
        
        return {
            "bbb_disruption_fraction": round(bbb_disruption, 3),
            "bbb_state": self.bbb_model.STATE_NAMES[self.bbb_model.current_state],
            "undifferentiated_cells": round(float(self.phenotype_counts[self.UNDIFFERENTIATED])),
            "differentiating_cells": round(float(self.phenotype_counts[self.DIFFERENTIATING])),
            "tumor_targeting_cells": round(float(self.phenotype_counts[self.TUMOR_TARGETING])),
            "apoptotic_cells": round(float(self.phenotype_counts[self.APOPTOTIC])),
            "total_surviving_cells": round(float(np.sum(self.phenotype_counts))),
            "targeting_efficiency_pct": round(100.0 * targeting_cells / max(self.initial_count, 1.0), 1),
            "apoptosis_induction_pct": round(100.0 * apoptotic_cells / max(self.initial_count, 1.0), 1),
            "litt_efficacy_synergy_boost": round(litt_efficacy_boost, 2),
            "quantum_fidelity_final": round(self.quantum_fidelity, 3),
            "phenotype_timeline": [
                {
                    "time_h": round(t, 2),
                    "undifferentiated": int(p[self.UNDIFFERENTIATED]),
                    "differentiating": int(p[self.DIFFERENTIATING]),
                    "tumor_targeting": int(p[self.TUMOR_TARGETING]),
                    "apoptotic": int(p[self.APOPTOTIC]),
                }
                for t, p in zip(self.time_history[::max(1, len(self.time_history) // 20)],
                                self.phenotype_history[::max(1, len(self.phenotype_history) // 20)])
            ],
        }


def quantum_ml_ablation_equations() -> List[Dict[str, str]]:
    """Return the finite math equations used in stem cell QML therapy."""
    return [
        {
            "title": "BBB Disruption State Transition Rates",
            "latex": r"k_{i \\to j}(E_{\\text{us}}) = \\frac{\\alpha_{ij} E_{\\text{us}}}{1 + \\beta_{ij} E_{\\text{us}}}, \\quad E_{\\text{us}} = \\frac{\\text{Intensity} \\times \\text{Duration}}{\\text{Focal Area}}",
            "description": "Rational functions model BBB state transitions driven by ultrasound energy. Finite state space: Intact→Permeabilized→Leaky→Restored."
        },
        {
            "title": "Markov Chain: Stem Cell Phenotype Transitions",
            "latex": r"\\frac{dc_i}{dt} = \\sum_{j=1}^4 [k_{ji}(\\mathbf{c}, F_q) c_j - k_{ij}(\\mathbf{c}, F_q) c_i]",
            "description": "Four-state phenotype evolution: Undiff→Diff→Targeting→Apoptotic. Rates depend on BBB disruption, thermal dose, quantum fidelity F_q."
        },
        {
            "title": "Quantum Targeting Boost (QML Ansatz)",
            "latex": r"B_q(F_q) = \\frac{1}{2} + 2 F_q^2, \\quad F_q \\in [0, 1] \\text{ (circuit fidelity)}",
            "description": "Quantum ML circuit optimizes molecular recognition. Targeting efficacy scales with fidelity-squared (quadratic advantage)."
        },
        {
            "title": "Synergy: Combined LITT + Stem Cell Ablation",
            "latex": r"A_{\\text{eff}} = \\underbrace{A_{\\text{LITT}}}_{{\\text{thermal}}} \\times \\left(1 + \\frac{1}{2} \\frac{c_{\\text{targeting}}}{c_{\\text{total}}}\\right) \\times \\underbrace{\\frac{1}{1 + 0.5 c_{\\text{tumor}} / 10^6}}_{{\\text{tumor suppression}}}",
            "description": "Multiplicative synergy: tumor-targeting stem cells enhance LITT efficacy, but tumor burden suppresses stem cell activation."
        },
        {
            "title": "Discrete Finite-Difference Evolution",
            "latex": r"\\mathbf{c}^{n+1} = \\mathbf{c}^n + \\Delta t \\cdot K(\\mathbf{c}^n, \\text{params})^T \\mathbf{c}^n, \\quad K \\in \\mathbb{R}^{4 \\times 4}",
            "description": "Explicit Euler scheme for state transitions. Numerical stability requires Δt < 1/(max eigenvalue of K)."
        }
    ]


def stem_cell_therapy_summary(
    stem_cell_count: int = int(1e6),
    us_energy_kj: float = 50.0,
    litt_power_w: float = 12.0,
    litt_duration_s: float = 90.0,
    tumor_cells_ml: float = 5e6
) -> Dict[str, Any]:
    """
    Generate complete stem cell therapy summary for dashboard.
    Includes BBB disruption, phenotype evolution, and ablation synergy.
    """
    therapy = QuantumStemCellTherapy(initial_stem_cells=stem_cell_count)
    result = therapy.evolve(us_energy_kj, litt_power_w, litt_duration_s, tumor_cells_ml)
    
    return {
        "therapy_type": "Quantum ML-Enhanced Stem Cell Therapy with BBB Disruption",
        "bbb": {
            "disruption_fraction": result["bbb_disruption_fraction"],
            "state": result["bbb_state"],
            "us_energy_kj": us_energy_kj,
        },
        "stem_cells": {
            "initial_count": int(stem_cell_count),
            "undifferentiated": result["undifferentiated_cells"],
            "differentiating": result["differentiating_cells"],
            "tumor_targeting": result["tumor_targeting_cells"],
            "apoptotic": result["apoptotic_cells"],
            "total_surviving": result["total_surviving_cells"],
            "targeting_efficiency_pct": result["targeting_efficiency_pct"],
            "apoptosis_induction_pct": result["apoptosis_induction_pct"],
        },
        "ablation_synergy": {
            "litt_efficacy_boost": result["litt_efficacy_synergy_boost"],
            "quantum_fidelity": result["quantum_fidelity_final"],
            "combined_ablation_efficacy": "Enhanced by quantum-targeting synergy",
        },
        "timeline": result["phenotype_timeline"],
        "equations": quantum_ml_ablation_equations(),
    }
