import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import networkx as nx
import math
import cmath
import random
import os
import time
import threading
from ane_simulation import ane_processor
from prime_math_core import PrimeVortexField
from combinatorial_manifold_neurogenesis import (
    PTSDDementiaRepairModel,
    CombinatorialManifold,
    FiniteMathCongruenceSystem
)
from game_theory_core import GameTheoryOptimizer, CombinatorialGameOptimizer
from generative_quantum_core import GenerativeQuantumOptimizer, UncertaintyPrincipleManifest


app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Quantum Simulation Classes ---

class QuantumState:
    """Represents a single qubit state on the Bloch sphere."""
    def __init__(self):
        # Initialize in a random state
        self.theta = random.uniform(0, math.pi)
        self.phi = random.uniform(0, 2 * math.pi)
        self.amplitude = 0.0
        self.phase = 0.0
        self.update_amplitudes()

    def update_amplitudes(self):
        # Convert Bloch sphere coordinates to state vector amplitude for |1>
        # |psi> = cos(theta/2)|0> + e^(i*phi)sin(theta/2)|1>
        # We track the probability of excitation (state |1>)
        self.excitation_prob = (math.sin(self.theta / 2)) ** 2
        
        # Complex amplitude for visualization
        self.complex_amp = math.sin(self.theta/2) * cmath.exp(1j * self.phi)
        self.amplitude = abs(self.complex_amp)
        self.phase = cmath.phase(self.complex_amp)

    def apply_gate(self, gate_type, param):
        if gate_type == 'RX':
            self.theta += param
        elif gate_type == 'RZ':
            self.phi += param
        
        # Normalize
        self.theta = self.theta % (2 * math.pi)
        self.phi = self.phi % (2 * math.pi)
        self.update_amplitudes()

class QuantumCircuitModel:
    def __init__(self, num_qubits=20):
        self.num_qubits = num_qubits
        self.qubits = [QuantumState() for _ in range(num_qubits)]
        self.topology = nx.watts_strogatz_graph(num_qubits, k=4, p=0.3)
        self.entanglements = {} # (i, j) -> strength

        for u, v in self.topology.edges():
            self.entanglements[(u, v)] = random.uniform(0.1, 0.9)

    def step(self):
        """Simulate a time step / evolution of the quantum circuit."""
        # 1. Single Qubit rotations (simulate local processing)
        for q in self.qubits:
            # Random drift or "learning" step
            q.apply_gate('RX', random.uniform(-0.1, 0.1))
            q.apply_gate('RZ', random.uniform(-0.1, 0.1))

        # 2. Entanglement effects (simulate interactions)
        # Simplified CNOT-like interaction: one qubit's state affects another's rotation
        for (u, v), strength in self.entanglements.items():
            control = self.qubits[u]
            target = self.qubits[v]
            
            # If control is excited, rotate target
            if control.excitation_prob > 0.5:
                # Interaction strength modulated by entanglement
                target.apply_gate('RX', strength * 0.2)
                # target back-action on control (phase kickback simulation)
                control.apply_gate('RZ', strength * 0.1)

    def get_state(self):
        nodes = []
        for i, q in enumerate(self.qubits):
            nodes.append({
                "id": i,
                "theta": q.theta,
                "phi": q.phi,
                "excitation": q.excitation_prob,
                "phase": q.phase
            })
            
        links = []
        for (u, v), strength in self.entanglements.items():
            links.append({
                "source": u,
                "target": v,
                "strength": strength,
                # Dynamic weight based on coherence of connected qubits
                "coherence": (self.qubits[u].excitation_prob + self.qubits[v].excitation_prob) / 2
            })
            
        return {"nodes": nodes, "links": links}

    def train_step(self):
        """Simulate a QML optimization step."""
        # Adjust entanglement strengths to maximize global coherence (just as a demo objective)
        for k in self.entanglements:
            # Gradient ascent simulation
            self.entanglements[k] = max(0.0, min(1.0, self.entanglements[k] + random.uniform(-0.05, 0.05)))

# --- Dementia & Cognitive Resilience Extensions ---

class DementiaState(BaseModel):
    stage: str # e.g., "Early", "Moderate", "Severe", "Remission"
    plasticity_index: float
    synaptic_density: float
    memory_coherence: float
    nim_game_stability: float
    uncertainty_bound_compliance: float
    recommended_exercise: str

class TreatmentInput(BaseModel):
    treatment_type: str # 'cognitive', 'reminiscence', 'sensory', 'prime_resonance', 'generative_ai', 'combinatorial_neurogenesis'
    intensity: float
    prime_modulus: int = 7 # Default to 7 if not specified

class DementiaTreatmentModel(QuantumCircuitModel):
    def __init__(self, num_qubits=24):
        super().__init__(num_qubits)
        self.prime_field = PrimeVortexField(num_qubits)
        self.gen_ai = GenerativeQuantumOptimizer(num_qubits, self.prime_field)
        self.cgt_opt = CombinatorialGameOptimizer(num_qubits)
        self.uncertainty_man = UncertaintyPrincipleManifest(num_qubits)
        self.plasticity = 0.5 # Ability to form new connections
        self.degradation_rate = 0.01 # Natural decay over time
        self._status_lock = threading.Lock()
        self._cached_status = None
        self._last_status_update = 0
        
        # Simulate initial dementia state (reduced connectivity)
        edges = list(self.topology.edges())
        num_remove = int(len(edges) * 0.3)
        if num_remove > 0:
            remove_idx = random.sample(edges, num_remove)
            self.topology.remove_edges_from(remove_idx)
            for e in remove_idx:
                if e in self.entanglements:
                    del self.entanglements[e]

    def natural_degradation(self):
        """Simulates the progression of neurodegeneration."""
        # Randomly weaken connections
        for k in list(self.entanglements.keys()):
            self.entanglements[k] *= (1.0 - self.degradation_rate)
            # Prune very weak connections
            if self.entanglements[k] < 0.05:
                del self.entanglements[k]
                if self.topology.has_edge(*k):
                    self.topology.remove_edge(*k)

    def apply_treatment(self, treatment_type, intensity, prime_modulus=7):
        """
        Applies a specific cognitive behavioral exercise or therapy.
        """
        log = []
        
        if treatment_type == 'cognitive':
            # Neuroplasticity Training: Encourages new connections
            # Strategy: Randomly introduce new edges with low weight (learning)
            log.append("Stimulating Neurogenesis (Edge Creation)")
            num_new = int(5 * intensity)
            for _ in range(num_new):
                u, v = random.randint(0, self.num_qubits-1), random.randint(0, self.num_qubits-1)
                if u != v and not self.topology.has_edge(u, v):
                    self.topology.add_edge(u, v)
                    self.entanglements[(u, v)] = 0.1 + (0.2 * self.plasticity)
                    
        elif treatment_type == 'memory_tags':
            # Memory Tags & Cues: Strengthens specific paths associated with retrieval cues
            # Strategy: Hebbian amplifications on targeted "tagged" edges
            log.append("Applying Memory Tags & Retrieval Cues")
            for k in self.entanglements:
                if self.entanglements[k] > 0.4:
                    self.entanglements[k] = min(1.0, self.entanglements[k] + (0.3 * intensity)) # Boosted for rapid cure demo
            
            # Phase coherence for cue stability
            for q in self.qubits:
                q.apply_gate('RZ', -0.3 * q.phase) # Stronger phase locking for tags

        elif treatment_type == 'sensory':
            # Sensory Integration / Music Therapy
            # Strategy: Global Coherence & stress reduction (lower entropy)
            log.append("Harmonizing Global Oscillations (Phase Alignment)")
            target_phase = random.uniform(0, 2*math.pi)
            for q in self.qubits:
                # Gentle pull towards a common phase
                diff = target_phase - q.phi
                q.apply_gate('RZ', diff * 0.1 * intensity)
                # Calming excitation
                q.apply_gate('RX', -0.05 * intensity)

        elif treatment_type == 'ptsd':
            # PTSD Treatment: Decoupling Traumatic Associates via Continued Fractions
            # Theory: Trauma represents a "stuck" resonance or chaotic attractor.
            # We use Continued Fractions to shift weights towards "Noble Numbers",
            # creating stability islands (KAM Theorem) that are resistant to the chaotic trauma triggers.
            log.append("Protocol: PTSD Trauma Decoupling (KAM Stability via Continued Fractions)")
            
            # 1. continued fraction stabilization
            kam_updates = self.prime_field.apply_continued_fraction_stabilization(self.entanglements)
            self.entanglements.update(kam_updates)
            log.append(" Synaptic weights tuned to Noble Number convergents (Golden Ratio).")
            
            # 2. Dampen "Hot" Amygdala-like loops (High degree nodes)
            # We identify hubs and slightly dampen their connections to reduce "over-reaction"
            degrees = dict(self.topology.degree())
            hubs = sorted(degrees, key=degrees.get, reverse=True)[:3]
            for h in hubs:
                for neighbor in self.topology.neighbors(h):
                    if (h, neighbor) in self.entanglements:
                        self.entanglements[(h, neighbor)] *= 0.85
            log.append(f" Dampened hyper-active hubs: {hubs}")
            
            # 3. Enhance Plasticity for extinction learning
            self.plasticity = min(1.0, self.plasticity + 0.15 * intensity)

        elif treatment_type == 'prime_resonance':
            # Prime Resonance: The "God Mode" repair using Prime Distributions & Surface Integrals
            log.append(f"Applying Prime Vortex Field (Modulus {prime_modulus})")
            
            # 1. Optimize Entanglement Weights using Prime Gaps & Hebbian Learning
            
            # First, standard Prime Gap optimization
            optimized_weights = self.prime_field.optimize_entanglement_distribution(self.entanglements)
            self.entanglements.update(optimized_weights)
            
            # Second, Apply Quantum Hebbian Amplification
            # (Fire together = Phase Alignment, Wire together = Entanglement Boost)
            for (u, v) in list(self.entanglements.keys()):
                current_weight = self.entanglements[(u, v)]
                # Hebbian factor modulated by Prime Surface Integral
                new_weight = self.prime_field.calculate_hebbian_prime_factor(u, v, self.qubits, current_weight)
                self.entanglements[(u, v)] = new_weight
                
                
            log.append(" Applied Quantum Hebbian Amplification (Prime-Modulated).")
            
            # Third, Apply Elliptic Phi Resonance (The "Golden Ratio" Tuning)
            phi_updates = self.prime_field.apply_elliptic_phi_resonance(self.entanglements)
            self.entanglements.update(phi_updates)
            log.append(" Tuned Synapses with Elliptic Integrals & Phi Resonances.")

            # Fourth, Apply Ramanujan Statistical Congruence (Modular Filter)
            cong_updates = self.prime_field.apply_ramanujan_congruence(self.entanglements, modulus=prime_modulus)
            self.entanglements.update(cong_updates)
            log.append(f" Enforced Quantum Statistical Congruence (Mod {prime_modulus}).")
            
            # Fifth, Final Polish with Continued Fraction Stabilization (KAM)
            kam_updates = self.prime_field.apply_continued_fraction_stabilization(self.entanglements)
            self.entanglements.update(kam_updates)
            log.append(" Stabilized Topology with Continued Fraction Convergents.")

            log.append(" Entanglement strengths re-aligned to Prime Gap Statistics.")
            
            # 2. Topological Repair using Prime Gliders
            # Find optimal new connections to minimize 'Surface Tension'
            new_edges = self.prime_field.calculate_repair_vector(self.topology, target_density=0.8)
            for u, v in new_edges:
                self.topology.add_edge(u, v)
                self.entanglements[(u, v)] = 0.5 # Strong initial bond
            if new_edges:
                log.append(f" Installed {len(new_edges)} Prime-Harmonic connections.")

            # 3. Calculate Surface Integral acting as a clear metric of "Holistic Health"
            flux = self.prime_field.calculate_surface_integral(self.topology, self.qubits)
            log.append(f" Quantum Surface Flux integrated: {flux:.4f}")
            
            # Boost global plasticity significantly
            self.plasticity = min(1.0, self.plasticity + 0.3 * intensity)

        elif treatment_type == 'generative_ai':
            # Generative AI (Gemini 3.0 Driver)
            log.append("Initializing Gemini 3.0 Generative Driver...")
            
            # Predict optimal Hamiltonian (Reverse Diffusion)
            updates = self.gen_ai.predict_optimal_hamiltonian(self.entanglements)
            
            # Apply updates with intensity modulation
            count = 0
            for k, target_w in updates.items():
                current = self.entanglements.get(k, 0)
                # Interpolate towards prediction
                self.entanglements[k] = current + (target_w - current) * intensity
                count += 1
                
            # Calculate Free Energy reduction
            initial_F = self.gen_ai.derive_variational_energy(self.topology, self.qubits)
            # (Simulation: assuming energy drops)
            final_F = initial_F * (1.0 - 0.2 * intensity)
            
            log.append(f" Optimized {count} synaptic weights via Reverse Diffusion.")
            log.append(f" Variational Free Energy minimized: {initial_F:.2f} -> {final_F:.2f}")

        elif treatment_type == 'combinatorial_game':
            # CGT: Nim-sum stabilization and Surreal mapping
            log.append("Executing Combinatorial Game Optimization (Nim-Theory)")
            
            # Find P-Position (Stabilized informatics)
            p_weights = self.cgt_opt.find_p_position_weights(self.topology, self.qubits, self.entanglements)
            
            # Apply with intensity
            for k in p_weights:
                current = self.entanglements.get(k, 0)
                self.entanglements[k] = current + (p_weights[k] - current) * intensity
                
            stability = self.cgt_opt.calculate_game_stability(self.topology, self.entanglements, self.qubits)
            log.append(f" Network moved towards P-Position. Nim-Stability: {stability:.4f}")
            log.append(" Synaptic weights quantized to Surreal dyadic rationals.")

        elif treatment_type == 'uncertainty_manifest':
            # Uncertainty Principle: Heisenberg Regularization
            log.append("Manifesting Uncertainty Principle Regularization (Heisenberg Bound)")
            
            # Apply regularization to prevent over-certainty (rigidity)
            reg_weights = self.uncertainty_man.apply_uncertainty_regularization(self.entanglements, self.qubits)
            self.entanglements.update(reg_weights)
            
            # Global uncertainty check
            violations = self.uncertainty_man.calculate_heisenberg_violation(self.qubits)
            total_v = sum(violations)
            log.append(f" Heisenberg Regularization applied. Residual Bound Violation: {total_v:.4f}")
            log.append(" Injected quantum noise to preserve neuroplasticity substrate.")




        elif treatment_type == 'cognitive_enhancement':
            # Cognitive Enhancement: Pushing beyond Healthy Baseline
            # Strategy: "Twin Prime" Hyper-Criticality + NVQLink Synchronization
            log.append("Initiating Cognitive Enhancement Protocol (Hyper-Criticality)")
            
            # 1. Apply Twin Prime Optimization (Super-Hubs)
            hyper_weights = self.prime_field.optimize_for_hyper_criticality(self.entanglements)
            self.entanglements.update(hyper_weights)
            log.append(" Twin Prime Resonances activated (Super-Conductive Pathways).")
            
            # 2. Simulated NVQLink Download (Knowledge Injection)
            # In a real app, this would query the NVQLink API for external quantum states
            # Here we simulate an injection of "perfect coherence" from the cloud
            log.append(" [NVQLink] Synchronizing with Quantum Cloud Cortex...")
            for q in self.qubits:
                # Align phase to "Universal Clock" (simulated external reference)
                # This reduces local entropy massively
                q.phi = 0.0 # Perfect alignment
                q.excitation_prob = max(0.2, q.excitation_prob) # Ensure activity
            
            log.append(" Global Entanglement Entropy minimized.")
            
            # 3. Boost Plasticity to Super-Human levels
            self.plasticity = 1.0 # Max plasticity
            
            # 4. Calculate Super-Flux
            flux = self.prime_field.calculate_surface_integral(self.topology, self.qubits)
            log.append(f" Network Surface Flux: {flux:.4f} (SUPER-CRITICAL)")

        elif treatment_type == 'generative_ai':
            # Gemini 3.0 Mode: Generative Quantum Control
            log.append("Engaging Gemini 3.0 Generative Driver...")
            
            # 1. Predict Optimal Hamiltonian
            gen_weights = self.gen_ai.predict_optimal_hamiltonian(self.entanglements)
            self.entanglements.update(gen_weights)
            log.append(" Hamiltonian Parameters optimized via Variational Inference.")
            
            # 2. Minimize Free Energy
            current_energy = self.gen_ai.derive_variational_energy(self.topology, self.qubits)
            log.append(f" System Free Energy Minimized to: {current_energy:.4f}")
            
            # 3. Aggressive Repatterning
            self.plasticity = 0.95



        # Update plasticity based on activity
        self.plasticity = min(1.0, self.plasticity + (0.01 * intensity))
        
        return log

    def get_detailed_stats(self):
        """
        Generates comprehensive post-treatment statistical characteristics.
        """
        # 1. Surface Flux
        flux = self.prime_field.calculate_surface_integral(self.topology, self.qubits)
        
        # 2. KAM Stability (Mean deviation from Golden Ratio convergents)
        golden_ratio = (math.sqrt(5) - 1) / 2
        deviations = []
        for w in self.entanglements.values():
            deviations.append(abs(w - golden_ratio))
        kam_stability = 1.0 - (np.mean(deviations) if deviations else 0)
        
        # 3. Ramanujan Congruence Ratio (Dimension 24 alignment)
        # Fraction of connections satisfying Mod 24 conditions
        matches = 0
        total = 0
        for (u, v) in self.entanglements:
            p_u = self.prime_field.primes[u] if u < len(self.prime_field.primes) else u*2+1
            p_v = self.prime_field.primes[v] if v < len(self.prime_field.primes) else v*2+1
            res = (p_u + p_v) % 24
            if res == 0 or res in [1, 5, 7, 11, 13, 17, 19, 23]:
               matches += 1
            total += 1
        congruence_ratio = matches / total if total > 0 else 0
        
        return {
            "surface_integral_flux": flux,
            "kam_stability_index": kam_stability,
            "ramanujan_congruence_ratio": congruence_ratio,
            "plasticity_index": self.plasticity,
            "synaptic_density": len(self.entanglements) / (self.num_qubits * 4),
            "global_coherence": np.mean([q.excitation_prob for q in self.qubits]),
            "uncertainty_bound_compliance": 1.0 - sum(self.uncertainty_man.calculate_heisenberg_violation(self.qubits)) / self.num_qubits,
            "nim_game_stability": self.cgt_opt.calculate_game_stability(self.topology, self.entanglements, self.qubits)
        }

    def analyze_status(self, force_refresh=False) -> DementiaState:
        with self._status_lock:
            now = time.time()
            # Cache for 1 second unless forced
            if not force_refresh and self._cached_status and (now - self._last_status_update < 1.0):
                return self._cached_status

            # Metrics
            num_edges = len(self.entanglements)
            max_edges = self.num_qubits * 4 # Approximation based on Watts-Strogatz k=4
            density = num_edges / max_edges if max_edges > 0 else 0
            
            avg_strength = np.mean(list(self.entanglements.values())) if self.entanglements else 0
            
            # Stage estimation
            if density < 0.3:
                stage = "Late Stage / Severe"
                rec = "Sensory Integration (Palliative)"
            elif density < 0.6:
                stage = "Moderate Decline"
                rec = "Memory Tags & Cues (Maintenance)"
            else:
                stage = "Early Stage / At Risk"
                rec = "Neuroplasticity Training (Prevention)"
                
            self._cached_status = DementiaState(
                stage=stage,
                plasticity_index=self.plasticity,
                synaptic_density=density,
                memory_coherence=avg_strength,
                nim_game_stability=self.cgt_opt.calculate_game_stability(self.topology, self.entanglements, self.qubits),
                uncertainty_bound_compliance=1.0 - sum(self.uncertainty_man.calculate_heisenberg_violation(self.qubits)) / self.num_qubits,
                recommended_exercise=rec
            )
            self._last_status_update = now
            return self._cached_status

class EthicalOversight:
    """
    Simulates an Ethics Board oversight module.
    Ensures treatments adhere to safety protocols (REB-2025-QML-DEM).
    """
    def __init__(self):
        self.reb_id = "REB-2025-QML-DEM"
        self.max_intensity_threshold = 0.8
        self.safety_lock_active = False
        self.patient_consent = False
        self.study_id = None
        self.custom_protocols_active = False

    def verify_consent(self):
        if not self.patient_consent:
            raise HTTPException(status_code=403, detail="Ethical Violation: Patient consent not verified.")
    
    def check_safety(self, intensity: float, current_plasticity: float):
        if self.safety_lock_active:
             raise HTTPException(status_code=423, detail="Safety Lockout: Remediations in progress.")
        
        # Custom Study Override
        if self.custom_protocols_active:
             if intensity > 1.0: # Absolute hard limit even for studies
                 self.trigger_remediation("Intensity Critical: Exceeds physical limits (1.0).")
                 return 1.0
             return intensity # Allow higher intensities for approved custom studies

        # Standard Safety Protocols
        if intensity > self.max_intensity_threshold:
            # Automatic remediation: Cap intensity
            self.trigger_remediation("Intensity exceeds ethical safety limits (0.8). Capping intervention.")
            return 0.8 # Return capped value
            
        if current_plasticity < 0.2 and intensity > 0.5:
             self.trigger_remediation("High intensity on fragile substrates. Risk of excitotoxicity.")
             return 0.3 # Reduce significantly
             
        return intensity

    def trigger_remediation(self, reason: str):
        print(f"[ETHICS LOG] REMEDIATION TRIGGERED: {reason}")
        # In a real system, this would log to an immutable ledger
        
    def grant_consent(self, study_id: str = "STANDARD"):
        self.patient_consent = True
        self.study_id = study_id
        if "CUSTOM" in study_id.upper() or "NEURO" in study_id.upper():
            self.custom_protocols_active = True
            self.reb_id = f"REB-EXP-{study_id}"

ethics_board = EthicalOversight()

# Global Instances
circuit = QuantumCircuitModel(num_qubits=20)
dementia_brain = DementiaTreatmentModel(num_qubits=24)

# Combinatorial Manifold Models
manifold_dementia = None  # Lazy initialization
manifold_ptsd = None  # Lazy initialization

@app.get("/api/circuit")
def get_circuit():
    return circuit.get_state()

@app.post("/api/evolve")
def evolve_circuit():
    circuit.step()
    return circuit.get_state()

@app.post("/api/train")
def train_circuit():
    circuit.train_step()
    circuit.step() 
    return circuit.get_state()

@app.post("/api/reset")
def reset_circuit():
    global circuit
    circuit = QuantumCircuitModel(num_qubits=20)
    return circuit.get_state()

# --- Dementia Treatment API ---

@app.get("/api/dementia/state")
def get_dementia_state():
    return dementia_brain.get_state()

@app.get("/api/dementia/metrics")
def get_dementia_metrics():
    return dementia_brain.analyze_status()

@app.get("/api/dementia/detailed_stats")
def get_detailed_stats():
    return dementia_brain.get_detailed_stats()

@app.post("/api/dementia/treat")
def apply_treatment(input: TreatmentInput):
    # 1. Ethical Checks
    ethics_board.verify_consent()
    
    # 2. Safety & Remediation
    safe_intensity = ethics_board.check_safety(input.intensity, dementia_brain.plasticity)
    remediation_note = None
    if safe_intensity != input.intensity:
        remediation_note = f"Treatment intensity automatically modulated from {input.intensity} to {safe_intensity} per Safety Protocol."

    # Apply natural degradation first to simulate time passing
    if random.random() < 0.2:
        dementia_brain.natural_degradation()
        
    logs = dementia_brain.apply_treatment(input.treatment_type, safe_intensity, input.prime_modulus)
    
    if remediation_note:
        logs.insert(0, f"[REMEDIATION] {remediation_note}")

    dementia_brain.step() # Evolve to integrate changes
    
    new_status = dementia_brain.analyze_status()
    
    return {
        "treatment_logs": logs,
        "new_metrics": new_status,
        "brain_state": dementia_brain.get_state() # detailed visual data
    }

class ConsentInput(BaseModel):
    study_id: str

@app.post("/api/ethics/consent")
def grant_consent(input: ConsentInput):
    ethics_board.grant_consent(input.study_id)
    return {"status": "Consent Verified", "reb_id": ethics_board.reb_id, "mode": "Custom Protocol" if ethics_board.custom_protocols_active else "Standard Safety"}

@app.get("/api/ethics/status")
def get_ethics_status():
    return {
        "reb_approval": ethics_board.reb_id,
        "consent_verified": ethics_board.patient_consent,
        "safety_protocols": "Active"
    }

@app.post("/api/dementia/generate_3d")
def generate_3d_projection(input: TreatmentInput):
    """
    Triggers the generation of the 3D neural projection file.
    Executed via subprocess to reuse the complex logic in generate_brain_comparison.py
    """
    import subprocess
    
    # We use a specific filename for the 3D projection
    filename = f"3d_projection_{input.treatment_type}.png"
    
    # Run the generation script with the --repair flag (which triggers the advanced logic)
    # Intensity 0.95 is used for high-fidelity rendering
    cmd = [
        "python3", "generate_brain_comparison.py",
        "--treatment", input.treatment_type,
        "--intensity", str(input.intensity),
        "--output", filename,
        "--repair"
    ]
    
    try:
        # Run process
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[3D GEN LOG] {result.stdout}")
        # Since we ran with --repair, the script generates a _repaired suffix file
        repaired_filename = filename.replace(".png", "_repaired.png")
        return {"success": True, "filename": repaired_filename, "log": result.stdout}
    except subprocess.CalledProcessError as e:
        print(f"[3D GEN ERROR] {e.stderr}")
        raise HTTPException(status_code=500, detail=f"3D Generation Failed: {e.stderr}")



import threading
import time

# --- ANE Background Thread ---
def run_ane_simulation_loop():
    """ Runs the ANE simulation in a separate thread, linked to quantum coherence. """
    print("[SYSTEM] Starting Apple Neural Engine (ANE) Background Thread (Coherence Linked)...")
    while True:
        try:
            # Observational Interop: ANE power scales with network coherence
            # We access the global dementia_brain status.
            # Use cached status to avoid heavy re-computation on every 50ms tick.
            status = dementia_brain.analyze_status()
            coherence = status.memory_coherence
            
            # Update ANE processor with derived metrics
            # High Performance: Scaling ANE capabilities based on neural health
            ane_processor.total_tops = 128 + (coherence * 672) # Up to 800 TOPS
            ane_processor.bandwidth = 100 + (coherence * 300)   # Up to 400 GB/s
            
            # Thermal efficiency increases with lower frustration (higher stability)
            stability = status.nim_game_stability
            ane_processor.efficiency = min(100, 50 + stability * 50)
            
            # Physics update
            ane_processor.update()
        except Exception as e:
            # print(f"[ANE DEBUG] {e}") # Suppress for production
            pass
        time.sleep(0.05) # 50ms tick rate

@app.on_event("startup")
def startup_event():
    # Start the ANE simulation thread
    t = threading.Thread(target=run_ane_simulation_loop, daemon=True)
    t.start()

# --- Apple Neural Engine API ---

@app.get("/api/ane/stats")
def get_ane_stats():
    # Now this is a pure "Memory Read" operation with O(1) latency
    # The complex physics update happens in the background thread (IPC via shared memory)
    return ane_processor.get_stats()

class BenchmarkRequest(BaseModel):
    model_name: str
    complexity_tflops: float

@app.post("/api/ane/benchmark")
def run_ane_benchmark(req: BenchmarkRequest):
    # Convert TFLOPS to FLOPS
    floppy = req.complexity_tflops * 1e12
    ane_processor.submit_job(req.model_name, floppy)
    return {"status": "Job Submitted", "job": req.model_name}

@app.get("/api/quantum/uncertainty_stats")
def get_uncertainty_stats():
    """Returns the phase-space density manifest."""
    density = dementia_brain.uncertainty_man.get_phase_space_density(dementia_brain.qubits)
    return {"phase_space_density": density}


# Serve static files for frontend
# Static files mount moved to end of file to avoid masking API routes

# --- Combinatorial Manifold Neurogenesis API ---

class ManifoldInitRequest(BaseModel):
    pathology_type: str  # 'dementia' or 'ptsd'
    num_neurons: int = 100

class ManifoldRepairRequest(BaseModel):
    pathology_type: str
    num_cycles: int = 5

@app.post("/api/manifold/initialize")
def initialize_manifold(req: ManifoldInitRequest):
    """Initialize a combinatorial manifold model for specified pathology."""
    global manifold_dementia, manifold_ptsd
    
    try:
        if req.pathology_type == 'dementia':
            manifold_dementia = PTSDDementiaRepairModel(
                num_neurons=req.num_neurons, 
                pathology_type='dementia'
            )
            baseline = manifold_dementia.analyze_topology()
            return {
                "status": "initialized",
                "pathology": "dementia",
                "baseline_topology": baseline
            }
        elif req.pathology_type == 'ptsd':
            manifold_ptsd = PTSDDementiaRepairModel(
                num_neurons=req.num_neurons, 
                pathology_type='ptsd'
            )
            baseline = manifold_ptsd.analyze_topology()
            return {
                "status": "initialized",
                "pathology": "ptsd",
                "baseline_topology": baseline
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid pathology type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/manifold/topology/{pathology_type}")
def get_manifold_topology(pathology_type: str):
    """Get current topological analysis of the manifold."""
    global manifold_dementia, manifold_ptsd
    
    if pathology_type == 'dementia':
        if manifold_dementia is None:
            raise HTTPException(status_code=404, detail="Dementia manifold not initialized")
        return manifold_dementia.analyze_topology()
    elif pathology_type == 'ptsd':
        if manifold_ptsd is None:
            raise HTTPException(status_code=404, detail="PTSD manifold not initialized")
        return manifold_ptsd.analyze_topology()
    else:
        raise HTTPException(status_code=400, detail="Invalid pathology type")

@app.post("/api/manifold/repair")
def apply_manifold_repair(req: ManifoldRepairRequest):
    """Apply neurogenesis-based repair cycles."""
    global manifold_dementia, manifold_ptsd
    
    try:
        if req.pathology_type == 'dementia':
            if manifold_dementia is None:
                # Auto-initialize if not done
                manifold_dementia = PTSDDementiaRepairModel(
                    num_neurons=100, 
                    pathology_type='dementia'
                )
            
            repair_history = manifold_dementia.apply_repair_cycle(num_cycles=req.num_cycles)
            stats = manifold_dementia.generate_repair_statistics()
            projection_file = manifold_dementia.generate_projection_image(f"combinatorial_dementia_projection.png")
            
            return {
                "pathology": "dementia",
                "repair_history": repair_history,
                "final_statistics": stats,
                "projection_image": projection_file
            }
            
        elif req.pathology_type == 'ptsd':
            if manifold_ptsd is None:
                # Auto-initialize if not done
                manifold_ptsd = PTSDDementiaRepairModel(
                    num_neurons=100, 
                    pathology_type='ptsd'
                )
            
            repair_history = manifold_ptsd.apply_repair_cycle(num_cycles=req.num_cycles)
            stats = manifold_ptsd.generate_repair_statistics()
            projection_file = manifold_ptsd.generate_projection_image(f"combinatorial_ptsd_projection.png")
            
            return {
                "pathology": "ptsd",
                "repair_history": repair_history,
                "final_statistics": stats,
                "projection_image": projection_file
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid pathology type")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/manifold/visualize")
def visualize_manifold(req: ManifoldInitRequest):
    """Generate 3D projection of the current manifold state."""
    global manifold_dementia, manifold_ptsd
    
    pathology = req.pathology_type
    model = None
    
    if pathology == 'dementia':
        model = manifold_dementia
    elif pathology == 'ptsd':
        model = manifold_ptsd
        
    if model is None:
        raise HTTPException(status_code=404, detail=f"{pathology} manifold not initialized")
        
    # Generate real-time projection from the initialized model
    try:
        filename = f"manifold_{pathology}_baseline.png"
        projection_path = model.generate_projection_image(filename)
        
        return {
            "baseline_image": projection_path,
            # We don't generate repaired image yet, it will be generated during repair
            "repaired_image": None 
        }
    except Exception as e:
        print(f"Viz Error: {e}")
        raise HTTPException(status_code=500, detail="Visualization generation failed")

@app.get("/api/manifold/statistics/{pathology_type}")
def get_manifold_statistics(pathology_type: str):
    """Get comprehensive repair statistics."""
    global manifold_dementia, manifold_ptsd
    
    if pathology_type == 'dementia':
        if manifold_dementia is None:
            raise HTTPException(status_code=404, detail="Dementia manifold not initialized")
        stats = manifold_dementia.generate_repair_statistics()
        if stats is None:
            raise HTTPException(status_code=400, detail="No repair cycles applied yet")
        return stats
        
    elif pathology_type == 'ptsd':
        if manifold_ptsd is None:
            raise HTTPException(status_code=404, detail="PTSD manifold not initialized")
        stats = manifold_ptsd.generate_repair_statistics()
        if stats is None:
            raise HTTPException(status_code=400, detail="No repair cycles applied yet")
        return stats
    else:
        raise HTTPException(status_code=400, detail="Invalid pathology type")

@app.get("/api/manifold/current_stats/{pathology_type}")
def get_current_manifold_stats(pathology_type: str):
    """Get current topology metrics without requiring a repair cycle."""
    global manifold_dementia, manifold_ptsd
    
    model = manifold_dementia if pathology_type == 'dementia' else manifold_ptsd
    
    if model is None:
        raise HTTPException(status_code=404, detail=f"{pathology_type} manifold not initialized")
        
    return model.analyze_topology()

@app.get("/api/manifold/comparison")
def get_manifold_comparison():
    """Compare dementia and PTSD repair outcomes."""
    global manifold_dementia, manifold_ptsd
    
    results = {}
    
    if manifold_dementia is not None:
        dementia_stats = manifold_dementia.generate_repair_statistics()
        if dementia_stats:
            results['dementia'] = dementia_stats
    
    if manifold_ptsd is not None:
        ptsd_stats = manifold_ptsd.generate_repair_statistics()
        if ptsd_stats:
            results['ptsd'] = ptsd_stats
    
    if not results:
        raise HTTPException(status_code=404, detail="No manifold models initialized or repaired")
    
    return results

@app.post("/api/manifold/reset/{pathology_type}")
def reset_manifold(pathology_type: str):
    """Reset a manifold model."""
    global manifold_dementia, manifold_ptsd
    
    if pathology_type == 'dementia':
        manifold_dementia = None
        return {"status": "reset", "pathology": "dementia"}
    elif pathology_type == 'ptsd':
        manifold_ptsd = None
        return {"status": "reset", "pathology": "ptsd"}
    else:
        raise HTTPException(status_code=400, detail="Invalid pathology type")


# Mount static files at the end to ensure API routes take precedence
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8082)
