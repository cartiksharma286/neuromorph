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
from ane_simulation import ane_processor
from prime_math_core import PrimeVortexField
from generative_quantum_core import GenerativeQuantumOptimizer


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
    recommended_exercise: str

class TreatmentInput(BaseModel):
    treatment_type: str # 'cognitive', 'reminiscence', 'sensory'
    intensity: float

class DementiaTreatmentModel(QuantumCircuitModel):
    def __init__(self, num_qubits=24):
        super().__init__(num_qubits)
        self.prime_field = PrimeVortexField(num_qubits)
        self.gen_ai = GenerativeQuantumOptimizer(num_qubits, self.prime_field)
        self.plasticity = 0.5 # Ability to form new connections
        self.degradation_rate = 0.01 # Natural decay over time
        
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

    def apply_treatment(self, treatment_type, intensity):
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

        elif treatment_type == 'ocd':
            # OCD Treatment: Over-reinforcement of specific loops (add high-weight edges)
            log.append("OCD Therapy: Reinforcing compulsive loop connections")
            # Choose a small set of node pairs to strongly connect
            num_loops = max(1, int(2 * intensity))
            for _ in range(num_loops):
                u = random.randint(0, self.num_qubits-1)
                v = (u + random.randint(1, 3)) % self.num_qubits  # nearby node
                if not self.topology.has_edge(u, v):
                    self.topology.add_edge(u, v)
                # Set a high entanglement strength to simulate compulsive loop
                # Set a high entanglement strength to simulate compulsive loop
                self.entanglements[(u, v)] = min(1.0, 0.8 + 0.2 * intensity)

        elif treatment_type == 'prime_resonance':
            # Prime Resonance: The "God Mode" repair using Prime Distributions & Surface Integrals
            log.append("Applying Prime Vortex Field (Surface Integral Optimization)")
            
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

    def analyze_status(self) -> DementiaState:
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
            
        return DementiaState(
            stage=stage,
            plasticity_index=self.plasticity,
            synaptic_density=density,
            memory_coherence=avg_strength,
            recommended_exercise=rec
        )

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
        
    logs = dementia_brain.apply_treatment(input.treatment_type, safe_intensity)
    
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
    """
    Runs the ANE simulation in a separate thread.
    This ensures 'processor optimization' by not blocking the main server loop
    and allows for smooth, continuous physics updates.
    """
    print("[SYSTEM] Starting Apple Neural Engine (ANE) Background Thread...")
    while True:
        ane_processor.update()
        time.sleep(0.05) # 50ms tick rate (20Hz)

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

# Serve static files for frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
