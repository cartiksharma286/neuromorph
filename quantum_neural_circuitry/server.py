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
    def __init__(self, num_qubits=10):
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

# Global Circuit Instance
circuit = QuantumCircuitModel(num_qubits=20)

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
    circuit.step() # Evolve after training
    return circuit.get_state()

@app.post("/api/reset")
def reset_circuit():
    global circuit
    circuit = QuantumCircuitModel(num_qubits=20)
    return circuit.get_state()

# Serve static files for frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
