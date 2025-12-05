
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import asyncio
from typing import List, Dict

from .qml_optimizer import QMLOptimizer
from .bbb_sim import BBBSimulation
from .statistical_measures import calculate_var, calculate_cvar
from .geodesics import compute_geodesic, variational_measure_weight
from .generative_field import GenerativeAcousticField
from .knot_theory import detect_vortex_topology

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
simulation = BBBSimulation()
optimizer = QMLOptimizer()
gen_ai = GenerativeAcousticField()
is_running = False

class TreatmentParams(BaseModel):
    frequency: float
    intensity: float
    duration: float

class StatusResponse(BaseModel):
    permeability: float
    temperature: float
    pressure: float
    cavitation: float
    risk_var: float
    optimization_cost: float

@app.post("/generate_field_pattern")
async def generate_field_pattern():
    """
    Uses Generative AI (Simulated GAN) + CUDA to create a safe acoustic pattern.
    """
    raw, effective = gen_ai.synthesize_optimal_field()
    
    # Analyze topology of the generated field
    # Normalize effective field to -pi -> pi range for phase analysis mock
    phase_map = (effective - np.mean(effective)) * np.pi 
    topology = detect_vortex_topology(phase_map)
    
    return {
        "status": "generated",
        "raw_field": raw.tolist(),
        "effective_field": effective.tolist(),
        "topology": topology
    }

@app.post("/start_treatment")
async def start_treatment():
    global is_running
    is_running = True
    return {"status": "started"}

@app.post("/stop_treatment")
async def stop_treatment():
    global is_running
    is_running = False
    return {"status": "stopped"}

@app.post("/update_params")
async def update_params(params: TreatmentParams):
    # In a real loop, these would be auto-optimized, 
    # but we allow manual override or initial seed here.
    return {"status": "updated", "params": params}

@app.get("/get_status")
async def get_status():
    global is_running, simulation, optimizer
    
    if not is_running:
        return {"status": "idle"}
    
    # 1. Run Optimization Step
    # Current simplistic implementation: treat current params as state to optimize
    # Norm params to 0-1 for ansatz
    current_norm = np.random.rand(3) # Mock current state reading
    best_params, cost = optimizer.optimize_parameters(current_norm)
    
    # 2. Apply Physics Step
    # Decode params: Freq (0.2-1.5 MHz), Int (0-5 W/cm2), Dur (0-100 ms)
    freq = best_params[0] * 1.3 + 0.2
    intensity = best_params[1] * 5.0
    duration = best_params[2] * 100.0
    
    phys_state = simulation.step(freq, intensity, duration)
    
    # 3. Calculate Risk Stats
    # Mock history of temperature for statistical analysis
    temp_window = np.random.normal(phys_state["temperature"], 0.5, 100)
    var_95 = calculate_var(temp_window, 0.95)
    
    return {
        "status": "active",
        "permeability": phys_state["permeability"],
        "temperature": phys_state["temperature"],
        "pressure": phys_state["pressure"],
        "cavitation": phys_state["cavitation_stable"],
        "risk_var_95": var_95,
        "optimization_cost": cost,
        "current_optimal_params": {
            "frequency": freq,
            "intensity": intensity,
            "duration": duration
        }
    }

@app.get("/get_geodesic_path")
async def get_geodesic_path(start_u: float, start_v: float, end_u: float, end_v: float):
    """
    Calculate optimal path on cortical surface.
    """
    start = np.array([start_u, start_v])
    end = np.array([end_u, end_v])
    
    path = compute_geodesic(start, end)
    
    # Evaluate measure weight
    # Example: how much "treatment value" does this path cover?
    weight = variational_measure_weight(path, None)
    
    return {
        "path": path.tolist(),
        "measure_weight": weight
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
