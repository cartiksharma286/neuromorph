from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import uvicorn
from contextlib import asynccontextmanager

from diffusion import PulseDiffusionModel
from simulator import MRISimulator

# Global State
model = PulseDiffusionModel()
simulator = MRISimulator()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    iterations: int = 1

@app.get("/")
def read_root():
    return {"status": "Client-Server MRI GenAI Active"}

@app.get("/generate")
def generate_pulse():
    """
    Run one forward pass of the Generative AI Model to create a pulse.
    Then simulate it to get feedback.
    """
    # 1. Generate Pulse
    pulse_waveform, history = model.sample()
    
    # 2. Simulate
    # Pulse is real-valued in this simple diffusion model, treat as B1_x
    # (Complex pulse generation would require 2 channel diffusion)
    sim_result = simulator.bloch_simulation(pulse_waveform)
    
    return {
        "pulse": pulse_waveform.tolist(),
        "diffusion_steps": history, # List of pulses showing noise -> signal
        "simulation": sim_result,
        "metadata": {
            "target_bw": model.target_bandwidth,
            "target_amp": model.target_amplitude
        }
    }

@app.post("/optimize")
def optimize_model(req: OptimizeRequest):
    """
    Runs an optimization loop:
    1. Generate
    2. Simulate to measure SNR
    3. Update Model Parameters (Adaptive Learning)
    """
    logs = []
    for _ in range(req.iterations):
        # Generate
        pulse, _ = model.sample()
        # Evaluate
        sim_result = simulator.bloch_simulation(pulse)
        current_snr = sim_result["snr"]
        current_angle = sim_result["flip_angle"]
        
        # Learn
        updates = model.optimize_step(current_snr, current_angle)
        logs.append({
            "snr": current_snr,
            "angle": current_angle,
            "updates": updates
        })
        
    return {"status": "optimized", "logs": logs}

@app.get("/export_seq")
def export_seq():
    """
    Generates and returns the latest pulse as a mock Pulseq file.
    """
    # Just generate a new sample for export or cache the last one?
    # For simplicity, we generate one. In a real app we'd use state.
    pulse, _ = model.sample()
    seq_data = simulator.export_pulseq(pulse, model.target_bandwidth, model.target_amplitude)
    return {"filename": "genai_pulse.seq", "content": seq_data}

@app.get("/reconstruct")
def reconstruct_image():
    """
    Returns a reconstruction image (base64) of the latest pulse's slice profile.
    """
    pulse, _ = model.sample()
    img_b64 = simulator.generate_reconstruction(pulse)
    return {"image": img_b64}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
