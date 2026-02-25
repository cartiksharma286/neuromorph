from diffusion import PulseDiffusionModel
from simulator import MRISimulator
import os
import sys

def save_sequence():
    print("Initializing Model and Simulator...", flush=True)
    try:
        model = PulseDiffusionModel()
        simulator = MRISimulator()
        
        # Optional: Run a few optimization steps to ensure it's a "good" pulse
        print("Optimizing pulse sequence...", flush=True)
        for i in range(5):
            pulse, _ = model.sample()
            sim = simulator.bloch_simulation(pulse)
            model.optimize_step(sim["snr"], sim["flip_angle"])
            
        print("Generating final pulse...", flush=True)
        pulse, _ = model.sample()
        
        print("Exporting to .seq format...", flush=True)
        seq_content = simulator.export_pulseq(pulse, model.target_bandwidth, model.target_amplitude)
        
        filename = "final_output.seq"
        with open(filename, "w") as f:
            f.write(seq_content)
            
        print(f"Successfully wrote pulse sequence to {os.path.abspath(filename)}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

if __name__ == "__main__":
    save_sequence()
