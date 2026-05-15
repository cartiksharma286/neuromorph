import os
import random
import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# --- Simulation Logic ---

def generate_ptsd_paradigm(veteran_profile):
    """
    Generates an optimal DBS treatment paradigm based on veteran characteristics.
    """
    severity = veteran_profile.get('severity', 0.7)
    duration = veteran_profile.get('duration_years', 5)
    
    # Mathematical modeling of recovery curves
    months = np.arange(0, 37, 1)
    # Efficacy follows a Gompertz-like growth function
    efficacy = 100 * (1 - np.exp(-0.15 * months)) * (1 - 0.2 * severity)
    # Trauma index reduction
    trauma_index = 100 * np.exp(-0.08 * months) * (1 + 0.1 * duration)
    
    return {
        "months": months.tolist(),
        "efficacy": efficacy.tolist(),
        "trauma_index": trauma_index.tolist(),
        "recommendations": [
            {"lobe": "vmPFC", "freq": "100-130Hz", "notes": "Target for inhibition of fear response."},
            {"lobe": "Amygdala", "freq": "20-40Hz", "notes": "Target for modulation of emotional arousal."},
            {"lobe": "Hippocampus", "freq": "5-15Hz", "notes": "Target for memory circuit stabilization."}
        ],
        "stages": [
            {"label": "Phase I: Calibration", "time": 0},
            {"label": "Phase II: Entrainment", "time": 12},
            {"label": "Phase III: Stabilization", "time": 24}
        ]
    }

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate-ptsd', methods=['POST'])
def simulate_ptsd():
    data = request.json or {}
    result = generate_ptsd_paradigm(data)
    return jsonify(result)

@app.route('/api/simulate-fea', methods=['POST'])
def simulate_fea():
    """
    Simulates cortical FEA field distributions.
    """
    # Simulate a 5x5 grid of field strengths across different lobes
    lobes = ["Frontal", "Temporal", "Parietal", "Occipital", "Insular"]
    field_data = []
    for lobe in lobes:
        field_data.append({
            "lobe": lobe,
            "stress": round(random.uniform(5, 20), 2),
            "conductivity": round(random.uniform(0.2, 0.4), 3),
            "vector": [round(random.uniform(0.1, 0.9), 2) for _ in range(5)]
        })
    
    return jsonify({
        "lobes": lobes,
        "field_data": field_data,
        "peak_field": round(random.uniform(15, 25), 2),
        "anisotropy_ratio": round(random.uniform(1.1, 1.5), 2)
    })

@app.route('/api/system-status')
def system_status():
    return jsonify({
        "status": "Active",
        "quantum_bridge": "Connected",
        "efficiency": "98.4%",
        "veterans_monitored": random.randint(120, 450)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=True)
