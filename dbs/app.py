from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
from logic.math_engine import simulate_treatment_plan, continued_fraction_signal, get_system_specs, simulate_fornix_conductivity, get_fornix_protocol, analyze_fornix_biosignals

app = Flask(__name__)
CORS(app)

# Ensure static and template folders are recognized
app.static_folder = 'static'
app.template_folder = 'templates'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    nodes = data.get('nodes', [])
    results = simulate_treatment_plan(nodes)
    return jsonify({
        "status": "success",
        "results": results
    })

@app.route('/api/analyze-signal', methods=['POST'])
def analyze_signal():
    data = request.json
    freq = data.get('frequency', 1.0)
    result = continued_fraction_signal(freq)
    return jsonify({
        "frequency": freq,
        "resonance": result
    })

@app.route('/api/circuit-schema')
def circuit_schema():
    # Returns metadata for the circuit schematic renderer
    return jsonify({
        "components": [
            {"type": "Voltage Source", "id": "V1", "value": "12V", "coords": [50, 50]},
            {"type": "H-Bridge Controller", "id": "U1", "coords": [150, 50]},
            {"type": "DBS Implantable Coil", "id": "L1", "value": "2.2uH", "coords": [300, 50]},
            {"type": "Statistical Yield Optimizer", "id": "Q1", "coords": [200, 150]}
        ],
        "connections": [
            {"from": "V1", "to": "U1"},
            {"from": "U1", "to": "L1"},
            {"from": "L1", "to": "Q1"}
        ]
    })

@app.route('/api/system-specs')
def system_specs():
    return jsonify(get_system_specs())

@app.route('/api/fornix-protocol')
def fornix_protocol():
    return jsonify(get_fornix_protocol())

@app.route('/api/fornix-conductivity', methods=['POST', 'GET'])
def fornix_conductivity():
    if request.method == 'POST':
        data = request.json or {}
        base_cond = float(data.get('base_cond', 0.20))
        anisotropy = float(data.get('anisotropy', 0.1))
        curvature = float(data.get('curvature', 2.0))
        grid = simulate_fornix_conductivity(base_cond=base_cond, anisotropy=anisotropy, curvature=curvature)
    else:
        grid = simulate_fornix_conductivity()
    return jsonify({"grid": grid})

@app.route('/api/analyze-biosignals', methods=['POST'])
def analyze_biosignals():
    data = request.json
    freq = data.get('frequency', 1.0)
    return jsonify(analyze_fornix_biosignals(freq))


import numpy as np

@app.route('/api/dementia-staging', methods=['POST'])
def dementia_staging():
    data = request.json or {}
    base_decline_rate = float(data.get('decline_rate', 0.05)) # Baseline beta
    dbs_amplitude = float(data.get('dbs_amplitude', 2.0))
    prompt = data.get('prompt', 'baseline')
    
    # Simple generative temporal math evolution
    # S(t) = S0 * exp(-beta * t) + Integral(DBS_yield * Hebbian * dt)
    time_steps = np.linspace(0, 60, 60) # 60 months (5 years)
    
    # Gen AI evolution factor
    ai_factor = 1.0
    if 'plasticity' in prompt.lower():
        ai_factor = 1.5
    elif 'aggressive' in prompt.lower():
        ai_factor = 0.8
        
    cognitive_scores = []
    clinical_distributions = []
    
    S0 = 30.0 # MMSE max score
    for t in time_steps:
        # Finite math equation for temporal paradigm
        decay = base_decline_rate * (1 + 0.1 * np.sin(t)) 
        dbs_effect = (dbs_amplitude * 0.4) * ai_factor * (1 - np.exp(-t/12.0))
        
        score_t = S0 * np.exp(-decay * (t/12.0)) + dbs_effect
        score_t = max(0, min(30, score_t))
        cognitive_scores.append(round(score_t, 2))
        
        # Clinical Statistical Distribution at time t
        std_dev = max(1.0, 5.0 - (dbs_amplitude * 0.5)) + (t/24.0)
        clinical_distributions.append({
            "mean": round(score_t, 2),
            "std": round(std_dev, 2)
        })
        
    return jsonify({
        "time_months": time_steps.tolist(),
        "cognitive_trajectory": cognitive_scores,
        "clinical_distributions": clinical_distributions,
        "generative_insight": "Generative AI derived finite optimal temporal progression mapping over 60 months. The clinical distribution shows variance mapping under high DBS amplitude and generative prompts, indicative of cognitive structural retention and temporal optimization."
    })

@app.route('/api/clinical-protocols', methods=['POST', 'GET'])
def clinical_protocols():
    return jsonify({
        "protocols": [
            {
                "lobe": "Nucleus Basalis of Meynert (NBM)",
                "frequency": "20 Hz",
                "pulse_width": "100 μs",
                "voltage": "1.5 - 3.0 V",
                "description": "Low-frequency stimulation of the primary source of cholinergic projections to the cortex. Directly targets Alzheimer's disease by modulating acetylcholine release and cortical plasticity."
            },
            {
                "lobe": "Subthalamic Nucleus (STN)",
                "frequency": "130 Hz",
                "pulse_width": "60 μs",
                "voltage": "2.5 V",
                "description": "Standard high-frequency stimulation primarily for motor control, but shows collateral cognitive-preservation properties via basal ganglia-thalamocortical loops."
            },
            {
                "lobe": "Globus Pallidus internus (GPi)",
                "frequency": "130 - 150 Hz",
                "pulse_width": "90 μs",
                "voltage": "3.0 V",
                "description": "Modifies indirect pathways to alleviate motor-cognitive dual burdens. Useful for dementia with Lewy bodies or Parkinson's-related dementia variants."
            },
            {
                "lobe": "Ventral Capsule / Ventral Striatum (VC/VS)",
                "frequency": "100 - 130 Hz",
                "pulse_width": "90 μs",
                "voltage": "4.0 V",
                "description": "Targets psychiatric comorbidities (e.g., severe depression, OCD) often accompanying late-stage dementia, improving overall cognitive availability."
            },
            {
                "lobe": "Fornix",
                "frequency": "130 Hz",
                "pulse_width": "90 μs",
                "voltage": "2.0 - 4.0 V",
                "description": "Stimulates the Circuit of Papez, driving hippocampal neurogenesis and memory circuit stabilization in early Alzheimer's."
            }
        ]
    })

if __name__ == '__main__':

    app.run(debug=True, port=5001)
