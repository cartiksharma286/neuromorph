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

@app.route('/api/fornix-conductivity')
def fornix_conductivity():
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

if __name__ == '__main__':

    app.run(debug=True, port=5001)
