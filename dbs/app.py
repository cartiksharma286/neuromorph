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

@app.route('/api/pareto_frontier', methods=['POST'])
def pareto_frontier():
    import numpy as np
    data = request.json or {}
    lambda_val = float(data.get('lambda', 0.5))
    
    # Simulate Combinatorial Game Theory Pareto frontier
    # Player 1: Striatal Activation
    # Player 2: Serotonin Release (SR)
    # Using continuous fraction bounds and finite math equilibria
    # Striatum(x) = (1 - exp(-x/lambda))
    # SR(y) = (1 - x^2) + epsilon*lambda
    
    x_vals = np.linspace(0, 1, 100)
    
    striatal_activation = (1.0 - np.exp(-x_vals / lambda_val)) * 100
    serotonin_release = ((1.0 - x_vals**1.5) + (0.1 * lambda_val)) * 100
    
    # Pareto optimal point (Nash intersection representation)
    # where Striatal ~ Serotonin
    diffs = np.abs(striatal_activation - serotonin_release)
    opt_idx = np.argmin(diffs)
    
    opt_striatal = float(striatal_activation[opt_idx])
    opt_serotonin = float(serotonin_release[opt_idx])
    
    return jsonify({
        "striatal": striatal_activation.tolist(),
        "serotonin": serotonin_release.tolist(),
        "optimal_x": float(x_vals[opt_idx]),
        "optimal_striatal": opt_striatal,
        "optimal_serotonin": opt_serotonin
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
                "lobe": "Frontal Lobe (dlPFC)",
                "frequency": "100-130 Hz",
                "pulse_width": "90-120 μs",
                "voltage": "3.0-5.0 V",
                "description": "Targets executive function, working memory, and behavioral regulation. High-frequency stimulation helps stabilize mood and cognitive flexibility in frontotemporal decline."
            },
            {
                "lobe": "Temporal Lobe (Hippocampus/EC)",
                "frequency": "50-130 Hz",
                "pulse_width": "60-90 μs",
                "voltage": "1.5-3.5 V",
                "description": "Directly interfaces with memory encoding structures. Stimulation enhances neurogenesis, synaptic plasticity, and spatial navigation consolidation."
            },
            {
                "lobe": "Parietal Lobe (Precuneus/PCC)",
                "frequency": "20-50 Hz",
                "pulse_width": "60-100 μs",
                "voltage": "2.0-4.0 V",
                "description": "Modulates the Default Mode Network (DMN). Low to mid-frequency stimulation increases local glucose metabolism and reverses hypometabolism seen in early AD."
            },
            {
                "lobe": "Insular Lobe (Anterior Insula)",
                "frequency": "100-130 Hz",
                "pulse_width": "90 μs",
                "voltage": "2.5-4.0 V",
                "description": "Crucial for emotional awareness and salience network regulation. Target for treating severe apathy and loss of empathy in FTD."
            },
            {
                "lobe": "Deep Brain (NBM / Fornix)",
                "frequency": "20-130 Hz",
                "pulse_width": "90-100 μs",
                "voltage": "1.5-4.0 V",
                "description": "Modulates acetylcholine release, cholinergic projections, and the Circuit of Papez, driving hippocampal neurogenesis and preserving cognitive structures."
            },
            {
                "lobe": "Deep Brain (STN / GPi)",
                "frequency": "130-150 Hz",
                "pulse_width": "60-90 μs",
                "voltage": "2.5-3.5 V",
                "description": "Modifies indirect pathways to alleviate motor-cognitive dual burdens. Preserves collateral cognitive function in Lewy Body / Parkinson's Dementia."
            }
        ]
    })

@app.route('/api/stage-gated-protocol')
def stage_gated_protocol():
    import numpy as np
    # M/M/1 queueing model: tau aggregation (lambda) vs glymphatic clearance (mu)
    # rho = lambda/mu (utilization), L_q = rho^2 / (1 - rho) (mean queue length)
    stages = [
        {
            "stage": 1,
            "name": "Stage I: Biomarker Audit",
            "desc": "Baseline theta-rhythm & hippocampal connectivity audit. Establish tau aggregation baseline and glymphatic throughput metrics.",
            "electrical": {
                "voltage_v": 1.0,
                "frequency_hz": "4-7 (Theta Sweep)",
                "pulse_width_us": 90,
                "target": "Fornix (Crucian Arch)"
            },
            "queueing": {
                "lambda_arrival": round(2.1, 2),
                "mu_clearance": round(3.5, 2),
                "rho_utilization": round(2.1 / 3.5, 3),
                "l_q": round((2.1 / 3.5) ** 2 / (1 - 2.1 / 3.5), 3)
            }
        },
        {
            "stage": 2,
            "name": "Stage II: Neural Priming",
            "desc": "Sub-threshold circuit sensitization via fixed-frequency stimulation. Enhance BDNF expression and synaptic receptor density.",
            "electrical": {
                "voltage_v": 2.0,
                "frequency_hz": "5 (Fixed)",
                "pulse_width_us": 90,
                "target": "NBM / Basal Forebrain"
            },
            "queueing": {
                "lambda_arrival": round(1.8, 2),
                "mu_clearance": round(4.2, 2),
                "rho_utilization": round(1.8 / 4.2, 3),
                "l_q": round((1.8 / 4.2) ** 2 / (1 - 1.8 / 4.2), 3)
            }
        },
        {
            "stage": 3,
            "name": "Stage III: Synaptic Entrainment",
            "desc": "Resonant plasticity modulation at quantum-optimal frequency. Drive long-term potentiation (LTP) across hippocampal-entorhinal circuits.",
            "electrical": {
                "voltage_v": 3.5,
                "frequency_hz": "Quantum Optimal (8-14 Hz)",
                "pulse_width_us": 100,
                "target": "Hippocampus / Entorhinal Cortex"
            },
            "queueing": {
                "lambda_arrival": round(1.2, 2),
                "mu_clearance": round(5.8, 2),
                "rho_utilization": round(1.2 / 5.8, 3),
                "l_q": round((1.2 / 5.8) ** 2 / (1 - 1.2 / 5.8), 3)
            }
        },
        {
            "stage": 4,
            "name": "Stage IV: Consolidation & Maintenance",
            "desc": "Long-term structural maintenance with adaptive closed-loop feedback. Minimize tau propagation while sustaining cognitive reserve.",
            "electrical": {
                "voltage_v": 2.5,
                "frequency_hz": "130 (HF Maintenance)",
                "pulse_width_us": 60,
                "target": "STN / GPi (Motor-Cognitive Bridge)"
            },
            "queueing": {
                "lambda_arrival": round(0.8, 2),
                "mu_clearance": round(6.5, 2),
                "rho_utilization": round(0.8 / 6.5, 3),
                "l_q": round((0.8 / 6.5) ** 2 / (1 - 0.8 / 6.5), 3)
            }
        }
    ]
    return jsonify({"protocol": stages})

if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=5007)
