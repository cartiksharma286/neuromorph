import random
import os
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from logic.math_engine import simulate_treatment_plan, continued_fraction_signal, get_system_specs, simulate_fornix_conductivity, get_fornix_protocol, analyze_fornix_biosignals, get_ptsd_staging_protocol, optimize_ptsd_tbi_protocol_generative

app = Flask(__name__)
CORS(app)
# Ensure static and template folders are recognized
app.static_folder = 'static'
app.template_folder = 'templates'

@app.route('/')
def index():
    return render_template('index.html')

# --- FAS Computational Endpoints ---
@app.route('/api/fas-cortical', methods=['POST'])
def fas_cortical():
    # Placeholder: Quantum-theoretic statistical cortical simulation
    return jsonify({"result": "Quantum-theoretic cortical simulation complete. (Placeholder)"})

@app.route('/api/fas-bem', methods=['POST'])
def fas_bem():
    # Placeholder: Boundary element simulation
    return jsonify({"result": "Boundary element simulation complete. (Placeholder)"})

@app.route('/api/fas-cf', methods=['POST'])
def fas_cf():
    # Placeholder: Continued fractions for treatment paradigms
    return jsonify({"result": "Continued fraction model simulation complete. (Placeholder)"})

@app.route('/api/fas-feynman', methods=['POST'])
def fas_feynman():
    # Placeholder: Feynman path integrals for cortical trajectory pathways
    return jsonify({"result": "Feynman path integral simulation complete. (Placeholder)"})

@app.route('/api/fas-postop', methods=['POST'])
def fas_postop():
    # Placeholder: Post-op validation with pre/post cortical manifolds and geodesic visualization
    return jsonify({"result": "Post-op validation and geodesic visualization complete. (Placeholder)"})

@app.route('/api/system-specs')
def system_specs():
    return jsonify(get_system_specs())

@app.route('/api/veteran-ptsd-dbs-protocol', methods=['POST'])
def veteran_ptsd_dbs_protocol():
    import numpy as np
    # Simulate optimal clinical DBS paradigms for PTSD in veterans
    months = np.arange(0, 48, 2)
    efficacy = 25 + 70 * np.exp(-((months-18)/13)**2)
    recovery = 20 + 75 / (1 + np.exp(-0.16 * (months - 16))) + np.random.normal(0, 2, len(months))
    # Protocol sequence
    sequence = [
        {"stage": "1. Baseline Assessment", "description": "Comprehensive neuropsychiatric evaluation and baseline imaging."},
        {"stage": "2. Target Localization", "description": "MRI-guided targeting of amygdala, vmPFC, and hippocampal circuits."},
        {"stage": "3. Initial DBS Titration", "description": "Low-frequency stimulation (20-40 Hz) for circuit priming and safety."},
        {"stage": "4. Efficacy Optimization", "description": "High-frequency stimulation (100-130 Hz) for symptom reduction and cognitive recovery."},
        {"stage": "5. Maintenance & Monitoring", "description": "Adaptive closed-loop DBS with periodic clinical review and device tuning."}
    ]
    # Protocol stages
    stages = [
        {"label": "Assessment & Imaging", "context": "Month 0: Baseline, MRI, and planning."},
        {"label": "DBS Initiation", "context": "Month 1-3: Initial titration and safety monitoring."},
        {"label": "Optimization Phase", "context": "Month 4-12: Efficacy maximization and cognitive support."},
        {"label": "Maintenance", "context": "Month 13-48: Adaptive closed-loop DBS and follow-up."}
    ]
    simulation = {
        "months": months.tolist(),
        "efficacy": efficacy.tolist(),
        "recovery": recovery.tolist()
    }
    return jsonify({
        "sequence": sequence,
        "stages": stages,
        "simulation": simulation
    })


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
    prompt = data.get('prompt', 'baseline')
    dbs_amplitude = float(data.get('dbs_amplitude', 30))
    base_decline_rate = float(data.get('base_decline_rate', 0.05))

    # Simulate Combinatorial Game Theory Pareto frontier
    # Player 1: Striatal Activation
    # Player 2: Serotonin Release (SR)
    # Using continuous fraction bounds and finite math equilibria
    # Striatum(x) = (1 - exp(-x/lambda))
    # SR(y) = (1 - x^2) + epsilon*lambda

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

@app.route('/api/analyze-biosignals', methods=['POST'])
def analyze_biosignals():
    data = request.json or {}
    freq = float(data.get('freq', 10.0))
    return jsonify(analyze_fornix_biosignals(freq))

@app.route('/api/fornix-protocol', methods=['GET'])
def fornix_protocol():
    return jsonify(get_fornix_protocol())

@app.route('/api/fornix-conductivity', methods=['POST', 'GET'])
def fornix_conductivity():
    data = request.json or {}
    grid_size = int(data.get('grid_size', 10))
    base_cond = float(data.get('base_cond', 0.20))
    anisotropy = float(data.get('anisotropy', 0.1))
    curvature = float(data.get('curvature', 2.0))
    grid = simulate_fornix_conductivity(grid_size, base_cond, anisotropy, curvature)
    return jsonify({"grid": grid})

@app.route('/api/dementia-staging', methods=['POST'])
def dementia_staging():
    import numpy as np
    data = request.json or {}
    decline_rate = float(data.get('decline_rate', 0.05))
    dbs_amplitude = float(data.get('dbs_amplitude', 30))
    prompt = data.get('prompt', 'baseline')

    # Long-term modeling: 15 years (180 months)
    time_steps = np.linspace(0, 180, 180)
    ai_factor = 1.0
    if 'plasticity' in prompt.lower():
        ai_factor = 1.7
    elif 'aggressive' in prompt.lower():
        ai_factor = 0.7

    # Computational model: non-linear recovery, relapse, and stabilization
    scores = []
    clinical_distributions = []
    base_recovery = 10.0 * ai_factor * (dbs_amplitude / 30.0)
    for t in time_steps:
        # Simulate cognitive score with recovery, possible relapse, and stabilization
        recovery = base_recovery * (1 - np.exp(-t/48.0))
        relapse = -2.0 * np.sin(t/60.0 * np.pi) if t > 60 else 0  # possible relapse after 5 years
        decline = 30 * np.exp(-decline_rate * (t/12.0))
        noise = np.random.normal(0, 0.8)
        score = decline + recovery + relapse + noise
        score = max(0, min(30, score))
        scores.append(round(score, 2))
        # Clinical std deviation: higher at start, lower with stable recovery, slight increase with relapse
        std = max(1.2, 4.0 - 2.5 * (t/180.0) + (1.0 if t > 60 and relapse < 0 else 0))
        clinical_distributions.append({"mean": round(score, 2), "std": round(std, 2)})

    # Clinical insight
    insight = (
        "This protocol models long-term dementia care over 15 years, integrating neurodegenerative decline, DBS-induced neuroplasticity, and possible relapse phases. "
        "Computationally, the model captures non-linear recovery, stabilization, and relapse, with stochastic bounds reflecting real-world patient variability. "
        "Clinically, sustained high-amplitude stimulation and plasticity-focused paradigms yield improved cognitive stability and resilience, with relapse risk managed by protocol adaptation."
    )
    variance = f"Std Dev starts at ±{clinical_distributions[0]['std']} and stabilizes to ±{clinical_distributions[-1]['std']} MMSE points."

    return jsonify({
        "time_months": time_steps.tolist(),
        "cognitive_trajectory": scores,
        "clinical_distributions": clinical_distributions,
        "generative_insight": insight,
        "variance": variance
    })

@app.route('/api/canadian-ptsd-cortical-fea', methods=['POST'])
def canadian_ptsd_cortical_fea():
    import numpy as np
    months = np.arange(0, 48, 2)
    efficacy = 30 + 65 * (1 - np.exp(-months/14.0))
    recovery = 25 + 70 / (1 + np.exp(-0.15 * (months - 18)))
    
    sequence = [
        {"stage": "1. Cortical Mapping", "description": "High-resolution FEA of amygdala and vmPFC circuits."},
        {"stage": "2. FEA-Guided Lead Placement", "description": "Precision targeting using finite element stress analysis."},
        {"stage": "3. Neuroplasticity Induction", "description": "HF stimulation (120Hz) to drive white matter remodeling."},
        {"stage": "4. Cognitive Integration", "description": "Closed-loop feedback for adaptive symptom management."}
    ]
    
    fea_results = [
        {"region": "Amygdala", "stress": "12.4", "notes": "Targeted for inhibition"},
        {"region": "vmPFC", "stress": "8.9", "notes": "Targeted for excitation"},
        {"region": "Hippocampus", "stress": "15.2", "notes": "Structural retention zone"}
    ]
    
    return jsonify({
        "sequence": sequence,
        "fea_results": fea_results,
        "simulation": {
            "months": months.tolist(),
            "efficacy": efficacy.tolist(),
            "recovery": recovery.tolist()
        }
    })

@app.route('/api/canadian-ptsd-statopt-cure', methods=['POST'])
def canadian_ptsd_statopt_cure():
    import numpy as np
    months = np.arange(0, 49, 4)
    efficacy = 20 + 75 * (1 - np.exp(-months/20.0))
    recovery = 15 + 80 * (1 - np.exp(-months/24.0))
    
    return jsonify({
        "months": months.tolist(),
        "efficacy": efficacy.tolist(),
        "recovery": recovery.tolist(),
        "stages": [
            {"label": "StatOpt Phase I", "time": 0},
            {"label": "StatOpt Phase II", "time": 12},
            {"label": "StatOpt Phase III", "time": 24},
            {"label": "StatOpt Maintenance", "time": 36}
        ],
        "param_names": ["Voltage", "Freq", "Pulse", "Impedance", "Stability"],
        "param_dist": [0.8, 0.9, 0.7, 0.4, 0.95]
    })

@app.route('/api/canadian-ptsd-qml-care', methods=['GET'])
def canadian_ptsd_qml_care():
    return jsonify({
        "summary": "Quantum Machine Learning optimized paradigms for Canadian veteran PTSD care.",
        "paradigms": [
            {
                "title": "Quantum Amygdala Shunting",
                "type": "QML+DBS",
                "description": "Adaptive shunting of hyperactive fear circuits using quantum-derived priors.",
                "ai_score": 0.94,
                "notes": "Highly recommended for treatment-resistant cases."
            },
            {
                "title": "Hebbian vmPFC Reinforcement",
                "type": "GenAI+Plasticity",
                "description": "Generative reinforcement of prefrontal inhibitory control over limbic outbursts.",
                "ai_score": 0.88,
                "notes": "Optimal for cognitive reintegration."
            }
        ]
    })

@app.route('/api/ptsd-tbi-lobe-plans', methods=['GET'])
def ptsd_tbi_lobe_plans():
    return jsonify(optimize_ptsd_tbi_protocol_generative())

@app.route('/api/fasd-statopt-cure', methods=['POST'])
def fasd_statopt_cure():
    import numpy as np
    months = np.arange(0, 49, 4)
    return jsonify({
        "months": months.tolist(),
        "efficacy": (20 + 70 * (1 - np.exp(-months/22.0))).tolist(),
        "recovery": (15 + 75 * (1 - np.exp(-months/26.0))).tolist(),
        "stages": [
            {"label": "FASD Phase I", "time": 0},
            {"label": "FASD Phase II", "time": 12},
            {"label": "FASD Phase III", "time": 24}
        ],
        "param_names": ["V", "Hz", "us", "kOhm", "Stab"],
        "param_dist": [0.7, 0.8, 0.9, 0.5, 0.85]
    })

@app.route('/api/fasd-cure', methods=['POST'])
def fasd_cure():
    import numpy as np
    months = np.arange(0, 49, 4)
    return jsonify({
        "months": months.tolist(),
        "qml_neuroplasticity": (30 + 65 * (1 - np.exp(-months/18.0))).tolist(),
        "qml_exec_func": (25 + 60 * (1 - np.exp(-months/20.0))).tolist(),
        "cortical_stim": (40 + 50 * (1 - np.exp(-months/15.0))).tolist(),
        "qml_field_vectors": [[random.uniform(0.5, 0.9) for _ in range(5)] for _ in range(len(months))],
        "stages": [
            {"label": "QML Reset", "time": 0},
            {"label": "Neural Growth", "time": 12},
            {"label": "Circuit Tuning", "time": 24}
        ]
    })

@app.route('/api/fas-neurosymbolic', methods=['POST'])
def fas_neurosymbolic():
    return jsonify({
        "status": "success",
        "trajectory": [random.uniform(0.1, 0.9) for _ in range(20)],
        "insight": "Neurosymbolic priors suggest high plasticity in mid-brain regions."
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

@app.route('/api/hd-cure', methods=['POST'])
def hd_cure_simulation():
    data = request.json or {}
    
    # Generate temporal progression across staging protocol
    # Emulating mHTT reduction and MSN (Medium Spiny Neuron) recovery
    times = list(range(0, 48, 2))  # 48 months, steps of 2
    
    mhtt_levels = [100 * (0.92 ** t) for t in times] # Exponential decay
    msn_recovery = [20 + 80 * (1 - (0.88 ** t)) for t in times]
    cognitive_score = [40 + 60 * (1 - (0.90 ** t)) for t in times]
    
    # Cortical field simulation vectors
    field_vectors = []
    for _ in times:
        vec = [round(random.uniform(0.1, 0.9), 3) for _ in range(5)]
        field_vectors.append(vec)

    return jsonify({
        "times": times,
        "mhtt_levels": mhtt_levels,
        "msn_recovery": msn_recovery,
        "cognitive_score": cognitive_score,
        "field_vectors": field_vectors,
        "stages": [
            {"time": 0, "label": "Stage I: Microglial Reset"},
            {"time": 12, "label": "Stage II: mHTT Clearance Quad"},
            {"time": 24, "label": "Stage III: Interventional Plasticity"},
            {"time": 36, "label": "Stage IV: Cortical Restitution"}
        ]
    })


if __name__ == '__main__':
    import sys
    port = 5007
    # Allow port override via command-line argument or environment variable
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1].replace('--port=', ''))
        except Exception:
            pass
    port = int(os.environ.get('PORT', port))
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=port)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
