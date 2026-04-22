from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    beer_units = float(data.get('beer', 0.0))  # Pints or units of <4.5% beer
    medication = float(data.get('medication', 0.0))
    nicotine = float(data.get('nicotine', 0.0))
    polymath_load = float(data.get('polymath_load', 0.0))  # Physics/Polymath structured cognition
    legal_trouble = float(data.get('legal_trouble', 0.0))  # Legal / Judicial Stressor
    
    # Base variances for neurotransmitters
    base_dopamine_var = 1.0
    
    # Mathematical Modeling:
    # Beer (<4.5% ABV) has a much milder impact on dopamine variance 
    beer_impact = beer_units * 0.35 
    nicotine_impact = nicotine * 0.4
    legal_impact = legal_trouble * 2.0  # High stress multiplier
    
    # Polymath physics-based cognitive load acts as a dopamine "sink",
    # channeling striatal activity into structured frontostriatal tracts
    # thereby actively reducing volatile pathologial variance while raising organized node activity.
    polymath_buffer = polymath_load * 0.6
    
    dopamine_var = base_dopamine_var + (beer_impact * 1.5) + (nicotine_impact * 1.5) + (legal_impact * 2.5) - (medication * 1.2) - polymath_buffer
    dopamine_var = max(0.1, dopamine_var)
    
    # QML Interventional Protocol Pipeline Simulation
    optimal_tms_freq = 10.0 + (dopamine_var * 8.5)
    treatment_yield = max(0, min(99.9, 65.0 - (beer_impact * 30.0) + (nicotine_impact * 15.0) - (legal_impact * 40.0) + (medication * 45.0) + (polymath_load * 20.0)))
    
    # Corrective Cognitive Behavioral Traits & Feedback Correlates
    cbt_optimization_score = min(100.0, max(0.0, 50.0 + (medication * 35.0) + (nicotine_impact * 25.0) + (polymath_load * 30.0) - (legal_impact * 50.0) - (beer_impact * 45.0) - (nicotine_impact * beer_impact * 40.0)))
    
    cbt_traits = []
    feedback_correlate = ""
    
    if cbt_optimization_score > 80:
        cbt_traits = ["Enhanced Emotional Regulation", "Optimized Sensory Gating", "High Cognitive Flexibility", "Hyper-Structured Reality Testing", "Striatal Pattern Anchoring"]
        feedback_correlate = "Optimal behavior state. Physics-based polymath processing actively grounds striatal dopamine into deterministic pathways, providing a profound cognitive buffer against volatility. Highly primed for advanced cognitive reframing."
    elif cbt_optimization_score > 50:
        cbt_traits = ["Moderate Frustration Tolerance", "Adequate Threat Assessment", "Transient Attentional Focus", "Logical Anchoring"]
        if legal_impact > 1.0:
            feedback_correlate = "Therapeutic window maintained, but under immense pressure from legal stressors. Polymath cognitive load is struggling to engage frontostriatal networks to offset this psychosocial volatility."
        else:
            feedback_correlate = "Therapeutic window maintained. Polymath cognitive load is successfully engaging frontostriatal networks to offset chemical/pathological volatility."
    else:
        cbt_traits = ["Impaired Executive Function", "Heightened Threat Perception", "Sensory Overload / Reduced Cortical Control"]
        if legal_impact > 1.0:
            cbt_traits.append("Paranoid Ideation / Institutional Threat Response")
            feedback_correlate = "Severe decompensation. Extreme legal and judicial stressors have completely overwhelmed the prefrontal cortex, triggering amygdalar hyperactivation and destroying cognitive buffering."
        else:
            feedback_correlate = "Sub-optimal. Volatile chemical administration is overwhelming the prefrontal cortex's ability to maintain structured polymathic/physics computations. Cognitive buffering collapses."

    # Statistical Distribution
    mean_symptom = (beer_impact * 2.5) + (nicotine_impact * 1.0) + (legal_impact * 4.0) - (medication * 2.0) - (polymath_buffer * 1.5)
    x = np.linspace(-10, 10, 200)
    y = (1 / (dopamine_var * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_symptom) / dopamine_var) ** 2)
    
    # Brain region simulated activity
    # Physics computation heavily drives up PFC and Striatum structurally.
    # Legal trouble heavily suppresses PFC and hyperactivates Amygdala.
    pfc_val = max(0.1, 1.0 - (beer_impact*0.8) + (nicotine_impact*1.2) - (legal_impact*1.5) + (medication*0.5) + (polymath_load * 1.2))
    neural_nodes = [
        {'id': 'Prefrontal Cortex (PFC)', 'val': pfc_val, 'x': 0, 'y': 2.5, 'z': 2},
        {'id': 'Amygdala', 'val': min(2.5, 0.8 + (beer_impact*1.2) + (nicotine_impact*0.5) + (legal_impact*2.0) - (medication*0.9) - (polymath_load*0.4)), 'x': -1, 'y': 1.0, 'z': -1},
        {'id': 'Hippocampus', 'val': min(2.5, 0.9 + (beer_impact*1.0) + (nicotine_impact*0.3) + (legal_impact*1.0) - (medication*0.7) + (polymath_load*0.5)), 'x': 1, 'y': 1.0, 'z': -1},
        {'id': 'Striatum', 'val': base_dopamine_var * (1.0 + beer_impact + nicotine_impact + legal_impact - medication*0.5) + (polymath_load * 2.0), 'x': 0, 'y': 1.5, 'z': 0}
    ]
    
    # Hebbian Amplification & Cortical Monitoring for Optimal Control
    # Hebbian learning correlates (fire together, wire together) - optimized when PFC is active and dopamine is stable
    hebbian_amplification = min(100.0, max(0.0, (pfc_val / dopamine_var) * 45.0))
    cortical_control_index = min(1.0, pfc_val / max(0.1, (neural_nodes[1]['val'] + neural_nodes[3]['val']) / 2.0))
    
    # Relapse Probability Analysis
    # Low cortical control and high variance lead to higher relapse probability
    relapse_prob = max(1.0, min(99.0, 100.0 - (cortical_control_index * 80.0) + (dopamine_var * 15.0) - (medication * 10.0)))
    if cbt_optimization_score > 70:
        relapse_prob -= 15.0 # Positive CBT traits buffer against relapse
    relapse_prob = max(1.0, relapse_prob)
    
    return jsonify({
        'dopamine_variance': float(dopamine_var),
        'opt_tms_freq': float(optimal_tms_freq),
        'treatment_yield': float(treatment_yield),
        'cbt_score': float(cbt_optimization_score),
        'cbt_traits': cbt_traits,
        'feedback_correlate': feedback_correlate,
        'hebbian_amplification': float(hebbian_amplification),
        'cortical_control_index': float(cortical_control_index),
        'relapse_probability': float(relapse_prob),
        'stats': {
            'x': x.tolist(),
            'y': y.tolist()
        },
        'neural_nodes': neural_nodes,
        'mean_symptom_score': float(mean_symptom)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5005)
