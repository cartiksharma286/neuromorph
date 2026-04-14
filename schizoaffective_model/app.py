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
    
    # Base variances for neurotransmitters
    base_dopamine_var = 1.0
    
    # Mathematical Modeling:
    # Beer (<4.5% ABV) has a much milder impact on dopamine variance and prefrontal inhibition 
    # compared to hard spirits. 
    beer_impact = beer_units * 0.35 
    
    dopamine_var = base_dopamine_var + (beer_impact * 1.5) - (medication * 1.2)
    dopamine_var = max(0.1, dopamine_var)
    
    # QML Interventional Protocol Pipeline Simulation
    optimal_tms_freq = 10.0 + (dopamine_var * 8.5)
    treatment_yield = max(0, min(99.9, 65.0 - (beer_impact * 30.0) + (medication * 45.0)))
    
    # Corrective Cognitive Behavioral Traits & Feedback Correlates
    cbt_optimization_score = min(100.0, max(0.0, 50.0 + (medication * 35.0) - (beer_impact * 45.0)))
    
    cbt_traits = []
    feedback_correlate = ""
    
    if cbt_optimization_score > 80:
        cbt_traits = ["Enhanced Emotional Regulation", "High Cognitive Flexibility", "Stable Reality Testing"]
        feedback_correlate = "Optimal behavior state. Mild beer intake (<4.5%) permits social reward networks without destabilizing prefrontal gating. Prime window for cognitive reframing protocols."
    elif cbt_optimization_score > 50:
        cbt_traits = ["Moderate Frustration Tolerance", "Adequate Threat Assessment", "Variable Social Processing"]
        feedback_correlate = "Therapeutic window maintained. Pharmacological stability counteracts mild alcoholic inhibition. Implement standard behavioral reinforcement."
    else:
        cbt_traits = ["Impaired Executive Function", "Heightened Threat Perception", "Reduced Cortical Control"]
        feedback_correlate = "Sub-optimal for CBT. Beer intake is overwhelming pharmacological safeguards, increasing mesolimbic dopamine volatility. Recommend pausing interventions."

    # Statistical Distribution
    mean_symptom = (beer_impact * 2.5) - (medication * 2.0)
    x = np.linspace(-10, 10, 200)
    y = (1 / (dopamine_var * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_symptom) / dopamine_var) ** 2)
    
    # Brain region simulated activity
    pfc_val = max(0.1, 1.0 - (beer_impact*0.8) + (medication*0.5))
    neural_nodes = [
        {'id': 'Prefrontal Cortex (PFC)', 'val': pfc_val, 'x': 0, 'y': 2.5, 'z': 2},
        {'id': 'Amygdala', 'val': min(2.5, 0.8 + (beer_impact*1.2) - (medication*0.9)), 'x': -1, 'y': 1.0, 'z': -1},
        {'id': 'Hippocampus', 'val': min(2.5, 0.9 + (beer_impact*1.0) - (medication*0.7)), 'x': 1, 'y': 1.0, 'z': -1},
        {'id': 'Striatum', 'val': base_dopamine_var * (1.0 + beer_impact - medication*0.5), 'x': 0, 'y': 1.5, 'z': 0}
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
    app.run(debug=True, port=5002)
