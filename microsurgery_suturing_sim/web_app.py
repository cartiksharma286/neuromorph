from flask import Flask, jsonify, send_file
from qml_suturing_sim_app import TissueForceDynamics, GenerativeScalpelAI, QuantumMLTissueSimulator, FormalVerifier
import numpy as np

app = Flask(__name__)

# Initialize single global simulation state
fd = TissueForceDynamics(width=20, height=20)
qml = QuantumMLTissueSimulator()
gen_ai = GenerativeScalpelAI()

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/step')
def step():
    # Generative AI proposes next scalpel position
    scalpel_pos = gen_ai.generate_next_step(fd.stress_field)
    
    # Calculate dynamic forces applied
    applied_force = np.random.uniform(5.0, 15.0)
    fd.apply_scalpel_force(scalpel_pos[0], scalpel_pos[1], applied_force)
    
    max_stress = float(np.max(fd.stress_field))
    
    # QML Prediction
    qml.apply_suture_gate(max_stress)
    integrity = float(qml.measure_tissue_integrity())
    
    try:
        is_safe = FormalVerifier.verify_tissue_safety(max_stress, integrity)
    except AssertionError:
        is_safe = False
        
    # Suture/healing rule: apply a counter force (negative stress) occasionally to simulate real-time suturing
    if max_stress > 15.0:
        fd.stress_field *= 0.5 # Tissue relaxes after suture

    return jsonify({
        'scalpel_x': float(scalpel_pos[0]),
        'scalpel_y': float(scalpel_pos[1]),
        'max_stress': max_stress,
        'integrity': integrity,
        'is_safe': bool(is_safe),
        'stress_field': fd.stress_field.tolist()
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5051)
