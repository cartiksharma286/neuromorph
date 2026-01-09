from flask import Flask, render_template, jsonify, request
import os
from crispr_simulation import Crispr3System, QMLProteinFolder

app = Flask(__name__)

# Initialize Systems
autoimmune_sys = Crispr3System(target_type='autoimmune')
hiv_sys = Crispr3System(target_type='hiv')
qml_folder = QMLProteinFolder()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/profiles/<target_type>')
def get_profiles(target_type):
    if target_type == 'autoimmune':
        return jsonify({"profiles": autoimmune_sys.get_profiles(), "conjugates": autoimmune_sys.get_conjugates()})
    elif target_type == 'hiv':
        return jsonify({"profiles": hiv_sys.get_profiles(), "conjugates": hiv_sys.get_conjugates()})
    else:
        return jsonify({"error": "Unknown target type"}), 400

@app.route('/api/simulate-therapy', methods=['POST'])
def simulate_therapy():
    data = request.json
    target_type = data.get('target_type')
    profile_id = data.get('profile_id')
    conjugate_id = data.get('conjugate_id')
    
    if target_type == 'autoimmune':
        result = autoimmune_sys.run_therapy_simulation(profile_id, conjugate_id)
    elif target_type == 'hiv':
        result = hiv_sys.run_therapy_simulation(profile_id, conjugate_id)
    else:
        return jsonify({"error": "Invalid target type"}), 400
        
    return jsonify(result)

@app.route('/api/qml-fold', methods=['POST'])
def qml_fold():
    data = request.json
    protein = data.get('protein', 'gp120')
    result = qml_folder.simulate_folding(protein)
    return jsonify(result)

if __name__ == '__main__':
    port = 5006
    print(f"Starting HIV/Autoimmune Simulator on http://127.0.0.1:{port}")
    app.run(port=port, debug=True)
