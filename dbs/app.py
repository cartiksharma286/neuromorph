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

if __name__ == '__main__':
    app.run(debug=True, port=5001)
