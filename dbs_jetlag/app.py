
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
from logic.simulator import JetLagSimulator

app = Flask(__name__)
CORS(app)

# Global simulator instance
sim = JetLagSimulator(num_neurons=150)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json or {}
    dbs_intensity = float(data.get('dbs_intensity', 0.5))
    hebbian_rate = float(data.get('hebbian_rate', 0.01))
    prime_mod = int(data.get('prime_mod', 17))
    target_phase = float(data.get('target_phase', 0.0))
    
    result = sim.step(
        dbs_intensity=dbs_intensity,
        target_phase=target_phase,
        hebbian_rate=hebbian_rate,
        prime_mod=prime_mod
    )
    return jsonify(result)

@app.route('/api/reset', methods=['POST'])
def reset():
    global sim
    data = request.json or {}
    num_neurons = int(data.get('num_neurons', 150))
    sim = JetLagSimulator(num_neurons=num_neurons)
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(debug=True, port=5005)
