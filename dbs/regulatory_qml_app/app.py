import os
import random
import numpy as np
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# --- Mock Data Generation ---

def generate_qml_data():
    """Generates mock Quantum Machine Learning anomaly detection data."""
    labels = [f"TxBlock-{i}" for i in range(1, 11)]
    # Quantum superposition states collapsed to probabilities
    probabilities = [round(random.uniform(0.1, 0.95), 2) for _ in range(10)]
    entanglement = [round(random.uniform(1.2, 3.5), 2) for _ in range(10)]
    
    return {
        "labels": labels,
        "probabilities": probabilities,
        "entanglement": entanglement,
        "threat_level": "HIGH" if max(probabilities) > 0.85 else "LOW"
    }

def generate_optimal_control_data():
    """Generates mock Optimal Control trajectories for anti-fraud intervention."""
    time_steps = np.arange(0, 100, 1)
    # Hamiltonian dynamics: State (Fraud volume) and Costate (Intervention friction)
    fraud_volume = 100 * np.exp(-0.05 * time_steps) + np.random.normal(0, 2, len(time_steps))
    intervention_cost = 50 * (1 - np.exp(-0.02 * time_steps)) + np.random.normal(0, 1, len(time_steps))
    
    optimal_intervention_point = 45 # Time step where cost = benefit
    
    return {
        "time": time_steps.tolist(),
        "fraud_volume": np.maximum(0, fraud_volume).tolist(),
        "intervention_cost": np.maximum(0, intervention_cost).tolist(),
        "optimal_point": optimal_intervention_point
    }

def generate_monetization_data():
    """Generates cost monetization and ROI metrics."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    funds_recovered = [random.randint(100000, 500000) for _ in range(6)]
    compliance_cost = [random.randint(50000, 150000) for _ in range(6)]
    roi = [(f - c) / c * 100 for f, c in zip(funds_recovered, compliance_cost)]
    
    return {
        "months": months,
        "funds_recovered": funds_recovered,
        "compliance_cost": compliance_cost,
        "roi": [round(r, 1) for r in roi],
        "total_recovered": sum(funds_recovered)
    }

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/qml-fraud-detection')
def qml_fraud_detection():
    return jsonify(generate_qml_data())

@app.route('/api/optimal-control')
def optimal_control():
    return jsonify(generate_optimal_control_data())

@app.route('/api/cost-monetization')
def cost_monetization():
    return jsonify(generate_monetization_data())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5015, debug=True)
