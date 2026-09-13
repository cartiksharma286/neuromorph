"""
Sunnybrook Neurology: Stroke Gait Kinematics & Neuronal Rehabilitation Web Application
====================================================================================
Point-of-Care Interventional Neuro-Kinematic Restoration Platform
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from stroke_gait_engine import stroke_gait_engine

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return render_template("stroke_gait_dashboard.html")

# 1. Stroke Gait Kinematics Endpoint
@app.route("/api/gait/kinematics", methods=["POST"])
def api_gait_kinematics():
    try:
        data = request.json or {}
        lesion_key = data.get("lesion_type", "mca_cortical")
        intervention = float(data.get("intervention_level", 0.35))
        cadence = float(data.get("cadence", 85.0))
        result = stroke_gait_engine.compute_gait_kinematics(
            stroke_lesion_key=lesion_key,
            intervention_level=intervention,
            cadence_steps_min=cadence
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# 2. Quantum Machine Learning (QML) Interventional Neuromodulation
@app.route("/api/qml/ansatz", methods=["POST"])
def api_qml_ansatz():
    try:
        data = request.json or {}
        n_qubits = int(data.get("n_qubits", 8))
        n_layers = int(data.get("n_layers", 4))
        entanglement = data.get("entanglement", "full_mesh_cnot")
        lr = float(data.get("learning_rate", 0.045))
        
        # Synthesize 8-dimensional kinematic state vector from request
        state_vec = np.array(data.get("state_vector", [0.42, -14.0, 32.0, 6.8, 18.0, 72.0, 0.48, 1.15]))
        result = stroke_gait_engine.execute_qml_interventional_repair(
            kinematic_state_vector=state_vec,
            n_qubits=n_qubits,
            n_layers=n_layers,
            entanglement_scheme=entanglement,
            learning_rate=lr
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# 3. Prime Moduli Arithmetic & Multi-Scale Harmonic Resonances
@app.route("/api/moduli/trajectories", methods=["POST"])
def api_moduli_trajectories():
    try:
        data = request.json or {}
        lesion_key = data.get("lesion_type", "mca_cortical")
        primes = data.get("prime_basis", None)
        if primes and isinstance(primes, list):
            primes = [int(p) for p in primes]
        weeks = int(data.get("session_weeks", 12))
        
        result = stroke_gait_engine.compute_prime_moduli_trajectories(
            stroke_lesion_key=lesion_key,
            prime_basis=primes,
            session_weeks=weeks
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# 4. Combinatorial Synaptic Hebbian Amplification & Long-Term Horizon Care
@app.route("/api/hebbian/synaptic", methods=["POST"])
def api_hebbian_synaptic():
    try:
        data = request.json or {}
        lesion_key = data.get("lesion_type", "mca_cortical")
        n_neurons = int(data.get("n_neurons", 16))
        epochs = int(data.get("training_epochs", 40))
        rate = float(data.get("hebbian_rate", 0.09))
        pruning = float(data.get("anti_hebbian_pruning", 0.015))
        stdp_tau = float(data.get("stdp_tau_ms", 20.0))
        qml_gain = float(data.get("qml_reinforcement_gain", 1.25))
        modality = data.get("target_rehab_modality", "coupled_robotic_fes")

        result = stroke_gait_engine.simulate_hebbian_synaptic_amplification(
            stroke_lesion_key=lesion_key,
            n_neurons=n_neurons,
            training_epochs=epochs,
            hebbian_rate=rate,
            anti_hebbian_pruning=pruning,
            stdp_tau_ms=stdp_tau,
            qml_reinforcement_gain=qml_gain,
            target_rehab_modality=modality
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# 5. Multi-Objective Exercise Prescription Optimization
@app.route("/api/optimization/prescription", methods=["POST"])
def api_optimization_prescription():
    try:
        data = request.json or {}
        lesion_key = data.get("lesion_type", "mca_cortical")
        fatigue_mets = float(data.get("fatigue_threshold_mets", 3.8))
        duration = int(data.get("max_minutes", 60))
        goal = data.get("rehab_goal", "maximum_speed_and_symmetry")

        result = stroke_gait_engine.optimize_exercise_prescription(
            stroke_lesion_key=lesion_key,
            patient_fatigue_threshold_mets=fatigue_mets,
            daily_session_max_minutes=duration,
            target_rehab_goal=goal
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# 6. 15-Year Long-Term Healthcare Neurological Horizons
@app.route("/api/horizons/projections", methods=["POST"])
def api_horizons_projections():
    try:
        data = request.json or {}
        lesion_key = data.get("lesion_type", "mca_cortical")
        adherence = float(data.get("adherence_rate", 0.85))
        age = int(data.get("patient_age", 62))
        cv_risk = data.get("cv_risk_tier", "moderate")

        result = stroke_gait_engine.compute_15yr_longterm_neurological_horizons(
            stroke_lesion_key=lesion_key,
            adherence_compliance_rate=adherence,
            patient_age=age,
            cardiovascular_risk_tier=cv_risk
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# Metadata catalog endpoint
@app.route("/api/metadata/archetypes", methods=["GET"])
def api_metadata_archetypes():
    return jsonify({
        "success": True,
        "lesion_types": stroke_gait_engine.stroke_lesion_types,
        "exercise_protocols": stroke_gait_engine.rehab_exercise_protocols
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5025, debug=True)
