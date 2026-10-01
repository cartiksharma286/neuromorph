"""Flask Application for Quantum-Statistical Traffic Signal Monitoring & Red Light Violation Mitigation.

Serves RESTful APIs and modern interactive geospatial analytics dashboard.
"""

import os
import math
from flask import Flask, render_template, jsonify, request, send_from_directory
from backend.data_loader import load_cameras_dataset
from backend.quantum_observables import QuantumTrafficObservables, get_camera_quantum_state
from backend.statistical_ml import StatisticalViolationModel
from backend.mitigation_engine import evaluate_mitigation_policy

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
)

# Load cameras once at startup
CAMERAS = load_cameras_dataset()
MODEL = StatisticalViolationModel(CAMERAS)

# Fit initial ML models and attach cluster and quantum information
GMM_CLUSTERS = MODEL.fit_spatial_gmm_clusters()
GLM_RESULTS = MODEL.fit_negative_binomial_glm()
EVT_RESULTS = MODEL.fit_extreme_value_tail()
MCMC_RESULTS = MODEL.simulate_mcmc_speed_posterior()
Q_ENGINE = QuantumTrafficObservables()
VQE_RESULTS = Q_ENGINE.simulate_vqe_convergence()

# Pre-calculate quantum state for each camera
for cam in CAMERAS:
    q_state = get_camera_quantum_state(cam)
    cam["quantum_observables"] = q_state


@app.route("/")
def index():
    """Serves the main interactive dashboard."""
    return render_template("index.html")


@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    """Returns all 301 camera nodes with metadata, risk ratings, and quantum observables."""
    district = request.args.get("district")
    risk_tier = request.args.get("risk_tier")

    filtered = CAMERAS
    if district and district != "all":
        filtered = [c for c in filtered if c["district"] == district]
    if risk_tier and risk_tier != "all":
        filtered = [c for c in filtered if c["risk_tier"] == risk_tier]

    return jsonify({
        "status": "success",
        "total_count": len(filtered),
        "total_city_cameras": len(CAMERAS),
        "cameras": filtered
    })


@app.route("/api/camera/<int:cid>", methods=["GET"])
def api_camera_detail(cid):
    """Returns deep telemetry for a specific camera intersection."""
    cam = next((c for c in CAMERAS if c["id"] == cid), None)
    if not cam:
        return jsonify({"status": "error", "message": f"Camera {cid} not found"}), 404

    # Hourly distribution of violations across 24-hour cycle
    hours = list(range(24))
    # Peak at rush hours (8am, 5pm) and late night weekend (11pm-2am)
    hourly_violations = []
    base_m = cam["monthly_violations"] / 30.0 # daily
    for h in hours:
        if 7 <= h <= 9: # AM Rush
            f = 1.65
        elif 16 <= h <= 18: # PM Rush
            f = 1.85
        elif 22 <= h or h <= 2: # Late night speed rush
            f = 1.30
        elif 3 <= h <= 5: # Overnight lulls
            f = 0.25
        else:
            f = 0.90
        hourly_violations.append(round(float(base_m * (f / 24.0) * 1.05), 2))

    # Stopping vs Clearing curves over approach speed
    speeds_sweep = list(range(30, 95, 5))
    v0_arr = [s / 3.6 for s in speeds_sweep]
    t_r = 1.2
    a = 3.2
    tau = cam["yellow_duration_s"] + cam["all_red_duration_s"]
    w = cam["intersection_width_m"]
    l = 5.5

    stopping_curves = [round(v * t_r + (v ** 2) / (2.0 * a), 1) for v in v0_arr]
    clearing_curves = [round(v * tau - (w + l), 1) for v in v0_arr]

    return jsonify({
        "status": "success",
        "camera": cam,
        "hourly_telemetry": {
            "hours": [f"{h:02d}:00" for h in hours],
            "violations": hourly_violations
        },
        "kinematic_curves": {
            "speeds_kmh": speeds_sweep,
            "stopping_dist_m": stopping_curves,
            "clearing_dist_m": clearing_curves
        }
    })


@app.route("/api/convergence", methods=["GET"])
def api_convergence():
    """Returns convergence characteristics for ML models and quantum variational eigensolver."""
    return jsonify({
        "status": "success",
        "quantum_vqe": VQE_RESULTS,
        "negative_binomial_glm": {
            "iterations": GLM_RESULTS["iterations"],
            "log_likelihood": GLM_RESULTS["convergence_log_likelihood"],
            "beta_norm_diff": GLM_RESULTS["convergence_beta_diff"],
            "pseudo_r2": GLM_RESULTS["pseudo_r2"],
            "dispersion_alpha": GLM_RESULTS["dispersion_alpha"],
            "coefficients": GLM_RESULTS["coefficients"]
        },
        "spatial_gmm_em": {
            "iterations": GMM_CLUSTERS["iterations"],
            "log_likelihood": GMM_CLUSTERS["em_log_likelihood_convergence"]
        },
        "mcmc_bayesian": MCMC_RESULTS
    })


@app.route("/api/extreme-values", methods=["GET"])
def api_extreme_values():
    """Returns Extreme Value Theory (EVT) Generalized Pareto Distribution metrics."""
    return jsonify({
        "status": "success",
        "evt": EVT_RESULTS
    })


@app.route("/api/clusters", methods=["GET"])
def api_clusters():
    """Returns the 4 spatial risk phenotypes from GMM clustering."""
    return jsonify({
        "status": "success",
        "clusters": GMM_CLUSTERS["clusters"]
    })


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Runs interactive scenario simulation modifying clearance timing, flow, and weather."""
    data = request.get_json(silent=True) or {}
    yellow_ext = float(data.get("yellow_extension_s", 0.6))
    all_red_ext = float(data.get("all_red_extension_s", 0.4))
    weather = str(data.get("weather_condition", "clear")).lower()
    flow_mult = float(data.get("flow_multiplier", 1.0))
    quantum_gating = bool(data.get("quantum_gating", True))

    # Apply weather friction factor to braking deceleration
    # Clear = 3.2 m/s^2, Rain = 2.6 m/s^2, Snow/Ice = 1.8 m/s^2
    decel_map = {"clear": 3.2, "rain": 2.6, "snow_ice": 1.9}
    eff_decel = decel_map.get(weather, 3.2)

    # Re-evaluate all cameras under simulated parameters
    sim_cameras = []
    for c in CAMERAS:
        c_copy = dict(c)
        v0 = c_copy["speed_85th_kmh"] / 3.6
        c_copy["aadt_volume"] = int(c["aadt_volume"] * flow_mult)

        # Updated stopping & clearing
        w = c_copy["intersection_width_m"]
        l = 5.5
        tau_new = c_copy["yellow_duration_s"] + yellow_ext + c_copy["all_red_duration_s"] + all_red_ext
        x_stop = v0 * 1.2 + (v0 ** 2) / (2.0 * eff_decel)
        x_clear = v0 * tau_new - (w + l)

        c_copy["stopping_dist_m"] = round(x_stop, 1)
        c_copy["clearing_dist_m"] = round(x_clear, 1)
        c_copy["dilemma_zone_gap_m"] = round(max(0.0, x_stop - x_clear), 1)

        sim_cameras.append(c_copy)

    mitigation = evaluate_mitigation_policy(
        sim_cameras,
        yellow_extension_s=yellow_ext,
        all_red_extension_s=all_red_ext,
        quantum_dilemma_gating=quantum_gating
    )

    return jsonify({
        "status": "success",
        "parameters": {
            "yellow_extension_s": yellow_ext,
            "all_red_extension_s": all_red_ext,
            "weather_condition": weather,
            "effective_deceleration_m_s2": eff_decel,
            "flow_multiplier": flow_mult,
            "quantum_gating": quantum_gating
        },
        "mitigation": mitigation
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "app_name": "Quantum-Statistical Red Light Camera Monitoring",
        "cameras_loaded": len(CAMERAS),
        "port": 5005
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)
