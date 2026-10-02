#!/usr/bin/env python3
"""Aggregate hate-crime prevention planning dashboard."""

import os
from flask import Flask, jsonify, render_template, request
from prevention_engine import PreventionPlanningEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
engine = PreventionPlanningEngine(BASE_DIR)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    summary = engine.model(1.0)
    return jsonify({
        "status": "ONLINE",
        "system": "SafeCity Prevention Planning Console",
        "records": summary["data_quality"]["hate_incident_records"],
        "scope": summary["data_quality"]["scope"],
        "model_target": "<2% aggregate planning index",
    })


@app.route("/api/model", methods=["GET", "POST"])
def model():
    payload = request.get_json(silent=True) or {}
    level = payload.get("intervention_level", request.args.get("intervention_level", 0.82))
    try:
        level = float(level)
    except (TypeError, ValueError):
        level = 0.82
    return jsonify(engine.model(level))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5095))
    print(f"Launching SafeCity Prevention Planning Console on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
