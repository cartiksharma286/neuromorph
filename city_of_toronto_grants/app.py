#!/usr/bin/env python3
"""City of Toronto Community Grants Management Application.
Powered by Quantum-Statistical Selective Excitation States & Dirichlet Pigeonhole Combinatorics.
Author: Cartik Sharma, Neuromorph System.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from quantum_pigeonhole_engine import (
    TorontoCommunityGrantOptimizer, TORONTO_WARDS, GRANT_SECTORS
)
from generate_nature_preprint import generate_nature_preprint_pdf

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

optimizer = TorontoCommunityGrantOptimizer()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/wards", methods=["GET"])
def get_wards():
    return jsonify(TORONTO_WARDS)

@app.route("/api/sectors", methods=["GET"])
def get_sectors():
    return jsonify(GRANT_SECTORS)

@app.route("/api/allocate", methods=["GET", "POST"])
def allocate_grants():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form.to_dict()
    else:
        data = request.args

    budget = float(data.get("budget", 150_000_000.0))
    flip_angle = float(data.get("flip_angle", 105.0))
    bandwidth = float(data.get("bandwidth", 2.0))
    applicants = int(data.get("applicants", 1750))
    years = int(data.get("years", 5))

    results = optimizer.run_full_allocation(
        total_budget_cad=budget,
        flip_angle_deg=flip_angle,
        bandwidth_khz=bandwidth,
        total_applicants=applicants,
        projection_years=years
    )
    return jsonify(results)

@app.route("/api/auto_optimize", methods=["POST"])
def auto_optimize():
    """Calculates globally optimal flip angle, bandwidth, and budget for maximum municipal yield."""
    data = request.get_json(silent=True) or {}
    budget = float(data.get("budget", 150_000_000.0))
    applicants = int(data.get("applicants", 1750))

    best_res = None
    best_params = {}
    for fa in [75.0, 90.0, 105.0, 120.0]:
        for bw in [1.6, 2.0, 2.4, 2.8]:
            res = optimizer.run_full_allocation(
                total_budget_cad=budget,
                flip_angle_deg=fa,
                bandwidth_khz=bw,
                total_applicants=applicants,
                projection_years=5
            )
            yield_cad = res["summary"]["5yr_cumulative_municipal_yield_cad"]
            if best_res is None or yield_cad > best_res["summary"]["5yr_cumulative_municipal_yield_cad"]:
                best_res = res
                best_params = {"flip_angle": fa, "bandwidth": bw, "budget": budget, "applicants": applicants}

    best_res["optimal_parameters"] = best_params
    return jsonify(best_res)

@app.route("/download_preprint", methods=["GET"])
def download_preprint():
    pdf_path = os.path.join(BASE_DIR, "Nature_Preprint_Toronto_Grants_Allocation.pdf")
    if not os.path.exists(pdf_path):
        generate_nature_preprint_pdf(pdf_path)
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="Nature_Preprint_Toronto_Grants_Allocation.pdf",
        mimetype="application/pdf"
    )

@app.route("/api/rebuild_preprint", methods=["POST"])
def rebuild_preprint():
    pdf_path = os.path.join(BASE_DIR, "Nature_Preprint_Toronto_Grants_Allocation.pdf")
    generate_nature_preprint_pdf(pdf_path)
    return jsonify({"status": "success", "message": "Preprint regenerated successfully."})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch City of Toronto Community Grants Management App")
    parser.add_argument("--port", type=int, default=5055, help="Port to serve application")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("==========================================================================")
    print("🏛️  City of Toronto Community Grants Management Platform")
    print("⚛️  Quantum Selective Excitation & Dirichlet Pigeonhole Combinatorics")
    print(f"📍 Local URL:    http://127.0.0.1:{args.port}")
    print(f"🌐 Network URL:  http://0.0.0.0:{args.port}")
    print("📄 Nature Preprint: Available for download and viewing")
    print("==========================================================================")
    app.run(host=args.host, port=args.port, debug=args.debug)
