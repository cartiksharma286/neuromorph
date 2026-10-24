"""
Flask Web Application: Invisalign Orthodontic Smile Engineering
==============================================================
Featuring Riemannian Geodesic Arches, Measure Theory Smile Signatures,
Polynomial Ring Staging, Quantum FEA (Cortical/Trabecular) Biomechanics,
and Long-term Health Economic & Revenue Projections.
"""

import os
import io
import json
import math
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from ortho_engine import smile_engine, fea_engine, health_econ_engine

app = Flask(__name__)

# ─── Smile Simulation ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/invisalign/simulate", methods=["POST"])
def api_simulate():
    try:
        data = request.json or {}
        result = smile_engine.simulate_full_treatment(
            crowding=data.get("crowding_severity", "moderate"),
            arch_form=data.get("arch_form", "catenary"),
            lip_line=float(data.get("lip_line_height_mm", 14.0)),
            smile_width=float(data.get("smile_width_mm", 62.0))
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/invisalign/signature", methods=["POST"])
def api_signature():
    try:
        data = request.json or {}
        coords = np.array(data.get("coords"))
        sig = smile_engine.compute_measure_theoretic_smile_signature(
            coords,
            float(data.get("lip_line_height_mm", 14.0)),
            float(data.get("smile_width_mm", 62.0))
        )
        return jsonify({"success": True, "signature": sig})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ─── Quantum FEA Routes ───────────────────────────────────────────────────────

@app.route("/api/fea/simulate", methods=["POST"])
def api_fea_simulate():
    """Full-arch QML-enhanced FEA across all tooth types."""
    try:
        data = request.json or {}
        crowding     = data.get("crowding_severity", "moderate")
        aligner_stage = int(data.get("aligner_stage", 1))
        result = fea_engine.simulate_full_arch_fea(crowding=crowding, aligner_stage=aligner_stage)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/fea/tooth", methods=["POST"])
def api_fea_tooth():
    """Single-tooth QML FEA."""
    try:
        data = request.json or {}
        result = fea_engine.simulate_tooth_fea(
            tooth_type    = data.get("tooth_type", "central_incisor"),
            crowding      = data.get("crowding_severity", "moderate"),
            aligner_stage = int(data.get("aligner_stage", 1)),
            rho_rel       = float(data.get("rho_rel", 0.22))
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ─── Health Economics Routes ──────────────────────────────────────────────────

@app.route("/api/economics/full", methods=["POST"])
def api_economics_full():
    """Full health economics model."""
    try:
        data = request.json or {}
        n_patients = int(data.get("n_patients", 1000))
        result = health_econ_engine.run_full_health_economics(n_patients=n_patients)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ─── Nature PDF Generation ────────────────────────────────────────────────────

@app.route("/api/preprint/download", methods=["GET"])
@app.route("/api/invisalign/preprint", methods=["GET"])
def api_preprint():
    pdf_path = os.path.join(os.path.dirname(__file__), "Nature_Preprint_Invisalign_Geodesic_Smile_Signatures.pdf")
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=False,
                         download_name="Nature_Preprint_Invisalign_Geodesic_Smile_Signatures.pdf")
    return "Nature preprint is being compiled...", 404

@app.route("/api/preprint/fea/generate", methods=["POST"])
def api_generate_fea_preprint():
    """Generate and return the FEA + Health Economics Nature preprint PDF."""
    try:
        from generate_nature_pdf_fea_economics import generate_fea_economics_preprint
        pdf_bytes = generate_fea_economics_preprint()
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=False,
            download_name="Nature_Preprint_QML_FEA_Tooth_Biomechanics_HealthEconomics.pdf"
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/preprint/fea/download", methods=["GET"])
def api_download_fea_preprint():
    """Download the pre-generated FEA preprint PDF."""
    pdf_path = os.path.join(os.path.dirname(__file__),
                            "Nature_Preprint_QML_FEA_Tooth_Biomechanics_HealthEconomics.pdf")
    if not os.path.exists(pdf_path):
        # Auto-generate on first request
        try:
            from generate_nature_pdf_fea_economics import generate_fea_economics_preprint
            pdf_bytes = generate_fea_economics_preprint()
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return send_file(pdf_path, mimetype="application/pdf", as_attachment=False,
                     download_name="Nature_Preprint_QML_FEA_Tooth_Biomechanics_HealthEconomics.pdf")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5020))
    app.run(host="0.0.0.0", port=port, debug=True)
