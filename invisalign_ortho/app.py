"""
Flask Web Application: Invisalign Orthodontic Smile Engineering
==============================================================
Featuring Riemannian Geodesic Arches, Measure Theory Smile Signatures,
and Polynomial Ring Staging for Craniofacial Smile Signatures.
"""

import os
import io
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from ortho_engine import smile_engine

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# 1. Simulate Full Treatment
@app.route("/api/invisalign/simulate", methods=["POST"])
def api_simulate():
    try:
        data = request.json or {}
        crowding = data.get("crowding_severity", "moderate")
        arch_form = data.get("arch_form", "catenary")
        lip_line = float(data.get("lip_line_height_mm", 14.0))
        smile_width = float(data.get("smile_width_mm", 62.0))

        result = smile_engine.simulate_full_treatment(
            crowding=crowding,
            arch_form=arch_form,
            lip_line=lip_line,
            smile_width=smile_width
        )
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# 2. Evaluate Specific Measure Theoretic Smile Signature
@app.route("/api/invisalign/signature", methods=["POST"])
def api_signature():
    try:
        data = request.json or {}
        coords = np.array(data.get("coords"))
        lip_line = float(data.get("lip_line_height_mm", 14.0))
        smile_width = float(data.get("smile_width_mm", 62.0))
        sig = smile_engine.compute_measure_theoretic_smile_signature(coords, lip_line, smile_width)
        return jsonify({"success": True, "signature": sig})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# 3. Nature Preprint PDF Download Route
@app.route("/api/preprint/download", methods=["GET"])
@app.route("/api/invisalign/preprint", methods=["GET"])
def api_preprint():
    pdf_path = os.path.join(os.path.dirname(__file__), "Nature_Preprint_Invisalign_Geodesic_Smile_Signatures.pdf")
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=False, download_name="Nature_Preprint_Invisalign_Geodesic_Smile_Signatures.pdf")
    return "Nature preprint is being compiled...", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5020))
    app.run(host="0.0.0.0", port=port, debug=True)
