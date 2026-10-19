"""
Flask Blueprint for Underwater Robot Camera, Quantum Vision & Elliptic Moduli Event Detection.
"""

from dataclasses import asdict
from io import BytesIO

from flask import Blueprint, current_app, jsonify, request, send_file

from backend.underwater_vision_engine import (
    LIMITS,
    SCENE_PRESETS,
    UnderwaterVisionEngine,
    VisionInputs,
    parse_vision_inputs,
)
from backend.vision_preprint import PDF_NAME, render_vision_preprint


vision = Blueprint("vision", __name__, url_prefix="/api/vision")
engine = UnderwaterVisionEngine()


@vision.before_request
def limit_request() -> None:
    request.max_content_length = 32 * 1024


@vision.errorhandler(ValueError)
def invalid_input(error):
    current_app.logger.warning("Underwater vision input rejected: %s", error)
    return jsonify(status="error", message=str(error)), 400


def get_inputs() -> VisionInputs:
    if not request.is_json:
        raise ValueError("Send a JSON object of vision parameters.")
    return parse_vision_inputs(request.get_json())


@vision.get("/defaults")
def defaults():
    return jsonify({
        "status": "success",
        "inputs": asdict(VisionInputs()),
        "limits": LIMITS,
        "scenes": {k: v["title"] for k, v in SCENE_PRESETS.items()},
    })


@vision.get("/scenes")
def list_scenes():
    return jsonify({
        "status": "success",
        "scenes": SCENE_PRESETS
    })


@vision.post("/simulate")
def simulate_vision():
    current_inputs = get_inputs()
    result = engine.simulate(current_inputs)
    return jsonify({
        "status": "success",
        "report": result
    })


@vision.route("/preprint", methods=["GET", "POST"])
def preprint():
    if request.method == "POST" and request.is_json:
        current_inputs = parse_vision_inputs(request.get_json())
    else:
        current_inputs = VisionInputs()
    report = engine.simulate(current_inputs)
    pdf_data = render_vision_preprint(report)
    return send_file(
        BytesIO(pdf_data),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=PDF_NAME
    )


@vision.get("/preprint/view")
def preprint_view():
    report = engine.simulate(VisionInputs())
    pdf_data = render_vision_preprint(report)
    return send_file(
        BytesIO(pdf_data),
        mimetype="application/pdf",
        as_attachment=False
    )
