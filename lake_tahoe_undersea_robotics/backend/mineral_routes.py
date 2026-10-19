"""
Flask Blueprint for Benthic Mineral Ore Excavation, QML Fermionic Parity & Revenue Projections.
"""

from dataclasses import asdict
from io import BytesIO

from flask import Blueprint, current_app, jsonify, request, send_file

from backend.mineral_excavation_engine import (
    COMMODITY_PRICES,
    DEPOSIT_PRESETS,
    LIMITS,
    MineralExcavationEngine,
    MineralInputs,
    parse_mineral_inputs,
)
from backend.mineral_preprint import PDF_NAME, render_mineral_preprint


mineral = Blueprint("mineral", __name__, url_prefix="/api/mineral")
engine = MineralExcavationEngine()


@mineral.before_request
def limit_request() -> None:
    request.max_content_length = 32 * 1024


@mineral.errorhandler(ValueError)
def invalid_input(error):
    current_app.logger.warning("Mineral excavation input rejected: %s", error)
    return jsonify(status="error", message=str(error)), 400


def get_inputs() -> MineralInputs:
    if not request.is_json:
        raise ValueError("Send a JSON object of mineral excavation parameters.")
    return parse_mineral_inputs(request.get_json())


@mineral.get("/defaults")
def defaults():
    return jsonify({
        "status": "success",
        "inputs": asdict(MineralInputs()),
        "limits": LIMITS,
        "deposits": {k: v["title"] for k, v in DEPOSIT_PRESETS.items()},
        "commodities": COMMODITY_PRICES,
    })


@mineral.get("/deposits")
def list_deposits():
    return jsonify({
        "status": "success",
        "deposits": DEPOSIT_PRESETS
    })


@mineral.post("/simulate")
def simulate_mineral():
    current_inputs = get_inputs()
    result = engine.simulate(current_inputs)
    return jsonify({
        "status": "success",
        "report": result
    })


@mineral.route("/preprint", methods=["GET", "POST"])
def preprint():
    if request.method == "POST" and request.is_json:
        current_inputs = parse_mineral_inputs(request.get_json())
    else:
        current_inputs = MineralInputs()
    report = engine.simulate(current_inputs)
    pdf_data = render_mineral_preprint(report)
    return send_file(
        BytesIO(pdf_data),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=PDF_NAME
    )


@mineral.get("/preprint/view")
def preprint_view():
    report = engine.simulate(MineralInputs())
    pdf_data = render_mineral_preprint(report)
    return send_file(
        BytesIO(pdf_data),
        mimetype="application/pdf",
        as_attachment=False
    )
