"""Input-validated spectroscopy analysis and matching PDF downloads."""

from dataclasses import asdict
from io import BytesIO

from flask import Blueprint, current_app, jsonify, request, send_file

from backend.ecoli_analysis import Inputs, LIMITS, analyze, parse_inputs
from backend.ecoli_preprint import PDF_NAME, render_preprint


ecoli = Blueprint("ecoli", __name__, url_prefix="/api/ontario/ecoli")


@ecoli.before_request
def limit_request() -> None:
    request.max_content_length = 16 * 1024


@ecoli.errorhandler(ValueError)
def invalid_input(error):
    current_app.logger.warning("Spectroscopy input rejected: %s", error)
    return jsonify(status="error", message=str(error)), 400


def inputs() -> Inputs:
    if not request.is_json:
        raise ValueError("Send a JSON object of analysis inputs.")
    return parse_inputs(request.get_json())


@ecoli.get("/defaults")
def defaults():
    return jsonify(status="success", inputs=asdict(Inputs()), limits=LIMITS)


@ecoli.post("/analyze")
def analysis():
    return jsonify(status="success", report=analyze(inputs()))


@ecoli.route("/preprint", methods=["GET", "POST"])
def preprint():
    if request.method == "POST":
        current_inputs = inputs()
    else:
        current_inputs = Inputs()
    return send_file(BytesIO(render_preprint(analyze(current_inputs))),
                     mimetype="application/pdf", as_attachment=True, download_name=PDF_NAME)


@ecoli.get("/preprint/view")
def preprint_view():
    return send_file(BytesIO(render_preprint(analyze(Inputs()))),
                     mimetype="application/pdf", as_attachment=False)
