"""Scoped economics/report verification endpoints."""

from dataclasses import asdict
from io import BytesIO

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from backend.carbon_economics import Assumptions, LIMITS, evaluate, parse_assumptions
from backend.carbon_preprint import PDF_NAME, render_preprint
from backend.pqc_reports import ReportSigner


carbon = Blueprint("carbon", __name__, url_prefix="/api/carbon")
signer = ReportSigner()


@carbon.before_request
def limit_request() -> None:
    request.max_content_length = 128 * 1024
    if request.content_length and request.content_length > request.max_content_length:
        raise RequestEntityTooLarge()


@carbon.errorhandler(ValueError)
def invalid_input(error):
    current_app.logger.warning("Carbon/PQC input rejected: %s", error)
    return jsonify(status="error", message=str(error)), 400


def input_assumptions() -> Assumptions:
    if not request.is_json:
        raise ValueError("Send a JSON object of numeric assumptions.")
    return parse_assumptions(request.get_json())


@carbon.get("/defaults")
def defaults():
    return jsonify(status="success", assumptions=asdict(Assumptions()), limits=LIMITS)


@carbon.post("/evaluate")
def evaluate_scenario():
    report = evaluate(input_assumptions())
    return jsonify(status="success", report=report, envelope=signer.sign(report))


@carbon.post("/pqc/self-test")
def pqc_self_test():
    return jsonify(status="success", verification=signer.self_test(evaluate(input_assumptions())))


@carbon.post("/pqc/verify")
def pqc_verify():
    if not request.is_json:
        raise ValueError("Send a signed report envelope as JSON.")
    valid = signer.verify(request.get_json())
    return jsonify(status="success", signature_valid=valid,
                   scope="Verified against this process's key; no external identity assurance.")


@carbon.post("/preprint")
def scenario_preprint():
    report = evaluate(input_assumptions())
    pdf = render_preprint(report, signer.self_test(report))
    return send_file(BytesIO(pdf), as_attachment=True, download_name=PDF_NAME,
                     mimetype="application/pdf")
