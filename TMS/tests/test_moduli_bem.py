import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "moduli_bem_paradigm",
    ROOT / "logic" / "moduli_bem_paradigm.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_moduli_reduction_and_stability():
    res = module.get_moduli_bem_paradigm("stroke")

    assert "optimal_protocol" in res
    opt = res["optimal_protocol"]
    assert opt["frequency_hz"] >= 1.0
    assert opt["intensity_pct"] >= 10.0
    assert 0.0 <= opt["stability_score"] <= 1.0
    assert "elliptic" in opt["nearest_elliptic_point"]


def test_bem_heatmap_and_attenuation():
    res = module.get_moduli_bem_paradigm("dementia")
    bem = res["bem_heatmap"]
    attenuation = res["bem_attenuation"]

    assert bem["peak_potential"] > 0
    assert len(bem["theta"]) == len(bem["potential"][0])
    assert len(attenuation["depths_mm"]) == len(attenuation["field_pct"])
    assert attenuation["field_pct"][-1] < attenuation["field_pct"][0]


def test_moduli_bem_api_and_preprint_routes():
    app_spec = importlib.util.spec_from_file_location("rtms_app", ROOT / "app.py")
    app_module = importlib.util.module_from_spec(app_spec)
    app_spec.loader.exec_module(app_module)
    client = app_module.app.test_client()

    response = client.get("/api/moduli-bem-paradigm?condition=tremor")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "success"

    pdf_response = client.get("/api/moduli-bem-preprint?condition=tremor")
    assert pdf_response.status_code == 200
    assert len(pdf_response.data) > 10000
