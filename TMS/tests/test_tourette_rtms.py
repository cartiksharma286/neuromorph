import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tourette_rtms",
    ROOT / "logic" / "tourette_rtms.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_tourette_simulation_is_deterministic_and_bounded():
    first = module.simulate_tourette_rtms()
    second = module.simulate_tourette_rtms()

    assert first == second
    assert len(first["weeks"]) == 21
    assert len(first["ygtss_synergistic"]) == 21
    assert len(first["motor_tic_score"]) == 21
    assert len(first["vocal_tic_score"]) == 21
    assert len(first["puts_urge_score"]) == 21
    assert all(0.0 <= val <= 1.0 for val in first["control_effort"])
    assert all(0.0 <= val <= 1.0 for val in first["tic_cluster_entropy"])
    assert first["metrics"]["final_ygtss"] < first["params"]["baseline_ygtss"]
    assert first["metrics"]["puts_reduction_pct"] > 30.0


def test_combinatorial_pulse_allocation():
    res = module.simulate_tourette_rtms(daily_pulses=3000)
    alloc = res["allocation"]

    assert alloc["total_pulses"] == 3000
    assert len(alloc["allocated_nodes"]) == 5
    assert sum(n["allocated_pulses"] for n in alloc["allocated_nodes"]) == 3000
    assert alloc["combinatorial_entropy"] > 1.0
    assert alloc["total_suppression_score"] > 0


def test_bem_presma_field_simulation():
    res = module.simulate_tourette_rtms()
    bem = res["bem_field"]

    assert bem["peak_surface_e_vm"] > 50.0
    assert len(bem["depths_mm"]) == 46
    assert len(bem["cstc_layers"]) == 5
    assert bem["skin_depth_delta_mm"] > 10.0


def test_tourette_api_and_preprint_routes():
    app_spec = importlib.util.spec_from_file_location("rtms_app", ROOT / "app.py")
    app_module = importlib.util.module_from_spec(app_spec)
    app_spec.loader.exec_module(app_module)
    client = app_module.app.test_client()

    response = client.get("/api/tourette-rtms?baseline_ygtss=35&treatment_weeks=18")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "success"
    assert payload["data"]["params"]["baseline_ygtss"] == 35.0
    assert len(payload["data"]["weeks"]) == 19

    pdf_response = client.get("/api/tourette-rtms-preprint?baseline_ygtss=35&treatment_weeks=18")
    assert pdf_response.status_code == 200
    assert len(pdf_response.data) > 10000
