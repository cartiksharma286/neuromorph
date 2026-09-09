import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "depression_rtms",
    ROOT / "logic" / "depression_rtms.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_depression_model_is_deterministic_and_bounded():
    first = module.simulate_depression_rtms()
    second = module.simulate_depression_rtms()

    assert first == second
    assert len(first["sessions"]) == 31
    assert len(first["phq9_adaptive_combined"]) == 31
    assert all(0.0 <= value <= 1.0 for value in first["control_effort"])
    assert all(2.0 <= value <= 27.0 for value in first["phq9_adaptive_combined"])
    assert first["metrics"]["final_phq9"] < first["metrics"]["baseline_phq9"]
    assert first["paradigm"]["status"].startswith("in-silico")


def test_number_signatures_are_finite_and_reproducible():
    result = module.simulate_depression_rtms(sessions=12, signature_ratio=1.61803398875)

    assert result["prime_sessions"] == [2, 3, 5, 7, 11]
    assert len(result["number_signature"]) == 13
    assert all(0 <= value < 17 for value in result["number_signature"])
    errors = [item["error"] for item in result["continued_fraction"]["convergents"]]
    assert errors[-1] < errors[0]


def test_optimal_staging_has_ordered_gates_and_ranked_costs():
    staging = module.simulate_depression_rtms()["staging"]

    assert 0 < staging["optimal_induction_end"] < staging["optimal_consolidation_end"] < 30
    assert len(staging["stage_index"]) == 31
    assert [stage["name"] for stage in staging["stages"]] == ["Induction", "Consolidation", "Maintenance"]
    assert staging["candidate_cost"] == sorted(staging["candidate_cost"])
    assert staging["candidate_count"] > len(staging["candidate_cost"])


def test_depression_api_and_preprint_routes():
    app_spec = importlib.util.spec_from_file_location("rtms_app", ROOT / "app.py")
    app_module = importlib.util.module_from_spec(app_spec)
    app_spec.loader.exec_module(app_module)
    client = app_module.app.test_client()

    response = client.get("/api/depression-rtms?baseline_phq9=21&sessions=24")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["params"]["baseline_phq9"] == 21.0
    assert len(payload["sessions"]) == 25

    pdf_response = client.get("/api/depression-rtms-preprint?baseline_phq9=21&sessions=24")
    assert pdf_response.status_code == 200
    assert pdf_response.content_type == "application/pdf"
    assert pdf_response.data.startswith(b"%PDF")
