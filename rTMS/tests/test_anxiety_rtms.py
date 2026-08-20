import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "anxiety_millennials_rtms",
    ROOT / "logic" / "anxiety_millennials_rtms.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_anxiety_model_is_deterministic_and_bounded():
    first = module.simulate_anxiety_rtms()
    second = module.simulate_anxiety_rtms()

    assert first == second
    assert len(first["weeks"]) == 25
    assert len(first["gad7_synergistic"]) == 25
    assert all(0.0 <= val <= 1.0 for val in first["control_effort"])
    assert all(1.5 <= val <= 21.0 for val in first["gad7_synergistic"])
    assert first["metrics"]["final_gad7"] < first["params"]["baseline_gad7"]
    assert first["metrics"]["cohen_d"] > 1.0


def test_cortical_surface_fea_and_eeg_processing():
    res = module.simulate_anxiety_rtms()
    fea = res["fea"]
    eeg = res["eeg"]

    assert fea["peak_surface_e_vm"] > 50.0
    assert len(fea["depths_mm"]) == 46
    assert len(fea["e_field_vm"]) == 46
    assert eeg["faa_pre"] < 0.0 # Negative FAA before treatment
    assert eeg["faa_post"] > 0.0 # Positive valence FAA after treatment
    assert eeg["delta_faa"] > 0.0
    assert len(eeg["frequencies"]) == len(eeg["psd_pre"])


def test_pharmacological_trials_and_staging():
    res = module.simulate_anxiety_rtms()
    trials = res["trials"]
    staging = res["staging"]

    assert len(trials["trial_arms"]) == 6
    assert trials["bayesian_posterior_prob_superiority"] > 0.99
    assert 0 < staging["optimal_induction_end"] < staging["optimal_consolidation_end"] <= 24
    assert len(staging["stages"]) == 3
    assert staging["candidate_costs"] == sorted(staging["candidate_costs"])


def test_anxiety_api_and_preprint_routes():
    app_spec = importlib.util.spec_from_file_location("rtms_app", ROOT / "app.py")
    app_module = importlib.util.module_from_spec(app_spec)
    app_spec.loader.exec_module(app_module)
    client = app_module.app.test_client()

    response = client.get("/api/anxiety-rtms?baseline_gad7=18&treatment_weeks=20")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["data"]["params"]["baseline_gad7"] == 18.0
    assert len(payload["data"]["weeks"]) == 21

    pdf_response = client.get("/api/anxiety-rtms-preprint?baseline_gad7=18&treatment_weeks=20")
    assert pdf_response.status_code == 200
    assert len(pdf_response.data) > 10000
