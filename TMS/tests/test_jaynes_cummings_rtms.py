import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rtms_engine",
    ROOT / "logic" / "rtms_engine.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_simulate_jaynes_cummings_rtms_returns_expected_structure():
    result = module.simulate_jaynes_cummings_rtms(omega_c=10.0, omega_a=9.0, coupling_g=0.5, n_photons=3)

    assert result["status"] == "ok"
    assert result["model"] == "Jaynes-Cummings"
    assert result["omega_c"] == 10.0
    assert result["omega_a"] == 9.0
    assert result["coupling_g"] == 0.5
    assert result["n_photons"] == 3
    assert len(result["oscillation_profile"]) == 5
    assert result["oscillation_profile"][0]["phase"] == 0
    assert result["neural_analogy"]["resonance_shift"] > 0
    assert result["combinatorial_weights"][0]["weight"] > 0
    assert result["photon_state_emissions"][0]["mass_equivalent_kg"] > 0
