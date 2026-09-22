import math


def simulate_pharmacological_equivalence(
    population: int = 1200,
    horizon_months: int = 18,
    molecular_dispersion: float = 0.72,
    rTMS_equivalence: float = 0.81,
):
    """Synthetic generative-AI discovery model for rTMS-equivalent therapeutics."""
    months = list(range(0, horizon_months + 1, 2))
    dispersion_curve = []
    clinical_horizon = []

    for month in months:
        phase_term = 0.24 * math.exp(-0.03 * month)
        oscillation = 0.08 * math.sin(month / 5.0)
        convective_term = 0.05 * math.cos(month / 7.0)
        dispersion_value = molecular_dispersion + phase_term + oscillation + convective_term
        dispersion_curve.append({
            "month": month,
            "dispersion": round(max(0.0, min(1.0, dispersion_value)), 4),
        })

        baseline_signal = 0.42 + 0.33 * math.exp(-0.02 * month)
        equivalence_gain = 0.14 * rTMS_equivalence
        horizon_signal = baseline_signal + equivalence_gain + 0.06 * math.sin(month / 6.0)
        clinical_horizon.append({
            "month": month,
            "equivalence_score": round(max(0.0, min(1.0, horizon_signal)), 4),
        })

    candidates = [
        {
            "name": "NeuroMorph Ligand A",
            "mechanism": "alpha-2A agonist + glutamatergic modulation",
            "equivalence_index": 0.91,
            "horizon_months": 18,
            "risk_level": "Low",
        },
        {
            "name": "Cortical Synapse Modulator",
            "mechanism": "GABA-B and NMDA balance optimizer",
            "equivalence_index": 0.86,
            "horizon_months": 15,
            "risk_level": "Moderate",
        },
        {
            "name": "Frontolimbic Stabilizer",
            "mechanism": "dopamine-serotonin receptor modulation",
            "equivalence_index": 0.79,
            "horizon_months": 12,
            "risk_level": "Low",
        },
        {
            "name": "Riboflavin Phospho-Target",
            "mechanism": "mitochondrial energy coupling / synaptic plasticity",
            "equivalence_index": 0.73,
            "horizon_months": 9,
            "risk_level": "Moderate",
        },
        {
            "name": "CBT-Paired Sleep Nootropic",
            "mechanism": "sleep architecture and cortical synchrony",
            "equivalence_index": 0.67,
            "horizon_months": 8,
            "risk_level": "Moderate",
        },
    ]

    equations = [
        {
            "title": "Molecular dispersion",
            "latex": r"D(t) = D_{0} e^{-\gamma t} + \lambda \int_{0}^{t} e^{-\gamma (t-s)} \Phi(s) \, ds",
            "description": "Diffusion and target engagement of the generative lead across longitudinal tissue exposure.",
        },
        {
            "title": "rTMS-equivalent potency",
            "latex": r"E_{eq}(t) = \alpha \cdot \frac{\mathrm{AUC}_{target}}{\mathrm{AUC}_{rTMS}} + \beta \cdot \Delta C_{syn} - \delta \cdot \mathrm{AE}(t)",
            "description": "Equivalent therapeutic effect is modeled from target coverage, synaptic modulation, and adverse-event burden.",
        },
        {
            "title": "Long-horizon response",
            "latex": r"R(t) = R_{0} \exp(-kt) + \eta \left(1 - e^{-kt}\right) \left(1 - \frac{\sigma_{risk}}{\sigma_{max}}\right)",
            "description": "Patient response evolves under adaptive treatment with risk-adjusted benefit persistence.",
        },
    ]

    trial_management = {
        "sites": 12,
        "active_patients": 428,
        "enrollment_goal": population,
        "completion_pct": 76,
        "ae_rate": 1.8,
        "status": "Active / DSMB review",
    }

    return {
        "population": population,
        "horizon_months": horizon_months,
        "molecular_dispersion": {
            "x": [point["month"] for point in dispersion_curve],
            "y": [point["dispersion"] for point in dispersion_curve],
        },
        "clinical_horizon": clinical_horizon,
        "candidates": candidates,
        "equations": equations,
        "trial_management": trial_management,
    }
