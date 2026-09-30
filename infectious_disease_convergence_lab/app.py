from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]



def bounded(value: Any, low: float, high: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return min(high, max(low, numeric))


def complete_elliptic_integral(modulus: float) -> float:
    """Finite midpoint approximation of K(k), sufficient for a display metric."""
    segments = 240
    safe_modulus = min(0.98, max(0.0, modulus))
    step = (math.pi / 2) / segments
    return sum(
        step / math.sqrt(1 - safe_modulus ** 2 * math.sin((index + 0.5) * step) ** 2)
        for index in range(segments)
    )


def build_long_term_horizon(
    transmission: float,
    intervention: float,
    repair_capacity: float,
    phase_shift: float,
    horizon_years: int = 5,
) -> list[dict[str, Any]]:
    """Long-term multi-annual horizon with waning immunity and recurrent prime seasonal forcing."""
    months = max(36, min(120, horizon_years * 12))
    susceptible = 0.88
    exposed = 0.02
    infectious = 0.01
    recovered = 0.09

    r_base = transmission * (1 - 0.65 * intervention) * (1 - 0.20 * repair_capacity)
    points = []

    for m in range(months):
        year_frac = m / 12.0
        prime_forcing = sum(
            (2.0 * ((-1) ** k) / p) * math.cos(2 * math.pi * m / (p * 2.4) + phase_shift)
            for k, p in enumerate(PRIMES[:6])
        )
        seasonality = 1.0 + 0.18 * math.sin(2 * math.pi * m / 12.0 + phase_shift) + 0.08 * prime_forcing
        effective_r = max(0.2, r_base * seasonality)

        contact = effective_r * infectious * susceptible
        new_exposed = max(0.0005, contact * 0.45)
        new_infectious = 0.65 * exposed
        new_recovered = (0.50 + 0.30 * intervention + 0.15 * repair_capacity) * infectious
        waning_recovery = 0.045 * recovered

        susceptible = max(0.01, susceptible - new_exposed + waning_recovery)
        exposed = max(0.001, exposed + new_exposed - new_infectious)
        infectious = max(0.0005, infectious + new_infectious - new_recovered)
        recovered = max(0.01, recovered + new_recovered - waning_recovery)

        total = susceptible + exposed + infectious + recovered
        susceptible /= total
        exposed /= total
        infectious /= total
        recovered /= total

        points.append({
            "month": m,
            "year": round(year_frac, 2),
            "susceptible": round(susceptible * 100, 2),
            "exposed": round(exposed * 100, 2),
            "infectious": round(infectious * 100, 2),
            "recovered": round(recovered * 100, 2),
            "effective_r": round(effective_r, 3),
        })
    return points


def build_health_economics(
    payload: dict[str, Any] | None = None,
    transmission: float = 1.35,
    intervention: float = 0.48,
    repair_capacity: float = 0.58,
) -> dict[str, Any]:
    """Health economics revenue projection with statistical learning (ridge regression) and value-based care cash flows."""
    payload = payload or {}
    cohort_size = int(bounded(payload.get("cohort_size"), 25000, 2000000, 250000))
    horizon_years = int(bounded(payload.get("horizon_years"), 3, 10, 5))
    intervention_cost = bounded(payload.get("intervention_cost"), 50.0, 1500.0, 350.0)
    hospitalization_cost = bounded(payload.get("hospitalization_cost"), 5000.0, 50000.0, 18500.0)
    reimbursement_rate = bounded(payload.get("reimbursement_rate"), 0.20, 0.95, 0.65)
    learning_regularizer = bounded(payload.get("learning_regularizer"), 0.01, 2.5, 0.25)
    bioremediation_adoption = bounded(payload.get("bioremediation_adoption"), 0.10, 0.95, 0.45)

    # 1. Statistical learning engine: Multi-cohort pilot training dataset
    rng = np.random.RandomState(42)
    M = 90
    t_samples = rng.uniform(1.0, 10.0, M)
    adop_samples = rng.uniform(0.1, 0.95, M)
    eff_samples = rng.uniform(0.2, 0.9, M)
    X_train = np.column_stack([
        np.ones(M),
        t_samples,
        np.sqrt(t_samples),
        adop_samples,
        eff_samples,
    ])
    true_beta = np.array([38.0, 14.5, 28.0, 95.0, 72.0])
    noise = rng.normal(0, 6.5, M)
    y_train = X_train @ true_beta + noise

    p = X_train.shape[1]
    XtX = X_train.T @ X_train
    ridge_inv = np.linalg.inv(XtX + learning_regularizer * np.eye(p))
    beta_hat = ridge_inv @ X_train.T @ y_train
    y_pred_train = X_train @ beta_hat
    residuals = y_train - y_pred_train
    sigma2_hat = float(np.sum(residuals ** 2) / max(1, M - p))
    cov_beta = sigma2_hat * (ridge_inv @ XtX @ ridge_inv)
    ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
    r2_score = float(1.0 - np.sum(residuals ** 2) / ss_tot)
    rmse = float(np.sqrt(sigma2_hat))
    cond_number = float(np.linalg.cond(XtX + learning_regularizer * np.eye(p)))

    # 2. Year-by-year projections over horizon
    discount_rate = 0.035
    annual_projections = []
    cumulative_gross = 0.0
    cumulative_npv = 0.0
    cumulative_admissions_averted = 0
    cumulative_qalys = 0.0
    cumulative_volume = 0

    learning_curve_points = []
    baseline_admission_rate = 0.034 * (transmission / 1.35)

    for year in range(1, horizon_years + 1):
        adoption_t = bioremediation_adoption / (1.0 + math.exp(-1.15 * (year - 2.1)))
        enrolled_population = int(cohort_size * adoption_t)
        cumulative_volume += enrolled_population

        efficacy_t = min(0.92, 0.32 + 0.42 * intervention + 0.22 * repair_capacity)
        admissions_averted = int(enrolled_population * baseline_admission_rate * efficacy_t)
        cumulative_admissions_averted += admissions_averted

        x_eval = np.array([1.0, float(year), math.sqrt(float(year)), adoption_t, efficacy_t])
        pred_unit_rev = float(x_eval @ beta_hat)
        pred_var = float(x_eval @ cov_beta @ x_eval + sigma2_hat)
        pred_std = math.sqrt(max(0.1, pred_var))

        direct_program_rev = (enrolled_population * pred_unit_rev) / 1e6
        shared_savings_rev = (admissions_averted * hospitalization_cost * reimbursement_rate) / 1e6
        gross_rev = direct_program_rev + shared_savings_rev

        unit_delivery_cost = intervention_cost * ((1.0 + cumulative_volume / cohort_size) ** -0.22)
        program_cost = (enrolled_population * unit_delivery_cost) / 1e6
        net_margin = gross_rev - program_cost

        discount_factor = (1.0 + discount_rate) ** year
        discounted_margin = net_margin / discount_factor
        cumulative_gross += gross_rev
        cumulative_npv += discounted_margin

        qaly_t = admissions_averted * 1.32 + enrolled_population * 0.016
        cumulative_qalys += qaly_t

        ci_scale = (enrolled_population / 1e6)
        rev_lower = max(0.0, gross_rev - 1.96 * pred_std * ci_scale)
        rev_upper = gross_rev + 1.96 * pred_std * ci_scale

        annual_projections.append({
            "year": year,
            "enrolled": enrolled_population,
            "adoption_pct": round(adoption_t * 100, 1),
            "gross_revenue_m": round(gross_rev, 3),
            "gross_lower_m": round(rev_lower, 3),
            "gross_upper_m": round(rev_upper, 3),
            "net_margin_m": round(net_margin, 3),
            "program_cost_m": round(program_cost, 3),
            "shared_savings_m": round(shared_savings_rev, 3),
            "admissions_averted": admissions_averted,
            "qalys_gained": round(qaly_t, 1),
            "unit_cost": round(unit_delivery_cost, 2),
        })

        learning_curve_points.append({
            "volume_k": round(cumulative_volume / 1000, 1),
            "unit_cost": round(unit_delivery_cost, 2),
        })

    total_program_cost_m = sum(item["program_cost_m"] for item in annual_projections)
    total_savings_m = sum(item["shared_savings_m"] for item in annual_projections)
    net_economic_cost = max(1000.0, (total_program_cost_m - total_savings_m) * 1e6)
    icer = round(net_economic_cost / max(1.0, cumulative_qalys), 2)

    payer_mix = [
        {"segment": "Commercial Insurers", "share_pct": 42, "revenue_m": round(cumulative_gross * 0.42, 2)},
        {"segment": "Medicare Advantage & Public", "share_pct": 36, "revenue_m": round(cumulative_gross * 0.36, 2)},
        {"segment": "Capitated ACO Value-Based", "share_pct": 15, "revenue_m": round(cumulative_gross * 0.15, 2)},
        {"segment": "Bioremediation Subscriptions", "share_pct": 7, "revenue_m": round(cumulative_gross * 0.07, 2)},
    ]

    return {
        "parameters": {
            "cohort_size": cohort_size,
            "horizon_years": horizon_years,
            "intervention_cost": intervention_cost,
            "hospitalization_cost": hospitalization_cost,
            "reimbursement_rate": reimbursement_rate,
            "learning_regularizer": learning_regularizer,
            "bioremediation_adoption": bioremediation_adoption,
        },
        "headline": {
            "cumulative_gross_revenue_m": round(cumulative_gross, 2),
            "net_present_value_m": round(cumulative_npv, 2),
            "statistical_r2": round(r2_score, 4),
            "icer_per_qaly": icer,
            "cumulative_admissions_averted": cumulative_admissions_averted,
            "cumulative_qalys": round(cumulative_qalys, 1),
        },
        "statistical_model": {
            "formula": "y = X β + ε,  β_hat = (X^T X + λ I)^(-1) X^T y",
            "features": ["Intercept", "Horizon (yr)", "sqrt(Year)", "Adoption", "Efficacy"],
            "weights": [round(float(w), 3) for w in beta_hat],
            "ridge_lambda": learning_regularizer,
            "r_squared": round(r2_score, 4),
            "rmse": round(rmse, 3),
            "condition_number": round(cond_number, 2),
        },
        "annual_projections": annual_projections,
        "payer_mix": payer_mix,
        "learning_curve": learning_curve_points,
    }


def build_simulation(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a synthetic population-level scenario, never clinical guidance."""
    payload = payload or {}
    transmission = bounded(payload.get("transmission"), 0.55, 2.4, 1.35)
    intervention = bounded(payload.get("intervention"), 0.0, 0.9, 0.48)
    diffusion = bounded(payload.get("diffusion"), 0.05, 1.0, 0.42)
    phase_shift = bounded(payload.get("phase_shift"), 0.0, math.pi, 1.2)
    repair_capacity = bounded(payload.get("repair_capacity"), 0.1, 1.0, 0.58)
    horizon_years = int(bounded(payload.get("horizon_years"), 3, 10, 5))

    effective_r = transmission * (1 - 0.68 * intervention) * (1 - 0.18 * repair_capacity)
    susceptible, exposed, infectious, recovered = 0.986, 0.008, 0.006, 0.0
    trajectory = []
    convergence = []
    phase_points = []
    residual = 0.74

    for day in range(42):
        contact = effective_r * infectious * susceptible
        phase_signal = 0.5 + 0.5 * math.cos(day * 0.54 + phase_shift)
        sifted_exposure = contact * (0.82 + 0.18 * phase_signal)
        diffusion_inflow = diffusion * 0.004 * math.sin(day * 0.37 + phase_shift) * susceptible
        new_exposed = max(0.0, sifted_exposure + diffusion_inflow)
        new_infectious = 0.29 * exposed
        new_recovered = (0.11 + 0.11 * intervention + 0.06 * repair_capacity) * infectious

        susceptible = max(0.0, susceptible - new_exposed)
        exposed = max(0.0, exposed + new_exposed - new_infectious)
        infectious = max(0.0, infectious + new_infectious - new_recovered)
        recovered = min(1.0, recovered + new_recovered)
        total = susceptible + exposed + infectious + recovered
        susceptible, exposed, infectious, recovered = (
            susceptible / total, exposed / total, infectious / total, recovered / total,
        )

        residual *= 0.87 + 0.05 * (1 - intervention)
        trajectory.append({
            "day": day,
            "susceptible": round(susceptible * 100, 3),
            "exposed": round(exposed * 100, 3),
            "infectious": round(infectious * 100, 3),
            "recovered": round(recovered * 100, 3),
            "intervention_index": round((intervention * 0.8 + repair_capacity * 0.2) * 100, 2),
        })
        convergence.append({"iteration": day + 1, "residual": round(residual, 6)})
        phase_points.append({
            "day": day,
            "amplitude": round(math.sin(day * diffusion + phase_shift), 5),
            "sift": round(phase_signal, 5),
        })

    signature_terms = []
    leibniz_sum = 0.0
    for index, prime in enumerate(PRIMES[:10]):
        term = 4 * ((-1) ** index) / prime
        leibniz_sum += term
        signature_terms.append({"prime": prime, "term": round(term, 6), "sum": round(leibniz_sum, 6)})

    occupancy = [round(14 + 9 * math.sin(index * 0.83 + phase_shift) + 8 * infectious * 100) for index in range(8)]
    capacity = 20
    collisions = sum(max(0, count - capacity) for count in occupancy)
    modulus = min(0.97, diffusion * (0.55 + infectious * 3.0))
    elliptic_k = complete_elliptic_integral(modulus)

    governance = [
        {"name": "Community repair routing", "state": "READY" if repair_capacity >= 0.45 else "REVIEW", "score": round(repair_capacity * 100)},
        {"name": "Data minimization ledger", "state": "ACTIVE", "score": 96},
        {"name": "Equity allocation audit", "state": "REVIEW" if collisions else "ACTIVE", "score": max(60, 94 - collisions * 3)},
        {"name": "Bioremediation readiness", "state": "MONITOR", "score": round((intervention + diffusion) * 50)},
    ]

    long_term = build_long_term_horizon(
        transmission=transmission,
        intervention=intervention,
        repair_capacity=repair_capacity,
        phase_shift=phase_shift,
        horizon_years=horizon_years,
    )

    health_econ = build_health_economics(
        payload=payload,
        transmission=transmission,
        intervention=intervention,
        repair_capacity=repair_capacity,
    )

    return {
        "disclaimer": "Synthetic, population-level research visualization only. Not medical advice, diagnosis, or a clinical decision system.",
        "parameters": {
            "transmission": transmission,
            "intervention": intervention,
            "diffusion": diffusion,
            "phase_shift": phase_shift,
            "repair_capacity": repair_capacity,
            **health_econ["parameters"],
        },
        "headline": {
            "effective_r": round(effective_r, 3),
            "peak_infectious_pct": round(max(point["infectious"] for point in trajectory), 2),
            "terminal_residual": convergence[-1]["residual"],
            "governance_readiness": round(sum(item["score"] for item in governance) / len(governance)),
            "health_economics": health_econ["headline"],
        },
        "trajectory": trajectory,
        "convergence": convergence,
        "phase_sifting": phase_points,
        "long_term_horizon": long_term,
        "health_economics": health_econ,
        "finite_math": {
            "pigeonhole_capacity": capacity,
            "regional_occupancy": occupancy,
            "overflow_collisions": collisions,
            "elliptic_modulus": round(modulus, 4),
            "elliptic_projection_k": round(elliptic_k, 5),
            "leibniz_prime_signatures": signature_terms,
        },
        "state_transitions": [
            {"from": "Susceptible", "to": "Exposed", "driver": "phase-sifted contact"},
            {"from": "Exposed", "to": "Infectious", "driver": "latent progression"},
            {"from": "Infectious", "to": "Recovered", "driver": "interventional repair"},
        ],
        "governance": governance,
    }


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.post("/api/simulate")
def simulate_api():
    return jsonify(build_simulation(request.get_json(silent=True)))


@app.get("/api/baseline")
def baseline_api():
    return jsonify(build_simulation())


@app.post("/api/health-economics")
def health_economics_api():
    body = request.get_json(silent=True) or {}
    return jsonify(build_health_economics(body))


@app.get("/preprint.pdf")
def preprint_pdf():
    # Prefer the long-term disease management preprint, fall back if not yet generated
    pdf_path = ROOT / "Nature_Preprint_Long_Term_Disease_Management_Health_Economics.pdf"
    if not pdf_path.exists():
        pdf_path = ROOT / "Nature_Preprint_Infectious_Disease_Convergence.pdf"
    if not pdf_path.exists():
        return jsonify({"error": "Preprint not generated yet. Run generate_preprint.py."}), 404
    return send_file(pdf_path, mimetype="application/pdf", download_name=pdf_path.name)


@app.get("/preprint-longterm.pdf")
def preprint_longterm_pdf():
    pdf_path = ROOT / "Nature_Preprint_Long_Term_Disease_Management_Health_Economics.pdf"
    if not pdf_path.exists():
        return jsonify({"error": "Long-term preprint not generated yet."}), 404
    return send_file(pdf_path, mimetype="application/pdf", download_name=pdf_path.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8094"))
    app.run(host="127.0.0.1", port=port, debug=False)
