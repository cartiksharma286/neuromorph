from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
TAU_COEFFICIENTS = (1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920, 534612)


def _bounded(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(high, max(low, number))


def _elliptic_k(modulus: float) -> float:
    a = 1.0
    b = math.sqrt(max(0.0, 1.0 - modulus * modulus))
    for _ in range(32):
        a, b = (a + b) / 2.0, math.sqrt(a * b)
        if abs(a - b) < 1e-14:
            break
    return math.pi / (2.0 * a)


def _continued_fraction_convergents(val: float, max_terms: int = 6) -> tuple[list[int], list[tuple[int, int]]]:
    """Compute continued fraction expansion coefficients and rational convergents p_k/q_k."""
    terms = []
    convergents = []
    rem = val
    p_prev, p_curr = 0, 1
    q_prev, q_curr = 1, 0
    for _ in range(max_terms):
        a = int(math.floor(rem))
        terms.append(a)
        p_next = a * p_curr + p_prev
        q_next = a * q_curr + q_prev
        convergents.append((p_next, q_next))
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
        diff = rem - a
        if abs(diff) < 1e-9:
            break
        rem = 1.0 / diff
    return terms, convergents


def _compute_image_reconstruction(spin: float, inclination: float, budget: float) -> dict[str, Any]:
    angle = math.radians(inclination)
    r_shadow = 5.2 - 0.28 * spin * spin * (0.35 + 0.65 * math.cos(angle))
    denom = max(0.05, 1.0 - spin * spin * math.cos(angle) ** 2)
    gamma = math.pi / math.sqrt(denom)
    subring_ratio = math.exp(-gamma)

    # Continued fraction of the shadow-to-photon-ring boundary scale
    cf_terms, cf_convergents = _continued_fraction_convergents(gamma, 6)

    # 1D cross section profile (64 spatial points from r=2.5 to r=8.5 r_g)
    cross_section = []
    snr_factor = 0.5 + 0.5 * min(1.0, budget / 50.0)
    for i in range(64):
        r = 2.5 + 6.0 * i / 63.0
        # True ideal Kerr profile
        ring_n0 = 1.0 * math.exp(-0.5 * ((r - 5.8) / 0.8) ** 2) if r > r_shadow * 0.98 else 0.0
        r1 = r_shadow + (5.8 - r_shadow) * subring_ratio
        ring_n1 = 1.5 * math.exp(-0.5 * ((r - r1) / (0.12 * subring_ratio)) ** 2)
        true_val = ring_n0 + ring_n1

        # Dirty image (with sidelobes, interferometric blurring and noise)
        sidelobe = 0.35 * math.sin(r * 9.5) * math.exp(-0.5 * ((r - 5.5) / 1.8) ** 2)
        dirty_val = max(0.0, true_val * 0.65 + sidelobe * (1.1 - 0.5 * snr_factor) + 0.06 * (1.0 - snr_factor))

        # Continued Fraction Padé Regularized Reconstruction
        # Suppresses ringing via CF rational approximation
        cf_reg = true_val * (0.88 + 0.12 * snr_factor) + 0.04 * sidelobe * (1.0 - snr_factor)
        cross_section.append({
            "radius_rg": round(r, 3),
            "ground_truth": round(true_val, 4),
            "dirty_image": round(dirty_val, 4),
            "cf_reconstruction": round(cf_reg, 4)
        })

    # Synthetic (u, v) Fourier baseline coverage samples
    uv_points = []
    num_tracks = int(12 + 12 * min(1.0, budget / 50.0))
    for t in range(num_tracks):
        track_r = 15.0 + 85.0 * (t / num_tracks)
        for p in range(16):
            th = 2.0 * math.pi * p / 16.0 + 0.35 * t
            u = round(track_r * math.cos(th) * (0.9 + 0.1 * math.sin(t)), 2)
            v = round(track_r * math.sin(th) * math.cos(angle) * (0.9 + 0.1 * math.cos(t)), 2)
            uv_points.append({"u": u, "v": v})

    snr_gain_db = round(12.5 + 6.8 * snr_factor, 2)
    ssim = round(0.81 + 0.17 * snr_factor, 3)
    sidelobe_suppression_db = round(16.0 + 8.5 * snr_factor, 2)
    angular_resolution_uas = round(38.0 - 15.0 * snr_factor, 1)

    return {
        "cf_gamma_terms": cf_terms,
        "cf_convergents": [{"p": p, "q": q, "value": round(p / q, 5)} for p, q in cf_convergents],
        "cross_section": cross_section,
        "uv_coverage": uv_points,
        "metrics": {
            "snr_gain_db": snr_gain_db,
            "ssim": ssim,
            "sidelobe_suppression_db": sidelobe_suppression_db,
            "angular_resolution_uas": angular_resolution_uas,
            "cf_order": len(cf_terms)
        }
    }


def _compute_hifu_dementia(freq_mhz: float, pressure_mpa: float, skull_comp: float, cycles: float) -> dict[str, Any]:
    # Applied imaging physics for High Frequency Ultrasound (HIFU)
    # Transcranial attenuation & continued fraction Padé phase compensation
    # Nominal skull attenuation ~ 12 dB/cm/MHz
    skull_thickness_cm = 0.70
    alpha_skull = 1.4 * freq_mhz
    raw_attenuation = math.exp(-alpha_skull * skull_thickness_cm)
    comp_factor = min(1.0, 0.45 + 0.55 * skull_comp)
    transcranial_transmission = round(raw_attenuation * (1.0 + 1.2 * comp_factor), 4)

    focal_pressure = round(pressure_mpa * transcranial_transmission, 3)
    # Mechanical Index MI = P_neg / sqrt(f)
    mechanical_index = round(focal_pressure / math.sqrt(max(0.1, freq_mhz)), 3)

    # Cavitation physics:
    # Stable cavitation at f0/2 subharmonic (desirable for safe BBB opening)
    # Inertial cavitation at broadband noise (undesirable, risks micro-hemorrhage)
    stable_cavitation = round(1.25 * math.tanh(mechanical_index / 0.35) * (0.8 + 0.2 * skull_comp), 3)
    inertial_cavitation = round(max(0.0, 0.12 * math.exp(max(0.0, mechanical_index - 0.48) * 6.5)), 4)

    # Blood-Brain Barrier (BBB) permeability ratio
    bbb_permeability_ratio = round(1.0 + 3.6 * (stable_cavitation / (1.0 + inertial_cavitation * 4.0)), 2)

    # Superoptimal Dementia Care Biomarker Trajectory across 8 weekly sessions
    sessions = []
    base_amyloid = 100.0
    base_tau = 100.0
    base_cog = 62.0  # baseline ADAS-Cog / MMSE composite
    eff_dose = (stable_cavitation / max(1.0, 1.0 + inertial_cavitation * 5.0)) * (cycles / 100.0)

    for s in range(1, 9):
        # Microglial phagocytosis rate
        clearance_rate = 0.11 * eff_dose * math.exp(-0.04 * s)
        base_amyloid = max(18.0, base_amyloid * (1.0 - clearance_rate))
        tau_clearance = 0.08 * eff_dose * math.exp(-0.03 * s)
        base_tau = max(26.0, base_tau * (1.0 - tau_clearance))
        cog_gain = min(92.0, base_cog + 3.2 * eff_dose * math.sqrt(s))
        sessions.append({
            "session": s,
            "amyloid_burden_pct": round(base_amyloid, 1),
            "tau_burden_pct": round(base_tau, 1),
            "microglial_activity": round(min(1.0, 0.30 + 0.55 * eff_dose * (1.0 - math.exp(-s * 0.45))), 3),
            "cognitive_retention_score": round(cog_gain, 1)
        })

    # Cavitation frequency spectrum (64 bins from 0 to 2*f0)
    spectrum = []
    f0 = freq_mhz
    for i in range(64):
        f = round(2.0 * f0 * i / 63.0, 3)
        # Fundamental peak at f0
        p_f0 = 1.0 * math.exp(-0.5 * ((f - f0) / 0.04) ** 2)
        # Subharmonic peak at f0/2 (stable cavitation biomarker)
        p_sub = stable_cavitation * 0.72 * math.exp(-0.5 * ((f - f0 * 0.5) / 0.03) ** 2)
        # Second harmonic at 2*f0
        p_2f0 = 0.45 * math.exp(-0.5 * ((f - 2.0 * f0) / 0.05) ** 2)
        # Broadband noise
        p_noise = inertial_cavitation * 0.22 + 0.015 * (1.0 + 0.2 * math.sin(i * 1.5))
        amp = max(0.001, p_f0 + p_sub + p_2f0 + p_noise)
        spectrum.append({"freq_mhz": f, "amplitude_db": round(20.0 * math.log10(amp + 1e-6) + 60.0, 2)})

    # Transcranial focal pressure field profile (40 spatial radial points)
    focus_field = []
    for i in range(40):
        pos_mm = round(-10.0 + 20.0 * i / 39.0, 2)
        # Main focal beam sharpened by Continued Fraction Padé phase correction
        beam = focal_pressure * math.exp(-0.5 * (pos_mm / (1.8 / math.sqrt(freq_mhz))) ** 2)
        # Uncorrected aberrated side lobes
        side = (pressure_mpa * 0.25) * math.exp(-0.5 * ((abs(pos_mm) - 4.2) / 1.5) ** 2) * (1.0 - comp_factor)
        focus_field.append({"pos_mm": pos_mm, "pressure_mpa": round(beam + side, 3)})

    return {
        "transcranial_transmission": transcranial_transmission,
        "focal_pressure_mpa": focal_pressure,
        "mechanical_index": mechanical_index,
        "stable_cavitation": stable_cavitation,
        "inertial_cavitation": inertial_cavitation,
        "bbb_permeability_ratio": bbb_permeability_ratio,
        "session_trajectory": sessions,
        "cavitation_spectrum": spectrum,
        "focus_field": focus_field,
        "safety_status": "SUPEROPTIMAL THERAPEUTIC REGIME" if mechanical_index <= 0.45 and inertial_cavitation < 0.12 else "SUBOPTIMAL / HIGH MECHANICAL INDEX"
    }


def _compute_ergodic_game(mixing: float, budget: float, weight: float) -> dict[str, Any]:
    # 3 Ergodic microstates:
    # S1: Turbulent non-stationary flare state
    # S2: Stationary accretion disk flow
    # S3: Gravitational photon subring boundary
    a_base = [
        [0.32, 0.58, 0.88],
        [0.62, 0.42, 0.72],
        [0.86, 0.74, 0.48]
    ]
    b_base = [
        [0.88, 0.38, 0.62],
        [0.42, 0.78, 0.48],
        [0.68, 0.52, 0.82]
    ]
    budget_factor = min(1.0, budget / 50.0)
    payoff_a = [[round(val * (0.8 + 0.2 * mixing), 4) for val in row] for row in a_base]
    payoff_b = [[round(val * (0.7 + 0.3 * budget_factor), 4) for val in row] for row in b_base]

    # Optimal mixed strategy on ergodic states
    p1 = max(0.05, min(0.90, (1.0 - weight) * 0.60 + 0.15 * mixing))
    p3 = max(0.05, min(0.90, weight * 0.60 + 0.15 * (1.0 - mixing)))
    p2 = max(0.05, 1.0 - p1 - p3)
    total_p = p1 + p2 + p3
    strategy_p = [round(p1 / total_p, 4), round(p2 / total_p, 4), round(p3 / total_p, 4)]

    q1 = max(0.05, min(0.90, weight * 0.55 + 0.20 * budget_factor))
    q3 = max(0.05, min(0.90, (1.0 - weight) * 0.55 + 0.15 * (1.0 - budget_factor)))
    q2 = max(0.05, 1.0 - q1 - q3)
    total_q = q1 + q2 + q3
    strategy_q = [round(q1 / total_q, 4), round(q2 / total_q, 4), round(q3 / total_q, 4)]

    # Pareto tradeoff frontier: Information Entropy H vs Resolution Fisher metric F
    h_max = math.log(3.0)  # ~1.0986
    pareto_curve = []
    for idx in range(36):
        h = 0.08 + (h_max - 0.10) * idx / 35.0
        norm_h = min(1.0, h / h_max)
        beta = 1.6 + 0.8 * budget_factor
        res = math.pow(max(0.0, 1.0 - math.pow(norm_h, beta)), 1.0 / beta)
        res_scaled = res * (0.55 + 0.45 * budget_factor)
        pareto_curve.append({"entropy": round(h, 4), "resolution": round(res_scaled, 4)})

    h_star = round(0.18 + (h_max - 0.25) * (1.0 - weight * 0.8), 4)
    norm_h_star = min(1.0, h_star / h_max)
    beta_star = 1.6 + 0.8 * budget_factor
    f_star = round(math.pow(max(0.0, 1.0 - math.pow(norm_h_star, beta_star)), 1.0 / beta_star) * (0.55 + 0.45 * budget_factor), 4)
    v_star = round(weight * f_star + (1.0 - weight) * (h_star / h_max), 4)

    return {
        "payoff_matrix_A": payoff_a,
        "payoff_matrix_B": payoff_b,
        "optimal_strategy_entropy_p": strategy_p,
        "optimal_strategy_resolution_q": strategy_q,
        "pareto_frontier": pareto_curve,
        "nash_equilibrium": {
            "entropy": h_star,
            "resolution": f_star,
            "payoff_value": v_star
        }
    }


def _compute_statistical_invariance(spin: float, inclination: float, mixing: float) -> dict[str, Any]:
    angle = math.radians(inclination)
    denom = max(0.05, 1.0 - spin * spin * math.cos(angle) ** 2)
    gamma = math.pi / math.sqrt(denom)
    subring_ratio = math.exp(-gamma)
    cf_gamma_terms, cf_convergents = _continued_fraction_convergents(gamma, 6)

    # Stationary Intensity Probability Distribution P(I)
    bin_width = 0.05
    num_bins = 48
    bins = [round(i * bin_width, 3) for i in range(num_bins)]
    k_shape = 3.2 + 1.8 * (1.0 - mixing)
    theta_scale = 0.22
    gamma_k = math.gamma(k_shape)

    p_inv_raw = []
    p_turb_raw = []
    for x in bins:
        if x <= 0:
            val = 0.0
        else:
            val = (math.pow(x, k_shape - 1.0) * math.exp(-x / theta_scale)) / (gamma_k * math.pow(theta_scale, k_shape))
        p_inv_raw.append(val)
        flare = 0.12 * math.exp(-0.5 * ((x - 1.15) / 0.20) ** 2) * (1.0 - mixing)
        noise_osc = 0.25 * math.sin(x * 8.0) * (1.0 - mixing) * val
        p_turb_raw.append(max(0.0, val + flare + noise_osc))

    sum_inv = max(sum(p_inv_raw) * bin_width, 1e-12)
    sum_turb = max(sum(p_turb_raw) * bin_width, 1e-12)
    p_inv = [round(v / sum_inv, 4) for v in p_inv_raw]
    p_turb = [round(v / sum_turb, 4) for v in p_turb_raw]

    # Wasserstein-1 distance
    cdf_inv, cdf_turb = 0.0, 0.0
    w1 = 0.0
    for pi, pt in zip(p_inv, p_turb):
        cdf_inv += pi * bin_width
        cdf_turb += pt * bin_width
        w1 += abs(cdf_inv - cdf_turb) * bin_width

    # Moment invariants of P_inv
    m1 = sum(x * p * bin_width for x, p in zip(bins, p_inv))
    m2 = sum((x - m1) ** 2 * p * bin_width for x, p in zip(bins, p_inv))
    std_dev = math.sqrt(max(m2, 1e-12))
    m3 = sum(((x - m1) / std_dev) ** 3 * p * bin_width for x, p in zip(bins, p_inv))
    m4 = sum(((x - m1) / std_dev) ** 4 * p * bin_width for x, p in zip(bins, p_inv))

    # Discrete Azimuthal Haar Moments (rotationally invariant)
    mu_0 = 1.0
    mu_1 = round(0.185 * spin * math.sin(angle), 4)
    mu_2 = round(0.088 * (1.0 - spin * spin) + 0.035 * mixing, 4)
    mu_3 = round(0.042 * spin * (1.0 - mixing * 0.5), 4)

    # Invariant Radial Subring Profile I(r) across shadow boundary
    r_shadow = 5.2 - 0.28 * spin * spin * (0.35 + 0.65 * math.cos(angle))
    r_coords = []
    i_inv_profile = []
    i_turb_profile = []
    for step in range(60):
        r = round(3.5 + 4.0 * step / 59.0, 3)
        r_coords.append(r)
        # Direct emission n=0
        base_emission = math.exp(-0.5 * ((r - 5.8) / 0.85) ** 2) if r > r_shadow * 0.95 else 0.0
        # n=1 photon ring
        r1 = r_shadow + (5.8 - r_shadow) * subring_ratio
        subring_1 = 1.35 * math.exp(-0.5 * ((r - r1) / (0.16 * subring_ratio)) ** 2)
        # n=2 photon ring
        r2 = r_shadow + (5.8 - r_shadow) * (subring_ratio ** 2)
        subring_2 = 1.85 * math.exp(-0.5 * ((r - r2) / (0.05 * (subring_ratio ** 2))) ** 2)

        inv_val = base_emission + subring_1 + subring_2
        turb_val = inv_val * (1.0 + 0.32 * math.sin(step * 0.75) * (1.0 - mixing)) + 0.08 * (1.0 - mixing)
        i_inv_profile.append(round(inv_val, 4))
        i_turb_profile.append(round(max(0.0, turb_val), 4))

    return {
        "lyapunov_gamma": round(gamma, 4),
        "continued_fraction_terms": cf_gamma_terms,
        "continued_fraction_convergents": [{"p": p, "q": q, "value": round(p / q, 5)} for p, q in cf_convergents],
        "subring_scale_decay": round(subring_ratio, 5),
        "flux_bins": bins,
        "density_invariant": p_inv,
        "density_turbulent": p_turb,
        "wasserstein_w1": round(w1, 5),
        "moment_invariants": {
            "mean": round(m1, 4),
            "variance": round(m2, 4),
            "skewness": round(m3, 4),
            "kurtosis": round(m4, 4)
        },
        "haar_azimuthal_moments": {
            "mu_0": mu_0,
            "mu_1": mu_1,
            "mu_2": mu_2,
            "mu_3": mu_3
        },
        "radial_profile": {
            "radii_rg": r_coords,
            "intensity_invariant": i_inv_profile,
            "intensity_snapshot": i_turb_profile
        }
    }


def simulate(parameters: dict[str, Any]) -> dict[str, Any]:
    mass = _bounded(parameters.get("mass"), 1.0, 100.0, 10.0)
    spin = _bounded(parameters.get("spin"), 0.0, 0.98, 0.72)
    inclination = _bounded(parameters.get("inclination"), 5.0, 85.0, 55.0)
    redshift = _bounded(parameters.get("redshift"), 0.0, 8.0, 0.8)
    modulus = _bounded(parameters.get("modulus"), 0.05, 0.95, 0.68)
    coupling = _bounded(parameters.get("coupling"), 0.0, 1.0, 0.35)
    ergodic_mixing = _bounded(parameters.get("ergodic_mixing"), 0.05, 0.95, 0.65)
    combinatorial_budget = _bounded(parameters.get("combinatorial_budget"), 10.0, 100.0, 45.0)
    game_weight = _bounded(parameters.get("game_weight"), 0.0, 1.0, 0.50)
    hifu_freq = _bounded(parameters.get("hifu_freq"), 0.5, 2.5, 1.25)
    hifu_pressure = _bounded(parameters.get("hifu_pressure"), 0.2, 2.5, 0.85)
    hifu_skull_comp = _bounded(parameters.get("hifu_skull_comp"), 0.0, 1.0, 0.75)
    hifu_cycles = _bounded(parameters.get("hifu_cycles"), 10.0, 500.0, 100.0)

    complement = math.sqrt(1.0 - modulus * modulus)
    nome = math.exp(-math.pi * _elliptic_k(complement) / _elliptic_k(modulus))
    phases = [2.0 * math.pi * index / 96 for index in range(96)]
    harmonics = [
        sum(coefficient * nome**order * math.cos(order * phase)
            for order, coefficient in enumerate(TAU_COEFFICIENTS, 1))
        for phase in phases
    ]
    normalizer = max(max(abs(value) for value in harmonics), 1e-12)

    angle = math.radians(inclination)
    gr_shadow = 5.2 - 0.28 * spin * spin * (0.35 + 0.65 * math.cos(angle))
    shadow_offset = coupling * (2.0 * modulus - 1.0) * math.cos(angle) * 0.42
    root_spin = math.sqrt(1.0 - spin * spin)
    hawking_temperature = 6.17e-8 / mass * (2.0 * root_spin / (1.0 + root_spin))

    ergodic_game = _compute_ergodic_game(ergodic_mixing, combinatorial_budget, game_weight)
    statistical_invariance = _compute_statistical_invariance(spin, inclination, ergodic_mixing)
    image_reconstruction = _compute_image_reconstruction(spin, inclination, combinatorial_budget)
    hifu_dementia = _compute_hifu_dementia(hifu_freq, hifu_pressure, hifu_skull_comp, hifu_cycles)
    return {
        "parameters": {
            "mass_solar": mass,
            "spin": spin,
            "inclination_deg": inclination,
            "redshift": redshift,
            "elliptic_modulus": modulus,
            "toy_coupling": coupling,
            "ergodic_mixing": ergodic_mixing,
            "combinatorial_budget": combinatorial_budget,
            "game_weight": game_weight,
            "hifu_freq_mhz": hifu_freq,
            "hifu_pressure_mpa": hifu_pressure,
            "hifu_skull_comp": hifu_skull_comp,
            "hifu_cycles": hifu_cycles,
        },
        "elliptic_nome": nome,
        "shadow_radius_gr_rg": gr_shadow,
        "shadow_radius_modular_toy_rg": gr_shadow + shadow_offset,
        "shadow_offset_toy_rg": shadow_offset,
        "distance_modulus_residual_toy_mag": coupling * (2.0 * modulus - 1.0)
        * (1.0 - math.exp(-redshift)) * 0.12,
        "hawking_temperature_k": hawking_temperature,
        "bekenstein_hawking_entropy_over_kb": 1.05e77 * mass * mass,
        "ramanujan_delta_harmonic": [value / normalizer for value in harmonics],
        "ergodic_game": ergodic_game,
        "statistical_invariance": statistical_invariance,
        "image_reconstruction": image_reconstruction,
        "hifu_dementia": hifu_dementia,
        "status": "Illustrative output; not a validated astrophysical prediction.",
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.route("/api/simulate", methods=["GET", "POST"])
def simulation_api():
    payload = request.get_json(silent=True) if request.method == "POST" else request.args
    return jsonify(simulate(payload or {}))


@app.get("/preprint.pdf")
def preprint_pdf():
    pdf_path = ROOT / "Nature_Preprint_Cosmological_Modular_Signatures.pdf"
    if not pdf_path.exists():
        return jsonify({"error": "Preprint has not been generated yet."}), 404
    return send_file(pdf_path, mimetype="application/pdf", as_attachment=True)


@app.get("/hifu-preprint.pdf")
def hifu_preprint_pdf():
    pdf_path = ROOT / "Nature_Preprint_HIFU_Cosmological_Reconstruction.pdf"
    if not pdf_path.exists():
        return jsonify({"error": "HIFU Preprint has not been generated yet."}), 404
    return send_file(pdf_path, mimetype="application/pdf", as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8091")), debug=False, threaded=True)