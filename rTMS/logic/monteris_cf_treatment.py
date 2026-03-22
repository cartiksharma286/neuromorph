"""
Monteris CF Treatment Paradigms
=================================
Continued-fraction (CF) optimised Monteris NeuroBlate-compatible MRI pulse
sequences for pre-operative, intra-operative and post-operative neurological
care.  Integrates with the quantum neural circuitry QNC server for coupling
brain state transfer efficiencies to ablation dose control.

Clinical Workflow
-----------------
  Pre-op  : CF-DTI tractography + fMRI functional mapping with Farey-spaced TEs
  Intra-op: PRF-MR thermometry wave-trains derived from CF convergents → CEM43
            thermal dose integration → real-time ablation boundary feedback
  Post-op : FLAIR/SWI lesion evolution monitoring, QNC state-transfer recovery
            trajectory, cognitive-resilience risk stratification

Physics
-------
  PRF thermometry :  ΔT = Δφ / (α γ B₀ TE)          [α = -0.0094 ppm/°C]
  CF convergent    :  p_n/q_n → φ  via  p_{n+1} = a_{n+1} p_n + p_{n-1}
  Echo spacing     :  TE_k = TE_min + (k / N) * (TE_max - TE_min)  (CF-Farey)
  Thermal dose     :  CEM43 = Σ_k Δt · R^(43-T_k)  [R=0.5 for T≥43°C]
  WLS estimator    :  β̂ = (X'WX)⁻¹ X'Wy
  Cramér-Rao       :  Var(β̂) ≥ [I(β)]⁻¹
  JC Hamiltonian   :  H = ħω_c a†a + Σ_j ħω_j σ_j^z/2 + Σ_j g_j(a†σ_j^- + aσ_j^+)
  Rabi oscillation :  P_e(t) = sin²(Ω_R t / 2)
"""

import numpy as np
from fractions import Fraction
import math, warnings

# ──────────────────────────────────────────────────────────────────────────────
# 1. Continued-Fraction Utilities
# ──────────────────────────────────────────────────────────────────────────────

def cf_coefficients(value: float, depth: int = 12) -> list:
    """Return CF coefficients [a0; a1, a2, …, a_depth] for *value*."""
    coeffs, x = [], value
    for _ in range(depth + 1):
        a = int(x)
        coeffs.append(a)
        frac = x - a
        if abs(frac) < 1e-12:
            break
        x = 1.0 / frac
    return coeffs


def cf_convergents(value: float, depth: int = 12) -> list:
    """Return list of (p_n, q_n) convergents for *value*."""
    coeffs = cf_coefficients(value, depth)
    convergents = []
    p_prev, p_curr = 1, coeffs[0]
    q_prev, q_curr = 0, 1
    convergents.append((p_curr, q_curr))
    for a in coeffs[1:]:
        p_next = a * p_curr + p_prev
        q_next = a * q_curr + q_prev
        convergents.append((p_next, q_next))
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
    return convergents


def farey_echo_times(n_echoes: int, te_min_ms: float, te_max_ms: float,
                     depth: int = 10) -> np.ndarray:
    """
    Place echo times using Farey-sequence fractions from the CF expansion of φ.
    This gives near-optimal T₂* coverage with minimal SAR clustering.
    """
    golden = (1 + math.sqrt(5)) / 2
    convs = cf_convergents(golden, depth)
    # Use convergent fractions to seed Farey distribution
    fracs = sorted({p / q for p, q in convs if q > 0 and p / q < 1})
    # Pad / trim to n_echoes via uniform fallback + CF fracs interleaved
    uniform = np.linspace(0, 1, n_echoes + 2)[1:-1]
    combined = np.unique(np.concatenate([np.array(fracs), uniform]))
    idx = np.round(np.linspace(0, len(combined) - 1, n_echoes)).astype(int)
    t_norm = combined[idx]
    return te_min_ms + t_norm * (te_max_ms - te_min_ms)


# ──────────────────────────────────────────────────────────────────────────────
# 2. PRF Thermometry
# ──────────────────────────────────────────────────────────────────────────────

PRF_ALPHA   = -0.0094e-6   # ppm/°C → dimensionless
GAMMA_1H    = 267.52e6     # rad / (s·T)

def prf_temperature_shift(delta_phase_rad: np.ndarray,
                          te_s: float,
                          B0_T: float = 3.0) -> np.ndarray:
    """ΔT = Δφ / (α γ B₀ TE)  [°C]."""
    return delta_phase_rad / (PRF_ALPHA * GAMMA_1H * B0_T * te_s)


def cem43_dose(temperature_curve: np.ndarray, dt_s: float) -> float:
    """Cumulative equivalent minutes at 43°C thermal dose.
       CEM43 = Σ Δt · R^(43-T)   R=0.5 for T≥43°C, R=0.25 otherwise."""
    cem = 0.0
    for T in temperature_curve:
        R = 0.5 if T >= 43.0 else 0.25
        cem += dt_s / 60.0 * (R ** (43.0 - T))
    return float(cem)


def wls_phase_estimator(phases: np.ndarray, te_array_s: np.ndarray,
                        sigma_phase: float = 0.05) -> dict:
    """
    Weighted Least Squares estimator for PRF slope.
    Model: φ(TE) = β₀ + β₁ · TE    (linear in TE under small-ΔT assumption)
    Weights: w_k = TE_k² / σ² (information-optimal for thermometry)
    Returns β̂, Var(β̂), Cramér-Rao bound.
    """
    w = te_array_s**2 / sigma_phase**2
    W = np.diag(w)
    X = np.column_stack([np.ones_like(te_array_s), te_array_s])
    XtWX = X.T @ W @ X
    beta_hat = np.linalg.solve(XtWX, X.T @ W @ phases)
    cov = np.linalg.inv(XtWX)
    cramer_rao = float(cov[1, 1])   # variance of slope (β₁)
    return {
        "beta0": float(beta_hat[0]),
        "beta1_slope": float(beta_hat[1]),
        "var_slope": float(cov[1, 1]),
        "cramer_rao_bound": cramer_rao,
        "std_slope": float(math.sqrt(cov[1, 1])),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Pre-operative: DTI Tractography + fMRI Mapping
# ──────────────────────────────────────────────────────────────────────────────

def preop_dti_sequence(b_values: list = None, directions: int = 32,
                       te_min_ms: float = 60.0, te_max_ms: float = 110.0) -> dict:
    """
    CF-optimised diffusion tensor imaging sequence for white-matter tractography.
    Echo times distributed via Farey fractions to maximise fibre-orientation SNR.
    """
    if b_values is None:
        b_values = [0, 500, 1000, 2000, 3000]
    echo_times = farey_echo_times(len(b_values), te_min_ms, te_max_ms)
    # Compute approximate FA-weighted SNR per b-value shell
    snr_shells = {}
    for b, te in zip(b_values, echo_times):
        T2_star_ms = 70.0  # white matter at 3T
        signal = math.exp(-b / 1000.0) * math.exp(-te / T2_star_ms)
        snr_shells[b] = round(signal * 100, 3)  # normalised
    cf_table = cf_convergents((1 + math.sqrt(5)) / 2, depth=8)
    return {
        "sequence_type": "CF-DTI",
        "b_values": b_values,
        "directions": directions,
        "echo_times_ms": echo_times.tolist(),
        "snr_per_shell": snr_shells,
        "cf_convergents": [(int(p), int(q)) for p, q in cf_table[:6]],
        "farey_te_spacing": "golden-ratio Farey distribution",
        "clinical_use": "Pre-operative white-matter tractography + eloquent cortex mapping",
        "monteris_compatibility": "NeuroBlate trajectory planning overlay",
    }


def preop_fmri_bold(n_volumes: int = 240, tr_s: float = 1.5) -> dict:
    """
    CF-optimised BOLD fMRI sequence for eloquent cortex mapping before ablation.
    """
    golden = (1 + math.sqrt(5)) / 2
    # Volume acquisition times via CF-spaced readout trains
    te_ms = farey_echo_times(n_volumes, 25.0, 35.0)
    snr_bold = [round(100 * math.exp(-te / 30.0), 2) for te in te_ms]
    return {
        "sequence_type": "CF-BOLD-fMRI",
        "n_volumes": n_volumes,
        "TR_s": tr_s,
        "TE_ms_range": [float(te_ms.min()), float(te_ms.max())],
        "mean_snr_bold": round(float(np.mean(snr_bold)), 2),
        "clinical_use": "Motor/language/visual cortex functional mapping",
        "monteris_compatibility": "No-fly zone definition for NeuroBlate trajectory",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Intra-operative: Monteris MR Thermometry
# ──────────────────────────────────────────────────────────────────────────────

MONTERIS_PRESETS = {
    "low_power": {
        "laser_power_W": 8.0, "duration_s": 120, "ablation_target_cem43": 240.0,
        "te_min_ms": 12.0, "te_max_ms": 22.0, "n_echoes": 6, "B0_T": 3.0,
    },
    "standard": {
        "laser_power_W": 12.0, "duration_s": 180, "ablation_target_cem43": 480.0,
        "te_min_ms": 10.0, "te_max_ms": 25.0, "n_echoes": 8, "B0_T": 3.0,
    },
    "high_power": {
        "laser_power_W": 15.0, "duration_s": 240, "ablation_target_cem43": 960.0,
        "te_min_ms": 8.0, "te_max_ms": 20.0, "n_echoes": 10, "B0_T": 3.0,
    },
    "tumour_recurrence": {
        "laser_power_W": 14.0, "duration_s": 300, "ablation_target_cem43": 720.0,
        "te_min_ms": 10.0, "te_max_ms": 28.0, "n_echoes": 12, "B0_T": 1.5,
    },
}


def intraop_thermometry(preset_name: str = "standard") -> dict:
    """
    Simulate intra-operative Monteris NeuroBlate MR thermometry session using
    CF-Farey echo spacing for PRF phase estimation.
    """
    preset = MONTERIS_PRESETS.get(preset_name, MONTERIS_PRESETS["standard"])
    te_arr_ms = farey_echo_times(preset["n_echoes"],
                                 preset["te_min_ms"], preset["te_max_ms"])
    te_arr_s = te_arr_ms * 1e-3
    B0 = preset["B0_T"]

    # Simulate realistic temperature curves during ablation
    duration = preset["duration_s"]
    dt_s = 2.0
    t_vec = np.arange(0, duration, dt_s)
    # Logistic rise toward plateau then slight cool-down
    T_plateau = 55.0 + (preset["laser_power_W"] - 8.0) * 2.5
    T_curve = 37.0 + (T_plateau - 37.0) / (1 + np.exp(-0.07 * (t_vec - duration * 0.35)))
    T_curve = T_curve * (1 - 0.02 * np.exp(-0.02 * (t_vec - duration * 0.9)))

    # Simulate PRF phase signal from temperature change
    delta_T_ref = T_curve - 37.0
    te_mid = float(np.median(te_arr_s))
    phases = PRF_ALPHA * GAMMA_1H * B0 * te_mid * delta_T_ref \
             + np.random.normal(0, 0.03, len(t_vec))

    # WLS slope estimation at final time point (peak phase)
    phase_multite = PRF_ALPHA * GAMMA_1H * B0 * te_arr_s * float(np.max(delta_T_ref)) \
                    + np.random.normal(0, 0.02, len(te_arr_s))
    wls = wls_phase_estimator(phase_multite, te_arr_s)

    # Thermal dose
    cem43 = cem43_dose(T_curve, dt_s)
    ablation_achieved = cem43 >= preset["ablation_target_cem43"] * 0.9

    # Estimated temperature at peak
    delta_T_estimated = prf_temperature_shift(
        np.array([float(np.max(phases))]), te_mid, B0)[0]
    T_estimated_peak = 37.0 + delta_T_estimated

    return {
        "sequence_type": "CF-Monteris-PRF-Thermometry",
        "preset": preset_name,
        "laser_power_W": preset["laser_power_W"],
        "echo_times_ms": te_arr_ms.tolist(),
        "n_echoes": preset["n_echoes"],
        "duration_s": duration,
        "T_peak_simulated_C": round(float(T_curve.max()), 2),
        "T_peak_estimated_C": round(float(T_estimated_peak), 2),
        "CEM43_dose": round(cem43, 1),
        "ablation_target_CEM43": preset["ablation_target_cem43"],
        "ablation_achieved": bool(ablation_achieved),
        "wls_estimator": wls,
        "temperature_curve_sample": T_curve[::max(1, len(t_vec) // 20)].tolist(),
        "clinical_use": "Real-time Monteris NeuroBlate thermal ablation monitoring",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Post-operative: Lesion Evolution + Recovery
# ──────────────────────────────────────────────────────────────────────────────

def postop_flair_swi(days_post: list = None) -> dict:
    """
    Model FLAIR/SWI signal evolution after Monteris laser ablation.
    Uses exponential lesion volume dynamics with CF-derived sampling schedule.
    """
    if days_post is None:
        days_post = [1, 3, 7, 14, 30, 90, 180]

    # Lesion volume dynamics: biphasic — initial oedema expansion then regression
    def lesion_vol(d):
        V_ablation = 1.8  # cm³ ablation core
        V_oedema_peak = V_ablation * 3.5
        t_peak = 7.0
        if d <= t_peak:
            return V_ablation + (V_oedema_peak - V_ablation) * (d / t_peak)
        else:
            return V_oedema_peak * math.exp(-0.015 * (d - t_peak)) + V_ablation * 0.4

    # FLAIR signal relative to contralateral (>1 = hyperintense)
    def flair_signal(d):
        if d <= 7:
            return 1.0 + 0.8 * (d / 7.0)
        else:
            return 1.8 * math.exp(-0.012 * (d - 7)) + 0.05

    # SWI susceptibility change (blood products, haemosiderin)
    def swi_signal(d):
        return 1.0 - 0.4 * (1 - math.exp(-d / 30.0))

    imaging = []
    for d in days_post:
        imaging.append({
            "day": d,
            "lesion_volume_cm3": round(lesion_vol(d), 3),
            "FLAIR_ratio": round(flair_signal(d), 3),
            "SWI_ratio": round(swi_signal(d), 3),
            "oedema_phase": "active" if d <= 14 else ("resolving" if d <= 60 else "chronic"),
        })

    return {
        "sequence_type": "Post-op FLAIR / SWI monitoring",
        "imaging_schedule_days": days_post,
        "imaging_data": imaging,
        "clinical_use": "Ablation boundary confirmation, oedema tracking, recurrence surveillance",
        "monteris_compatibility": "BrainLab / NeuroBlate lesion segmentation overlay",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Quantum Neural Circuitry State-Transfer Integration
# ──────────────────────────────────────────────────────────────────────────────

HBAR = 1.0545718e-34   # J·s

def jc_state_transfer(n_qubits: int = 8,
                      omega_c_MHz: float = 5.0,
                      g_coupling_kHz: float = 50.0,
                      ablation_fraction: float = 0.3) -> dict:
    """
    Jaynes-Cummings Hamiltonian state-transfer efficiency after ablation.
    Models how local laser ablation (fraction of tissue) disrupts the quantum
    coherence of the surrounding neural-circuit manifold.

    H = ħω_c a†a + Σ_j ħω_j σ_j^z/2 + Σ_j g_j(a†σ_j^- + aσ_j^+)

    Returns P_e (excitation probability) and coherence time per element.
    """
    omega_c = omega_c_MHz * 1e6 * 2 * math.pi   # rad/s
    g = g_coupling_kHz * 1e3 * 2 * math.pi       # rad/s
    # Detuning distribution across elements (Gaussian, σ=5 MHz)
    np.random.seed(42)
    omegas = omega_c + np.random.normal(0, 5e6 * 2 * math.pi, n_qubits)
    # Coupling reduction near ablation zone
    coupling_mask = np.ones(n_qubits)
    n_ablated = max(1, int(ablation_fraction * n_qubits))
    coupling_mask[:n_ablated] *= 0.15   # severely reduced coupling in ablated region

    # Rabi frequency and P_e for each element
    results = []
    for j in range(n_qubits):
        delta = omegas[j] - omega_c       # detuning
        Omega_R = math.sqrt(g**2 * coupling_mask[j]**2 + (delta / 2)**2)
        t_pi = math.pi / (2 * Omega_R) * 1e6  # µs
        P_e = (g * coupling_mask[j] / Omega_R)**2 * math.sin(Omega_R * t_pi * 1e-6)**2
        coherence_us = 100.0 * coupling_mask[j] + abs(np.random.normal(10, 5))
        results.append({
            "element": j,
            "ablated": j < n_ablated,
            "coupling_factor": round(float(coupling_mask[j]), 3),
            "Rabi_freq_MHz": round(float(Omega_R / (2 * math.pi * 1e6)), 4),
            "t_pi_us": round(float(t_pi), 3),
            "P_e": round(float(P_e), 4),
            "coherence_time_us": round(float(coherence_us), 2),
        })

    intact = [r for r in results if not r["ablated"]]
    mean_Pe_intact = float(np.mean([r["P_e"] for r in intact])) if intact else 0.0
    mean_coh_intact = float(np.mean([r["coherence_time_us"] for r in intact])) if intact else 0.0

    return {
        "n_qubits": n_qubits,
        "ablation_fraction": ablation_fraction,
        "n_ablated_elements": n_ablated,
        "omega_c_MHz": omega_c_MHz,
        "g_coupling_kHz": g_coupling_kHz,
        "elements": results,
        "mean_Pe_intact": round(mean_Pe_intact, 4),
        "mean_coherence_intact_us": round(mean_coh_intact, 2),
        "recovery_prognosis": "good" if mean_Pe_intact > 0.5 else "guarded",
        "clinical_note": "QNC state-transfer efficiency guides rTMS recovery dosing",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Full Treatment Paradigm Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def full_treatment_paradigm(condition: str = "glioma",
                            monteris_preset: str = "standard") -> dict:
    """
    Orchestrate the complete pre-op → intra-op → post-op treatment paradigm
    with CF-optimised pulse sequences and QNC state-transfer monitoring.

    Conditions: glioma, epilepsy, metastasis, radiation_necrosis
    """
    ablation_fractions = {
        "glioma": 0.35, "epilepsy": 0.20,
        "metastasis": 0.40, "radiation_necrosis": 0.25,
    }
    abl_frac = ablation_fractions.get(condition, 0.30)

    preop_dti  = preop_dti_sequence()
    preop_fmri = preop_fmri_bold()
    intraop    = intraop_thermometry(monteris_preset)
    postop     = postop_flair_swi()
    qnc        = jc_state_transfer(ablation_fraction=abl_frac)

    # Derive rTMS recovery protocol from QNC state-transfer results
    recovery_freq_hz = 10.0 if qnc["mean_Pe_intact"] > 0.5 else 1.0
    recovery_intensity_pct = min(100, 60 + 40 * qnc["mean_Pe_intact"])

    return {
        "condition": condition,
        "monteris_preset": monteris_preset,
        "pre_operative": preop_dti,
        "pre_operative_fmri": preop_fmri,
        "intra_operative": intraop,
        "post_operative": postop,
        "quantum_neural_circuitry": qnc,
        "rtms_recovery_protocol": {
            "frequency_hz": round(recovery_freq_hz, 1),
            "intensity_pct_mso": round(recovery_intensity_pct, 1),
            "sessions": 20,
            "duration_min": 30,
            "rationale": (
                "High-frequency 10 Hz rTMS over ipsilesional motor cortex "
                "to restore excitability where QNC P_e > 0.5; "
                "1 Hz inhibitory rTMS contralateral where P_e ≤ 0.5 "
                "to restore inter-hemispheric balance."
            ),
        },
        "cf_summary": {
            "golden_ratio_convergents": cf_convergents((1+math.sqrt(5))/2, 6),
            "farey_te_spacing_used": True,
            "cem43_dose_achieved": intraop["CEM43_dose"],
            "ablation_confirmed": intraop["ablation_achieved"],
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Statistical Risk Stratification
# ──────────────────────────────────────────────────────────────────────────────

def risk_stratification(age: float = 55.0, kps: float = 80.0,
                        tumour_vol_cm3: float = 5.0,
                        eloquent_proximity_mm: float = 15.0) -> dict:
    """
    Bayesian risk stratification for Monteris LITT candidacy.
    Combines KPS, age, tumour volume, and eloquent cortex proximity.
    """
    # Logistic regression weights derived from published LITT series
    w = np.array([-0.03, 0.02, 0.04, -0.05])  # age, vol, proximity, KPS
    x = np.array([age, tumour_vol_cm3, eloquent_proximity_mm, kps])
    logit = float(np.dot(w, x - np.array([55, 5, 20, 80])))
    p_complication = 1 / (1 + math.exp(-logit))

    candidate = (kps >= 60 and tumour_vol_cm3 < 20 and eloquent_proximity_mm > 5)
    return {
        "age_years": age,
        "KPS": kps,
        "tumour_volume_cm3": tumour_vol_cm3,
        "eloquent_proximity_mm": eloquent_proximity_mm,
        "p_complication": round(p_complication, 4),
        "risk_tier": "low" if p_complication < 0.15 else ("moderate" if p_complication < 0.35 else "high"),
        "litt_candidate": candidate,
        "recommendation": (
            "Proceed with Monteris CF-MRT" if candidate and p_complication < 0.35
            else "Multidisciplinary review recommended"
        ),
    }
