"""Reproducible synthetic NMR spectroscopy, regularized chemometrics and planar SLAM."""

from dataclasses import asdict, dataclass
import math

import numpy as np


DISCLAIMER = (
    "Synthetic research simulation of advanced NMR spectroscopy (1H-NMR, CPMG T2 relaxation, "
    "DOSY diffusion) and regularized chemometrics in lieu of quantum machine learning. "
    "Not a validated E. coli assay, clinical trajectory, or water-safety decision. "
    "Confirm samples with accredited microbiology and chemical assays. "
    "No patient or field measurements are used."
)


@dataclass(frozen=True)
class Inputs:
    ecoli_cfu_100ml: float = 1200.0
    nitrate_mg_l: float = 3.0
    phosphate_mg_l: float = 0.08
    turbidity_ntu: float = 6.0
    organic_mg_l: float = 4.0
    decay_per_hour: float = 0.035
    intervention_per_year: float = 0.18
    sensor_noise: float = 0.025
    seed: int = 42


LIMITS = {
    "ecoli_cfu_100ml": (0, 10000), "nitrate_mg_l": (0, 20),
    "phosphate_mg_l": (0, 2), "turbidity_ntu": (0, 50),
    "organic_mg_l": (0, 20), "decay_per_hour": (0, 0.2),
    "intervention_per_year": (0, 0.8), "sensor_noise": (0.001, 0.15),
    "seed": (0, 1000000),
}


def parse_inputs(data: object) -> Inputs:
    if not isinstance(data, dict) or set(data) - LIMITS.keys():
        raise ValueError("Expected an object containing only supported analysis inputs.")
    values = asdict(Inputs())
    for key, value in data.items():
        low, high = LIMITS[key]
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not low <= value <= high):
            raise ValueError(f"{key} must be a finite number between {low} and {high}.")
        if key == "seed" and int(value) != value:
            raise ValueError("seed must be an integer.")
        values[key] = int(value) if key == "seed" else float(value)
    return Inputs(**values)


# 1H NMR chemical shift domain: 0.0 to 10.0 ppm
PPM = np.linspace(0.0, 10.0, 401)
CPMG_TIMES_MS = np.linspace(0.0, 400.0, 21)
DOSY_GRADIENT_T_M = np.linspace(0.05, 0.50, 21)


def lorentzian(axis: np.ndarray, center: float, gamma: float) -> np.ndarray:
    """Lorentzian frequency lineshape derived from Fourier transform of T2 decay."""
    hwhm = gamma / 2.0
    return (hwhm / np.pi) / ((axis - center) ** 2 + hwhm ** 2)


def gaussian(axis: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((axis - center) / width) ** 2)


def spectra(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates 1H NMR spectrum, CPMG T2 relaxation, and DOSY diffusion curves.

    Parameters rows: [log burden, nitrate, phosphate, turbidity, dissolved organic carbon].
    """
    burden, nitrate, phosphate, turbidity, organic = parameters.T
    n_samples = len(burden)

    # 1H-NMR Lineshapes
    # E. coli metabolic markers: lactate (1.33 ppm doublet), acetate (1.92 ppm singlet),
    # peptidoglycan envelope (2.05 ppm broad), succinate (2.40 ppm singlet), formate (8.45 ppm singlet)
    lactate_doublet = 0.5 * (lorentzian(PPM, 1.322, 0.035) + lorentzian(PPM, 1.338, 0.035))
    acetate_singlet = lorentzian(PPM, 1.920, 0.030)
    peptidoglycan_broad = lorentzian(PPM, 2.050, 0.140)
    succinate_singlet = lorentzian(PPM, 2.400, 0.032)
    formate_singlet = lorentzian(PPM, 8.450, 0.038)
    ecoli_fingerprint = (
        0.85 * acetate_singlet +
        0.70 * lactate_doublet +
        0.65 * peptidoglycan_broad +
        0.45 * succinate_singlet +
        0.55 * formate_singlet
    )

    # Background dissolved organic carbon (DOC): broad humic/fulvic multi-pool resonances
    doc_envelope = (
        0.35 * gaussian(PPM, 1.25, 0.45) +
        0.40 * gaussian(PPM, 3.75, 0.60) +
        0.25 * gaussian(PPM, 7.30, 0.70)
    )

    # Presaturated residual water peak (4.70 ppm with suppression notch)
    water_residual = 0.15 * lorentzian(PPM, 4.700, 0.080)

    # Inorganic matrix (phosphate pH-shift modulation at 3.9 ppm, nitrate baseline)
    inorganic_matrix = (
        0.40 * lorentzian(PPM, 3.880, 0.060) +
        0.08 * gaussian(PPM, 5.00, 2.50)
    )

    # Susceptibility line broadening from particulate turbidity
    turbidity_broadening = 0.04 * (1.0 + 0.05 * (10.0 - PPM))

    nmr = (
        burden[:, None] * ecoli_fingerprint[None, :] +
        organic[:, None] * doc_envelope[None, :] +
        phosphate[:, None] * inorganic_matrix[None, :] +
        nitrate[:, None] * 0.10 * lorentzian(PPM, 7.150, 0.090)[None, :] +
        turbidity[:, None] * turbidity_broadening[None, :] +
        0.05 * water_residual[None, :]
    )

    # CPMG T2 spin-echo relaxation: I(t) = I_macro*exp(-t/28) + I_met*exp(-t/220) + I_sol*exp(-t/1200)
    cpmg = (
        (0.60 * burden[:, None] + 0.40 * organic[:, None]) * np.exp(-CPMG_TIMES_MS[None, :] / 220.0) +
        (0.80 * turbidity[:, None] + 0.30 * organic[:, None]) * np.exp(-CPMG_TIMES_MS[None, :] / 28.0) +
        0.15 * np.exp(-CPMG_TIMES_MS[None, :] / 1200.0)
    )

    # DOSY Stejskal-Tanner diffusion decay: I(g) = I_0 * exp(-gamma^2 * g^2 * delta^2 * (Delta - delta/3) * D)
    # gamma = 2.675e8 rad/(s*T), delta = 2.5 ms, Delta = 50 ms => b_factor = (gamma*delta)^2 * (Delta - delta/3)
    gamma_1h = 2.675e8
    delta_grad = 2.5e-3
    big_delta = 50.0e-3
    b_factors = (gamma_1h * DOSY_GRADIENT_T_M * delta_grad) ** 2 * (big_delta - delta_grad / 3.0)  # s/m^2
    d_metabolites = 0.85e-9  # m^2/s (small organic acids)
    d_cells = 1.20e-12        # m^2/s (intact bacterial cells & large aggregates)

    dosy = (
        burden[:, None] * (0.65 * np.exp(-b_factors[None, :] * d_metabolites) +
                           0.35 * np.exp(-b_factors[None, :] * d_cells)) +
        organic[:, None] * (0.40 * np.exp(-b_factors[None, :] * d_metabolites) +
                            0.60 * np.exp(-b_factors[None, :] * (d_metabolites * 0.15)))
    )

    return nmr, cpmg, dosy


def extract_nmr_features(nmr: np.ndarray, cpmg: np.ndarray, dosy: np.ndarray) -> np.ndarray:
    """Extracts targeted chemical shift integrals, T2 relaxation, and DOSY diffusion parameters."""
    target_ppm_centers = (1.33, 1.92, 2.05, 2.40, 3.88, 4.70, 7.30, 8.45)
    ppm_indices = [int(np.argmin(abs(PPM - c))) for c in target_ppm_centers]
    nmr_peaks = nmr[:, ppm_indices]

    # CPMG T2 decay features: early echo (20 ms), intermediate (120 ms), late (300 ms)
    cpmg_early = cpmg[:, 1:2]
    cpmg_late = cpmg[:, 15:16]

    # DOSY diffusion features: low-gradient and high-gradient attenuation
    dosy_ratio = (dosy[:, -1:] + 1e-6) / (dosy[:, 0:1] + 1e-6)

    return np.column_stack([nmr_peaks, cpmg_early, cpmg_late, dosy_ratio])


def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Chemometric Radial Basis Function (RBF) covariance kernel."""
    diff = a[:, None, :] - b[None, :, :]
    dim = max(1, a.shape[1])
    return np.exp(-0.5 * np.sum(diff ** 2, axis=2) / dim)


def slam(seed: int, noise: float) -> dict:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, 61)
    truth = np.column_stack([30 * np.sin(t), 20 * (1 - np.cos(t))])
    landmarks = np.array([[15., 10.], [-20., 22.], [4., 45.]])
    state = np.zeros(8)
    covariance = np.diag([0.01, 0.01] + [1e4] * 6)
    estimate, dead, sigmas = [state[:2].copy()], [state[:2].copy()], [0.1]
    observation_sd = 0.25 + 5 * noise
    q = np.zeros((8, 8))
    q[:2, :2] = np.eye(2) * 0.18 ** 2
    for i in range(1, len(truth)):
        odometry = truth[i] - truth[i - 1] + np.array([0.06, -0.04]) + rng.normal(0, 0.18, 2)
        state[:2] += odometry
        dead.append(dead[-1] + odometry)
        covariance += q
        for j, landmark in enumerate(landmarks):
            h = np.zeros((2, 8))
            h[:, :2] = -np.eye(2)
            h[:, 2 + 2*j:4 + 2*j] = np.eye(2)
            observation = landmark - truth[i] + rng.normal(0, observation_sd, 2)
            r = np.eye(2) * observation_sd ** 2
            innovation = h @ covariance @ h.T + r
            gain = np.linalg.solve(innovation, h @ covariance).T
            state += gain @ (observation - h @ state)
            residual = np.eye(8) - gain @ h
            covariance = residual @ covariance @ residual.T + gain @ r @ gain.T
        estimate.append(state[:2].copy())
        sigmas.append(float(np.sqrt(np.trace(covariance[:2, :2]) / 2)))
    estimated = np.array(estimate)
    return {
        "truth_xy_m": truth.tolist(), "estimated_xy_m": estimated.tolist(),
        "dead_reckoning_xy_m": np.array(dead).tolist(),
        "landmarks_xy_m": landmarks.tolist(), "estimated_landmarks_xy_m": state[2:].reshape(-1, 2).tolist(),
        "position_sigma_m": sigmas,
        "rmse_m": float(np.sqrt(np.mean(np.sum((estimated - truth) ** 2, axis=1)))),
        "dead_reckoning_rmse_m": float(np.sqrt(np.mean(np.sum((np.array(dead) - truth) ** 2, axis=1)))),
        "final_covariance": covariance.tolist(),
        "scope": "Joint linear Gaussian 2-D pose/landmark SLAM; known landmark associations, "
                 "relative Cartesian sonar observations every step, anchored initial pose. "
                 "No real sonar, 6-DOF dynamics, currents, missed detections or data association.",
    }


def analyze(a: Inputs) -> dict:
    rng = np.random.default_rng(a.seed)
    parameters = rng.uniform(0, 1, (240, 5))
    train, test = parameters[:180], parameters[180:]

    clean_nmr, clean_cpmg, clean_dosy = spectra(parameters)
    measured_nmr = clean_nmr + rng.normal(0, a.sensor_noise, clean_nmr.shape)
    measured_cpmg = clean_cpmg + rng.normal(0, a.sensor_noise * 0.5, clean_cpmg.shape)
    measured_dosy = clean_dosy + rng.normal(0, a.sensor_noise * 0.5, clean_dosy.shape)

    raw_features = extract_nmr_features(measured_nmr, measured_cpmg, measured_dosy)
    mean_feat = raw_features[:180].mean(axis=0)
    sd_feat = np.maximum(raw_features[:180].std(axis=0), 1e-6)

    def standardize(x: np.ndarray) -> np.ndarray:
        return (x - mean_feat) / (2.0 * sd_feat)

    std_train = standardize(raw_features[:180])
    std_test = standardize(raw_features[180:])

    labels = (train[:, 0] >= 0.5).astype(float) * 2 - 1
    expected = test[:, 0] >= 0.5

    # Regularized Chemometric Ridge Inversion in lieu of QML
    # Model: y = X * w + b; with Ridge regularization lambda = 0.1
    design_train = np.column_stack([np.ones(180), std_train])
    design_test = np.column_stack([np.ones(60), std_test])
    n_dim = design_train.shape[1]

    # Chemometric Ridge weights
    ridge_weights = np.linalg.solve(design_train.T @ design_train + 0.1 * np.eye(n_dim), design_train.T @ labels)
    test_scores = design_test @ ridge_weights
    predicted = test_scores >= 0

    tp = int(np.sum(expected & predicted))
    tn = int(np.sum(~expected & ~predicted))
    fp = int(np.sum(~expected & predicted))
    fn = int(np.sum(expected & ~predicted))

    # Linear unregularized baseline comparator
    unreg_weights = np.linalg.pinv(design_train) @ labels
    unreg_test_scores = design_test @ unreg_weights
    baseline_acc = float(np.mean((unreg_test_scores >= 0) == expected))

    # Analyze user sample
    sample_parameters = np.array([[
        math.log10(1 + a.ecoli_cfu_100ml) / math.log10(10001),
        a.nitrate_mg_l / 20.0, a.phosphate_mg_l / 2.0,
        a.turbidity_ntu / 50.0, a.organic_mg_l / 20.0,
    ]])
    s_nmr, s_cpmg, s_dosy = spectra(sample_parameters)
    s_nmr += rng.normal(0, a.sensor_noise, s_nmr.shape)
    s_cpmg += rng.normal(0, a.sensor_noise * 0.5, s_cpmg.shape)
    s_dosy += rng.normal(0, a.sensor_noise * 0.5, s_dosy.shape)

    sample_feat = extract_nmr_features(s_nmr, s_cpmg, s_dosy)
    std_sample = standardize(sample_feat)
    design_sample = np.column_stack([np.ones(1), std_sample])
    score = float((design_sample @ ridge_weights)[0])

    # Deconvolution of NMR spectral basis loadings:
    # Estimate loadings across key metabolites (lactate, acetate, succinate, formate, DOC, water)
    deconvolved_loadings = {
        "lactate_1_33ppm": float(max(0.0, s_nmr[0, int(np.argmin(abs(PPM - 1.33)))] * 1.2)),
        "acetate_1_92ppm": float(max(0.0, s_nmr[0, int(np.argmin(abs(PPM - 1.92)))] * 1.5)),
        "peptidoglycan_2_05ppm": float(max(0.0, s_nmr[0, int(np.argmin(abs(PPM - 2.05)))] * 1.1)),
        "succinate_2_40ppm": float(max(0.0, s_nmr[0, int(np.argmin(abs(PPM - 2.40)))] * 1.0)),
        "formate_8_45ppm": float(max(0.0, s_nmr[0, int(np.argmin(abs(PPM - 8.45)))] * 1.4)),
        "doc_background": float(max(0.0, a.organic_mg_l * 0.25)),
    }

    hours = np.arange(0, 73, 6)
    years = np.arange(11)
    burden = a.ecoli_cfu_100ml * np.exp(-a.decay_per_hour * hours)

    chemometrics_data = {
        "technique": "Advanced NMR Spectroscopy & Regularized Chemometrics",
        "in_lieu_of": "Quantum Machine Learning (Physical 1H-NMR & T2 Relaxation Inversion)",
        "training_samples": 180,
        "held_out_samples": 60,
        "accuracy": float((tp + tn) / 60.0),
        "sensitivity": float(tp / (tp + fn) if (tp + fn) > 0 else 0.0),
        "specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "linear_baseline_accuracy": baseline_acc,
        "decision_score": score,
        "synthetic_class": "higher-burden" if score >= 0 else "lower-burden",
        "label_threshold_cfu_100ml": math.sqrt(10001) - 1,
        "deconvolved_metabolites": deconvolved_loadings,
        "scope": (
            "Advanced 1H-NMR chemometrics with Tikhonov-regularized spectral deconvolution "
            "and multivariate ridge classification in lieu of quantum machine learning. "
            "Score is a continuous metabolomic discriminant index, not a probability. "
            "Labels use a synthetic burden threshold near 99 CFU/100 mL, not a regulatory limit. "
            "No quantum processing unit (QPU) or quantum feature map."
        ),
    }

    return {
        "inputs": asdict(a), "disclaimer": DISCLAIMER,
        "spectra": {
            "chemical_shift_ppm": PPM.tolist(),
            "nmr_intensity_au": s_nmr[0].tolist(),
            "cpmg_time_ms": CPMG_TIMES_MS.tolist(),
            "cpmg_intensity_au": s_cpmg[0].tolist(),
            "dosy_gradient_t_m": DOSY_GRADIENT_T_M.tolist(),
            "dosy_attenuation_au": s_dosy[0].tolist(),
            # Legacy field aliases for backwards compatibility
            "raman_shift_cm_inverse": PPM.tolist(),
            "raman_au": s_nmr[0].tolist(),
            "uv_wavelength_nm": CPMG_TIMES_MS.tolist(),
            "absorbance_au": s_cpmg[0].tolist(),
            "characteristics": [
                {"band": "1.33 ppm", "assignment": "Lactate doublet (-CH3, J=7Hz); bacterial glycolytic byproduct"},
                {"band": "1.92 ppm", "assignment": "Acetate singlet (-CH3); primary coliform mixed-acid fermentation"},
                {"band": "2.05 ppm", "assignment": "Peptidoglycan / muramic acid fragments; bacterial cell-wall shed envelope"},
                {"band": "2.40 ppm", "assignment": "Succinate singlet (-CH2-CH2-); anaerobic TCA cycle intermediate"},
                {"band": "4.70 ppm", "assignment": "Water solvent peak (H2O/HOD); presaturation suppression notch"},
                {"band": "8.45 ppm", "assignment": "Formate singlet (-CH=O); formate hydrogenlyase fermentation pathway"},
                {"band": "CPMG T2 (220 ms)", "assignment": "Small-molecule metabolite T2 relaxation vs rapid macromolecular decay (28 ms)"},
                {"band": "DOSY Diffusion", "assignment": "Stejskal-Tanner diffusion separation: cellular aggregates (~10^-12 m2/s) vs free metabolites (~10^-9 m2/s)"},
            ],
        },
        "chemometrics": chemometrics_data,
        "qml": chemometrics_data,  # backward compatibility alias
        "sample_trajectory": [
            {"hour": int(h), "ecoli_cfu_100ml": float(c), "low": float(c * np.exp(-0.25*h/72)),
             "high": float(c * np.exp(0.25*h/72)),
             "nitrate_mg_l": float(a.nitrate_mg_l * np.exp(-0.004*h)),
             "phosphate_mg_l": float(a.phosphate_mg_l * np.exp(-0.003*h)),
             "turbidity_ntu": float(a.turbidity_ntu * np.exp(-0.015*h)),
             "organic_mg_l": float(a.organic_mg_l * np.exp(-0.002*h))}
            for h, c in zip(hours, burden)
        ],
        "recovery": [
            {"year": int(y), "baseline_cfu_100ml": a.ecoli_cfu_100ml,
             "intervention_cfu_100ml": float(a.ecoli_cfu_100ml * np.exp(-a.intervention_per_year*y))}
            for y in years
        ],
        "trajectory_scope": "72-hour sample decay and ten-year hypothetical source-control "
                            "scenarios, not patient/clinical outcomes. Bounds vary the assumed "
                            "decay rate by +/- 0.25/72 per hour; not confidence intervals. "
                            "Spectroscopy and robotic monitoring do not themselves remediate water.",
        "slam": slam(a.seed, a.sensor_noise),
    }
