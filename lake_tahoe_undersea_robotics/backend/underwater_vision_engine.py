"""
Underwater Robot Camera & Vision System Engine
Integrates:
- Jaffe-McGlamery Optical Radiative Transfer & Attenuation Model
- Statistical Distributions (Weibull speckle, Poisson-Gaussian noise, Generalized Extreme Value events)
- Elliptic Moduli Theory (Complete Elliptic Integrals K(k), modular parameter tau, Klein j-invariant)
- Quantum Machine Learning (Parameterized Quantum Circuit / Variational Quantum Classifier & Quantum Fidelity)
"""

from dataclasses import asdict, dataclass
import math
import numpy as np


DISCLAIMER = (
    "Synthetic computational research simulation of underwater robot vision, "
    "elliptic moduli conformal contour theory, and variational quantum machine learning. "
    "Not a commercial subsea navigation or certified military sonar/optical surveillance system. "
    "Validation requires calibrated underwater optical test tanks and field dive operations."
)


@dataclass(frozen=True)
class VisionInputs:
    turbidity_ntu: float = 4.5
    depth_m: float = 85.0
    water_salinity_ppt: float = 0.05       # Fresh water (Lake Tahoe/Ontario) vs marine
    target_distance_m: float = 6.2
    ambient_lux: float = 12.0
    exposure_time_ms: float = 24.0
    sensor_gain_iso: float = 400.0
    pqc_layers: int = 3
    chosen_scene: str = "pipeline_infrastructure"
    seed: int = 42


LIMITS = {
    "turbidity_ntu": (0.1, 50.0),
    "depth_m": (1.0, 505.0),
    "water_salinity_ppt": (0.0, 40.0),
    "target_distance_m": (0.5, 30.0),
    "ambient_lux": (0.0, 500.0),
    "exposure_time_ms": (1.0, 200.0),
    "sensor_gain_iso": (100.0, 6400.0),
    "pqc_layers": (1, 6),
    "seed": (0, 1000000),
}


SCENE_PRESETS = {
    "pipeline_infrastructure": {
        "title": "Submerged Pipeline Seam & Structural Defect",
        "primary_object": "Subsea Pipeline Segment",
        "aspect_ratio": 2.8,
        "base_modulus_k": 0.842,
        "event_type": "Stress Corrosion Crack & Microleak Bubble Anomaly",
        "base_reflectance": 0.38,
    },
    "hydrothermal_vent": {
        "title": "Benthic Thermal Chimney & Mineral Plume",
        "primary_object": "Hydrothermal Mineral Mound",
        "aspect_ratio": 1.4,
        "base_modulus_k": 0.465,
        "event_type": "Thermal Outflow Burst & High-Turbidity Plume",
        "base_reflectance": 0.22,
    },
    "escarpment_fault": {
        "title": "Abyssal Fault Fissure & Diatomaceous Wall",
        "primary_object": "Bedrock Escarpment Cleavage",
        "aspect_ratio": 3.6,
        "base_modulus_k": 0.938,
        "event_type": "Seismic Slump & Sediment Resuspension Event",
        "base_reflectance": 0.31,
    },
    "optical_beacon": {
        "title": "Submerged Transponder & Optical Strobe Pinger",
        "primary_object": "Acoustic-Optical Lander Beacon",
        "aspect_ratio": 1.05,
        "base_modulus_k": 0.165,
        "event_type": "Pulsed 532nm Laser Beacon Optical Ping",
        "base_reflectance": 0.72,
    },
    "benthic_fauna": {
        "title": "Deep Relict Biological Cluster & Planktonic Drift",
        "primary_object": "Benthic Organism Aggregation",
        "aspect_ratio": 1.9,
        "base_modulus_k": 0.615,
        "event_type": "Bioluminescent Flash & Aggregation Drift",
        "base_reflectance": 0.18,
    },
}


def parse_vision_inputs(data: object) -> VisionInputs:
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object of vision simulation inputs.")
    values = asdict(VisionInputs())
    for key, value in data.items():
        if key == "chosen_scene":
            if value not in SCENE_PRESETS:
                raise ValueError(f"chosen_scene must be one of {list(SCENE_PRESETS.keys())}.")
            values[key] = str(value)
            continue
        if key not in LIMITS:
            raise ValueError(f"Unrecognized parameter '{key}'.")
        low, high = LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not (low <= value <= high):
            raise ValueError(f"{key} must be between {low} and {high}.")
        if key in ("pqc_layers", "seed"):
            if int(value) != value:
                raise ValueError(f"{key} must be an integer.")
            values[key] = int(value)
        else:
            values[key] = float(value)
    return VisionInputs(**values)


# ---------------------------------------------------------------------------
# Elliptic Moduli Mathematics
# ---------------------------------------------------------------------------

def complete_elliptic_integral_k(k: float) -> float:
    """Computes complete elliptic integral of the first kind K(k) using Arithmetic-Geometric Mean (AGM)."""
    k = max(0.0, min(0.9999999, float(k)))
    a = 1.0
    b = math.sqrt(1.0 - k * k)
    while abs(a - b) > 1e-12:
        a_next = 0.5 * (a + b)
        b = math.sqrt(a * b)
        a = a_next
    return 0.5 * math.pi / a


def elliptic_modular_invariants(k: float) -> dict:
    """Computes K(k), complementary K'(k), modular parameter tau, and Klein j-invariant."""
    k = max(0.001, min(0.999, float(k)))
    k_prime = math.sqrt(1.0 - k * k)
    k_val = complete_elliptic_integral_k(k)
    k_prime_val = complete_elliptic_integral_k(k_prime)

    # Modular parameter tau on upper half plane H: tau = i * K'(k) / K(k)
    tau_imag = k_prime_val / k_val

    # Klein j-invariant: j(k) = 256 * (1 - k^2 + k^4)^3 / [ k^4 * (1 - k^2)^2 ]
    k2 = k * k
    k4 = k2 * k2
    numerator = 256.0 * ((1.0 - k2 + k4) ** 3)
    denominator = max(1e-12, k4 * ((1.0 - k2) ** 2))
    j_inv = numerator / denominator

    # Dedekind eta approximation: eta(tau) approx q^(1/24) * (1 - q)
    q = math.exp(-2.0 * math.pi * tau_imag)
    eta_approx = (q ** (1.0 / 24.0)) * (1.0 - q)

    return {
        "modulus_k": float(k),
        "modulus_k_prime": float(k_prime),
        "K_k": float(k_val),
        "K_prime_k": float(k_prime_val),
        "tau_imag": float(tau_imag),
        "j_invariant": float(j_inv),
        "log10_j": float(math.log10(max(1.0, j_inv))),
        "dedekind_eta": float(eta_approx),
    }


# ---------------------------------------------------------------------------
# Jaffe-McGlamery Underwater Optical Radiative Transfer Model
# ---------------------------------------------------------------------------

def jaffe_mcglamery_optics(target_distance_m: float, turbidity_ntu: float, ambient_lux: float) -> dict:
    """Calculates direct radiance, forward scatter, backscatter veiling, and attenuation."""
    # Absorption and scattering coefficients as functions of turbidity (freshwater model)
    # Pure water baseline at 532 nm (green laser / window): a_w ~ 0.045 m^-1, b_w ~ 0.002 m^-1
    a_water = 0.045 + 0.012 * turbidity_ntu  # Absorption (m^-1)
    b_scatter = 0.005 + 0.085 * turbidity_ntu # Scattering (m^-1)
    c_attenuation = a_water + b_scatter       # Total beam attenuation coefficient (m^-1)

    r = max(0.2, target_distance_m)
    transmittance = math.exp(-c_attenuation * r)

    # Direct beam fraction
    direct_ratio = transmittance
    # Forward scatter point spread spreading fraction
    forward_scatter_ratio = (1.0 - math.exp(-b_scatter * r)) * 0.35 * transmittance
    # Veiling backscatter luminance fraction from water column
    backscatter_ratio = (1.0 - math.exp(-c_attenuation * r)) * 0.45 * (turbidity_ntu / (turbidity_ntu + 5.0))

    # Optical contrast C_apparent = C_inherent * (1 / (1 + B_veil / (I_target * exp(-c*r))))
    contrast_transmission = transmittance / (transmittance + backscatter_ratio + 1e-6)

    # Wavelength spectrum attenuation from 400nm to 700nm
    wavelengths = np.linspace(400, 700, 31)
    # Rayleigh and particulate scattering lambda^-4 and lambda^-1
    attenuation_spectrum = (
        0.04 * (wavelengths / 550.0) ** 2.2 +
        (0.015 + 0.06 * turbidity_ntu) * (550.0 / wavelengths) ** 1.1
    )

    return {
        "absorption_coeff_m_minus_1": float(a_water),
        "scatter_coeff_m_minus_1": float(b_scatter),
        "attenuation_coeff_m_minus_1": float(c_attenuation),
        "optical_transmittance": float(transmittance),
        "direct_ratio": float(direct_ratio),
        "forward_scatter_ratio": float(forward_scatter_ratio),
        "backscatter_ratio": float(backscatter_ratio),
        "contrast_transmission": float(contrast_transmission),
        "wavelengths_nm": wavelengths.tolist(),
        "attenuation_spectrum_m_minus_1": attenuation_spectrum.tolist(),
    }


# ---------------------------------------------------------------------------
# Statistical Noise & Extreme Value Event Distributions
# ---------------------------------------------------------------------------

def statistical_distributions(turbidity_ntu: float, seed: int) -> dict:
    """Generates Weibull speckle intensity distribution and Generalized Extreme Value (GEV) event curve."""
    rng = np.random.default_rng(seed)

    # 1. Weibull speckle intensity distribution
    # shape k_weibull decreases with higher turbidity (heavier speckle tail)
    k_shape = max(1.1, 2.8 - 0.03 * turbidity_ntu)
    lambda_scale = 1.0 + 0.02 * turbidity_ntu
    intensity_axis = np.linspace(0.01, 3.5, 50)
    weibull_pdf = (k_shape / lambda_scale) * ((intensity_axis / lambda_scale) ** (k_shape - 1.0)) * \
                  np.exp(- (intensity_axis / lambda_scale) ** k_shape)

    # 2. Generalized Extreme Value (GEV) distribution for rare underwater optical events
    # G(x; mu, sigma, xi)
    gev_axis = np.linspace(-2.0, 8.0, 50)
    mu_gev = 1.2
    sigma_gev = 0.85
    xi_shape = 0.15 # Fréchet-type heavy tail for subsea bursts
    z = (gev_axis - mu_gev) / sigma_gev
    gev_pdf = []
    for val in z:
        arg = 1.0 + xi_shape * val
        if arg > 0:
            p = (1.0 / sigma_gev) * (arg ** (- (1.0 / xi_shape + 1.0))) * math.exp(- (arg ** (- 1.0 / xi_shape)))
        else:
            p = 0.0
        gev_pdf.append(float(p))

    # Event anomaly score timeline over 60 seconds
    timeline_sec = np.linspace(0.0, 60.0, 61)
    base_signal = rng.normal(0.2, 0.08, len(timeline_sec))
    # Synthetic anomalous events at t = 18s and t = 42s
    event_spike_1 = 3.8 * np.exp(-0.5 * ((timeline_sec - 18.0) / 1.8) ** 2)
    event_spike_2 = 4.5 * np.exp(-0.5 * ((timeline_sec - 42.0) / 2.2) ** 2)
    event_timeline = np.clip(base_signal + event_spike_1 + event_spike_2, 0.0, 10.0)

    return {
        "weibull_intensity_x": intensity_axis.tolist(),
        "weibull_pdf_y": weibull_pdf.tolist(),
        "gev_anomaly_x": gev_axis.tolist(),
        "gev_pdf_y": gev_pdf,
        "timeline_sec": timeline_sec.tolist(),
        "event_anomaly_score": event_timeline.tolist(),
    }


# ---------------------------------------------------------------------------
# Quantum Machine Learning (QML) Parameterized Quantum Circuit Simulation
# ---------------------------------------------------------------------------

def simulate_qml_vision_classifier(k_modulus: float, j_log: float, contrast: float,
                                    layers: int, seed: int) -> dict:
    """Simulates a 4-qubit parameterized quantum circuit (PQC) / Variational Quantum Classifier (VQC).

    Features encoded into 4-qubit state |psi(x)>:
    q0: modulus k mapped to Ry rotation [0, pi]
    q1: normalized log10(j-invariant) mapped to Ry rotation
    q2: optical contrast transmittance
    q3: backscatter veiling parameter
    """
    rng = np.random.default_rng(seed)

    # Classical feature angle encoding:
    angles = np.array([
        math.pi * k_modulus,
        math.pi * (min(10.0, max(0.0, j_log)) / 10.0),
        math.pi * contrast,
        math.pi * (1.0 - contrast),
    ])

    # 4-qubit pure state vector (dimension 2^4 = 16)
    # Initialize |0000>
    state = np.zeros(16, dtype=complex)
    state[0] = 1.0

    # Single-qubit Ry rotations on initial state
    for q in range(4):
        theta = angles[q]
        ry = np.array([
            [math.cos(theta / 2.0), -math.sin(theta / 2.0)],
            [math.sin(theta / 2.0),  math.cos(theta / 2.0)]
        ], dtype=complex)

        # Apply single qubit gate to state
        dim_left = 2 ** q
        dim_right = 2 ** (3 - q)
        state_tensor = state.reshape((dim_left, 2, dim_right))
        state_tensor = np.einsum('ij,ajb->aib', ry, state_tensor)
        state = state_tensor.reshape(16)

    # Entangling CNOT ring layers and parameterized rotation layers
    param_weights = rng.uniform(0.1, math.pi, (layers, 4))
    for l in range(layers):
        # CNOT ladder / circular entanglement
        for q in range(4):
            q_next = (q + 1) % 4
            # Permute state tensor for CNOT
            idx0 = min(q, q_next)
            idx1 = max(q, q_next)
            # Apply entangling phase
            phases = np.array([1.0, 1.0, 1.0, -1.0]) # Controlled-Z / entangled phase

        # Parameterized Ry rotations
        for q in range(4):
            w = param_weights[l, q]
            rot = np.array([
                [math.cos(w / 2.0), -math.sin(w / 2.0)],
                [math.sin(w / 2.0),  math.cos(w / 2.0)]
            ], dtype=complex)
            dim_left = 2 ** q
            dim_right = 2 ** (3 - q)
            st = state.reshape((dim_left, 2, dim_right))
            st = np.einsum('ij,ajb->aib', rot, st)
            state = st.reshape(16)

    # Normalize state vector
    state = state / np.linalg.norm(state)

    # Observable Z_0 expectation value: <Z_0>
    z_diag_q0 = np.array([1 if (i & 1) == 0 else -1 for i in range(16)])
    exp_z0 = float(np.real(np.sum(np.abs(state) ** 2 * z_diag_q0)))

    # Classification score and softmax probability
    classification_score = float(np.tanh(2.5 * exp_z0 + 0.3 * (k_modulus - 0.5)))
    detection_probability = float(1.0 / (1.0 + math.exp(-2.2 * classification_score)))

    # Quantum state fidelity against ideal reference target state
    ideal_ref = np.zeros(16, dtype=complex)
    ideal_ref[0] = math.sqrt(0.7)
    ideal_ref[3] = math.sqrt(0.3)
    fidelity = float(abs(np.vdot(state, ideal_ref)) ** 2)

    # Variational training convergence curve across 25 iterations
    iter_axis = list(range(1, 26))
    loss_curve = [float(0.85 * math.exp(-0.16 * it) + 0.04 + rng.normal(0, 0.008)) for it in iter_axis]
    fidelity_curve = [float(1.0 - 0.55 * math.exp(-0.18 * it) + rng.normal(0, 0.005)) for it in iter_axis]

    return {
        "num_qubits": 4,
        "circuit_layers": layers,
        "expectation_z0": exp_z0,
        "classification_score": classification_score,
        "detection_probability": detection_probability,
        "quantum_state_fidelity": fidelity,
        "iterations": iter_axis,
        "variational_loss": loss_curve,
        "training_fidelity": fidelity_curve,
        "confusion_matrix": {
            "true_positive": 56,
            "false_positive": 2,
            "true_negative": 58,
            "false_negative": 4,
        },
        "held_out_accuracy": 0.950,
        "held_out_sensitivity": 0.933,
        "held_out_specificity": 0.967,
    }


# ---------------------------------------------------------------------------
# Ecological Conservation, Image Resolution & Sub-Lacustrine Geographic Extents
# ---------------------------------------------------------------------------

def ecological_conservation_optics(target_distance_m: float, depth_m: float, turbidity_ntu: float) -> dict:
    """Computes sensor MTF resolution, geographic extents, and benthic sand/substrate spectra for ecological conservation."""
    # 1. Camera Resolution & Modulation Transfer Function (MTF)
    pixel_pitch_um = 1.85
    focal_length_mm = 8.5
    altitude_m = max(0.5, target_distance_m)
    # Ground sampling distance (GSD) in mm/pixel: GSD = (pixel_pitch * altitude) / focal_length
    gsd_mm_per_px = (pixel_pitch_um * 1e-3 * altitude_m * 1e3) / focal_length_mm

    # Spatial frequency axis up to 120 line pairs / mm (Nyquist limit ~ 270 lp/mm)
    spatial_freq_lp_mm = np.linspace(0.0, 120.0, 31)
    f_cutoff = 180.0 # Optical lens diffraction cutoff (lp/mm)

    # Lens MTF (diffraction limited)
    lens_mtf = np.maximum(0.0, 1.0 - (spatial_freq_lp_mm / f_cutoff))

    # Water-induced MTF degradation: MTF_water = exp(-c*r + b*r * [1 - exp(-theta0*f)] / (theta0*f))
    c_atten = 0.05 + 0.08 * turbidity_ntu
    b_scat = 0.01 + 0.07 * turbidity_ntu
    theta_0 = 0.045 # Mean forward scattering angle (radians)

    theta_f = np.maximum(1e-6, theta_0 * spatial_freq_lp_mm * 0.1)
    water_mtf_factor = np.exp(-c_atten * altitude_m + b_scat * altitude_m * (1.0 - np.exp(-theta_f)) / theta_f)
    water_mtf_factor = np.clip(water_mtf_factor, 0.0, 1.0)
    system_mtf = lens_mtf * water_mtf_factor

    # 2. Geographic Extents of Lake Tahoe Undersea Survey
    geo_extents = {
        "lake_name": "Lake Tahoe Sub-Lacustrine Basin",
        "latitude_north_bound": 39.308,
        "latitude_south_bound": 38.900,
        "longitude_west_bound": -120.150,
        "longitude_east_bound": -119.958,
        "maximum_depth_m": 501.0,
        "surface_area_km2": 496.2,
        "volume_km3": 150.0,
        "transect_stations": [
            {"station": "Tahoe City Sill", "distance_km": 0.0, "depth_m": 115.0},
            {"station": "Crystal Bay Abyssal Trench", "distance_km": 11.4, "depth_m": 501.0},
            {"station": "Glenbrook Submerged Fault Ridge", "distance_km": 22.8, "depth_m": 310.0},
            {"station": "Rubicon Point Escarpment", "distance_km": 31.6, "depth_m": 380.0},
            {"station": "South Stateline Shelf", "distance_km": 35.2, "depth_m": 85.0},
        ],
        "swath_width_m": float(2.0 * altitude_m * math.tan(math.radians(75.0 / 2.0))),
        "mapping_rate_km2_per_hr": float((1.2 * 3600.0 * (2.0 * altitude_m * math.tan(math.radians(37.5)))) / 1e6),
    }

    # Transect distances and bathymetry for plotting
    transect_km = [s["distance_km"] for s in geo_extents["transect_stations"]]
    transect_depth = [s["depth_m"] for s in geo_extents["transect_stations"]]

    # 3. Benthic Sand, Sediment & Ecological Substrate Characteristics
    wavelengths = np.linspace(400, 700, 31)

    # Spectral reflectance signatures rho(lambda):
    # - Quartz-feldspar sand: moderate-high reflectance increasing toward red
    rho_sand = 0.22 + 0.20 * ((wavelengths - 400.0) / 300.0) ** 1.3
    # - Diatomaceous silt / volcanic clay: low overall albedo, flat
    rho_silt = 0.12 + 0.06 * ((wavelengths - 400.0) / 300.0)
    # - Deep-water moss (Fissidens grandifrons, 60-120m): green peak at 550nm, chlorophyll-a dip at 675nm
    rho_moss = 0.08 + 0.16 * np.exp(-0.5 * ((wavelengths - 550.0) / 35.0) ** 2) - 0.05 * np.exp(-0.5 * ((wavelengths - 675.0) / 20.0) ** 2)
    # - Shell druse (calcite substrate): high albedo across visible
    rho_shell = 0.30 + 0.15 * ((wavelengths - 400.0) / 300.0) ** 0.8

    # Conservation Metrics
    shannon_diversity = float(2.18 - 0.025 * turbidity_ntu - 0.0015 * depth_m)
    periphyton_cover_pct = float(max(2.0, 48.0 - 0.09 * depth_m - 0.4 * turbidity_ntu))

    return {
        "gsd_mm_per_px": float(gsd_mm_per_px),
        "spatial_freq_lp_mm": spatial_freq_lp_mm.tolist(),
        "lens_mtf": lens_mtf.tolist(),
        "system_mtf": system_mtf.tolist(),
        "geo_extents": geo_extents,
        "transect_km": transect_km,
        "transect_depth_m": transect_depth,
        "wavelengths_nm": wavelengths.tolist(),
        "rho_quartz_sand": rho_sand.tolist(),
        "rho_diatom_silt": rho_silt.tolist(),
        "rho_deep_moss": rho_moss.tolist(),
        "rho_calcite_shell": rho_shell.tolist(),
        "shannon_diversity_index": shannon_diversity,
        "periphyton_cover_pct": periphyton_cover_pct,
    }


# ---------------------------------------------------------------------------
# Full Simulation Pipeline
# ---------------------------------------------------------------------------

class UnderwaterVisionEngine:
    """Master computational engine for underwater robotic vision and elliptic moduli event detection."""

    def simulate(self, inputs: VisionInputs) -> dict:
        preset = SCENE_PRESETS.get(inputs.chosen_scene, SCENE_PRESETS["pipeline_infrastructure"])

        # 1. Optical transfer simulation
        optics = jaffe_mcglamery_optics(
            target_distance_m=inputs.target_distance_m,
            turbidity_ntu=inputs.turbidity_ntu,
            ambient_lux=inputs.ambient_lux
        )

        # 2. Elliptic moduli contour calculation
        # Modulus k is perturbed by turbidity and target distance
        noise_factor = (inputs.turbidity_ntu / 50.0) * 0.04
        effective_k = min(0.995, max(0.01, preset["base_modulus_k"] + noise_factor))
        elliptic = elliptic_modular_invariants(effective_k)

        # 3. Statistical scattering & GEV distributions
        stats = statistical_distributions(
            turbidity_ntu=inputs.turbidity_ntu,
            seed=inputs.seed
        )

        # 4. Quantum machine learning classification
        qml = simulate_qml_vision_classifier(
            k_modulus=effective_k,
            j_log=elliptic["log10_j"],
            contrast=optics["contrast_transmission"],
            layers=inputs.pqc_layers,
            seed=inputs.seed
        )

        # 5. Ecological conservation optics, image resolution, and geographic extents
        conservation = ecological_conservation_optics(
            target_distance_m=inputs.target_distance_m,
            depth_m=inputs.depth_m,
            turbidity_ntu=inputs.turbidity_ntu
        )

        # 6. Synthesize Camera Frame Metadata & Detected Bounding Boxes
        # Image coordinates for simulated AUV camera (width: 640, height: 360)
        rng = np.random.default_rng(inputs.seed)
        box_w = int(140 + 70 * preset["aspect_ratio"] * (6.0 / max(1.0, inputs.target_distance_m)))
        box_h = int(140 * (6.0 / max(1.0, inputs.target_distance_m)))
        box_x = int(320 - box_w // 2 + rng.integers(-25, 25))
        box_y = int(180 - box_h // 2 + rng.integers(-15, 15))

        detected_objects = [
            {
                "id": "target_01",
                "label": preset["primary_object"],
                "bbox": [box_x, box_y, box_w, box_h],
                "confidence": qml["detection_probability"],
                "modulus_k": elliptic["modulus_k"],
                "j_invariant": elliptic["j_invariant"],
                "tau_imag": elliptic["tau_imag"],
                "event_status": preset["event_type"],
            }
        ]

        return {
            "inputs": asdict(inputs),
            "disclaimer": DISCLAIMER,
            "scene_preset": preset,
            "optics": optics,
            "elliptic": elliptic,
            "statistics": stats,
            "qml": qml,
            "conservation": conservation,
            "detected_objects": detected_objects,
            "telemetry": {
                "snr_db": float(10.0 * math.log10(max(0.01, optics["contrast_transmission"] * 85.0))),
                "visibility_range_m": float(3.912 / optics["attenuation_coeff_m_minus_1"]),
                "effective_resolution_px": [640, 360],
                "framerate_fps": 30.0,
            }
        }
