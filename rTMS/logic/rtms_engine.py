import numpy as np
import time

def simulate_fea(resolution=20):
    """
    Simulates Finite Element Analysis (FEA) of cortical manifolds for E-field distribution.
    Returns a mock 2D grid of electromagnetic field intensities.
    """
    # Create a 2D grid representing cortical surface E-field
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Simulate a focal hotspot (e.g., motor cortex or DLPFC)
    hotspot_x, hotspot_y = np.random.uniform(-0.5, 0.5, 2)
    Z = np.exp(-((X - hotspot_x)**2 + (Y - hotspot_y)**2) / 0.1)
    
    # Add some structural noise (sulci/gyri geometry effects)
    noise = np.sin(X * 10) * np.cos(Y * 10) * 0.1
    Z = Z + noise
    
    return Z.tolist()

def simulate_bem(nodes=50):
    """
    Simulates Boundary Element Method (BEM) for head tissue boundaries to account
    for skin, skull, CSF, and gray matter conductivity differences.
    """
    # Return a 3D boundary geometry array (x,y,z) with scalar potentials
    theta = np.linspace(0, 2 * np.pi, nodes)
    phi = np.linspace(0, np.pi, nodes // 2)
    THETA, PHI = np.meshgrid(theta, phi)
    
    R = 1.0 + 0.05 * np.sin(4 * THETA) * np.cos(4 * PHI)  # Simple deformed sphere model
    
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)
    
    # Flatten it to just a list of vertices and a mock potential value
    vertices = []
    for i in range(len(X)):
        for j in range(len(X[0])):
            potential = np.exp(-Z[i][j]**2) * 10.0 # simulated BEM potential
            vertices.append({
                "x": round(float(X[i][j]), 3),
                "y": round(float(Y[i][j]), 3),
                "z": round(float(Z[i][j]), 3),
                "c": round(float(potential), 3)
            })
            
    # Sample down to keep payload manageable
    np.random.shuffle(vertices)
    return vertices[:500]

def optimize_protocol(condition="stroke"):
    """
    Uses statistical optimization techniques to dynamically adjust rTMS frequency,
    intensity, and pulse train duration for optimal neuromodulation based on condition.
    """
    iterations = []
    current_freq = 1.0
    current_intensity = 40.0

    if condition == "stroke":
        target_freq, target_intensity = 10.0, 80.0
    elif condition == "dementia":
        target_freq, target_intensity = 20.0, 100.0
    elif condition == "tremor":
        # Essential tremor: low-freq inhibitory rTMS targeting cerebello-thalamo-cortical loop
        # 1 Hz inhibitory rTMS over primary motor cortex / cerebellum
        target_freq, target_intensity = 1.0, 70.0
    else:
        target_freq, target_intensity = 10.0, 80.0

    for i in range(20):
        current_freq += (target_freq - current_freq) * 0.2 + np.random.normal(0, 0.3)
        current_intensity += (target_intensity - current_intensity) * 0.15 + np.random.normal(0, 1.0)

        fitness = 1.0 - (
            abs(target_freq - current_freq) / max(target_freq, 0.1) +
            abs(target_intensity - current_intensity) / target_intensity
        ) / 2

        iterations.append({
            "iteration": i + 1,
            "frequency_hz": round(max(0.5, current_freq), 2),
            "intensity_mso": round(max(10.0, current_intensity), 2),
            "fitness": round(max(0.0, float(fitness)), 4)
        })

    return {
        "final_parameters": iterations[-1],
        "convergence_trajectory": iterations,
        "protocol_summary": f"Optimized {condition.capitalize()} Protocol: {iterations[-1]['frequency_hz']} Hz at {iterations[-1]['intensity_mso']}% MSO"
    }

def run_full_simulation(condition="stroke"):
    """
    Simulates Google Cloud Server execution of complex BEM/FEA and Statistical Optimization.
    """
    return {
        "condition": condition,
        "timestamp": time.time(),
        "cloud_node": "gcp-us-central1-c-tensor-node",
        "optimization": optimize_protocol(condition),
        "fea_grid": simulate_fea(),
        "bem_mesh": simulate_bem()
    }


def get_tremor_clinical_data():
    """
    Returns Essential Tremor clinical data: tremor frequency spectrum,
    VIM thalamus targeting coordinates, session metrics, and evidence levels.
    """
    # Tremor frequency power spectrum (3–12 Hz band is pathological ET range)
    freqs = np.linspace(0, 30, 200)
    spectrum = (
        2.5 * np.exp(-((freqs - 6.5)**2) / 2.5) +   # ET peak ~4–8 Hz
        0.8 * np.exp(-((freqs - 18)**2) / 4.0) +      # Harmonic
        np.random.normal(0, 0.05, 200)                 # Baseline noise
    )
    spectrum = np.clip(spectrum, 0, None)

    # VIM thalamus 3D target cloud (MNI coordinates centred on bilateral VIM)
    n = 120
    vim_left  = np.random.normal([-14, -18, 4],  [1.5, 1.5, 1.5], (n, 3))
    vim_right = np.random.normal([ 14, -18, 4],  [1.5, 1.5, 1.5], (n, 3))
    vim_pts   = np.vstack([vim_left, vim_right])
    vim_intensity = np.exp(-np.sum((vim_pts - [0, -18, 4])**2, axis=1) / 30)

    # Session-by-session tremor reduction (% improvement over 10 sessions)
    sessions = list(range(1, 11))
    tremor_reduction = []
    val = 0.0
    for _ in sessions:
        val += np.random.uniform(4, 9)
        tremor_reduction.append(round(min(val, 72.0), 1))

    # TETRAS (The Essential Tremor Rating Assessment Scale) score per session
    tetras_start = 32
    tetras_scores = [round(max(8, tetras_start - i * np.random.uniform(1.8, 3.2)), 1)
                     for i in range(10)]

    # Clinical evidence levels by target region
    evidence = [
        {"region": "Cerebellum (ipsilateral)",    "level": "Level A", "pct": 88},
        {"region": "M1 Motor Cortex",              "level": "Level B", "pct": 74},
        {"region": "VIM Thalamus (indirect)",      "level": "Level B", "pct": 70},
        {"region": "Premotor Cortex",              "level": "Level C", "pct": 55},
        {"region": "Supplementary Motor Area",     "level": "Level C", "pct": 48},
    ]

    return {
        "tremor_spectrum": {
            "frequencies": [round(float(f), 2) for f in freqs],
            "power":        [round(float(p), 4) for p in spectrum]
        },
        "vim_target": {
            "x":         [round(float(v), 2) for v in vim_pts[:, 0]],
            "y":         [round(float(v), 2) for v in vim_pts[:, 1]],
            "z":         [round(float(v), 2) for v in vim_pts[:, 2]],
            "intensity": [round(float(v), 4) for v in vim_intensity]
        },
        "session_outcomes": {
            "sessions":          sessions,
            "tremor_reduction":  tremor_reduction,
            "tetras_scores":     tetras_scores
        },
        "clinical_evidence":  evidence,
        "recommended_protocol": {
            "target":         "Ipsilateral Cerebellum + Contralateral M1",
            "frequency_hz":   1,
            "intensity_mso":  70,
            "pulses_session": 1200,
            "sessions_total": 10,
            "inter_train_s":  10,
            "coil_type":      "Figure-8 (70 mm)",
            "pulse_type":     "Monophasic"
        }
    }


def get_equipment_list():
    """
    Returns a comprehensive list of rTMS equipment with clinical operating characteristics.
    """
    equipment = [
        {
            "id": "EQ-001",
            "name": "MagVenture MagPro X100",
            "category": "Stimulator Unit",
            "description": "High-performance biphasic/monophasic TMS stimulator with cTBS and TBS capabilities.",
            "specs": {
                "Max Output (% MSO)": "100%",
                "Peak E-Field (V/m)": "220",
                "Pulse Width (µs)": "280",
                "Frequency Range (Hz)": "0.1 – 100",
                "Max Continuous Duty Cycle": "50%",
                "Cooling System": "Active liquid cooling",
                "Power Supply Voltage (V)": "200 – 240 VAC",
                "Weight (kg)": "32"
            },
            "operating_characteristics": {
                "op_temp_c": 25,
                "max_temp_c": 40,
                "efficiency_pct": 92,
                "heat_dissipation_w": 180,
                "emi_shielding_db": 45
            }
        },
        {
            "id": "EQ-002",
            "name": "Magstim Horizon 3.0",
            "category": "Stimulator Unit",
            "description": "Next-gen triple-pulse TMS system with integrated neuronavigation readiness.",
            "specs": {
                "Max Output (% MSO)": "100%",
                "Peak E-Field (V/m)": "200",
                "Pulse Width (µs)": "290",
                "Frequency Range (Hz)": "0.1 – 50",
                "Max Continuous Duty Cycle": "40%",
                "Cooling System": "Forced air + heat sink",
                "Power Supply Voltage (V)": "110 – 240 VAC",
                "Weight (kg)": "28"
            },
            "operating_characteristics": {
                "op_temp_c": 22,
                "max_temp_c": 38,
                "efficiency_pct": 88,
                "heat_dissipation_w": 150,
                "emi_shielding_db": 42
            }
        },
        {
            "id": "EQ-003",
            "name": "Figure-8 Coil (70mm Air-Cooled)",
            "category": "Stimulation Coil",
            "description": "Standard focal coil for precise cortical targeting in stroke motor rehab therapy.",
            "specs": {
                "Coil Diameter (mm)": "70",
                "Inductance (µH)": "16.4",
                "Resistance (mΩ)": "105",
                "Max Surface Temperature (°C)": "41",
                "Focal Depth (mm)": "20 – 35",
                "Max Repetition Rate (Hz)": "30",
                "Cooling System": "Natural convection",
                "Weight (g)": "320"
            },
            "operating_characteristics": {
                "op_temp_c": 36,
                "max_temp_c": 41,
                "efficiency_pct": 85,
                "heat_dissipation_w": 40,
                "emi_shielding_db": 30
            }
        },
        {
            "id": "EQ-004",
            "name": "H7 Deep TMS Coil (Brainsway)",
            "category": "Stimulation Coil",
            "description": "H-coil geometry enabling bilateral deep prefrontal cortex activation — ideal for dementia and depression.",
            "specs": {
                "Coil Geometry": "H-shaped (bilateral)",
                "Focal Depth (mm)": "50 – 70",
                "Max Surface Temperature (°C)": "43",
                "Inductance (µH)": "22.1",
                "Resistance (mΩ)": "130",
                "Max Repetition Rate (Hz)": "20",
                "Cooling System": "Fluid-cooled helmet insert",
                "Weight (g)": "850"
            },
            "operating_characteristics": {
                "op_temp_c": 38,
                "max_temp_c": 43,
                "efficiency_pct": 78,
                "heat_dissipation_w": 70,
                "emi_shielding_db": 28
            }
        },
        {
            "id": "EQ-005",
            "name": "Localite TMS Navigator 4.0",
            "category": "Neuronavigation System",
            "description": "Real-time optical tracking neuronavigation system synced with patient MRI for precise coil targeting.",
            "specs": {
                "Tracking Technology": "Infrared Optical",
                "Spatial Accuracy (mm)": "< 1.5",
                "Update Rate (Hz)": "60",
                "MRI Compatibility": "T1 / T2 / FLAIR",
                "Coil Interfaces": "Universal (Magstim, MagVenture, Deymed)",
                "Display Resolution": "4K",
                "OS": "Windows 11 Embedded"
            },
            "operating_characteristics": {
                "op_temp_c": 23,
                "max_temp_c": 35,
                "efficiency_pct": 97,
                "heat_dissipation_w": 25,
                "emi_shielding_db": 55
            }
        },
        {
            "id": "EQ-006",
            "name": "64-Channel EEG Amplifier (BrainProducts)",
            "category": "EEG Monitoring",
            "description": "High-resolution TMS-compatible EEG system for real-time cortical excitability monitoring.",
            "specs": {
                "Channels": "64 + 8 AUX",
                "Sampling Rate (kHz)": "25",
                "Input Impedance (MΩ)": "> 1000",
                "CMRR (dB)": "130",
                "ADC Resolution (bit)": "24",
                "TMS Artifact Recovery (ms)": "< 5",
                "Bandwidth (Hz)": "DC – 5000"
            },
            "operating_characteristics": {
                "op_temp_c": 22,
                "max_temp_c": 35,
                "efficiency_pct": 99,
                "heat_dissipation_w": 12,
                "emi_shielding_db": 60
            }
        },
        {
            "id": "EQ-007",
            "name": "Robotic Coil Positioning Arm (RPA-3)",
            "category": "Positioning System",
            "description": "6 DOF robotic arm for automated, reproducible coil placement with < 1mm repeatability.",
            "specs": {
                "Degrees of Freedom": "6",
                "Repeatability (mm)": "0.8",
                "Max Payload (kg)": "5",
                "Reach (mm)": "900",
                "Control Interface": "USB 3.0 / LAN",
                "Force Sensor": "6-axis, 0.01 N resolution",
                "Safety Standard": "ISO 13849 PLd"
            },
            "operating_characteristics": {
                "op_temp_c": 21,
                "max_temp_c": 40,
                "efficiency_pct": 95,
                "heat_dissipation_w": 55,
                "emi_shielding_db": 35
            }
        },
        {
            "id": "EQ-008",
            "name": "GCP rTMS Cloud Processing Node",
            "category": "Cloud Infrastructure",
            "description": "Tensor Processing Unit (TPU) cluster for real-time FEA/BEM statistical optimization bursts.",
            "specs": {
                "Node Type": "n2-highmem-32",
                "vCPUs": "32",
                "RAM (GB)": "256",
                "TPU Version": "v4",
                "Network Bandwidth (Gbps)": "100",
                "Region": "us-central1-c",
                "Latency (ms)": "< 12",
                "SLA Uptime": "99.99%"
            },
            "operating_characteristics": {
                "op_temp_c": 20,
                "max_temp_c": 28,
                "efficiency_pct": 98,
                "heat_dissipation_w": 850,
                "emi_shielding_db": 70
            }
        },
        {
            "id": "EQ-009",
            "name": "Boston Scientific Vercise Genus",
            "category": "Neuromodulation Device (DBS)",
            "description": "Directional Deep Brain Stimulation system with multiple independent current control, designed for precise targeting in essential tremor and advanced research in dementia cure and stroke repair.",
            "specs": {
                "Lead Type": "Directional 8-contact",
                "Stimulation Type": "Multiple Independent Current Control (MICC)",
                "Pulse Width (µs)": "10 – 400",
                "Frequency Range (Hz)": "2 – 255",
                "Current Output (mA)": "0 – 25.5",
                "MRI Compatibility": "1.5T and 3T Full Body",
                "Battery Type": "Rechargeable (Vercise Genus R16) / Non-Rechargeable (P16)"
            },
            "operating_characteristics": {
                "op_temp_c": 37,
                "max_temp_c": 41,
                "efficiency_pct": 98,
                "heat_dissipation_w": 2,
                "emi_shielding_db": 65
            }
        }
    ]
    return equipment


# ─────────────────────────────────────────────────────────────────────────────
# Treatment Paradigm Engine
# Stage Gating  ·  Hebbian-DBS Amplification  ·  Continued Fractions
# ─────────────────────────────────────────────────────────────────────────────

def _continued_fraction_convergents(target_freq, depth=8):
    """
    Expand target_freq as a continued fraction and return the first `depth`
    convergents.  These are used as candidate stimulation frequencies whose
    rational approximations minimise phase-locking artefacts.

      f = a0 + 1/(a1 + 1/(a2 + ...))

    Returns list of dicts {iteration, a_k, p/q, approx, error_pct}.
    """
    x = float(target_freq)
    a_list, p, q = [], [1, 0], [0, 1]
    convergents = []
    for k in range(depth):
        a = int(x)
        a_list.append(a)
        p_k = a * p[-1] + p[-2]
        q_k = a * q[-1] + q[-2]
        approx = p_k / q_k if q_k != 0 else float('inf')
        err = abs(approx - target_freq) / target_freq * 100
        convergents.append({
            "iteration": k,
            "a_k": a,
            "numerator": int(p_k),
            "denominator": int(q_k),
            "approx_freq": round(approx, 6),
            "error_pct": round(float(err), 6)
        })
        p.append(p_k); q.append(q_k)
        frac = x - a
        if abs(frac) < 1e-10:
            break
        x = 1.0 / frac
    return convergents


def _hebbian_amplification(n_sessions=10, eta=0.04, dbs_gain=1.6, noise_std=0.05):
    """
    Simulate Hebbian synaptic weight amplification coupled with DBS bursting.

    Hebbian rule:   Δwᵢⱼ = η · xᵢ(t) · xⱼ(t)
    DBS coupling:   xᵢ(t) += dbs_gain · sin(2π f_dbs t)   (burst injection)
    Decay:          w(t+1) = w(t) · (1 - λ) + Δw           (λ = forgetting)

    Returns per-session pre/post firing rates, synaptic weights,
    and DBS burst amplitudes.
    """
    lam = 0.02          # synaptic forgetting rate
    f_dbs = 130         # DBS carrier frequency (Hz), typical STN/VIM target
    t = np.linspace(0, 1, 200)   # 1 second time window per session

    sessions, weights, pre_rates, post_rates, dbs_bursts = [], [], [], [], []
    w = 0.1  # initial synaptic weight

    for s in range(n_sessions):
        # Pre-synaptic firing: baseline cortical oscillation + noise
        pre = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * 5 * t)
                      + np.random.normal(0, noise_std, len(t)), 0, 1)

        # DBS burst injection into post-synaptic neuron
        dbs_burst = dbs_gain * np.abs(np.sin(2 * np.pi * f_dbs * t)) * (s / n_sessions)
        post = np.clip(pre * w + dbs_burst * 0.3
                       + np.random.normal(0, noise_std, len(t)), 0, 1)

        # Hebbian update
        delta_w = eta * np.mean(pre * post)
        w = w * (1 - lam) + delta_w
        w = float(np.clip(w, 0, 1))

        sessions.append(s + 1)
        weights.append(round(w, 4))
        pre_rates.append(round(float(np.mean(pre)), 4))
        post_rates.append(round(float(np.mean(post)), 4))
        dbs_bursts.append(round(float(np.mean(dbs_burst)), 4))

    return {
        "sessions":    sessions,
        "weights":     weights,
        "pre_rates":   pre_rates,
        "post_rates":  post_rates,
        "dbs_bursts":  dbs_bursts
    }


def _stage_gates(condition):
    """
    Optimal stage-gating paradigm.  Three therapeutic phases gated by
    biomarker thresholds.

    Gate function:   G(t) = H(m(t) − θ_gate)
    where H is the Heaviside step and m(t) is the outcome metric at session t.

    Phase I  — rTMS induction        (sessions 1–N₁)
    Phase II — rTMS + DBS integration (sessions N₁+1–N₂)
    Phase III— DBS maintenance        (sessions N₂+1–N₃)
    """
    if condition == "stroke":
        N1, N2, N3 = 5, 10, 15
        metric_name = "Motor Recovery Score (Fugl-Meyer)"
        theta = [40, 65, 85]           # gate thresholds
        dbs_target = "Globus Pallidus internus (GPi)"
        dbs_freq_hz, dbs_pw_us = 130, 60
    elif condition == "dementia":
        N1, N2, N3 = 6, 12, 18
        metric_name = "ADAS-Cog Score (inverted %)"
        theta = [30, 55, 75]
        dbs_target = "Nucleus Basalis of Meynert (NBM) / Fornix"
        dbs_freq_hz, dbs_pw_us = 80, 90
    else:  # tremor
        N1, N2, N3 = 4, 8, 12
        metric_name = "TETRAS Improvement (%)"
        theta = [25, 50, 70]
        dbs_target = "Ventral Intermediate Nucleus (VIM) / STN"
        dbs_freq_hz, dbs_pw_us = 185, 60

    total = N3
    sessions = list(range(1, total + 1))
    metric = []
    phases = []
    val = 0.0
    for s in sessions:
        noise = np.random.normal(0, 2.5)
        if s <= N1:
            val += np.random.uniform(5, 9) + noise
            phase = "I — rTMS Induction"
        elif s <= N2:
            val += np.random.uniform(6, 11) + noise   # DBS adds amplification
            phase = "II — rTMS + DBS Integration"
        else:
            val += np.random.uniform(2, 5) + noise    # DBS maintenance plateau
            phase = "III — DBS Maintenance"
        metric.append(round(float(np.clip(val, 0, 100)), 1))
        phases.append(phase)

    # Gate events (session at which each threshold is first crossed)
    gate_events = []
    for i, thr in enumerate(theta):
        crossed = next((s for s, m in zip(sessions, metric) if m >= thr), None)
        gate_events.append({"phase": i + 1, "threshold": thr,
                             "session_crossed": crossed})

    return {
        "sessions":    sessions,
        "metric":      metric,
        "metric_name": metric_name,
        "phases":      phases,
        "N1": N1, "N2": N2, "N3": N3,
        "gate_thresholds": theta,
        "gate_events": gate_events,
        "dbs_target":  dbs_target,
        "dbs_freq_hz": dbs_freq_hz,
        "dbs_pw_us":   dbs_pw_us
    }


# ─────────────────────────────────────────────────────────────────────────────
# Long-Term Dementia Care — Cortical Surface Geodesics & BEM Simulation
# ─────────────────────────────────────────────────────────────────────────────

def _cortical_surface_geodesics(n_paths=12, resolution=80):
    """
    Compute geodesic paths on a simulated cortical surface manifold.
    Models the shortest paths between cortical ROIs on a curved surface
    parameterised as a Riemannian manifold S ⊂ R³.

    Geodesic equation:  d²xᵅ/dt² + Γᵅ_βγ (dxᵝ/dt)(dxᵞ/dt) = 0

    Returns surface mesh + geodesic path coordinates for visualisation.
    """
    # Generate cortical surface as a deformed sphere (gyri/sulci)
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution // 2)
    U, V = np.meshgrid(u, v)

    R = 1.0 + 0.08 * np.sin(5 * U) * np.cos(4 * V) + 0.05 * np.sin(8 * U + 3 * V)
    X = R * np.sin(V) * np.cos(U)
    Y = R * np.sin(V) * np.sin(U)
    Z = R * np.cos(V)

    # Compute geodesic paths between random cortical ROI pairs
    geodesics = []
    roi_labels = [
        "DLPFC-L", "DLPFC-R", "Hippocampus-L", "Hippocampus-R",
        "Entorhinal-L", "Entorhinal-R", "Precuneus", "PCC",
        "Temporal Pole-L", "Temporal Pole-R", "Parietal-L", "Parietal-R"
    ]
    for i in range(n_paths):
        t_param = np.linspace(0, 1, 40)
        u0, v0 = np.random.uniform(0.3, 5.8), np.random.uniform(0.3, 2.8)
        u1, v1 = np.random.uniform(0.3, 5.8), np.random.uniform(0.3, 2.8)

        # Linear interpolation on parameter space + geodesic curvature perturbation
        u_path = u0 + (u1 - u0) * t_param + 0.15 * np.sin(np.pi * t_param) * np.random.randn()
        v_path = v0 + (v1 - v0) * t_param + 0.1 * np.sin(np.pi * t_param) * np.random.randn()

        r_path = 1.0 + 0.08 * np.sin(5 * u_path) * np.cos(4 * v_path)
        gx = r_path * np.sin(v_path) * np.cos(u_path)
        gy = r_path * np.sin(v_path) * np.sin(u_path)
        gz = r_path * np.cos(v_path)

        # Compute arc-length (geodesic distance)
        ds = np.sqrt(np.diff(gx)**2 + np.diff(gy)**2 + np.diff(gz)**2)
        arc_length = float(np.sum(ds))

        geodesics.append({
            "id": i,
            "from_roi": roi_labels[i % len(roi_labels)],
            "to_roi": roi_labels[(i + n_paths // 2) % len(roi_labels)],
            "x": [round(float(v), 4) for v in gx],
            "y": [round(float(v), 4) for v in gy],
            "z": [round(float(v), 4) for v in gz],
            "arc_length": round(arc_length, 4)
        })

    return {
        "surface": {
            "x": X.tolist(),
            "y": Y.tolist(),
            "z": Z.tolist()
        },
        "geodesics": geodesics
    }


def _dementia_bem_simulation(n_layers=4, resolution=30, include_dbs=True):
    """
    Boundary Element Method simulation for dementia-specific head model.
    Models concentric tissue boundaries (scalp, skull, CSF, grey matter)
    and computes rTMS-induced electric field attenuation at each layer.
    Also models Deep Brain Stimulation (DBS) outwards field from the newly added
    Boston Scientific Vercise Genus device.
    """
    conductivities = {
        "Scalp": 0.33,
        "Skull": 0.0042,
        "CSF": 1.79,
        "Grey Matter": 0.33,
        "Deep Brain (DBS Target)": 0.45
    }
    radii = [1.0, 0.92, 0.87, 0.80, 0.25]
    
    layers = []
    for idx, (name, sigma) in enumerate(conductivities.items()):
        r = radii[idx]
        theta = np.linspace(0, 2 * np.pi, resolution)
        phi = np.linspace(0, np.pi, resolution // 2)
        TH, PH = np.meshgrid(theta, phi)

        if name == "Grey Matter":
            R_layer = r + 0.04 * np.sin(6 * TH) * np.cos(5 * PH) - 0.02
        elif name == "Deep Brain (DBS Target)":
            R_layer = r + 0.01 * np.sin(2 * TH) * np.cos(2 * PH)
        else:
            R_layer = r + 0.02 * np.sin(3 * TH) * np.cos(3 * PH)

        X = R_layer * np.sin(PH) * np.cos(TH)
        Y = R_layer * np.sin(PH) * np.sin(TH)
        Z = R_layer * np.cos(PH)

        # Baseline rTMS incoming potential
        potential_rtms = sigma * np.exp(-Z**2 / 0.5) * (1 + 0.3 * np.sin(4 * TH))
        
        # DBS out-radiating potential from center (z=0, x=0, y=0)
        # The Boston Scientific Vercise Genus (MICC) delivers highly directional currents
        dist_from_center = np.sqrt(X**2 + Y**2 + Z**2)
        dbs_potential = 0.0
        if include_dbs:
            # High intensity near center (directional field via MICC tuning)
            dbs_potential = 1.5 * np.exp(-dist_from_center / 0.3) * (1 + 0.5 * np.cos(PH))

        potential = potential_rtms + dbs_potential

        layers.append({
            "name": name,
            "conductivity": sigma,
            "radius": r,
            "x": X.tolist(),
            "y": Y.tolist(),
            "z": Z.tolist(),
            "potential": potential.tolist()
        })

    depths = np.linspace(0, 1, 50)
    attenuation = []
    for d in depths:
        v = 100.0
        for idx, (name, sigma) in enumerate(conductivities.items()):
            boundary = radii[idx]
            if d >= (1 - boundary):
                v *= np.exp(-sigma * (d - (1 - boundary)) * 5)
        # Add DBS amplification deep
        if include_dbs and d > 0.6:
            v += 45.0 * np.exp(-(d - 0.75)**2 / 0.05)
            
        attenuation.append(round(float(v), 2))

    return {
        "layers": layers,
        "attenuation": {
            "depths": [round(float(d), 3) for d in depths],
            "field_pct": attenuation
        }
    }


def get_dementia_longterm_care():
    """
    Returns comprehensive long-term dementia care data including:
    - Clinical protocols for staged care
    - Cortical surface geodesic analysis
    - BEM tissue simulation for rTMS targeting
    - Smart aging biomarker tracking
    - Treatment optimization metrics
    """
    # Longitudinal cognitive tracking (ADAS-Cog, MMSE, MoCA)
    months = list(range(1, 25))
    adas_baseline = 35
    mmse_baseline = 22
    moca_baseline = 18

    adas_scores, mmse_scores, moca_scores = [], [], []
    a, m, mc = float(adas_baseline), float(mmse_baseline), float(moca_baseline)
    for mo in months:
        # rTMS treatment effect: gradual improvement with plateau
        a -= np.random.uniform(0.3, 1.2) * np.exp(-mo / 20)
        m += np.random.uniform(0.1, 0.6) * np.exp(-mo / 18)
        mc += np.random.uniform(0.1, 0.5) * np.exp(-mo / 22)
        adas_scores.append(round(float(np.clip(a, 8, 40)), 1))
        mmse_scores.append(round(float(np.clip(m, 18, 30)), 1))
        moca_scores.append(round(float(np.clip(mc, 14, 30)), 1))

    # Smart aging biomarkers
    biomarkers = {
        "months": months,
        "amyloid_pet_suvr": [round(1.4 - 0.015 * mo + np.random.normal(0, 0.02), 3) for mo in months],
        "tau_pet_suvr": [round(1.8 - 0.01 * mo + np.random.normal(0, 0.03), 3) for mo in months],
        "hippocampal_volume_ml": [round(3.2 + 0.008 * mo + np.random.normal(0, 0.02), 3) for mo in months],
        "cortical_thickness_mm": [round(2.5 + 0.005 * mo + np.random.normal(0, 0.01), 3) for mo in months]
    }

    # Clinical protocol stages for long-term dementia care
    protocols = [
        {
            "stage": "Stage I — Early Intervention",
            "duration": "Months 1–6",
            "target": "Bilateral DLPFC",
            "frequency_hz": 20,
            "intensity_mso": 80,
            "sessions_per_week": 5,
            "pulses_session": 3000,
            "coil": "H7 Deep TMS Coil",
            "adjunct": "Cognitive rehabilitation therapy (CRT)",
            "biomarker_gate": "ADAS-Cog improvement ≥ 3 points"
        },
        {
            "stage": "Stage II — Consolidation",
            "duration": "Months 7–12",
            "target": "DLPFC + Precuneus",
            "frequency_hz": 10,
            "intensity_mso": 90,
            "sessions_per_week": 3,
            "pulses_session": 2000,
            "coil": "Figure-8 (70mm) + Neuronavigation",
            "adjunct": "rTMS + Fornix DBS integration",
            "biomarker_gate": "Amyloid PET SUVR reduction ≥ 5%"
        },
        {
            "stage": "Stage III — Maintenance",
            "duration": "Months 13–18",
            "target": "DLPFC + Hippocampal circuit",
            "frequency_hz": 5,
            "intensity_mso": 70,
            "sessions_per_week": 2,
            "pulses_session": 1500,
            "coil": "H7 Deep TMS Coil",
            "adjunct": "DBS maintenance + pharmacotherapy",
            "biomarker_gate": "MMSE stable ≥ 24"
        },
        {
            "stage": "Stage IV — Smart Aging Extension",
            "duration": "Months 19–24+",
            "target": "Personalised geodesic-optimised ROI",
            "frequency_hz": 10,
            "intensity_mso": 60,
            "sessions_per_week": 1,
            "pulses_session": 1000,
            "coil": "Conformal geodesic-guided coil",
            "adjunct": "AI-adaptive closed-loop neuromodulation",
            "biomarker_gate": "Cortical thickness stable; tau PET declining"
        }
    ]

    # Treatment optimization convergence
    opt_iterations = []
    score = 0.0
    for i in range(1, 21):
        score += np.random.uniform(2.5, 6.0) * np.exp(-i / 25)
        opt_iterations.append({
            "iteration": i,
            "composite_score": round(float(np.clip(score, 0, 100)), 2),
            "geodesic_coverage_pct": round(float(np.clip(40 + score * 0.8 + np.random.normal(0, 2), 40, 98)), 1),
            "bem_field_efficiency": round(float(np.clip(0.3 + score * 0.012 + np.random.normal(0, 0.02), 0.3, 0.98)), 3)
        })

    return {
        "cognitive_tracking": {
            "months": months,
            "adas_cog": adas_scores,
            "mmse": mmse_scores,
            "moca": moca_scores
        },
        "biomarkers": biomarkers,
        "protocols": protocols,
        "geodesics": _cortical_surface_geodesics(),
        "bem_simulation": _dementia_bem_simulation(),
        "optimization": opt_iterations
    }


def get_treatment_paradigm(condition="stroke"):
    """
    Returns the full optimal treatment paradigm for a given condition:
    - Stage-gating trajectory with Heaviside gate crossings
    - Hebbian-DBS synaptic amplification over sessions
    - Continued fraction frequency convergents for rTMS + DBS
    - DBS hardware protocol recommendation
    """
    target_freqs = {"stroke": 10.0, "dementia": 20.0, "tremor": 130.0}
    tf = target_freqs.get(condition, 10.0)

    return {
        "condition": condition,
        "stage_gates": _stage_gates(condition),
        "hebbian_dbs": _hebbian_amplification(
            n_sessions={"stroke": 15, "dementia": 18, "tremor": 12}.get(condition, 10)
        ),
        "cf_convergents": _continued_fraction_convergents(tf, depth=8),
        "dbs_hardware": {
            "stroke":  {"device": "Medtronic Percept PC", "lead": "SenSight 4-contact",
                        "target": "GPi / M1", "voltage_v": 3.0, "impedance_ohm": 1200},
            "dementia":{"device": "Abbott Infinity DBS", "lead": "Directional 8-contact",
                        "target": "Fornix / NBM",  "voltage_v": 2.5, "impedance_ohm": 1400},
            "tremor":  {"device": "Boston Scientific Vercise", "lead": "8-contact directional",
                        "target": "VIM / cZI",     "voltage_v": 3.5, "impedance_ohm": 1100},
        }.get(condition, {})
    }
