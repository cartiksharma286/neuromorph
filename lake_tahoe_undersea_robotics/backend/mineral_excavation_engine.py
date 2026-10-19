"""
Benthic Mineral Ore Excavation & Quantum Perception Engine
Simulates:
- Underwater robotic excavation of Iron (Fe), Uranium (U), Rare Earth Elements (REE), Silver (Ag), and Cobalt (Co)
- Integrated Computer Vision & Hyperspectral Reflectance Spectroscopy (400 - 1000 nm)
- Quantum Machine Learning with Fermionic Parity Operators (Jordan-Wigner transformation, parity preservation)
- 10-Year Discounted Cash Flow (DCF), Net Present Value (NPV), and Long-Term Commodity Revenue Projections
"""

from dataclasses import asdict, dataclass
import math
import numpy as np


DISCLAIMER = (
    "Synthetic computational research and financial simulation of benthic mineral ore detection, "
    "fermionic parity quantum machine learning, and sub-lacustrine excavation economics. "
    "Not a commercial mining feasibility study, NI 43-101 certified resource estimate, "
    "or environmental permit approval. Field extraction in lacustrine basins is subject to strict "
    "environmental protection laws and international/state regulatory oversight."
)


@dataclass(frozen=True)
class MineralInputs:
    excavation_rate_tpd: float = 850.0        # Metric tonnes of ore excavated per day
    operating_days_year: float = 300.0        # Operating days per year
    discount_rate: float = 0.075              # 7.5% real discount rate
    capital_capex_usd_m: float = 48.0         # Initial CapEx ($M USD) for AUV fleet & topside barge
    operating_opex_per_tonne: float = 82.0    # Operating expense ($/tonne excavated)
    environmental_levy_pct: float = 0.08      # 8% of gross revenue dedicated to environmental remediation
    sensor_turbidity_ntu: float = 4.2         # Water turbidity during optical/spectral scanning
    pqc_qubits: int = 5                       # 5-qubit Jordan-Wigner fermionic register
    pqc_depth: int = 4                        # Variational circuit depth
    chosen_deposit: str = "abyssal_nodule_field"
    seed: int = 42


LIMITS = {
    "excavation_rate_tpd": (100.0, 5000.0),
    "operating_days_year": (150.0, 365.0),
    "discount_rate": (0.0, 0.25),
    "capital_capex_usd_m": (10.0, 250.0),
    "operating_opex_per_tonne": (20.0, 300.0),
    "environmental_levy_pct": (0.01, 0.25),
    "sensor_turbidity_ntu": (0.1, 50.0),
    "pqc_qubits": (4, 8),
    "pqc_depth": (1, 8),
    "seed": (0, 1000000),
}


DEPOSIT_PRESETS = {
    "abyssal_nodule_field": {
        "title": "Abyssal Polymetallic Nodule Field (Crystal Bay Basin, 480m)",
        "geological_setting": "Benthic ferromanganese nodule pavement overlying diatomaceous clay",
        "iron_grade_pct": 34.5,
        "uranium_grade_ppm": 180.0,
        "ree_grade_ppm": 1250.0,
        "silver_grade_g_tonne": 115.0,
        "cobalt_grade_pct": 0.85,
        "recovery_rate": 0.88,
        "modulus_k": 0.725,
    },
    "hydrothermal_vein": {
        "title": "Hydrothermal Polymetallic Fault Vein (Rubicon Escarpment, 360m)",
        "geological_setting": "Epithermal quartz-sulfide fissure vein with argentite and cobaltite",
        "iron_grade_pct": 22.0,
        "uranium_grade_ppm": 95.0,
        "ree_grade_ppm": 680.0,
        "silver_grade_g_tonne": 320.0,
        "cobalt_grade_pct": 0.62,
        "recovery_rate": 0.91,
        "modulus_k": 0.885,
    },
    "uraniferous_sapropel": {
        "title": "Uraniferous Pelagic Reducing Sapropel (Western Deep Basin, 220m)",
        "geological_setting": "Anoxic organic-rich mud with organo-uranyl and heavy REE complexes",
        "iron_grade_pct": 18.5,
        "uranium_grade_ppm": 640.0,
        "ree_grade_ppm": 2150.0,
        "silver_grade_g_tonne": 75.0,
        "cobalt_grade_pct": 0.32,
        "recovery_rate": 0.85,
        "modulus_k": 0.420,
    },
    "granitic_placer": {
        "title": "Granitic Heavy-Mineral Placer Shelf (Stateline Sill, 95m)",
        "geological_setting": "Littoral-pelagic paleochannel placer with monazite, xenotime, and magnetite",
        "iron_grade_pct": 42.0,
        "uranium_grade_ppm": 140.0,
        "ree_grade_ppm": 1820.0,
        "silver_grade_g_tonne": 65.0,
        "cobalt_grade_pct": 0.28,
        "recovery_rate": 0.89,
        "modulus_k": 0.640,
    },
    "cobalt_crust_ridge": {
        "title": "Cobalt-Rich Ferromanganese Ridge (Glenbrook Ridge, 310m)",
        "geological_setting": "Submerged bedrock horst capped by thick cobalt-manganese hydrogenetic crust",
        "iron_grade_pct": 28.0,
        "uranium_grade_ppm": 210.0,
        "ree_grade_ppm": 1450.0,
        "silver_grade_g_tonne": 90.0,
        "cobalt_grade_pct": 1.15,
        "recovery_rate": 0.87,
        "modulus_k": 0.795,
    },
}


# Commodity Market Benchmark Prices (USD)
COMMODITY_PRICES = {
    "iron_usd_per_tonne": 125.0,              # Fe 62% benchmark ore ($/tonne)
    "uranium_usd_per_kg": 185.0,              # U3O8 yellowcake ($/kg)
    "ree_usd_per_kg": 95.0,                   # Blended Rare Earth Oxides ($/kg)
    "silver_usd_per_gram": 1.05,              # Fine silver ($1.05/g ~ $32.6/oz)
    "cobalt_usd_per_tonne": 34500.0,          # High-grade LME cobalt ($/tonne)
}


def parse_mineral_inputs(data: object) -> MineralInputs:
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object of mineral excavation inputs.")
    values = asdict(MineralInputs())
    for key, value in data.items():
        if key == "chosen_deposit":
            if value not in DEPOSIT_PRESETS:
                raise ValueError(f"chosen_deposit must be one of {list(DEPOSIT_PRESETS.keys())}.")
            values[key] = str(value)
            continue
        if key not in LIMITS:
            raise ValueError(f"Unrecognized parameter '{key}'.")
        low, high = LIMITS[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not (low <= value <= high):
            raise ValueError(f"{key} must be between {low} and {high}.")
        if key in ("pqc_qubits", "pqc_depth", "seed"):
            if int(value) != value:
                raise ValueError(f"{key} must be an integer.")
            values[key] = int(value)
        else:
            values[key] = float(value)
    return MineralInputs(**values)


# ---------------------------------------------------------------------------
# Vision & Hyperspectral Reflectance Spectroscopy of Mineral Ores
# ---------------------------------------------------------------------------

def simulate_ore_spectroscopy(deposit_preset: dict, turbidity_ntu: float, seed: int) -> dict:
    """Simulates 400 - 1000 nm hyperspectral optical reflectance signatures for Fe, U, REE, Ag, and Co."""
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(400.0, 1000.0, 61)

    # 1. Iron (Fe): Fe3+ crystal field absorption dip at 860-920 nm, blue absorption
    fe_spectrum = (
        0.18 + 0.28 * ((wavelengths - 400.0) / 600.0) ** 1.4 -
        0.14 * np.exp(-0.5 * ((wavelengths - 890.0) / 45.0) ** 2)
    )

    # 2. Uranium (U): Uranyl (UO2)2+ vibronic progression bands 420-490 nm, fluorescence peak at 530 nm
    u_spectrum = (
        0.15 + 0.12 * np.exp(-0.5 * ((wavelengths - 530.0) / 30.0) ** 2) +
        0.08 * np.sin(2.0 * math.pi * (wavelengths - 420.0) / 45.0) * np.exp(-0.5 * ((wavelengths - 460.0) / 60.0) ** 2) +
        0.05 * ((wavelengths - 400.0) / 600.0)
    )

    # 3. Rare Earth Elements (REE - Nd, Dy, Er): Sharp 4f-4f electronic absorption multiplets at 520, 580, 740, 800 nm
    ree_spectrum = (
        0.25 + 0.10 * ((wavelengths - 400.0) / 600.0) -
        0.12 * np.exp(-0.5 * ((wavelengths - 580.0) / 14.0) ** 2) -  # Nd3+ band
        0.09 * np.exp(-0.5 * ((wavelengths - 520.0) / 12.0) ** 2) -  # Er3+ band
        0.11 * np.exp(-0.5 * ((wavelengths - 740.0) / 16.0) ** 2) -  # Dy3+ band
        0.10 * np.exp(-0.5 * ((wavelengths - 800.0) / 18.0) ** 2)    # Nd3+ band
    )

    # 4. Silver (Ag): Metallic high broadband specular reflectance rise above 500 nm
    ag_spectrum = 0.35 + 0.45 * (1.0 / (1.0 + np.exp(-(wavelengths - 520.0) / 45.0)))

    # 5. Cobalt (Co): Broad Co2+ d-d ligand field absorption envelope across 540 - 640 nm
    co_spectrum = (
        0.20 + 0.18 * ((wavelengths - 400.0) / 600.0) -
        0.13 * np.exp(-0.5 * ((wavelengths - 590.0) / 50.0) ** 2)
    )

    # Composite Benthic Ore Reflectance based on normalized grade loadings
    fe_norm = deposit_preset["iron_grade_pct"] / 50.0
    u_norm = deposit_preset["uranium_grade_ppm"] / 1000.0
    ree_norm = deposit_preset["ree_grade_ppm"] / 2500.0
    ag_norm = deposit_preset["silver_grade_g_tonne"] / 400.0
    co_norm = deposit_preset["cobalt_grade_pct"] / 1.5

    sum_norm = max(1e-6, fe_norm + u_norm + ree_norm + ag_norm + co_norm)
    w_fe = fe_norm / sum_norm
    w_u = u_norm / sum_norm
    w_ree = ree_norm / sum_norm
    w_ag = ag_norm / sum_norm
    w_co = co_norm / sum_norm

    composite_ore = (
        w_fe * fe_spectrum +
        w_u * u_spectrum +
        w_ree * ree_spectrum +
        w_ag * ag_spectrum +
        w_co * co_spectrum
    )

    # Water column absorption and turbidity noise overlay
    turbidity_haze = 0.04 * (turbidity_ntu / 10.0)
    noise = rng.normal(0.0, 0.008, len(wavelengths))
    composite_ore = np.clip(composite_ore + turbidity_haze + noise, 0.01, 0.98)

    return {
        "wavelengths_nm": wavelengths.tolist(),
        "iron_spectrum": fe_spectrum.tolist(),
        "uranium_spectrum": u_spectrum.tolist(),
        "ree_spectrum": ree_spectrum.tolist(),
        "silver_spectrum": ag_spectrum.tolist(),
        "cobalt_spectrum": co_spectrum.tolist(),
        "composite_ore_spectrum": composite_ore.tolist(),
        "elemental_weights": {
            "iron": float(w_fe),
            "uranium": float(w_u),
            "ree": float(w_ree),
            "silver": float(w_ag),
            "cobalt": float(w_co),
        }
    }


# ---------------------------------------------------------------------------
# Quantum Machine Learning: Jordan-Wigner Fermionic Parity Simulation
# ---------------------------------------------------------------------------

def simulate_fermionic_parity_qml(deposit_preset: dict, num_qubits: int, depth: int, seed: int) -> dict:
    """Simulates 5-qubit Jordan-Wigner fermionic state and parity operator P = Z1 (x) Z2 (x) ... (x) ZN.

    Fermionic annihilation/creation operators:
    c_j = (prod_{k<j} Z_k) * (X_j + i Y_j) / 2
    Fermionic parity operator P = prod_j (-1)^(n_j) = prod_j (1 - 2 c_j^dag c_j) = Z_1 Z_2 ... Z_N
    """
    rng = np.random.default_rng(seed)
    n = min(8, max(4, num_qubits))
    dim = 2 ** n

    # Normalized elemental grades into rotation angles theta_j in [0, pi]
    g_fe = deposit_preset["iron_grade_pct"] / 50.0
    g_u = deposit_preset["uranium_grade_ppm"] / 1000.0
    g_ree = deposit_preset["ree_grade_ppm"] / 2500.0
    g_ag = deposit_preset["silver_grade_g_tonne"] / 400.0
    g_co = deposit_preset["cobalt_grade_pct"] / 1.5

    base_angles = [math.pi * min(1.0, max(0.05, g)) for g in [g_fe, g_u, g_ree, g_ag, g_co]]
    while len(base_angles) < n:
        base_angles.append(0.5 * math.pi)

    # Initialize state |0...0>
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0

    # 1. Angle encoding with Ry rotations
    for q in range(n):
        theta = base_angles[q]
        ry = np.array([
            [math.cos(theta / 2.0), -math.sin(theta / 2.0)],
            [math.sin(theta / 2.0),  math.cos(theta / 2.0)]
        ], dtype=complex)
        dim_left = 2 ** q
        dim_right = 2 ** (n - 1 - q)
        st = state.reshape((dim_left, 2, dim_right))
        st = np.einsum('ij,ajb->aib', ry, st)
        state = st.reshape(dim)

    # 2. Parity-preserving fermionic excitation layers U_f = exp[-i (phi/2) (X_j Y_{j+1} - Y_j X_{j+1})]
    for d in range(depth):
        for q in range(n - 1):
            phi = rng.uniform(0.2, math.pi * 0.8)
            # 2-qubit Givens rotation preserving particle number and parity
            givens = np.eye(4, dtype=complex)
            givens[1, 1] = math.cos(phi / 2.0)
            givens[1, 2] = -math.sin(phi / 2.0)
            givens[2, 1] = math.sin(phi / 2.0)
            givens[2, 2] = math.cos(phi / 2.0)

            dim_left = 2 ** q
            dim_right = 2 ** (n - 2 - q)
            st = state.reshape((dim_left, 4, dim_right))
            st = np.einsum('ij,ajb->aib', givens, st)
            state = st.reshape(dim)

    # Normalize state
    state = state / np.linalg.norm(state)

    # Compute Fermionic Parity Operator expectation: P = Z_0 (x) Z_1 (x) ... (x) Z_{N-1}
    parity_diagonal = np.zeros(dim)
    even_parity_prob = 0.0
    odd_parity_prob = 0.0

    for idx in range(dim):
        # Count number of 1s in binary representation (fermion occupation number N_occ)
        num_ones = bin(idx).count('1')
        parity_eigenval = 1.0 if (num_ones % 2 == 0) else -1.0
        parity_diagonal[idx] = parity_eigenval
        prob = float(abs(state[idx]) ** 2)
        if parity_eigenval > 0:
            even_parity_prob += prob
        else:
            odd_parity_prob += prob

    parity_expectation = float(np.sum(abs(state) ** 2 * parity_diagonal))

    # Fermionic parity distribution across occupation sectors N_occ = 0, 1, 2, 3, 4, 5
    occupation_distribution = {}
    for idx in range(dim):
        num_ones = bin(idx).count('1')
        key = f"{num_ones}_fermions"
        occupation_distribution[key] = occupation_distribution.get(key, 0.0) + float(abs(state[idx]) ** 2)

    # Quantum classification confidence based on parity fidelity
    qml_confidence = float(0.82 + 0.16 * abs(parity_expectation))

    # QML parity expectation vs circuit depth sweep
    depth_axis = list(range(1, 9))
    depth_parity_sweep = [
        float(parity_expectation * math.exp(-0.06 * (dp - 1)) + rng.normal(0, 0.015))
        for dp in depth_axis
    ]

    return {
        "num_qubits": n,
        "circuit_depth": depth,
        "parity_expectation": parity_expectation,
        "even_parity_probability": float(even_parity_prob),
        "odd_parity_probability": float(odd_parity_prob),
        "occupation_distribution": occupation_distribution,
        "qml_confidence": qml_confidence,
        "depth_axis": depth_axis,
        "depth_parity_sweep": depth_parity_sweep,
        "scope": (
            "5-qubit Jordan-Wigner fermionic parity transformation. Operator P = Z1...ZN "
            "evaluates total fermionic parity conservation across valence d/f electronic transitions "
            "for heavy metal and rare earth ore identification."
        )
    }


# ---------------------------------------------------------------------------
# 10-Year Discounted Cash Flow & Mineral Revenue Projections
# ---------------------------------------------------------------------------

def compute_10yr_mineral_revenue(inputs: MineralInputs, deposit: dict) -> dict:
    """Computes annual production tonnage, gross revenue by commodity, OpEx, environmental fund, and 10-year NPV."""
    annual_ore_tonnes = inputs.excavation_rate_tpd * inputs.operating_days_year
    rec = deposit["recovery_rate"]

    # Annual Metal Production
    fe_tonnes = annual_ore_tonnes * (deposit["iron_grade_pct"] / 100.0) * rec
    u_kg = annual_ore_tonnes * (deposit["uranium_grade_ppm"] / 1e6) * 1e3 * rec
    ree_kg = annual_ore_tonnes * (deposit["ree_grade_ppm"] / 1e6) * 1e3 * rec
    ag_grams = annual_ore_tonnes * deposit["silver_grade_g_tonne"] * rec
    co_tonnes = annual_ore_tonnes * (deposit["cobalt_grade_pct"] / 100.0) * rec

    # Annual Gross Revenue by Commodity (USD)
    rev_fe = fe_tonnes * COMMODITY_PRICES["iron_usd_per_tonne"]
    rev_u = u_kg * COMMODITY_PRICES["uranium_usd_per_kg"]
    rev_ree = ree_kg * COMMODITY_PRICES["ree_usd_per_kg"]
    rev_ag = ag_grams * COMMODITY_PRICES["silver_usd_per_gram"]
    rev_co = co_tonnes * COMMODITY_PRICES["cobalt_usd_per_tonne"]

    annual_gross_revenue = rev_fe + rev_u + rev_ree + rev_ag + rev_co
    annual_opex = annual_ore_tonnes * inputs.operating_opex_per_tonne
    annual_env_fund = annual_gross_revenue * inputs.environmental_levy_pct
    annual_net_cash_flow = annual_gross_revenue - annual_opex - annual_env_fund

    # 10-Year Discounted Cash Flow (DCF) trajectory
    r = inputs.discount_rate
    initial_capex = inputs.capital_capex_usd_m * 1e6

    trajectory = []
    cumulative_cash = -initial_capex
    cumulative_discounted_cash = -initial_capex

    trajectory.append({
        "year": 0,
        "gross_revenue_usd": 0.0,
        "opex_usd": 0.0,
        "env_fund_usd": 0.0,
        "net_cash_flow_usd": -initial_capex,
        "discounted_cash_flow_usd": -initial_capex,
        "cumulative_discounted_npv_usd": -initial_capex,
    })

    # Slight real efficiency gain / grade variability trajectory
    rng = np.random.default_rng(inputs.seed)
    for y in range(1, 11):
        # 1.5% technological efficiency improvement per year
        efficiency_factor = (1.0 + 0.015 * (y - 1)) + rng.normal(0.0, 0.01)
        y_rev = annual_gross_revenue * efficiency_factor
        y_opex = annual_opex * (1.0 + 0.008 * (y - 1))
        y_env = y_rev * inputs.environmental_levy_pct
        y_net = y_rev - y_opex - y_env

        discount_factor = 1.0 / ((1.0 + r) ** y)
        discounted_flow = y_net * discount_factor
        cumulative_discounted_cash += discounted_flow

        trajectory.append({
            "year": y,
            "gross_revenue_usd": float(y_rev),
            "opex_usd": float(y_opex),
            "env_fund_usd": float(y_env),
            "net_cash_flow_usd": float(y_net),
            "discounted_cash_flow_usd": float(discounted_flow),
            "cumulative_discounted_npv_usd": float(cumulative_discounted_cash),
        })

    ten_year_npv = cumulative_discounted_cash
    payback_years = None
    for item in trajectory:
        if item["year"] > 0 and item["cumulative_discounted_npv_usd"] >= 0:
            payback_years = item["year"]
            break

    # Internal Rate of Return (IRR) approximation via bisection
    irr = None
    if ten_year_npv > 0:
        low_irr = 0.0
        high_irr = 1.5
        for _ in range(30):
            mid_irr = (low_irr + high_irr) / 2.0
            npv_mid = -initial_capex + sum(
                (item["net_cash_flow_usd"]) / ((1.0 + mid_irr) ** item["year"])
                for item in trajectory if item["year"] > 0
            )
            if npv_mid > 0:
                low_irr = mid_irr
            else:
                high_irr = mid_irr
        irr = float((low_irr + high_irr) / 2.0)

    return {
        "annual_ore_excavated_tonnes": float(annual_ore_tonnes),
        "annual_metal_production": {
            "iron_tonnes": float(fe_tonnes),
            "uranium_kg": float(u_kg),
            "ree_kg": float(ree_kg),
            "silver_kg": float(ag_grams / 1e3),
            "cobalt_tonnes": float(co_tonnes),
        },
        "annual_revenue_by_commodity_usd": {
            "iron": float(rev_fe),
            "uranium": float(rev_u),
            "ree": float(rev_ree),
            "silver": float(rev_ag),
            "cobalt": float(rev_co),
        },
        "annual_gross_revenue_usd": float(annual_gross_revenue),
        "annual_opex_usd": float(annual_opex),
        "annual_environmental_remediation_usd": float(annual_env_fund),
        "annual_net_cash_flow_usd": float(annual_net_cash_flow),
        "ten_year_discounted_npv_usd": float(ten_year_npv),
        "payback_period_years": payback_years,
        "internal_rate_of_return_irr": irr,
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# Master Benthic Mineral Excavation Engine
# ---------------------------------------------------------------------------

class MineralExcavationEngine:
    """Master engine for benthic mineral excavation, spectroscopy, QML fermionic parity, and revenue modeling."""

    def simulate(self, inputs: MineralInputs) -> dict:
        preset = DEPOSIT_PRESETS.get(inputs.chosen_deposit, DEPOSIT_PRESETS["abyssal_nodule_field"])

        # 1. Hyperspectral optical reflectance simulation
        spectroscopy = simulate_ore_spectroscopy(
            deposit_preset=preset,
            turbidity_ntu=inputs.sensor_turbidity_ntu,
            seed=inputs.seed
        )

        # 2. QML Fermionic Parity Operator simulation
        qml_parity = simulate_fermionic_parity_qml(
            deposit_preset=preset,
            num_qubits=inputs.pqc_qubits,
            depth=inputs.pqc_depth,
            seed=inputs.seed
        )

        # 3. 10-Year Discounted Cash Flow & Mineral Economics
        economics = compute_10yr_mineral_revenue(
            inputs=inputs,
            deposit=preset
        )

        return {
            "inputs": asdict(inputs),
            "disclaimer": DISCLAIMER,
            "deposit_preset": preset,
            "spectroscopy": spectroscopy,
            "qml_parity": qml_parity,
            "economics": economics,
            "commodities": COMMODITY_PRICES,
            "telemetry": {
                "excavation_rate_tph": float(inputs.excavation_rate_tpd / 24.0),
                "annual_ore_tonnes": economics["annual_ore_excavated_tonnes"],
                "ten_year_npv_usd_m": float(economics["ten_year_discounted_npv_usd"] / 1e6),
                "parity_eigenvalue_sign": 1 if qml_parity["parity_expectation"] >= 0 else -1,
                "qml_confidence_pct": float(qml_parity["qml_confidence"] * 100.0),
            }
        }
