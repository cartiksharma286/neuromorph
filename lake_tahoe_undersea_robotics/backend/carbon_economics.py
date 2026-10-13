"""Finite-horizon scenario accounting, not a calibrated lifecycle assessment."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Assumptions:
    fleet: int = 4
    robot_capex: float = 250000
    annual_opex: float = 35000
    hours: float = 1800
    robot_kw: float = 0.8
    qml_kw: float = 2
    qml_capex: float = 100000
    qml_embodied: float = 5000
    qml_reduction: float = 0
    electricity_price: float = 0.15
    grid_kg_kwh: float = 0.04
    embodied_kg: float = 12000
    baseline_cost: float = 250000
    baseline_kg: float = 25000
    annual_benefit: float = 100000
    carbon_price: float = 100
    discount: float = 0.04
    prime: int = 5


LIMITS = {
    "fleet": (1, 100), "robot_capex": (0, 1e7), "annual_opex": (0, 1e6),
    "hours": (0, 8760), "robot_kw": (0, 1000), "qml_kw": (0, 10000),
    "qml_capex": (0, 1e8), "qml_embodied": (0, 1e8),
    "qml_reduction": (0, 0.5), "electricity_price": (0, 10),
    "grid_kg_kwh": (0, 2), "embodied_kg": (0, 1e7),
    "baseline_cost": (0, 1e9), "baseline_kg": (0, 1e9),
    "annual_benefit": (0, 1e9), "carbon_price": (0, 10000),
    "discount": (0, 0.3), "prime": (2, 11),
}


def parse_assumptions(data: dict) -> Assumptions:
    if not isinstance(data, dict):
        raise ValueError("Assumptions must be a JSON object.")
    unknown = set(data) - set(LIMITS)
    if unknown:
        raise ValueError(f"Unknown assumptions: {', '.join(sorted(unknown))}")
    values = asdict(Assumptions())
    for name, raw in data.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name} must be a number.")
        lo, hi = LIMITS[name]
        if not lo <= raw <= hi or not math.isfinite(raw):
            raise ValueError(f"{name} must be finite and between {lo} and {hi}.")
        if name in ("fleet", "prime"):
            if raw != int(raw):
                raise ValueError(f"{name} must be an integer.")
            values[name] = int(raw)
        else:
            values[name] = float(raw)
    if values["prime"] not in (2, 3, 5, 7, 11):
        raise ValueError("prime must be one of 2, 3, 5, 7, 11.")
    return Assumptions(**values)


def quantum_kernel_demo(a: Assumptions) -> dict:
    """Exact product-state kernel, trained on synthetic accounting examples."""
    train = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 1], [1, 1, 0], [1, 1, 1],
    ], dtype=float)
    targets = train @ np.array([0.6, 0.3, 0.1])
    angles = math.pi * train
    kernel = np.prod(np.cos((angles[:, None, :] - angles[None, :, :]) / 2) ** 2, axis=2)
    alpha = np.linalg.solve(kernel + 0.01 * np.eye(len(train)), targets)
    query = np.array([a.robot_kw / 1000, a.hours / 8760, a.qml_reduction / 0.5])
    cross = np.prod(np.cos((angles - math.pi * query) / 2) ** 2, axis=1)
    return {
        "synthetic_score": float(cross @ alpha),
        "training_rmse": float(np.sqrt(np.mean((kernel @ alpha - targets) ** 2))),
        "samples": len(train), "qubits": 3,
        "method": "Classically evaluated product-state fidelity kernel; synthetic labels.",
        "used_for_financial_forecast": False,
    }


def evaluate(a: Assumptions) -> dict:
    capital = a.fleet * a.robot_capex + a.qml_capex
    energy = a.hours * (a.fleet * a.robot_kw * (1 - a.qml_reduction) + a.qml_kw)
    annual_cost = a.fleet * a.annual_opex + energy * a.electricity_price
    annual_emissions = energy * a.grid_kg_kwh
    embodied = a.fleet * a.embodied_kg + a.qml_embodied
    avoided = a.baseline_kg - annual_emissions
    cash = a.baseline_cost + a.annual_benefit - annual_cost
    annual_carbon_value = avoided / 1000 * a.carbon_price
    trajectory = [{
        "year": n,
        "npv_cad": -capital + sum(cash / (1 + a.discount) ** t for t in range(1, n + 1)),
        "carbon_valued_npv_cad": -capital - embodied / 1000 * a.carbon_price + sum(
            (cash + annual_carbon_value) / (1 + a.discount) ** t for t in range(1, n + 1)),
        "project_kg": embodied + n * annual_emissions,
        "baseline_kg": n * a.baseline_kg,
    } for n in range(11)]
    carbon_break_even = embodied / avoided if avoided > 0 else (0.0 if embodied == 0 and avoided == 0 else None)
    cost_sensitivity = []
    for scale in (0.5, 0.75, 1, 1.25, 1.5):
        cost_sensitivity.append({
            "capex_multiplier": scale,
            "npv_cad": trajectory[-1]["npv_cad"] - capital * (scale - 1),
        })
    return {
        "schema": "ontario-carbon-economics-v1",
        "assumptions": asdict(a), "horizon_years": 10,
        "scope": "Illustrative CAD scenario; no carbon credits, field calibration or proven QML advantage.",
        "capital_cad": capital, "annual_kwh": energy,
        "annual_cost_cad": annual_cost, "annual_operational_kg": annual_emissions,
        "embodied_kg": embodied, "annual_avoided_kg": avoided,
        "financial_npv_cad": trajectory[-1]["npv_cad"],
        "carbon_valued_npv_cad": trajectory[-1]["carbon_valued_npv_cad"],
        "net_avoided_kg_10yr": trajectory[-1]["baseline_kg"] - trajectory[-1]["project_kg"],
        "carbon_break_even_years": carbon_break_even,
        "viable_under_assumptions": trajectory[-1]["npv_cad"] >= 0
        and trajectory[-1]["baseline_kg"] >= trajectory[-1]["project_kg"],
        "trajectory": trajectory, "capex_sensitivity": cost_sensitivity,
        "qml": quantum_kernel_demo(a),
        "number_theory": {
            "prime": a.prime,
            "station_groups": [{"station_id": n, "residue": n % a.prime}
                               for n in range(1, a.fleet + 1)],
            "interpretation": "Modular scheduling labels, not geographic optimality or cryptographic keys.",
        },
    }
