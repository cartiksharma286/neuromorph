#!/usr/bin/env python3
"""
ontario_wildfire_qml_app - Ontario and Northern Ontario wildfire management
simulator with statistical quantum machine learning, ecological restoration,
and East Coast smoke propagation analytics.
"""

import math
import os

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

REGION_PROFILES = {
    "ontario": {
        "label": "Central and Southern Ontario",
        "baseline_burn_area_ha": 1380.0,
        "remoteness": 0.32,
        "corridor_pm": 18.0,
        "restoration_lag": 0.88,
        "community_exposure": 1.12,
        "watershed_recovery": 1.04,
    },
    "northern_ontario": {
        "label": "Northern Ontario Boreal Belt",
        "baseline_burn_area_ha": 1860.0,
        "remoteness": 0.54,
        "corridor_pm": 23.0,
        "restoration_lag": 1.02,
        "community_exposure": 0.96,
        "watershed_recovery": 0.94,
    },
}

EAST_COAST_CITIES = [
    {"city": "Buffalo", "distance_km": 920.0, "x": 0.38, "y": 0.45},
    {"city": "Albany", "distance_km": 1080.0, "x": 0.52, "y": 0.39},
    {"city": "Boston", "distance_km": 1340.0, "x": 0.74, "y": 0.30},
    {"city": "New York City", "distance_km": 1235.0, "x": 0.66, "y": 0.44},
    {"city": "Philadelphia", "distance_km": 1315.0, "x": 0.63, "y": 0.54},
    {"city": "Washington DC", "distance_km": 1485.0, "x": 0.69, "y": 0.66},
    {"city": "Portland, ME", "distance_km": 1425.0, "x": 0.82, "y": 0.24},
]

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

START_MONTH = 5
START_YEAR = 2026
HORIZON_MONTHS = 24


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def build_month_labels(months=HORIZON_MONTHS, start_year=START_YEAR, start_month=START_MONTH):
    labels = []
    current_month = start_month
    current_year = start_year
    for _ in range(months):
        labels.append(f"{MONTH_NAMES[current_month - 1]} {current_year}")
        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    return labels


MONTH_LABELS = build_month_labels()


def month_number_for_index(index, start_month=START_MONTH):
    return ((start_month - 1 + index) % 12) + 1


def is_summer_month(index):
    return month_number_for_index(index) in {6, 7, 8, 9}


SUMMER_INDICES = [idx for idx in range(HORIZON_MONTHS) if is_summer_month(idx)]


def seasonal_fire_pressure(index):
    month_number = month_number_for_index(index)
    summer_peak = math.exp(-((month_number - 7.5) ** 2) / 3.6)
    shoulder = math.exp(-((month_number - 6.0) ** 2) / 11.5)
    return 0.72 + 0.48 * summer_peak + 0.16 * shoulder


def qml_alignment_kernel(qml_weight, remoteness, humidity_factor, index):
    phase = 0.34 + 0.42 * qml_weight + 0.11 * remoteness + 0.09 * humidity_factor + 0.06 * (index / HORIZON_MONTHS)
    interference = math.cos(math.pi * phase) ** 2
    kernel = 0.72 + 0.15 * interference + 0.08 * qml_weight + 0.06 * humidity_factor - 0.05 * remoteness
    return clamp(kernel, 0.62, 1.08)


def summer_to_summer_delta(series):
    first_summer = [series[idx] for idx in SUMMER_INDICES[:4]]
    second_summer = [series[idx] for idx in SUMMER_INDICES[4:8]]
    if not first_summer or not second_summer:
        return 0.0
    return float(np.mean(first_summer) - np.mean(second_summer))


def compute_wildfire_command(region="northern_ontario", crews=92, tankers=18, humidity=52.0, qml_weight=0.62):
    profile = REGION_PROFILES.get(region, REGION_PROFILES["northern_ontario"])
    humidity_factor = clamp(float(humidity) / 100.0, 0.2, 0.95)
    crews = clamp(float(crews), 35.0, 180.0)
    tankers = clamp(float(tankers), 4.0, 34.0)
    qml_weight = clamp(float(qml_weight), 0.0, 1.0)

    crew_factor = crews / 120.0
    tanker_factor = tankers / 30.0

    active_fire_hectares = []
    suppression_efficiency = []
    local_aqi = []
    qml_kernel = []
    readiness_score = []

    for idx in range(HORIZON_MONTHS):
        season = seasonal_fire_pressure(idx)
        kernel = qml_alignment_kernel(qml_weight, profile["remoteness"], humidity_factor, idx)
        response = 0.18 + 0.12 * crew_factor + 0.10 * tanker_factor + 0.18 * qml_weight + 0.08 * humidity_factor

        suppression = 47.0 + 39.0 * (1.0 - math.exp(-(idx + 1) * response / 6.8))
        suppression += 5.5 * (kernel - 0.75) - 7.0 * profile["remoteness"] - 6.5 * (season - 0.82)
        suppression = clamp(suppression, 34.0, 95.0)

        area = profile["baseline_burn_area_ha"] * season * math.exp(-0.05 * idx) * (1.18 - suppression / 100.0)
        area *= 1.0 + 0.08 * profile["community_exposure"]
        area = max(120.0, area)

        aqi = 44.0 + 16.0 * season + 0.024 * area - 12.0 * humidity_factor - 7.0 * qml_weight
        aqi = clamp(aqi, 24.0, 145.0)

        readiness = 55.0 + 0.32 * suppression + 9.0 * qml_weight - 4.0 * profile["remoteness"]
        readiness = clamp(readiness, 48.0, 98.0)

        active_fire_hectares.append(round(area, 2))
        suppression_efficiency.append(round(suppression, 2))
        local_aqi.append(round(aqi, 2))
        qml_kernel.append(round(kernel, 3))
        readiness_score.append(round(readiness, 2))

    air_quality_gain = summer_to_summer_delta(local_aqi)

    first_summer_idxs = [i for i in SUMMER_INDICES if i < 12]
    second_summer_idxs = [i for i in SUMMER_INDICES if i >= 12]
    first_summer_aqi = round(float(np.mean([local_aqi[i] for i in first_summer_idxs])), 1) if first_summer_idxs else round(local_aqi[0], 1)
    second_summer_aqi = round(float(np.mean([local_aqi[i] for i in second_summer_idxs])), 1) if second_summer_idxs else round(local_aqi[-1], 1)
    first_summer_fire = round(float(np.mean([active_fire_hectares[i] for i in first_summer_idxs])), 1) if first_summer_idxs else round(active_fire_hectares[0], 1)
    second_summer_fire = round(float(np.mean([active_fire_hectares[i] for i in second_summer_idxs])), 1) if second_summer_idxs else round(active_fire_hectares[-1], 1)
    fire_reduction_pct = round((active_fire_hectares[0] - active_fire_hectares[-1]) / max(active_fire_hectares[0], 1.0) * 100.0, 1)
    peak_fire_idx = int(np.argmax(active_fire_hectares))
    first_summer_suppression = round(float(np.mean([suppression_efficiency[i] for i in first_summer_idxs])), 1) if first_summer_idxs else 0.0
    second_summer_suppression = round(float(np.mean([suppression_efficiency[i] for i in second_summer_idxs])), 1) if second_summer_idxs else 0.0

    characteristics = {
        "avg_suppression_efficiency": round(float(np.mean(suppression_efficiency[-6:])), 1),
        "current_active_fire_area_ha": round(active_fire_hectares[-1], 1),
        "local_aqi_index": round(local_aqi[-1], 1),
        "qml_stability_score": round(float(np.mean(qml_kernel) * 100.0), 1),
        "summer_air_quality_gain": round(air_quality_gain, 1),
        "fire_reduction_pct": fire_reduction_pct,
        "peak_fire_month_label": MONTH_LABELS[peak_fire_idx],
        "first_summer_aqi": first_summer_aqi,
        "second_summer_aqi": second_summer_aqi,
        "first_summer_fire_ha": first_summer_fire,
        "second_summer_fire_ha": second_summer_fire,
        "first_summer_suppression": first_summer_suppression,
        "second_summer_suppression": second_summer_suppression,
    }

    narrative = (
        f"{profile['label']} converges toward a lower-burn operating regime over the 24-month summer horizon. "
        f"With {int(crews)} crews, {int(tankers)} tanker sorties, humidity support of {float(humidity):.0f}% and "
        f"a quantum machine learning alignment weight of {qml_weight:.2f}, the command layer stabilizes suppression at "
        f"{characteristics['avg_suppression_efficiency']:.1f}% while improving the summer AQI profile by "
        f"{characteristics['summer_air_quality_gain']:.1f} index points between the first and second summer." 
    )

    return {
        "region_label": profile["label"],
        "month_labels": MONTH_LABELS,
        "active_fire_hectares": active_fire_hectares,
        "suppression_efficiency": suppression_efficiency,
        "local_aqi": local_aqi,
        "qml_kernel": qml_kernel,
        "readiness_score": readiness_score,
        "characteristics": characteristics,
        "narrative": narrative,
    }


def compute_containment_strategy(
    region="northern_ontario",
    crews=118,
    firebreak_km=145.0,
    restoration=70.0,
    prescribed_burn=54.0,
    qml_weight=0.66,
):
    profile = REGION_PROFILES.get(region, REGION_PROFILES["northern_ontario"])
    crews = clamp(float(crews), 40.0, 200.0)
    firebreak_km = clamp(float(firebreak_km), 20.0, 320.0)
    restoration = clamp(float(restoration), 10.0, 100.0)
    prescribed_burn = clamp(float(prescribed_burn), 5.0, 100.0)
    qml_weight = clamp(float(qml_weight), 0.0, 1.0)

    crew_factor = crews / 150.0
    firebreak_factor = firebreak_km / 180.0
    restoration_factor = restoration / 100.0
    prescribed_burn_factor = prescribed_burn / 100.0

    convergence_alpha = (
        0.058
        + 0.014 * crew_factor
        + 0.020 * firebreak_factor
        + 0.018 * prescribed_burn_factor
        + 0.028 * qml_weight
        - 0.010 * profile["remoteness"]
    )

    mitigation_convergence_pct = []
    active_risk_hectares = []
    cumulative_restored_hectares = []
    land_recovery_index = []
    regional_pm25 = []

    restored_total = 0.0

    for idx in range(HORIZON_MONTHS):
        season = seasonal_fire_pressure(idx)
        kernel = qml_alignment_kernel(qml_weight, profile["remoteness"], 0.52 + restoration_factor * 0.22, idx)

        convergence = 30.0 + 56.0 * (1.0 - math.exp(-convergence_alpha * (idx + 1) * kernel))
        convergence -= 8.5 * (season - 0.80)
        convergence += 4.0 * prescribed_burn_factor + 2.0 * firebreak_factor
        convergence = clamp(convergence, 28.0, 98.0)

        risk_hectares = profile["baseline_burn_area_ha"] * season * (1.28 - convergence / 100.0)
        risk_hectares *= 1.0 + 0.04 * profile["remoteness"]
        risk_hectares = max(90.0, risk_hectares)

        monthly_restoration = 42.0 + 2.4 * crews + 1.35 * firebreak_km + 3.4 * restoration + 2.1 * prescribed_burn
        monthly_restoration *= (0.55 + 0.45 * convergence / 100.0) / profile["restoration_lag"]
        monthly_restoration *= 1.03 - 0.08 * max(0.0, season - 1.0)
        restored_total += monthly_restoration

        recovery = 38.0 + 0.0052 * restored_total + 0.22 * restoration - 5.5 * profile["restoration_lag"]
        recovery += 2.5 * profile["watershed_recovery"]
        recovery = clamp(recovery, 35.0, 97.0)

        pm25 = 23.0 + 13.0 * season + 0.018 * risk_hectares - 0.12 * recovery - 5.0 * qml_weight
        pm25 = clamp(pm25, 8.0, 75.0)

        mitigation_convergence_pct.append(round(convergence, 2))
        active_risk_hectares.append(round(risk_hectares, 2))
        cumulative_restored_hectares.append(round(restored_total, 2))
        land_recovery_index.append(round(recovery, 2))
        regional_pm25.append(round(pm25, 2))

    convergence_month = next((idx + 1 for idx, value in enumerate(mitigation_convergence_pct) if value >= 85.0), HORIZON_MONTHS)
    mitigation_half_life = next((idx + 1 for idx, value in enumerate(active_risk_hectares) if value <= 0.5 * active_risk_hectares[0]), HORIZON_MONTHS)
    air_quality_gain = summer_to_summer_delta(regional_pm25)
    convergence_stability = clamp(100.0 - float(np.std(mitigation_convergence_pct[-6:]) * 4.0), 55.0, 99.0)

    characteristics = {
        "final_convergence_pct": round(mitigation_convergence_pct[-1], 1),
        "convergence_month": int(convergence_month),
        "mitigation_half_life_month": int(mitigation_half_life),
        "restored_hectares": round(cumulative_restored_hectares[-1], 1),
        "land_recovery_score": round(float(np.mean(land_recovery_index[-6:])), 1),
        "convergence_stability_score": round(convergence_stability, 1),
        "summer_pm25_reduction": round(air_quality_gain, 1),
    }

    narrative = (
        f"The containment strategy tab couples direct attack, firebreak creation, prescribed burning and ecological restoration over a 24-month horizon. "
        f"For {profile['label']}, the current operating point reaches {characteristics['final_convergence_pct']:.1f}% mitigation convergence by month "
        f"{characteristics['convergence_month']}, restores {characteristics['restored_hectares']:.0f} hectares, and reduces summer PM2.5 by "
        f"{characteristics['summer_pm25_reduction']:.1f} ug/m3 between the first and second summer windows."
    )

    return {
        "region_label": profile["label"],
        "month_labels": MONTH_LABELS,
        "mitigation_convergence_pct": mitigation_convergence_pct,
        "active_risk_hectares": active_risk_hectares,
        "cumulative_restored_hectares": cumulative_restored_hectares,
        "land_recovery_index": land_recovery_index,
        "regional_pm25": regional_pm25,
        "characteristics": characteristics,
        "narrative": narrative,
    }


def compute_smoke_propagation(
    region="northern_ontario",
    source_intensity=1.15,
    easterly_flow=24.0,
    humidity_scrub=58.0,
    qml_dissipation=0.64,
):
    profile = REGION_PROFILES.get(region, REGION_PROFILES["northern_ontario"])
    source_intensity = clamp(float(source_intensity), 0.5, 2.0)
    easterly_flow = clamp(float(easterly_flow), 8.0, 40.0)
    humidity_scrub = clamp(float(humidity_scrub), 20.0, 90.0)
    qml_dissipation = clamp(float(qml_dissipation), 0.0, 1.0)

    humidity_factor = humidity_scrub / 100.0

    east_coast_aqi = []
    transported_pm25 = []
    plume_half_life_hours = []
    city_series = {city["city"]: [] for city in EAST_COAST_CITIES}

    for idx in range(HORIZON_MONTHS):
        season = seasonal_fire_pressure(idx)
        kernel = qml_alignment_kernel(qml_dissipation, profile["remoteness"], humidity_factor, idx)
        source_load = profile["corridor_pm"] * source_intensity * season * (1.15 + 0.12 * profile["remoteness"])
        dissipation = 0.22 + 0.0042 * humidity_scrub + 0.20 * qml_dissipation + 0.018 * kernel
        corridor = 0.84 + 0.012 * easterly_flow

        city_month_pm = []
        for city in EAST_COAST_CITIES:
            transport = source_load * corridor * math.exp(-dissipation * city["distance_km"] / 900.0)
            transport *= 1.0 + 0.12 * season
            pm25 = clamp(4.0 + 1.8 * transport, 3.5, 72.0)
            city_series[city["city"]].append(round(pm25, 2))
            city_month_pm.append(pm25)

        avg_pm = float(np.mean(city_month_pm))
        aqi = clamp(28.0 + 2.25 * avg_pm - 6.5 * qml_dissipation - 0.09 * humidity_scrub, 22.0, 170.0)
        half_life = clamp(17.0 - 0.12 * humidity_scrub - 4.5 * qml_dissipation - 0.08 * easterly_flow + 2.1 * profile["remoteness"], 4.5, 18.0)

        transported_pm25.append(round(avg_pm, 2))
        east_coast_aqi.append(round(aqi, 2))
        plume_half_life_hours.append(round(half_life, 2))

    city_snapshot = []
    for city in EAST_COAST_CITIES:
        summer_pm = float(np.mean([city_series[city["city"]][idx] for idx in SUMMER_INDICES]))
        city_snapshot.append(
            {
                "city": city["city"],
                "pm25": round(summer_pm, 2),
                "aqi": round(clamp(26.0 + 2.35 * summer_pm, 24.0, 170.0), 1),
                "x": city["x"],
                "y": city["y"],
            }
        )

    top_city = max(city_snapshot, key=lambda item: item["pm25"])
    safeguard_score = clamp(
        100.0 - max(0.0, float(np.mean(east_coast_aqi)) - 40.0) * 1.25 - max(transported_pm25) * 0.55 + 14.0 * qml_dissipation,
        38.0,
        99.0,
    )

    characteristics = {
        "east_coast_aqi_safeguard": round(safeguard_score, 1),
        "peak_transport_pm25": round(max(transported_pm25), 1),
        "average_plume_half_life_hours": round(float(np.mean(plume_half_life_hours)), 1),
        "highest_risk_city": top_city["city"],
        "current_east_coast_aqi": round(east_coast_aqi[-1], 1),
    }

    narrative = (
        f"Smoke transport from {profile['label']} is propagated eastward over the same 24-month summer horizon. "
        f"Under the current source load, wind corridor and quantum dissipation settings, the highest recurring East Coast exposure is projected at "
        f"{top_city['city']} with {top_city['pm25']:.1f} ug/m3 summer PM2.5, while the mean plume half-life compresses to "
        f"{characteristics['average_plume_half_life_hours']:.1f} hours to uphold downstream air quality metrics."
    )

    return {
        "region_label": profile["label"],
        "month_labels": MONTH_LABELS,
        "east_coast_aqi": east_coast_aqi,
        "transported_pm25": transported_pm25,
        "plume_half_life_hours": plume_half_life_hours,
        "city_series": city_series,
        "city_snapshot": city_snapshot,
        "characteristics": characteristics,
        "narrative": narrative,
    }


@app.route("/api/command-forecast", methods=["GET"])
def api_command_forecast():
    payload = compute_wildfire_command(
        region=request.args.get("region", "northern_ontario"),
        crews=request.args.get("crews", 92),
        tankers=request.args.get("tankers", 18),
        humidity=request.args.get("humidity", 52),
        qml_weight=request.args.get("qml", 0.62),
    )
    return jsonify(payload)


@app.route("/api/containment-strategy", methods=["GET"])
def api_containment_strategy():
    payload = compute_containment_strategy(
        region=request.args.get("region", "northern_ontario"),
        crews=request.args.get("crews", 118),
        firebreak_km=request.args.get("firebreak", 145),
        restoration=request.args.get("restoration", 70),
        prescribed_burn=request.args.get("burn", 54),
        qml_weight=request.args.get("qml", 0.66),
    )
    return jsonify(payload)


@app.route("/api/smoke-propagation", methods=["GET"])
def api_smoke_propagation():
    payload = compute_smoke_propagation(
        region=request.args.get("region", "northern_ontario"),
        source_intensity=request.args.get("source", 1.15),
        easterly_flow=request.args.get("flow", 24),
        humidity_scrub=request.args.get("humidity", 58),
        qml_dissipation=request.args.get("qml", 0.64),
    )
    return jsonify(payload)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9100))
    app.run(debug=True, host="0.0.0.0", port=port)
