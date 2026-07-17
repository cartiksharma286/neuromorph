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


def compute_bioremediation(
    region="northern_ontario",
    myco_dose=65.0,
    phyto_coverage=72.0,
    bio_density=58.0,
    biostim_factor=50.0,
    optimal_weight=0.55,
):
    """
    Classical LQR-optimal bioremediation for post-wildfire soil and ecosystem recovery.
    State: r(t) = soil remediation index in [0, 100]
    Cost:  J = sum_t [ Q*(100 - r(t))^2 + R*||u(t)||^2 ]
    Optimal gain: K* = sqrt(Q / R)  (infinite-horizon Riccati solution)
    Control: u*(t) = K* * (100 - r(t)) / 100  (state feedback)
    """
    profile = REGION_PROFILES.get(region, REGION_PROFILES["northern_ontario"])
    myco_dose = clamp(float(myco_dose), 0.0, 100.0)
    phyto_coverage = clamp(float(phyto_coverage), 0.0, 100.0)
    bio_density = clamp(float(bio_density), 0.0, 100.0)
    biostim_factor = clamp(float(biostim_factor), 0.0, 100.0)
    optimal_weight = clamp(float(optimal_weight), 0.0, 1.0)

    u_m = myco_dose / 100.0
    u_p = phyto_coverage / 100.0
    u_b = bio_density / 100.0
    u_s = biostim_factor / 100.0

    # LQR cost weights — higher optimal_weight penalises control effort more
    Q = 1.0 + 0.5 * (1.0 - optimal_weight)
    R = 0.3 + 0.7 * optimal_weight

    # Process effectiveness coefficients
    gamma = 0.04 + 0.02 * profile["remoteness"]
    alpha_m = 0.038 + 0.012 * u_s      # mycoremediation
    alpha_p = 0.029 + 0.008 * profile["watershed_recovery"]  # phytoremediation
    alpha_b = 0.022 + 0.010 * u_s      # bioaugmentation

    # Infinite-horizon LQR optimal gain
    K_star = math.sqrt(Q / R)

    soil_remediation_index = []
    microbial_activity = []
    phyto_cover_pct = []
    toxin_degradation_pct = []
    carbon_sequestration_kgha = []
    optimal_control_effort = []

    r = 12.0 + 5.0 * (1.0 - profile["remoteness"])  # post-fire initial soil index
    toxin = 0.0
    carbon_total = 0.0
    J_running = 0.0
    phyto = 4.0

    for idx in range(HORIZON_MONTHS):
        season = seasonal_fire_pressure(idx)

        # Optimal state-feedback control
        error = 100.0 - r
        u_opt = clamp(K_star * error / 100.0, 0.0, 1.2)

        # Euler integration of soil remediation state
        dr = (alpha_m * u_m + alpha_p * u_p + alpha_b * u_b) * u_opt * 100.0
        dr -= gamma * r
        dr -= 3.5 * max(0.0, season - 0.85)   # fire season disturbance
        dr += 0.8 * u_s                         # biostimulation boost
        r = clamp(r + dr, r * 0.90, 100.0)

        # Microbial activity (sigmoidal response to soil index)
        microbe = 22.0 + 60.0 / (1.0 + math.exp(-0.12 * (r - 44.0)))
        microbe += 8.0 * u_m + 5.0 * u_s - 4.5 * max(0.0, season - 0.9)
        microbe = clamp(microbe, 16.0, 97.0)

        # Phytoremediation cover — logistic growth
        phyto += u_p * 3.8 * (1.0 - phyto / 95.0) * (0.6 + 0.4 * profile["watershed_recovery"])
        phyto -= 1.2 * max(0.0, season - 0.88)
        phyto = clamp(phyto, 2.0, 95.0)

        # Cumulative toxin degradation
        dtoxin = 0.9 * u_m + 0.5 * u_b + 0.3 * microbe / 100.0 + 0.2 * u_s
        dtoxin *= 0.7 + 0.3 * r / 100.0
        toxin = clamp(toxin + dtoxin, 0.0, 100.0)

        # Carbon sequestration (kg/ha/month)
        monthly_carbon = 18.0 * u_p * (phyto / 100.0) * profile["watershed_recovery"]
        monthly_carbon += 6.0 * u_m * (r / 100.0)
        monthly_carbon = clamp(monthly_carbon, 0.5, 38.0)
        carbon_total += monthly_carbon

        # Normalised aggregate control effort
        u_total = u_opt * (u_m + u_p + u_b + u_s * 0.5) / 3.5
        J_running += Q * error ** 2 + R * u_total ** 2

        soil_remediation_index.append(round(r, 2))
        microbial_activity.append(round(microbe, 2))
        phyto_cover_pct.append(round(phyto, 2))
        toxin_degradation_pct.append(round(toxin, 2))
        carbon_sequestration_kgha.append(round(carbon_total, 1))
        optimal_control_effort.append(round(u_total, 3))

    recovery_month = next((i + 1 for i, v in enumerate(soil_remediation_index) if v >= 85.0), HORIZON_MONTHS)
    control_efficiency = clamp(100.0 - J_running / (HORIZON_MONTHS * 800.0), 38.0, 98.0)
    first_summer_idxs = [i for i in SUMMER_INDICES if i < 12]
    second_summer_idxs = [i for i in SUMMER_INDICES if i >= 12]
    first_s = round(float(np.mean([soil_remediation_index[i] for i in first_summer_idxs])), 1) if first_summer_idxs else soil_remediation_index[0]
    second_s = round(float(np.mean([soil_remediation_index[i] for i in second_summer_idxs])), 1) if second_summer_idxs else soil_remediation_index[-1]

    characteristics = {
        "final_soil_index": round(soil_remediation_index[-1], 1),
        "optimal_recovery_month": int(recovery_month),
        "total_carbon_seq_kgha": round(carbon_total, 1),
        "toxin_clearance_pct": round(toxin_degradation_pct[-1], 1),
        "control_efficiency_score": round(control_efficiency, 1),
        "final_phyto_cover": round(phyto_cover_pct[-1], 1),
        "first_summer_soil_index": first_s,
        "second_summer_soil_index": second_s,
        "final_microbial_index": round(microbial_activity[-1], 1),
    }

    narrative = (
        f"Classical LQR-optimal bioremediation for {profile['label']} applies mycoremediation (u_m={u_m:.2f}), "
        f"phytoremediation (u_p={u_p:.2f}), and bioaugmentation (u_b={u_b:.2f}) under cost J = sum Q*(100-r)^2 + R*u^2 "
        f"with LQR gain K* = sqrt(Q/R) = {K_star:.3f}. "
        f"The soil remediation index reaches {characteristics['final_soil_index']:.1f}/100 by month {HORIZON_MONTHS}, "
        f"with the 85/100 optimal threshold attained at month {characteristics['optimal_recovery_month']}. "
        f"Cumulative carbon sequestration is {characteristics['total_carbon_seq_kgha']:.0f} kg/ha "
        f"with {characteristics['toxin_clearance_pct']:.1f}% toxin clearance across the 24-month horizon."
    )

    return {
        "region_label": profile["label"],
        "month_labels": MONTH_LABELS,
        "soil_remediation_index": soil_remediation_index,
        "microbial_activity": microbial_activity,
        "phyto_cover_pct": phyto_cover_pct,
        "toxin_degradation_pct": toxin_degradation_pct,
        "carbon_sequestration_kgha": carbon_sequestration_kgha,
        "optimal_control_effort": optimal_control_effort,
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


@app.route("/api/bioremediation", methods=["GET"])
def api_bioremediation():
    payload = compute_bioremediation(
        region=request.args.get("region", "northern_ontario"),
        myco_dose=request.args.get("myco", 65),
        phyto_coverage=request.args.get("phyto", 72),
        bio_density=request.args.get("bio", 58),
        biostim_factor=request.args.get("biostim", 50),
        optimal_weight=request.args.get("weight", 0.55),
    )
    return jsonify(payload)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9100))
    app.run(debug=True, host="0.0.0.0", port=port)
