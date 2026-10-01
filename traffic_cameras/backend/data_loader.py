"""Data Loader and Feature Enrichment for Toronto Red Light Cameras GeoJSON.

Loads 301 camera nodes from 'Red Light Cameras Data - 4326.geojson', parses coordinates,
and synthesizes traffic flow, dilemma-zone kinematics, and historical enforcement baselines.
"""

import json
import os
import math
import numpy as np
from typing import Dict, List, Any


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Red Light Cameras Data - 4326.geojson")


def load_cameras_dataset() -> List[Dict[str, Any]]:
    """Loads GeoJSON and enriches each intersection node with kinematic and traffic features."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"GeoJSON dataset not found at {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        geo = json.load(f)

    features = geo.get("features", [])
    cameras = []

    np.random.seed(42)

    for idx, feat in enumerate(features):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])

        # GeoJSON is MultiPoint: [[lon, lat]]
        if coords and len(coords) > 0 and len(coords[0]) >= 2:
            lon = float(coords[0][0])
            lat = float(coords[0][1])
        else:
            continue

        cid = props.get("INTERSECTION_ID") or props.get("_id") or (idx + 1)
        name = props.get("NAME") or f"Camera #{cid}"
        st1 = props.get("LINEAR_NAME_FULL_1") or props.get("CLIENT_STREET_1") or "Main St"
        st2 = props.get("LINEAR_NAME_FULL_2") or props.get("CLIENT_STREET_2") or "Cross St"
        district = props.get("DISTRICT") or "Toronto and East York"
        ward = props.get("WARD_1") or "Ward 1"
        police_div = props.get("POLICE_DIVISION_1") or "51"
        activation = props.get("ACTIVATION_DATE") or "2010-01-01"
        rlc_id = props.get("RLC") or str(cid)
        tcs_id = props.get("TCS") or "0000"

        # Categorize speed limit and arterial class
        high_speed_names = ["LAKE SHORE", "STEELES", "EGLINTON", "FINCH", "SHEPPARD", "LAWRENCE", "BLOOR", "KEELE", "DON MILLS", "MARKHAM", "DUFFERIN", "WESTON", "KENNEDY", "VICTORIA PARK"]
        is_high_speed = any(h in str(st1).upper() or h in str(st2).upper() for h in high_speed_names)

        speed_limit_kmh = 60.0 if is_high_speed else (50.0 if district != "Toronto and East York" else 40.0)
        speed_85th_kmh = speed_limit_kmh + float(np.random.normal(7.5, 2.2))
        v0 = speed_85th_kmh / 3.6 # m/s

        # Cross street & geometric dimensions
        cross_lanes = 6 if is_high_speed else 4
        intersection_width_m = round(cross_lanes * 3.5 + 4.0, 1) # width W
        vehicle_length_m = 5.5 # vehicle length L

        # Signal clearance parameters
        yellow_duration_s = round(3.8 if speed_limit_kmh <= 50 else 4.5, 1)
        all_red_duration_s = round(1.5 if speed_limit_kmh <= 50 else 2.0, 1)
        tau_clearance = yellow_duration_s + all_red_duration_s

        # Perception-reaction and deceleration
        reaction_time_s = 1.2
        deceleration_a = 3.2 # m/s^2 (ITE recommended comfortable deceleration)

        # ITE dilemma zone boundaries:
        # Stopping distance X_0 = v0 * t_r + v0^2 / (2 * a)
        # Clearing distance X_c = v0 * tau_clearance - (W + L)
        stopping_dist_m = v0 * reaction_time_s + (v0 ** 2) / (2.0 * deceleration_a)
        clearing_dist_m = v0 * tau_clearance - (intersection_width_m + vehicle_length_m)
        dilemma_zone_gap_m = max(0.0, stopping_dist_m - clearing_dist_m)
        has_dilemma_zone = dilemma_zone_gap_m > 2.0

        # Annual Average Daily Traffic (AADT)
        base_aadt = 32000 if is_high_speed else 21000
        aadt = int(base_aadt + np.random.normal(0, 4500))
        aadt = max(8000, aadt)

        # Base violation rate per month (typical Toronto RLC detects 80 to 450 violations/month)
        # Heavily modulated by dilemma zone gap and 85th percentile speed
        viol_rate_base = 65.0 + 8.5 * dilemma_zone_gap_m + 0.0035 * aadt + float(np.random.normal(0, 12))
        monthly_violations = max(18, int(round(viol_rate_base)))

        # Risk classification
        if monthly_violations > 280 or dilemma_zone_gap_m > 14.0:
            risk_tier = "Severe Risk"
            risk_color = "#dc2626" # red-600
            hazard_index = round(min(1.0, 0.75 + 0.01 * dilemma_zone_gap_m), 3)
        elif monthly_violations > 160 or dilemma_zone_gap_m > 7.0:
            risk_tier = "High Risk"
            risk_color = "#f97316" # orange-500
            hazard_index = round(0.50 + 0.015 * dilemma_zone_gap_m, 3)
        elif monthly_violations > 90 or dilemma_zone_gap_m > 3.0:
            risk_tier = "Moderate Risk"
            risk_color = "#eab308" # yellow-500
            hazard_index = round(0.30 + 0.01 * dilemma_zone_gap_m, 3)
        else:
            risk_tier = "Low Risk"
            risk_color = "#10b981" # emerald-500
            hazard_index = round(0.12 + 0.008 * dilemma_zone_gap_m, 3)

        cameras.append({
            "id": cid,
            "name": name,
            "street_1": st1,
            "street_2": st2,
            "district": district,
            "ward": ward,
            "police_division": police_div,
            "activation_date": str(activation)[:10],
            "rlc_code": rlc_id,
            "tcs_code": tcs_id,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "speed_limit_kmh": speed_limit_kmh,
            "speed_85th_kmh": round(speed_85th_kmh, 1),
            "aadt_volume": aadt,
            "intersection_width_m": intersection_width_m,
            "yellow_duration_s": yellow_duration_s,
            "all_red_duration_s": all_red_duration_s,
            "stopping_dist_m": round(stopping_dist_m, 1),
            "clearing_dist_m": round(clearing_dist_m, 1),
            "dilemma_zone_gap_m": round(dilemma_zone_gap_m, 1),
            "has_dilemma_zone": has_dilemma_zone,
            "monthly_violations": monthly_violations,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "hazard_index": hazard_index
        })

    return cameras
