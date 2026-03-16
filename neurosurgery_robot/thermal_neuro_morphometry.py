"""Thermal neuro-morphometry analysis with conformal invariants."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter, laplace


class ThermalNeuroMorphometry:
    """Extract conformal and continued-fraction descriptors from thermal fields."""

    def __init__(self, width: int = 128, height: int = 128, baseline_temp_c: float = 37.0):
        self.width = width
        self.height = height
        self.baseline_temp_c = baseline_temp_c
        self._last_result = self._empty_result()

    def _empty_result(self) -> Dict[str, object]:
        zeros = np.zeros((self.height, self.width), dtype=float)
        return {
            'centroid': np.array([0.5, 0.5], dtype=float),
            'conformal_invariant_map': zeros,
            'beltrami_map': zeros,
            'continued_fraction_terms': [1, 1, 1, 1],
            'continued_fraction_value': 1.0,
            'conformal_stability': 1.0,
            'distortion_mean': 0.0,
            'hotspot_area_px': 0,
            'thermal_shape_index': 0.0,
            'continued_fraction_depth': 4,
            'principal_ratio': 1.0,
            'curvature_energy': 0.0,
            'hotspot_mask': np.zeros((self.height, self.width), dtype=bool),
            'delta_t': zeros,
        }

    def _continued_fraction_terms(self, value: float, depth: int = 5) -> List[int]:
        value = float(max(value, 1e-6))
        terms: List[int] = []
        for _ in range(depth):
            integer = int(math.floor(value))
            terms.append(integer)
            frac = value - integer
            if frac < 1e-6:
                break
            value = 1.0 / frac
        if not terms:
            return [1]
        return terms

    def _continued_fraction_value(self, terms: List[int]) -> float:
        acc = 0.0
        for term in reversed(terms):
            acc = float(term) if acc == 0.0 else float(term) + 1.0 / acc
        return acc if acc > 0.0 else 1.0

    def analyze(
        self,
        temperature_map: np.ndarray,
        damage_map: np.ndarray,
        tissue_map: np.ndarray,
    ) -> Dict[str, object]:
        temp = np.asarray(temperature_map, dtype=float)
        damage = np.asarray(damage_map, dtype=float)
        tissue = np.asarray(tissue_map, dtype=float)

        smooth_temp = gaussian_filter(temp, sigma=1.1)
        delta_t = np.clip(smooth_temp - self.baseline_temp_c, 0.0, None)
        damage_norm = damage / (np.max(damage) + 1e-6)

        grad_y, grad_x = np.gradient(delta_t)
        h_xx = np.gradient(grad_x, axis=1)
        h_yy = np.gradient(grad_y, axis=0)
        h_xy = np.gradient(grad_x, axis=0)

        g11 = 1.0 + grad_x * grad_x + 0.20 * damage_norm
        g22 = 1.0 + grad_y * grad_y + 0.20 * damage_norm
        g12 = grad_x * grad_y + 0.05 * h_xy
        trace_g = g11 + g22
        det_g = np.clip(g11 * g22 - g12 * g12, 1e-9, None)

        conformal_invariant = np.clip((2.0 * np.sqrt(det_g)) / (trace_g + 1e-9), 0.0, 1.0)
        beltrami = np.sqrt((g11 - g22) ** 2 + 4.0 * g12 ** 2) / (trace_g + 1e-9)
        curvature = laplace(smooth_temp)
        curvature_energy = float(np.mean(curvature * curvature))

        hotspot_threshold = max(4.0, float(np.percentile(delta_t, 92)))
        hotspot_mask = delta_t >= hotspot_threshold
        if np.sum(hotspot_mask) < 9:
            hotspot_mask = delta_t >= max(2.0, float(np.percentile(delta_t, 80)))

        weights = delta_t * (1.0 + 0.25 * (tissue == 2.0))
        yy, xx = np.indices(temp.shape)
        weight_sum = float(np.sum(weights * hotspot_mask))
        if weight_sum > 1e-6:
            cx = float(np.sum(xx * weights * hotspot_mask) / weight_sum) / max(temp.shape[1] - 1, 1)
            cy = float(np.sum(yy * weights * hotspot_mask) / weight_sum) / max(temp.shape[0] - 1, 1)
        else:
            cx, cy = 0.5, 0.5

        x_centered = (xx / max(temp.shape[1] - 1, 1)) - cx
        y_centered = (yy / max(temp.shape[0] - 1, 1)) - cy
        weighted = weights * hotspot_mask
        mu20 = float(np.sum(weighted * x_centered * x_centered)) + 1e-9
        mu02 = float(np.sum(weighted * y_centered * y_centered)) + 1e-9
        mu11 = float(np.sum(weighted * x_centered * y_centered))

        principal_ratio = (mu20 + mu02 + abs(mu11)) / (min(mu20, mu02) + 1e-9)
        cf_terms = self._continued_fraction_terms(principal_ratio, depth=5)
        cf_value = self._continued_fraction_value(cf_terms)

        thermal_shape_index = float(
            np.mean(conformal_invariant[hotspot_mask]) - 0.5 * np.mean(beltrami[hotspot_mask])
        ) if np.any(hotspot_mask) else 0.0

        self._last_result = {
            'centroid': np.array([cx, cy], dtype=float),
            'conformal_invariant_map': conformal_invariant,
            'beltrami_map': beltrami,
            'continued_fraction_terms': cf_terms,
            'continued_fraction_value': float(cf_value),
            'conformal_stability': float(np.mean(conformal_invariant)),
            'distortion_mean': float(np.mean(beltrami)),
            'hotspot_area_px': int(np.sum(hotspot_mask)),
            'thermal_shape_index': thermal_shape_index,
            'continued_fraction_depth': len(cf_terms),
            'principal_ratio': float(principal_ratio),
            'curvature_energy': curvature_energy,
            'hotspot_mask': hotspot_mask,
            'delta_t': delta_t,
        }
        return self._last_result

    def get_last_result(self) -> Dict[str, object]:
        return self._last_result