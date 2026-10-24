"""
Research-only dental cavity drill-path and finite-element surrogate simulator.

The model renders a deterministic planning visualization. It is not connected to
hardware, does not consume patient data, and must not be used as a clinical plan.
"""

from __future__ import annotations

import math

import numpy as np


class RecurrentPrimeDrillPathEngine:
    """Construct a bounded path using continued-fraction phase corrections."""

    _PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)

    @staticmethod
    def _sqrt_continued_fraction(radicand: int, terms: int = 6) -> list[int]:
        """Return the first terms in the periodic continued fraction of sqrt(n)."""
        a0 = math.isqrt(radicand)
        if a0 * a0 == radicand:
            return [a0]

        values = [a0]
        m, denominator, coefficient = 0, 1, a0
        for _ in range(terms - 1):
            m = denominator * coefficient - m
            denominator = (radicand - m * m) // denominator
            coefficient = (a0 + m) // denominator
            values.append(coefficient)
        return values

    @staticmethod
    def _convergent(coefficients: list[int]) -> tuple[int, int]:
        numerator_prev, numerator = 0, 1
        denominator_prev, denominator = 1, 0
        for coefficient in coefficients:
            numerator_prev, numerator = numerator, coefficient * numerator + numerator_prev
            denominator_prev, denominator = denominator, coefficient * denominator + denominator_prev
        return numerator, denominator

    def simulate(
        self,
        planned_depth_mm: float = 4.0,
        recurrence_amplitude_mm: float = 0.075,
        prime_count: int = 8,
    ) -> dict:
        """
        Return a visual-only optimal-path proxy and a 2-D FEA-style material field.

        A squared terminal taper keeps entry and terminal coordinates unchanged,
        while recurrent prime harmonics are phase-corrected by sqrt(p) convergents.
        """
        prime_count = min(prime_count, len(self._PRIMES))
        primes = self._PRIMES[:prime_count]

        prime_terms = []
        for prime in primes:
            coefficients = self._sqrt_continued_fraction(prime)
            numerator, denominator = self._convergent(coefficients)
            prime_terms.append(
                {
                    "prime": prime,
                    "continued_fraction": coefficients,
                    "convergent": f"{numerator}/{denominator}",
                    "phase_fraction": round((numerator % denominator) / denominator, 7),
                }
            )

        path_points = []
        trace = []
        for progress in np.linspace(0.0, 1.0, 81):
            recurrent_sum = 0.0
            for term in prime_terms:
                phase = 2.0 * math.pi * term["phase_fraction"]
                weight = math.log(term["prime"]) / math.sqrt(term["prime"])
                recurrent_sum += weight * math.sin(
                    2.0 * math.pi * term["prime"] * progress + phase
                )

            normalized_sum = recurrent_sum / sum(
                math.log(prime) / math.sqrt(prime) for prime in primes
            )
            terminal_taper = progress * (1.0 - progress) ** 2
            lateral_offset = recurrence_amplitude_mm * terminal_taper * normalized_sum
            depth_mm = planned_depth_mm * progress

            cutting_force_n = 4.8 + 1.7 * (1.0 - math.exp(-2.4 * progress))
            cutting_force_n += 0.24 * abs(normalized_sum)
            tissue_stress_mpa = 0.34 + 0.48 * (progress**1.15) + 0.10 * abs(normalized_sum)
            tool_stress_mpa = 52.0 + 31.0 * progress + 3.6 * abs(normalized_sum)

            path_points.append(
                {
                    "depth_mm": round(float(depth_mm), 5),
                    "lateral_offset_mm": round(float(lateral_offset), 6),
                }
            )
            trace.append(
                {
                    "depth_mm": round(float(depth_mm), 5),
                    "cutting_force_n": round(float(cutting_force_n), 4),
                    "tissue_stress_mpa": round(float(tissue_stress_mpa), 4),
                    "tool_stress_mpa": round(float(tool_stress_mpa), 4),
                    "lateral_offset_mm": round(float(lateral_offset), 6),
                }
            )

        x_coordinates = np.linspace(-4.5, 4.5, 52)
        depth_coordinates = np.linspace(-0.8, planned_depth_mm + 1.2, 38)
        stress_field = []
        material_map = []

        for depth in depth_coordinates:
            stress_row = []
            material_row = []
            depth_fraction = min(max(depth / planned_depth_mm, 0.0), 1.0)
            interpolated_path = np.interp(
                depth_fraction,
                [point["depth_mm"] / planned_depth_mm for point in path_points],
                [point["lateral_offset_mm"] for point in path_points],
            )
            for x_coord in x_coordinates:
                radial_distance = abs(x_coord - interpolated_path)
                is_drill = 0.0 <= depth <= planned_depth_mm and radial_distance <= 0.26
                is_soft_tissue = -0.8 <= depth < 0.0
                is_pulp = depth >= planned_depth_mm + 0.45 and abs(x_coord) < 1.25

                if is_drill:
                    material = 4
                elif is_soft_tissue:
                    material = 1
                elif is_pulp:
                    material = 3
                else:
                    material = 2

                tip_distance = ((radial_distance / 0.48) ** 2) + (
                    ((depth - planned_depth_mm) / 0.72) ** 2
                )
                interface_distance = ((radial_distance / 0.88) ** 2) + (
                    ((depth - 0.12) / 0.55) ** 2
                )
                stress = 0.08 + 1.15 * math.exp(-tip_distance) + 0.34 * math.exp(
                    -interface_distance
                )
                if is_soft_tissue:
                    stress *= 0.42
                if is_drill:
                    stress = 0.0

                stress_row.append(round(float(stress), 5))
                material_row.append(material)
            stress_field.append(stress_row)
            material_map.append(material_row)

        offsets = [abs(point["lateral_offset_mm"]) for point in path_points]
        peak_tissue_stress = max(
            stress
            for row, materials in zip(stress_field, material_map)
            for stress, material in zip(row, materials)
            if material != 4
        )

        return {
            "simulation_scope": (
                "Research visualization only; not patient-specific, not a clinical plan, "
                "and not connected to drill hardware."
            ),
            "model": {
                "name": "Continued-fraction recurrent-prime path optimizer",
                "equation": (
                    "x(u)=A*u*(1-u)^2*sum_k[w_k sin(2*pi*p_k*u+2*pi*c_k)] / sum_k(w_k), "
                    "where c_k is the fractional part of a sqrt(p_k) convergent."
                ),
                "planned_depth_mm": planned_depth_mm,
                "recurrence_amplitude_mm": recurrence_amplitude_mm,
                "prime_terms": prime_terms,
            },
            "path_points": path_points,
            "trace": trace,
            "finite_element_field": {
                "x_coordinates_mm": [round(float(value), 4) for value in x_coordinates],
                "depth_coordinates_mm": [round(float(value), 4) for value in depth_coordinates],
                "stress_mpa": stress_field,
                "material_map": material_map,
                "materials": {
                    "1": "Soft tissue / gingival boundary",
                    "2": "Tooth hard tissue proxy",
                    "3": "Pulp-protection boundary",
                    "4": "Drill bit envelope",
                },
            },
            "characteristics": {
                "entry_error_um": 0.0,
                "terminal_error_um": 0.0,
                "maximum_lateral_correction_um": round(max(offsets) * 1000.0, 2),
                "peak_tissue_stress_mpa": round(float(peak_tissue_stress), 3),
                "peak_tool_stress_mpa": round(max(point["tool_stress_mpa"] for point in trace), 2),
                "peak_cutting_force_n": round(max(point["cutting_force_n"] for point in trace), 2),
                "finite_element_model": "2-D deterministic material-field surrogate",
            },
        }
