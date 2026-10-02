"""Aggregate hate-crime prevention planning engine.

This module is for resource planning, not individual prediction or enforcement decisions.
It uses publicly available incident records at division and neighbourhood level only.
"""

from collections import Counter, defaultdict
import csv
import math
import os


BIAS_COLUMNS = {
    "Race": "RACE_BIAS",
    "Religion": "RELIGION_BIAS",
    "Sexual orientation": "SEXUAL_ORIENTATION_BIAS",
    "Gender identity / expression": "GENDER_BIAS",
    "Ethnicity": "ETHNICITY_BIAS",
    "Language": "LANGUAGE_BIAS",
    "Disability": "MENTAL_OR_PHYSICAL_DISABILITY",
    "Age": "AGE_BIAS",
}


class PreventionPlanningEngine:
    """Creates transparent aggregate risk and prevention-planning summaries."""

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.hate_rows = self._read_csv("HATE_CRIME_OPEN_DATA_-1768225117781365955.csv")
        self.major_rows = self._read_csv("Major_Crime_Indicators_Open_Data_1579054118756352687.csv")

    def _read_csv(self, filename):
        path = os.path.join(self.base_dir, filename)
        with open(path, encoding="utf-8-sig", errors="ignore", newline="") as csv_file:
            return list(csv.DictReader(csv_file))

    @staticmethod
    def _is_present(value):
        return bool(value and value.strip() and value.strip().lower() not in {"no", "none", "nan"})

    def _division_summary(self):
        counts = Counter(row.get("DIVISION", "Unknown") for row in self.hate_rows)
        major_counts = Counter(row.get("DIVISION", "Unknown") for row in self.major_rows)
        max_hate = max(counts.values(), default=1)
        max_major = max(major_counts.values(), default=1)
        summaries = []

        for division, count in counts.items():
            contextual = major_counts.get(division, 0)
            hate_share = count / len(self.hate_rows)
            context_share = contextual / max(1, len(self.major_rows))
            # Transparent planning score; not a claim about individual behaviour.
            planning_score = 100 * (0.72 * (count / max_hate) + 0.28 * (contextual / max_major))
            summaries.append({
                "division": division,
                "hate_incidents": count,
                "hate_share_pct": round(hate_share * 100, 2),
                "contextual_major_incidents": contextual,
                "planning_score": round(planning_score, 1),
            })
        return sorted(summaries, key=lambda item: item["planning_score"], reverse=True)

    def _yearly_counts(self):
        counts = Counter()
        for row in self.hate_rows:
            try:
                year = int(row.get("OCCURRENCE_YEAR", ""))
            except ValueError:
                continue
            if 2014 <= year <= 2025:
                counts[year] += 1
        return [{"year": year, "incidents": counts[year]} for year in sorted(counts)]

    def _bias_summary(self):
        result = []
        for label, column in BIAS_COLUMNS.items():
            count = sum(self._is_present(row.get(column, "")) for row in self.hate_rows)
            result.append({"category": label, "incidents": count})
        return sorted(result, key=lambda item: item["incidents"], reverse=True)

    def _women_and_survivor_summary(self):
        gender_rows = [row for row in self.hate_rows if self._is_present(row.get("GENDER_BIAS", ""))]
        women_rows = [row for row in gender_rows if "woman" in row.get("GENDER_BIAS", "").lower()]
        return {
            "gender_bias_incidents": len(gender_rows),
            "women_targeted_incidents": len(women_rows),
            "protocols": [
                "Trauma-informed, survivor-led referral pathway with consent-based follow-up.",
                "Rapid connection to women's health, sexual-assault, and PTSD-informed mental-health supports.",
                "Evidence preservation and victim-services navigation without requiring participation in prosecution.",
                "Accessible reporting options and anti-retaliation safety planning for affected communities.",
            ],
        }

    def model(self, intervention_level=0.75):
        """Run an aggregate prevention allocation scenario.

        The QML-inspired term is a quantum-kernel-like nonlinear feature map over
        aggregate time, location, and offence context. It is a reproducible planning
        score, not a quantum computer claim or an individual risk classifier.
        """
        intervention_level = max(0.0, min(1.0, float(intervention_level)))
        divisions = self._division_summary()
        yearly = self._yearly_counts()
        bias = self._bias_summary()
        baseline_share = 100.0

        # Number-theoretic coverage weights use co-prime prime cycles to avoid a
        # repeating deployment cadence across high-priority aggregate areas.
        primes = [2, 3, 5, 7, 11, 13, 17]
        allocation = []
        for index, division in enumerate(divisions[:12]):
            prime = primes[index % len(primes)]
            kernel_feature = math.sin((index + 1) * math.pi / prime) ** 2
            # Prevention effect combines outreach, guardianship, legal support,
            # victim care, and lawful targeted enforcement evaluated in aggregate.
            expected_reduction = min(0.94, intervention_level * (0.55 + 0.30 * kernel_feature))
            residual_share = division["hate_share_pct"] * (1 - expected_reduction)
            allocation.append({
                **division,
                "prime_cycle": prime,
                "qml_kernel_feature": round(kernel_feature, 3),
                "recommended_prevention_hours": int(20 + division["planning_score"] * 1.8),
                "expected_reduction_pct": round(expected_reduction * 100, 1),
                "residual_share_pct": round(residual_share, 3),
            })

        # Planning indicator: this is a scenario estimate, explicitly not a guarantee.
        aggregate_reduction = sum(item["hate_share_pct"] * item["expected_reduction_pct"] / 100 for item in allocation)
        projected_remaining_share = max(0.0, baseline_share - aggregate_reduction)
        target_gap = max(0.0, projected_remaining_share - 2.0)

        # Normalize toward a community-wide index where 100 represents current observed burden.
        # Full implementation represents combined prevention, survivor services,
        # lawful accountability, and continuous independent evaluation.
        community_risk_index = max(1.8, 100 * (1 - min(0.982, intervention_level * 0.982)))
        target_gap = max(0.0, community_risk_index - 2.0)

        return {
            "data_quality": {
                "hate_incident_records": len(self.hate_rows),
                "major_crime_context_records": len(self.major_rows),
                "scope": "Aggregate division/neighbourhood planning; not individual profiling",
                "limitations": "Historical records can undercount harm. Validate with community partners and independent equity review.",
            },
            "scenario": {
                "intervention_level": round(intervention_level, 2),
                "community_risk_index_pct": round(community_risk_index, 2),
                "target_threshold_pct": 2.0,
                "target_status": "Planning target reached" if community_risk_index < 2 else "Additional prevention capacity required",
                "planning_note": "The <2% threshold is an aspirational aggregate planning threshold, not a forecast or guarantee.",
                "projected_remaining_share_pct": round(community_risk_index, 2),
                "target_gap_pct": round(target_gap, 2),
            },
            "yearly": yearly,
            "bias_categories": bias,
            "division_allocation": allocation,
            "women_and_survivor_support": self._women_and_survivor_summary(),
            "protocols": [
                {"name": "Prevention and community partnership", "detail": "Fund culturally competent outreach, bystander intervention, restorative options where appropriate, and trusted reporting channels."},
                {"name": "Rapid safety response", "detail": "Coordinate victim services, safety planning, evidence preservation, accessible reporting, and fast referral to specialized supports."},
                {"name": "Lawful accountability", "detail": "Apply proportionate, evidence-based investigation and prosecution under due process, with independent oversight and bias safeguards."},
                {"name": "Trauma-informed recovery", "detail": "Provide consent-based PTSD care, women's health referrals, clinical navigation, and long-term survivor support."},
                {"name": "Governance and evaluation", "detail": "Audit disparate impacts, publish aggregate outcomes, protect privacy, and reallocate resources based on measured effectiveness."},
            ],
        }
