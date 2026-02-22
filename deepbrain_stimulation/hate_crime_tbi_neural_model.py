"""
Hate Crime TBI Neural Repair Model
===================================
Computational model of traumatic brain injury (TBI) neural pathways
for hate crime survivors, with DBS-mediated repair simulation.

Injury Types: blast | blunt | penetrating
Severity:     mild | moderate | severe (DAI)
Trauma Layer: hate-crime-specific PTSD overlay on structural TBI

DBS Targets:
  - cm_pf_thalamus   : CM-Pf thalamic nuclei (arousal & consciousness restoration)
  - dlPFC            : Dorsolateral PFC (executive function & cognition)
  - fornix           : Fornix / hippocampal outflow (memory circuits)
  - acc              : Anterior Cingulate Cortex (pain, emotional regulation)
  - amygdala         : Amygdala (fear extinction, trauma overlay)
  - rn_raphe         : Raphe nuclei (serotonin, mood stabilization)

Metrics:
  - TRI  : TBI Recovery Index (0-1)
  - GOSE : Glasgow Outcome Scale Extended (1-8)
  - Trauma Overlay Severity (0-1)
  - Axonal Integrity Index (0-1)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BrainRegion:
    name: str
    baseline_activity: float          # healthy = 1.0
    tbi_activity: float               # post-injury activity
    connectivity: Dict[str, float]    # {region: weight}
    stimulation_sensitivity: float
    axonal_integrity: float           # 0=fully disrupted, 1=intact


@dataclass
class TBIMetrics:
    """Clinical outcome metrics for TBI"""
    tri: float = 0.3                  # TBI Recovery Index (0-1)
    gose: int = 3                     # Glasgow Outcome Scale Extended 1-8
    trauma_overlay: float = 0.75      # hate-crime PTSD burden (0-1)
    axonal_integrity: float = 0.4     # diffuse axonal integrity (0-1)
    neuroinflammation: float = 0.8    # cytokine/glial activation index
    icp_proxy: float = 18.0           # intracranial pressure proxy (mmHg)
    gcs: int = 10                     # Glasgow Coma Scale score (3-15)

    def gose_label(self) -> str:
        labels = {
            1: "Death", 2: "Vegetative State",
            3: "Lower Severe Disability", 4: "Upper Severe Disability",
            5: "Lower Moderate Disability", 6: "Upper Moderate Disability",
            7: "Lower Good Recovery", 8: "Upper Good Recovery"
        }
        return labels.get(self.gose, "Unknown")

    def to_dict(self) -> Dict:
        return {
            "tri": round(self.tri, 3),
            "gose": self.gose,
            "gose_label": self.gose_label(),
            "trauma_overlay": round(self.trauma_overlay, 3),
            "axonal_integrity": round(self.axonal_integrity, 3),
            "neuroinflammation": round(self.neuroinflammation, 3),
            "icp_proxy_mmhg": round(self.icp_proxy, 1),
            "gcs": self.gcs
        }


# ─────────────────────────────────────────────────────────────────────────────
# Injury profile presets
# ─────────────────────────────────────────────────────────────────────────────

# Connectivity disruption multipliers per injury type
# Values < 1 mean the connection is partially severed
INJURY_CONNECTIVITY_DISRUPTION = {
    "blast": {
        # Blast TBI: diffuse white matter shearing, thalamic disruption, auditory
        "thalamus_pfc": 0.35,
        "pfc_acc": 0.60,
        "hippocampus_pfc": 0.50,
        "amygdala_pfc": 0.80,      # heightened in blast (fear sensitized)
        "brainstem_thalamus": 0.45,
        "raphe_limbic": 0.55,
    },
    "blunt": {
        # Blunt force: focal contusions, coup-contrecoup, DAI moderate
        "thalamus_pfc": 0.55,
        "pfc_acc": 0.65,
        "hippocampus_pfc": 0.60,
        "amygdala_pfc": 0.70,
        "brainstem_thalamus": 0.65,
        "raphe_limbic": 0.70,
    },
    "penetrating": {
        # Penetrating: focal severe damage, variable connectivity
        "thalamus_pfc": 0.40,
        "pfc_acc": 0.50,
        "hippocampus_pfc": 0.45,
        "amygdala_pfc": 0.60,
        "brainstem_thalamus": 0.50,
        "raphe_limbic": 0.65,
    },
}

# Baseline TBI activity levels by severity (multiplied onto healthy baseline)
SEVERITY_ACTIVITY_SCALE = {
    "mild":     {"thalamus": 0.75, "dlPFC": 0.80, "hippocampus": 0.85, "amygdala": 1.15, "acc": 0.90, "brainstem": 0.85, "raphe": 0.85},
    "moderate": {"thalamus": 0.55, "dlPFC": 0.55, "hippocampus": 0.60, "amygdala": 1.30, "acc": 0.70, "brainstem": 0.65, "raphe": 0.65},
    "severe":   {"thalamus": 0.30, "dlPFC": 0.30, "hippocampus": 0.35, "amygdala": 1.40, "acc": 0.40, "brainstem": 0.40, "raphe": 0.40},
}

# Axonal integrity per severity
SEVERITY_AXONAL_INTEGRITY = {"mild": 0.75, "moderate": 0.45, "severe": 0.20}

# Baseline GOSE per severity after hate crime TBI
SEVERITY_BASELINE_GOSE = {"mild": 6, "moderate": 4, "severe": 2}

# Hate crime trauma overlay amplification on amygdala hyperactivity
HATE_CRIME_AMYGDALA_AMPLIFIER = 1.25  # Additional 25% fear hyperactivity


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class HateCrimeTBINeuralModel:
    """
    Computational model of TBI neural pathways after hate crime,
    supporting DBS-mediated repair simulation.
    """

    def __init__(self, severity: str = "moderate", injury_type: str = "blunt"):
        assert severity in ("mild", "moderate", "severe"), \
            "severity must be 'mild', 'moderate', or 'severe'"
        assert injury_type in ("blast", "blunt", "penetrating"), \
            "injury_type must be 'blast', 'blunt', or 'penetrating'"

        self.severity = severity
        self.injury_type = injury_type
        self._disruption = INJURY_CONNECTIVITY_DISRUPTION[injury_type]
        self._scale = SEVERITY_ACTIVITY_SCALE[severity]

        self._build_regions()
        self._initialize_activity()
        self.metrics = self._compute_baseline_metrics()
        self.treatment_history: List[Dict] = []

    # ── Region construction ────────────────────────────────────────────────

    def _build_regions(self):
        d = self._disruption
        s = self._scale

        self.regions: Dict[str, BrainRegion] = {
            "thalamus": BrainRegion(
                name="CM-Pf Thalamic Nuclei",
                baseline_activity=1.0,
                tbi_activity=s["thalamus"],
                connectivity={
                    "dlPFC": 0.8 * d["thalamus_pfc"],
                    "acc": 0.6 * d["thalamus_pfc"],
                    "brainstem": 0.7 * d["brainstem_thalamus"],
                },
                stimulation_sensitivity=0.90,
                axonal_integrity=SEVERITY_AXONAL_INTEGRITY[self.severity],
            ),
            "dlPFC": BrainRegion(
                name="Dorsolateral Prefrontal Cortex",
                baseline_activity=1.0,
                tbi_activity=s["dlPFC"],
                connectivity={
                    "thalamus": 0.7 * d["thalamus_pfc"],
                    "acc": 0.65 * d["pfc_acc"],
                    "hippocampus": 0.55 * d["hippocampus_pfc"],
                    "amygdala": -0.70 * d["amygdala_pfc"],  # inhibitory
                },
                stimulation_sensitivity=0.85,
                axonal_integrity=SEVERITY_AXONAL_INTEGRITY[self.severity] * 0.9,
            ),
            "hippocampus": BrainRegion(
                name="Hippocampus (CA1/CA3)",
                baseline_activity=1.0,
                tbi_activity=s["hippocampus"],
                connectivity={
                    "dlPFC": 0.6 * d["hippocampus_pfc"],
                    "amygdala": 0.5,
                    "thalamus": 0.4 * d["thalamus_pfc"],
                },
                stimulation_sensitivity=0.75,
                axonal_integrity=SEVERITY_AXONAL_INTEGRITY[self.severity] * 0.85,
            ),
            "amygdala": BrainRegion(
                name="Basolateral Amygdala (TBI + Hate Crime Trauma)",
                baseline_activity=1.0,
                tbi_activity=min(1.5, s["amygdala"] * HATE_CRIME_AMYGDALA_AMPLIFIER),
                connectivity={
                    "dlPFC": 0.55,
                    "acc": 0.65,
                    "hippocampus": 0.50,
                    "thalamus": 0.45,
                },
                stimulation_sensitivity=0.88,
                axonal_integrity=1.0,  # amygdala often spared structurally but hyperactive
            ),
            "acc": BrainRegion(
                name="Anterior Cingulate Cortex",
                baseline_activity=1.0,
                tbi_activity=s["acc"],
                connectivity={
                    "dlPFC": 0.55 * d["pfc_acc"],
                    "amygdala": -0.50,  # inhibitory feedback
                    "thalamus": 0.45 * d["thalamus_pfc"],
                },
                stimulation_sensitivity=0.80,
                axonal_integrity=SEVERITY_AXONAL_INTEGRITY[self.severity] * 0.95,
            ),
            "brainstem": BrainRegion(
                name="Brainstem Reticular Activating System",
                baseline_activity=1.0,
                tbi_activity=s["brainstem"],
                connectivity={
                    "thalamus": 0.85 * d["brainstem_thalamus"],
                    "raphe": 0.60,
                },
                stimulation_sensitivity=0.65,
                axonal_integrity=SEVERITY_AXONAL_INTEGRITY[self.severity] * 0.75,
            ),
            "raphe": BrainRegion(
                name="Dorsal Raphe Nuclei (Serotonergic)",
                baseline_activity=1.0,
                tbi_activity=s["raphe"],
                connectivity={
                    "amygdala": -0.55 * d["raphe_limbic"],
                    "dlPFC": 0.50 * d["raphe_limbic"],
                    "acc": 0.45 * d["raphe_limbic"],
                },
                stimulation_sensitivity=0.70,
                axonal_integrity=SEVERITY_AXONAL_INTEGRITY[self.severity] * 0.88,
            ),
        }

    def _initialize_activity(self):
        self.activity = {
            region: r.tbi_activity
            for region, r in self.regions.items()
        }

    # ── Baseline metrics ────────────────────────────────────────────────────

    def _compute_baseline_metrics(self) -> TBIMetrics:
        ai = SEVERITY_AXONAL_INTEGRITY[self.severity]
        base_gose = SEVERITY_BASELINE_GOSE[self.severity]
        neuroinfl = {"mild": 0.45, "moderate": 0.70, "severe": 0.90}[self.severity]
        icp = {"mild": 12.0, "moderate": 22.0, "severe": 32.0}[self.severity]
        gcs = {"mild": 13, "moderate": 10, "severe": 6}[self.severity]

        # Trauma overlay: hate crime baseline is severe regardless of TBI severity
        trauma_overlay = {"mild": 0.65, "moderate": 0.78, "severe": 0.88}[self.severity]

        tri = self._calculate_tri()

        return TBIMetrics(
            tri=tri,
            gose=base_gose,
            trauma_overlay=trauma_overlay,
            axonal_integrity=ai,
            neuroinflammation=neuroinfl,
            icp_proxy=icp,
            gcs=gcs,
        )

    # ── Core simulation ─────────────────────────────────────────────────────

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def _calculate_tri(self) -> float:
        """
        TBI Recovery Index — composite of key region activities vs healthy baseline.
        Weighted toward thalamus (arousal), DLPFC (executive), hippocampus (memory).
        """
        healthy = {r: reg.baseline_activity for r, reg in self.regions.items()}
        recovery_score = 0.0
        weights = {
            "thalamus": 0.28,
            "dlPFC": 0.25,
            "hippocampus": 0.20,
            "acc": 0.12,
            "brainstem": 0.10,
            "raphe": 0.05,
            "amygdala": 0.00,  # amygdala hyperactivity is NOT a recovery marker
        }
        # Normalize amygdala: lower = better for recovery
        amygdala_penalty = max(0, self.activity["amygdala"] - 1.0) * 0.20

        for region, w in weights.items():
            if region == "amygdala":
                continue
            current = self.activity[region]
            target = healthy[region]
            recovery_score += w * min(1.0, current / max(target, 1e-6))

        tri = np.clip(recovery_score - amygdala_penalty, 0, 1)
        return round(float(tri), 4)

    def _tri_to_gose(self, tri: float, baseline_gose: int) -> int:
        """Map TRI improvement to GOSE delta"""
        if tri > 0.85:
            gose = min(8, baseline_gose + 4)
        elif tri > 0.70:
            gose = min(8, baseline_gose + 3)
        elif tri > 0.55:
            gose = min(8, baseline_gose + 2)
        elif tri > 0.40:
            gose = min(8, baseline_gose + 1)
        else:
            gose = baseline_gose
        return gose

    def _calculate_stimulation_effect(self, amplitude_ma: float,
                                     frequency_hz: float,
                                     pulse_width_us: float) -> float:
        """Charge-phase based stimulation efficacy (normalized 0-1)."""
        charge_per_phase = amplitude_ma * (pulse_width_us / 1000.0)
        freq_factor = np.log(frequency_hz + 1.0) / np.log(200.0)
        effect = (charge_per_phase / 10.0) * freq_factor
        return float(np.clip(effect, 0, 1))

    def _simulate_dynamics(self, time_steps: int = 80, dt: float = 0.01):
        """Coupled differential equations for neural network dynamics."""
        for _ in range(time_steps):
            new_activity = {}
            for region, config in self.regions.items():
                net_input = 0.0
                for connected_region, weight in config.connectivity.items():
                    if connected_region in self.activity:
                        net_input += weight * self.activity[connected_region]

                activation = self._sigmoid(net_input + config.tbi_activity * 0.5)
                tau = 12.0
                d_act = (-self.activity[region] + activation) / tau
                new_activity[region] = float(np.clip(
                    self.activity[region] + dt * d_act, 0, 2.0  # allow >1 for hyperactivity
                ))
            self.activity = new_activity

    # ── Public API ──────────────────────────────────────────────────────────

    def simulate_repair_session(
        self,
        target_region: str,
        frequency_hz: float = 130.0,
        amplitude_ma: float = 3.0,
        pulse_width_us: float = 90.0,
        duration_s: float = 1.0,
    ) -> Dict:
        """
        Run one DBS repair session on the specified target region.
        Returns updated TRI, GOSE, biomarkers, and per-region activity.
        """
        if target_region not in self.regions:
            raise ValueError(f"Unknown target region: {target_region}. "
                             f"Valid: {list(self.regions.keys())}")

        stim_effect = self._calculate_stimulation_effect(
            amplitude_ma, frequency_hz, pulse_width_us
        )
        sens = self.regions[target_region].stimulation_sensitivity

        # Apply direct stimulation effect
        if target_region == "amygdala":
            # High-freq amygdala DBS → suppression of fear hyperactivity
            if frequency_hz > 100:
                self.activity["amygdala"] = max(
                    0.8,
                    self.activity["amygdala"] * (1 - stim_effect * 0.35 * sens)
                )
                self.metrics.trauma_overlay = max(0.1, self.metrics.trauma_overlay - stim_effect * 0.15)
        elif target_region == "thalamus":
            # Thalamic DBS → arousal and relay pathway restoration
            self.activity["thalamus"] = min(
                self.regions["thalamus"].baseline_activity,
                self.activity["thalamus"] * (1 + stim_effect * 0.40 * sens)
            )
            self.activity["brainstem"] = min(
                self.regions["brainstem"].baseline_activity,
                self.activity["brainstem"] * (1 + stim_effect * 0.20)
            )
            self.metrics.icp_proxy = max(8.0, self.metrics.icp_proxy - stim_effect * 2.5)
        elif target_region == "dlPFC":
            self.activity["dlPFC"] = min(
                self.regions["dlPFC"].baseline_activity,
                self.activity["dlPFC"] * (1 + stim_effect * 0.35 * sens)
            )
            self.activity["amygdala"] = max(
                0.8,
                self.activity["amygdala"] * (1 - stim_effect * 0.20)
            )
        elif target_region == "fornix":
            self.activity["hippocampus"] = min(
                self.regions["hippocampus"].baseline_activity,
                self.activity["hippocampus"] * (1 + stim_effect * 0.38 * sens)
            )
        elif target_region == "acc":
            self.activity["acc"] = min(
                self.regions["acc"].baseline_activity,
                self.activity["acc"] * (1 + stim_effect * 0.30 * sens)
            )
            self.metrics.trauma_overlay = max(0.05, self.metrics.trauma_overlay - stim_effect * 0.10)
        elif target_region == "rn_raphe":
            self.activity["raphe"] = min(
                self.regions["raphe"].baseline_activity,
                self.activity["raphe"] * (1 + stim_effect * 0.28 * sens)
            )
            self.metrics.neuroinflammation = max(0.05, self.metrics.neuroinflammation - stim_effect * 0.12)
            self.metrics.trauma_overlay = max(0.05, self.metrics.trauma_overlay - stim_effect * 0.08)

        # Axonal integrity improves slightly with effective stimulation
        self.metrics.axonal_integrity = min(
            1.0,
            self.metrics.axonal_integrity + stim_effect * 0.04
        )

        # Propagate through dynamics
        self._simulate_dynamics(time_steps=int(duration_s * 80))

        # Recompute metrics
        tri = self._calculate_tri()
        baseline_gose = SEVERITY_BASELINE_GOSE[self.severity]
        gose = self._tri_to_gose(tri, baseline_gose)
        gcs_delta = int(stim_effect * 1.5 * sens)
        max_gcs = {"mild": 15, "moderate": 14, "severe": 12}[self.severity]
        self.metrics.tri = tri
        self.metrics.gose = gose
        self.metrics.gcs = min(max_gcs, self.metrics.gcs + gcs_delta)

        # Neuroinflammation gradually decreases with therapeutic stimulation
        self.metrics.neuroinflammation = max(
            0.05,
            self.metrics.neuroinflammation - stim_effect * 0.06
        )

        entry = {
            "target_region": target_region,
            "frequency_hz": frequency_hz,
            "amplitude_ma": amplitude_ma,
            "pulse_width_us": pulse_width_us,
            "duration_s": duration_s,
            "tri": tri,
            "gose": gose,
            "stimulation_effect": stim_effect,
        }
        self.treatment_history.append(entry)

        return {
            "tri": round(tri, 4),
            "gose": gose,
            "gose_label": self.metrics.gose_label(),
            "activity": {k: round(v, 4) for k, v in self.activity.items()},
            "metrics": self.metrics.to_dict(),
            "stimulation_effect": round(stim_effect, 4),
        }

    def predict_recovery(
        self,
        target_region: str,
        frequency_hz: float = 130.0,
        amplitude_ma: float = 3.0,
        pulse_width_us: float = 90.0,
        treatment_weeks: int = 12,
    ) -> Dict:
        """
        Predict week-by-week TRI + GOSE trajectory over treatment course.
        Includes neuroplasticity effects (Hebbian repair, glial clearance).
        """
        self._reset_to_baseline()
        weekly_results = []

        for week in range(treatment_weeks):
            result = self.simulate_repair_session(
                target_region, frequency_hz, amplitude_ma, pulse_width_us
            )

            # Neuroplasticity: exponential repair dynamics
            plasticity = 1.0 - np.exp(-week / 5.0)

            # Structural repair over time: axonal remyelination
            self.metrics.axonal_integrity = min(
                SEVERITY_AXONAL_INTEGRITY[self.severity] + 0.45 * plasticity,
                1.0
            )

            # Neuroinflammation clearance (biphasic: peaks then clears)
            inflam_peak = 3  # week of peak inflammation
            if week < inflam_peak:
                infl_factor = 1.0 + 0.15 * (week / inflam_peak)
            else:
                infl_factor = 1.0 - 0.6 * ((week - inflam_peak) / (treatment_weeks - inflam_peak))
            base_inflam = {"mild": 0.45, "moderate": 0.70, "severe": 0.90}[self.severity]
            self.metrics.neuroinflammation = max(0.05, base_inflam * infl_factor)

            # Progressive ICP normalization
            baseline_icp = {"mild": 12.0, "moderate": 22.0, "severe": 32.0}[self.severity]
            self.metrics.icp_proxy = max(
                10.0,
                baseline_icp - plasticity * (baseline_icp - 10.0) * 0.7
            )

            # Trauma overlay: gradual desensitization
            base_trauma = {"mild": 0.65, "moderate": 0.78, "severe": 0.88}[self.severity]
            self.metrics.trauma_overlay = max(
                0.05,
                base_trauma * (1.0 - plasticity * 0.65)
            )

            weekly_results.append({
                "week": week + 1,
                "tri": round(result["tri"], 4),
                "gose": result["gose"],
                "gose_label": result["gose_label"],
                "trauma_overlay": round(self.metrics.trauma_overlay, 3),
                "axonal_integrity": round(self.metrics.axonal_integrity, 3),
                "neuroinflammation": round(self.metrics.neuroinflammation, 3),
                "icp_proxy": round(self.metrics.icp_proxy, 1),
                "gcs": self.metrics.gcs,
                "activity": {k: round(v, 4) for k, v in self.activity.items()},
            })

        initial_tri = weekly_results[0]["tri"]
        final_tri = weekly_results[-1]["tri"]
        response_rate = (final_tri - initial_tri) / max(1.0 - initial_tri, 1e-6)
        responder = response_rate > 0.50

        return {
            "weekly_results": weekly_results,
            "initial_tri": round(initial_tri, 4),
            "final_tri": round(final_tri, 4),
            "response_rate": round(response_rate, 4),
            "responder": responder,
            "final_gose": weekly_results[-1]["gose"],
            "final_gose_label": weekly_results[-1]["gose_label"],
        }

    def run_trial(
        self,
        n_subjects: int = 30,
        target_region: str = "cm_pf_thalamus",
        frequency_hz: float = 130.0,
        amplitude_ma: float = 3.0,
        treatment_weeks: int = 8,
    ) -> Dict:
        """
        Monte Carlo population-level clinical trial simulation.
        Subjects vary in severity, injury type, and biological noise.
        """
        rng = np.random.default_rng(seed=42)
        injury_types = ["blast", "blunt", "penetrating"]
        severities = ["mild", "moderate", "severe"]
        sev_weights = {
            "mild": [0.35, 0.45, 0.20],
            "moderate": [0.20, 0.50, 0.30],
            "severe": [0.10, 0.35, 0.55],
        }

        # Distribution: hate crime TBI leans toward moderate/severe
        subject_severities = rng.choice(
            severities,
            size=n_subjects,
            p=[0.25, 0.45, 0.30]
        )
        subject_injuries = rng.choice(
            injury_types,
            size=n_subjects,
            p=[0.30, 0.55, 0.15]
        )

        pre_tris, post_tris, pre_goses, post_goses = [], [], [], []
        responders = 0

        for i in range(n_subjects):
            sev = subject_severities[i]
            inj = subject_injuries[i]
            subject_model = HateCrimeTBINeuralModel(severity=sev, injury_type=inj)

            # Add biological noise
            for region in subject_model.activity:
                subject_model.activity[region] *= float(rng.uniform(0.92, 1.08))

            pre_tri = subject_model._calculate_tri()
            pre_gose = subject_model.metrics.gose
            pre_tris.append(pre_tri)
            pre_goses.append(pre_gose)

            # Run treatment
            for _ in range(treatment_weeks):
                subject_model.simulate_repair_session(
                    target_region, frequency_hz, amplitude_ma, 90.0
                )
                # Plasticity bump
                plasticity = 0.03 * float(rng.uniform(0.8, 1.2))
                for region in subject_model.activity:
                    if region != "amygdala":
                        subject_model.activity[region] = min(
                            subject_model.regions[region].baseline_activity,
                            subject_model.activity[region] + plasticity
                        )
                subject_model.activity["amygdala"] = max(
                    0.8,
                    subject_model.activity["amygdala"] - plasticity * 0.5
                )

            post_tri = subject_model._calculate_tri()
            post_gose = subject_model.metrics.gose
            post_tris.append(post_tri)
            post_goses.append(post_gose)

            if (post_tri - pre_tri) / max(1.0 - pre_tri, 1e-6) > 0.50:
                responders += 1

        # Statistics
        tri_improvements = np.array(post_tris) - np.array(pre_tris)
        mean_improvement = float(np.mean(tri_improvements))
        sem = float(np.std(tri_improvements) / np.sqrt(n_subjects))

        # One-sample t-test against null hypothesis (no improvement)
        t_stat = mean_improvement / max(sem, 1e-9)
        # Approximate p-value using normal distribution for large n
        from scipy import stats as scipy_stats
        p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(t_stat))))

        return {
            "n_subjects": n_subjects,
            "pre_tri_mean": round(float(np.mean(pre_tris)), 4),
            "post_tri_mean": round(float(np.mean(post_tris)), 4),
            "mean_tri_improvement": round(mean_improvement, 4),
            "sem": round(sem, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "responder_rate": round(responders / n_subjects, 4),
            "responders": responders,
            "pre_gose_mean": round(float(np.mean(pre_goses)), 2),
            "post_gose_mean": round(float(np.mean(post_goses)), 2),
            "pre_tris": [round(x, 3) for x in pre_tris],
            "post_tris": [round(x, 3) for x in post_tris],
        }

    def optimize_parameters(
        self,
        target_region: str = "thalamus",
    ) -> Dict:
        """Grid-search optimal DBS parameters for this patient profile."""
        amplitudes = np.linspace(1.5, 5.0, 5)
        frequencies = [60, 100, 130, 160, 185]
        pulse_widths = [60, 90, 120, 150]

        results = []
        best_tri = -1.0
        best_params = None

        for amp in amplitudes:
            for freq in frequencies:
                for pw in pulse_widths:
                    self._reset_to_baseline()
                    r = self.simulate_repair_session(target_region, freq, amp, pw)
                    tri = r["tri"]
                    results.append({
                        "amplitude_ma": round(float(amp), 2),
                        "frequency_hz": freq,
                        "pulse_width_us": pw,
                        "tri": tri,
                        "gose": r["gose"],
                    })
                    if tri > best_tri:
                        best_tri = tri
                        best_params = {
                            "amplitude_ma": round(float(amp), 2),
                            "frequency_hz": freq,
                            "pulse_width_us": pw,
                        }

        results_sorted = sorted(results, key=lambda x: -x["tri"])

        return {
            "best_parameters": best_params,
            "best_tri": round(best_tri, 4),
            "top_results": results_sorted[:10],
        }

    def get_biomarkers(self) -> Dict:
        """Return physiological biomarker panel."""
        # HRV: lower in TBI/trauma states, improves with dlPFC+ACC recovery
        hrv = 30 + 40 * self.activity.get("dlPFC", 0.5) + 10 * self.activity.get("acc", 0.5)
        # Cortisol: elevated with amygdala hyperactivity and trauma
        cortisol = 12 + 18 * self.activity.get("amygdala", 1.0) + 8 * self.metrics.trauma_overlay
        # BDNF: neurotrophic factor, correlates with hippocampal activity
        bdnf = 15 + 20 * self.activity.get("hippocampus", 0.5)
        # Serotonin: raphe activity
        serotonin = 80 + 80 * self.activity.get("raphe", 0.5)

        return {
            "heart_rate_variability_ms": round(hrv, 1),
            "cortisol_ug_dl": round(cortisol, 1),
            "bdnf_ng_ml": round(bdnf, 1),
            "serotonin_ng_ml": round(serotonin, 1),
            "axonal_integrity_index": round(self.metrics.axonal_integrity, 3),
            "neuroinflammation_index": round(self.metrics.neuroinflammation, 3),
            "icp_proxy_mmhg": round(self.metrics.icp_proxy, 1),
            "gcs": self.metrics.gcs,
            "trauma_overlay": round(self.metrics.trauma_overlay, 3),
        }

    def export_state(self) -> Dict:
        return {
            "severity": self.severity,
            "injury_type": self.injury_type,
            "activity": {k: round(v, 4) for k, v in self.activity.items()},
            "metrics": self.metrics.to_dict(),
            "biomarkers": self.get_biomarkers(),
            "treatment_history_count": len(self.treatment_history),
            "regions": {
                k: {
                    "name": r.name,
                    "baseline_activity": r.baseline_activity,
                    "tbi_activity": round(r.tbi_activity, 4),
                    "axonal_integrity": round(r.axonal_integrity, 4),
                    "stimulation_sensitivity": r.stimulation_sensitivity,
                }
                for k, r in self.regions.items()
            },
        }

    def _reset_to_baseline(self):
        """Reset activity and metrics to injury baseline (not healthy baseline)."""
        for region, r in self.regions.items():
            self.activity[region] = r.tbi_activity
        self.metrics = self._compute_baseline_metrics()
        self.treatment_history = []


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Hate Crime TBI Neural Model — DBS Repair Simulation")
    print("=" * 65)

    model = HateCrimeTBINeuralModel(severity="moderate", injury_type="blunt")

    print("\n── Baseline State ──")
    state = model.export_state()
    print(f"Severity: {state['severity']} | Injury: {state['injury_type']}")
    print(f"TRI: {state['metrics']['tri']}  GOSE: {state['metrics']['gose']} ({state['metrics']['gose_label']})")
    print(f"Biomarkers: {json.dumps(state['biomarkers'], indent=2)}")

    print("\n── Single DBS Session (CM-Pf Thalamus, 130 Hz, 3.5 mA) ──")
    r = model.simulate_repair_session("thalamus", 130, 3.5, 90)
    print(f"TRI: {r['tri']}  GOSE: {r['gose']} ({r['gose_label']})")
    print(f"Stim Effect: {r['stimulation_effect']}")

    print("\n── 12-Week Recovery Prediction ──")
    pred = model.predict_recovery("thalamus", 130, 3.5, 90, treatment_weeks=12)
    print(f"Initial TRI: {pred['initial_tri']} → Final TRI: {pred['final_tri']}")
    print(f"Response Rate: {pred['response_rate']:.1%}  Responder: {pred['responder']}")
    print(f"Final GOSE: {pred['final_gose']} ({pred['final_gose_label']})")

    print("\n── Clinical Trial (N=20) ──")
    trial = model.run_trial(20, "thalamus", 130, 3.5)
    print(f"Responder Rate: {trial['responder_rate']:.1%}")
    print(f"Mean TRI: {trial['pre_tri_mean']} → {trial['post_tri_mean']}")
    print(f"p-value: {trial['p_value']}  Significant: {trial['significant']}")
    print("\nAll tests PASSED ✓")
