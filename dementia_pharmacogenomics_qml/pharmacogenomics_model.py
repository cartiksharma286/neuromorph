#!/usr/bin/env python3
"""
Pharmacogenomics & Multi-Compartment PK/PD Engine for Dementia Therapeutics
Models genomic polymorphisms (APOE, CYP2D6, CYP2C19, APP, PSEN1, PSEN2, TREM2),
drug clearance kinetics, ARIA-E risk, and photonic blood-brain barrier permeability modulation.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional

class PharmacogenomicsEngine:
    """
    Translates genetic profile into drug metabolism, synaptic target engagement,
    and blood-brain barrier transport kinetics.
    """
    
    # APOE Allele risk and amyloid-beta 42 clearance coefficients
    APOE_PROFILES = {
        "e2/e2": {"risk_multiplier": 0.4, "clearance_eff": 1.25, "aria_risk_base": 0.02, "label": "APOE ε2/ε2 (Protective)"},
        "e2/e3": {"risk_multiplier": 0.6, "clearance_eff": 1.15, "aria_risk_base": 0.04, "label": "APOE ε2/ε3 (Low Risk)"},
        "e3/e3": {"risk_multiplier": 1.0, "clearance_eff": 1.00, "aria_risk_base": 0.08, "label": "APOE ε3/ε3 (Neutral/Wild-Type)"},
        "e3/e4": {"risk_multiplier": 3.4, "clearance_eff": 0.65, "aria_risk_base": 0.22, "label": "APOE ε3/ε4 (Heterozygous High Risk)"},
        "e4/e4": {"risk_multiplier": 12.0, "clearance_eff": 0.35, "aria_risk_base": 0.42, "label": "APOE ε4/ε4 (Homozygous Severe Risk)"}
    }
    
    # CYP2D6 activity scores and hepatic clearance factors
    CYP2D6_PROFILES = {
        "PM": {"activity_score": 0.0, "clearance_factor": 0.25, "half_life_mult": 2.8, "desc": "Poor Metabolizer (High AUC, toxicity risk)"},
        "IM": {"activity_score": 0.75, "clearance_factor": 0.65, "half_life_mult": 1.5, "desc": "Intermediate Metabolizer"},
        "NM": {"activity_score": 1.5, "clearance_factor": 1.00, "half_life_mult": 1.0, "desc": "Normal/Extensive Metabolizer (Baseline)"},
        "UM": {"activity_score": 2.5, "clearance_factor": 2.10, "half_life_mult": 0.5, "desc": "Ultrarapid Metabolizer (Sub-therapeutic risk)"}
    }

    # Drug pharmacological reference parameters
    DRUG_LIBRARY = {
        "Donepezil": {
            "type": "AChE_Inhibitor",
            "cyp_dependence": "CYP2D6",
            "base_clearance_L_h": 9.6,
            "volume_dist_L": 840.0,
            "base_half_life_h": 70.0,
            "std_daily_dose_mg": 10.0,
            "therapeutic_window_ng_ml": (30.0, 75.0),
            "mechanism": "Reversible inhibition of acetylcholinesterase, elevating synaptic ACh."
        },
        "Memantine": {
            "type": "NMDA_Antagonist",
            "cyp_dependence": "Renal_Predominant",
            "base_clearance_L_h": 8.1,
            "volume_dist_L": 650.0,
            "base_half_life_h": 65.0,
            "std_daily_dose_mg": 20.0,
            "therapeutic_window_ng_ml": (70.0, 150.0),
            "mechanism": "Uncompetitive NMDA receptor open-channel blocker, mitigating excitotoxicity."
        },
        "Galantamine": {
            "type": "AChE_Inhibitor_Allosteric",
            "cyp_dependence": "CYP2D6",
            "base_clearance_L_h": 14.5,
            "volume_dist_L": 175.0,
            "base_half_life_h": 7.0,
            "std_daily_dose_mg": 16.0,
            "therapeutic_window_ng_ml": (25.0, 80.0),
            "mechanism": "Dual AChE inhibition and allosteric nicotinic acetylcholine modulation."
        },
        "Rivastigmine": {
            "type": "Dual_AChE_BuChE_Inhibitor",
            "cyp_dependence": "Cholinesterase_Hydrolysis",
            "base_clearance_L_h": 1.8,
            "volume_dist_L": 35.0,
            "base_half_life_h": 1.5,
            "std_daily_dose_mg": 9.5,
            "therapeutic_window_ng_ml": (5.0, 20.0),
            "mechanism": "Pseudo-irreversible inhibition of both AChE and butyrylcholinesterase."
        },
        "Lecanemab": {
            "type": "Anti_Abeta_Protofibril_mAb",
            "cyp_dependence": "RES_Catabolism",
            "base_clearance_L_h": 0.018,
            "volume_dist_L": 4.5,
            "base_half_life_h": 168.0, # ~7 days
            "std_daily_dose_mg": 10.0, # mg/kg biweekly
            "therapeutic_window_ng_ml": (50000.0, 150000.0),
            "mechanism": "Humanized IgG1 mAb targeting soluble neurotoxic amyloid-beta protofibrils."
        },
        "Donanemab": {
            "type": "Anti_Pyroglutamate_Abeta_mAb",
            "cyp_dependence": "RES_Catabolism",
            "base_clearance_L_h": 0.015,
            "volume_dist_L": 4.2,
            "base_half_life_h": 288.0, # ~12 days
            "std_daily_dose_mg": 1400.0, # mg monthly
            "therapeutic_window_ng_ml": (80000.0, 220000.0),
            "mechanism": "Targets established amyloid plaques via N-terminally truncated pyroglutamate-3 Aβ."
        }
    }

    @classmethod
    def evaluate_patient_profile(cls,
                                 apoe: str = "e3/e4",
                                 cyp2d6: str = "IM",
                                 early_onset_mutation: bool = False,
                                 trem2_variant: bool = False,
                                 selected_drugs: Optional[List[str]] = None,
                                 photonic_assisted_bbb: bool = True) -> Dict[str, Any]:
        """
        Calculates precision dosing, clearance modifications, toxicity risks,
        and blood-brain barrier transport coefficients.
        """
        if selected_drugs is None:
            selected_drugs = ["Donepezil", "Lecanemab"]
            
        apoe_info = cls.APOE_PROFILES.get(apoe, cls.APOE_PROFILES["e3/e4"])
        cyp_info = cls.CYP2D6_PROFILES.get(cyp2d6, cls.CYP2D6_PROFILES["IM"])
        
        # Calculate early onset factor
        eoad_severity_mult = 1.85 if early_onset_mutation else 1.0
        trem2_mult = 1.45 if trem2_variant else 1.0
        
        # Aggregate amyloid-beta pathology drive
        abeta_pathology_drive = (
            apoe_info["risk_multiplier"] * eoad_severity_mult * trem2_mult
        )
        
        # Photonic enhancement of BBB permeability:
        # Pulsed 810 nm / 1064 nm light induces localized endothelial NO release,
        # reversibly relaxing claudin-5 tight junctions and boosting BBB penetration
        bbb_permeability_ratio = 3.25 if photonic_assisted_bbb else 1.0
        
        drug_evaluations = {}
        for drug_name in selected_drugs:
            if drug_name not in cls.DRUG_LIBRARY:
                continue
            d_spec = cls.DRUG_LIBRARY[drug_name]
            
            # Determine clearance based on pharmacogenomic status
            if d_spec["cyp_dependence"] == "CYP2D6":
                cl_factor = cyp_info["clearance_factor"]
                effective_hl = d_spec["base_half_life_h"] * cyp_info["half_life_mult"]
            else:
                cl_factor = 1.0
                effective_hl = d_spec["base_half_life_h"]
                
            # Adjusted optimal daily dose (mg)
            adjusted_dose = d_spec["std_daily_dose_mg"] * cl_factor
            
            # If photonic excitation assists BBB transport, reduce mAb requirement to spare ARIA risk
            if "mAb" in d_spec["type"] and photonic_assisted_bbb:
                adjusted_dose *= 0.65  # 35% dose reduction with equal or superior central ISF concentration
                
            # Calculate ARIA-E risk for monoclonal antibodies
            aria_risk = 0.0
            if "mAb" in d_spec["type"]:
                aria_risk = apoe_info["aria_risk_base"] * (1.3 if early_onset_mutation else 1.0)
                if photonic_assisted_bbb:
                    # Dose sparing reduces microvascular inflammatory amyloid leakage
                    aria_risk *= 0.42
                    
            drug_evaluations[drug_name] = {
                "base_dose_mg": d_spec["std_daily_dose_mg"],
                "precision_dose_mg": round(float(adjusted_dose), 2),
                "effective_half_life_hours": round(float(effective_hl), 1),
                "clearance_factor": round(float(cl_factor), 2),
                "brain_penetration_factor": round(float(bbb_permeability_ratio), 2),
                "aria_e_risk_probability": round(float(aria_risk), 3),
                "mechanism": d_spec["mechanism"]
            }
            
        return {
            "apoe_genotype": apoe,
            "apoe_label": apoe_info["label"],
            "cyp2d6_status": cyp2d6,
            "cyp2d6_desc": cyp_info["desc"],
            "early_onset_mutation": early_onset_mutation,
            "trem2_variant": trem2_variant,
            "pathology_progression_index": round(float(abeta_pathology_drive), 2),
            "photonic_bbb_enhanced": photonic_assisted_bbb,
            "bbb_flux_multiplier": round(float(bbb_permeability_ratio), 2),
            "drugs": drug_evaluations
        }

    @classmethod
    def simulate_pharmacokinetic_trajectory(cls,
                                           drug_name: str,
                                           dose_mg: float,
                                           cyp_status: str = "IM",
                                           photonic_active: bool = True,
                                           hours: int = 168,
                                           dt: float = 0.5) -> Dict[str, List[float]]:
        """
        Simulates two-compartment (Plasma and Brain Interstitial Fluid) PK curve.
        dC_plasma/dt = ka * F * D / Vd - (CL / Vd) * C_plasma - k_in * C_plasma + k_out * C_brain
        dC_brain/dt  = k_in * C_plasma - (k_out + k_elim) * C_brain
        """
        d_spec = cls.DRUG_LIBRARY.get(drug_name, cls.DRUG_LIBRARY["Donepezil"])
        cyp = cls.CYP2D6_PROFILES.get(cyp_status, cls.CYP2D6_PROFILES["IM"])
        
        cl = d_spec["base_clearance_L_h"] * (cyp["clearance_factor"] if d_spec["cyp_dependence"] == "CYP2D6" else 1.0)
        vd = d_spec["volume_dist_L"]
        
        # BBB transfer rate constants
        k_in_base = 0.045
        if photonic_active:
            k_in_base *= 3.25  # Photonic enhancement
        k_out = 0.08
        k_elim_brain = 0.03
        
        time_steps = int(hours / dt)
        t_arr = [i * dt for i in range(time_steps)]
        
        c_plasma = 0.0
        c_brain = 0.0
        plasma_trace = []
        brain_trace = []
        
        # Daily dosing event interval (every 24h)
        dose_interval = int(24.0 / dt)
        
        for step, t in enumerate(t_arr):
            if step % dose_interval == 0:
                # Instant absorption approximation into plasma
                c_plasma += (dose_mg * 1000.0) / vd
                
            d_cp = - (cl / vd) * c_plasma - k_in_base * c_plasma + k_out * c_brain
            d_cb = k_in_base * c_plasma - (k_out + k_elim_brain) * c_brain
            
            c_plasma = max(0.0, c_plasma + d_cp * dt)
            c_brain = max(0.0, c_brain + d_cb * dt)
            
            plasma_trace.append(round(float(c_plasma), 3))
            brain_trace.append(round(float(c_brain), 3))
            
        return {
            "time_hours": t_arr,
            "plasma_concentration_ng_ml": plasma_trace,
            "brain_concentration_ng_ml": brain_trace
        }
