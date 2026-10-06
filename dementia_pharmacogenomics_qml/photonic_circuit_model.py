#!/usr/bin/env python3
"""
Combinatorial Photonic Neural Circuit Excitation Engine
Simulates multi-wavelength (660, 810, 1064 nm) optogenetic and transcranial photobiomodulation,
40 Hz gamma resonance with parvalbumin interneurons, cytochrome c oxidase activation,
and targeted neural circuit clearance (Entorhinal, Hippocampus, NBM, DMN).
"""

import math
import numpy as np
from typing import Dict, List, Any

class PhotonicNeuralCircuitEngine:
    """
    Computes optical fluence transport through brain tissue, mitochondrial Cytochrome c Oxidase
    activation kinetics, 40 Hz microglial phagocytic entrainment, and combinatorial circuit resonance.
    """
    
    # Anatomical circuit coordinates, depth from skull surface (cm), and baseline degradation rate
    CIRCUITS = {
        "Entorhinal_Cortex": {
            "depth_cm": 3.8,
            "role": "Memory consolidation & spatial navigation (earliest tau tangle nidus)",
            "baseline_connectivity": 0.38,
            "gamma_vulnerability": 0.88,
            "opt_wavelength_nm": 1064
        },
        "Hippocampus_CA1": {
            "depth_cm": 4.5,
            "role": "Synaptic LTP, episodic memory, place cell encoding",
            "baseline_connectivity": 0.42,
            "gamma_vulnerability": 0.92,
            "opt_wavelength_nm": 1064
        },
        "Nucleus_Basalis_Meynert": {
            "depth_cm": 5.2,
            "role": "Subcortical cholinergic projection system to neocortex",
            "baseline_connectivity": 0.30,
            "gamma_vulnerability": 0.85,
            "opt_wavelength_nm": 1064
        },
        "Default_Mode_Network": {
            "depth_cm": 1.5,
            "role": "Posterior cingulate & precuneus resting-state cognitive hub",
            "baseline_connectivity": 0.48,
            "gamma_vulnerability": 0.78,
            "opt_wavelength_nm": 810
        },
        "Dorsolateral_PFC": {
            "depth_cm": 1.8,
            "role": "Executive function, working memory, cognitive flexibility",
            "baseline_connectivity": 0.52,
            "gamma_vulnerability": 0.72,
            "opt_wavelength_nm": 810
        }
    }
    
    # Optical attenuation properties (tissue absorption mu_a and reduced scattering mu_s' in cm^-1)
    OPTICAL_PROPERTIES = {
        660:  {"mu_a": 0.35, "mu_s_prime": 14.5, "cox_affinity": 0.65, "penetration_window": "Cortical Superfic"},
        810:  {"mu_a": 0.16, "mu_s_prime": 9.2,  "cox_affinity": 1.00, "penetration_window": "Optimal NIR-I Deep"},
        1064: {"mu_a": 0.22, "mu_s_prime": 5.8,  "cox_affinity": 0.82, "penetration_window": "High Deep NIR-II"}
    }

    @classmethod
    def compute_tissue_irradiance(cls, wavelength: int, depth_cm: float, surface_irradiance_mw: float) -> float:
        """
        Calculates localized light intensity at depth z via diffusion approximation of radiative transfer:
        mu_eff = sqrt(3 * mu_a * (mu_a + mu_s'))
        I(z) = I_0 * k_geom * exp(-mu_eff * z)
        """
        opt = cls.OPTICAL_PROPERTIES.get(wavelength, cls.OPTICAL_PROPERTIES[810])
        mu_eff = math.sqrt(3.0 * opt["mu_a"] * (opt["mu_a"] + opt["mu_s_prime"]))
        
        # Geometry backscattering factor k_geom ~ 2.8 at tissue boundary
        k_geom = 2.85
        irradiance_at_depth = surface_irradiance_mw * k_geom * math.exp(-mu_eff * depth_cm)
        return float(irradiance_at_depth)

    @classmethod
    def evaluate_combinatorial_excitation(cls,
                                          wavelengths: List[int] = [810, 1064],
                                          frequency_hz: float = 40.0,
                                          surface_irradiance_mw: float = 85.0,
                                          exposure_time_min: float = 30.0,
                                          target_circuits: List[str] = None) -> Dict[str, Any]:
        """
        Evaluates combinatorial multi-circuit excitation response:
        - 40 Hz gamma resonance with parvalbumin-positive interneurons
        - Cytochrome c oxidase (COX) stimulation & ATP rate enhancement
        - Microglial phagocytosis acceleration for Abeta & hyperphosphorylated tau
        - Synaptogenesis & dendritic spine preservation indices
        """
        if target_circuits is None:
            target_circuits = list(cls.CIRCUITS.keys())
            
        # Total fluence delivered (J/cm^2) = Irradiance (W/cm^2) * Time (s)
        surface_fluence_j_cm2 = (surface_irradiance_mw / 1000.0) * (exposure_time_min * 60.0)
        
        # 40 Hz resonance factor: sharp resonance curve around 40 Hz gamma
        # Gamma entrainment drives coordinated microglial engulfment of oligomers
        gamma_bandwidth = 5.0  # Hz
        gamma_resonance = math.exp(-((frequency_hz - 40.0)**2) / (2 * (gamma_bandwidth**2)))
        
        circuit_results = {}
        mean_atp_boost = 0.0
        mean_microglial_activation = 0.0
        mean_clearance_acceleration = 0.0
        
        for circuit_name in target_circuits:
            if circuit_name not in cls.CIRCUITS:
                continue
            c_info = cls.CIRCUITS[circuit_name]
            depth = c_info["depth_cm"]
            
            # Combine light contribution across selected wavelengths
            local_irradiance_total = 0.0
            cox_stimulation_accum = 0.0
            
            for wl in wavelengths:
                irr_depth = cls.compute_tissue_irradiance(wl, depth, surface_irradiance_mw)
                local_irradiance_total += irr_depth
                affinity = cls.OPTICAL_PROPERTIES.get(wl, cls.OPTICAL_PROPERTIES[810])["cox_affinity"]
                cox_stimulation_accum += irr_depth * affinity
                
            # Local fluence at depth (J/cm^2)
            local_fluence = (local_irradiance_total / 1000.0) * (exposure_time_min * 60.0)
            
            # Mitochondrial ATP boost: Biphasic Arndt-Schulz curve
            # Optimal local fluence is ~ 3 to 6 J/cm^2
            opt_local_fluence = 4.5
            atp_factor = (local_fluence / opt_local_fluence) * math.exp(1.0 - local_fluence / opt_local_fluence)
            atp_boost_percent = min(88.0, max(5.0, atp_factor * 68.0))
            
            # Microglial phagocytic clearance index (synergistic with 40 Hz gamma)
            microglial_clearance_mult = 1.0 + (3.4 * gamma_resonance * (atp_boost_percent / 60.0))
            
            # Synaptic connectivity restoration index (0.0 to 1.0)
            conn_restored = min(0.96, c_info["baseline_connectivity"] + 0.38 * gamma_resonance * (atp_boost_percent / 100.0))
            
            circuit_results[circuit_name] = {
                "depth_cm": depth,
                "role": c_info["role"],
                "local_irradiance_mw_cm2": round(float(local_irradiance_total), 3),
                "local_fluence_j_cm2": round(float(local_fluence), 3),
                "atp_synthesis_boost_pct": round(float(atp_boost_percent), 1),
                "gamma_entrainment_fidelity": round(float(gamma_resonance), 3),
                "microglial_clearance_multiplier": round(float(microglial_clearance_mult), 2),
                "post_treatment_connectivity": round(float(conn_restored), 3)
            }
            
            mean_atp_boost += atp_boost_percent
            mean_microglial_activation += gamma_resonance
            mean_clearance_acceleration += microglial_clearance_mult
            
        n_circ = max(1, len(circuit_results))
        mean_atp_boost /= n_circ
        mean_microglial_activation /= n_circ
        mean_clearance_acceleration /= n_circ
        
        return {
            "selected_wavelengths_nm": wavelengths,
            "pulse_frequency_hz": frequency_hz,
            "surface_irradiance_mw_cm2": surface_irradiance_mw,
            "surface_fluence_j_cm2": round(float(surface_fluence_j_cm2), 2),
            "exposure_time_min": exposure_time_min,
            "overall_atp_boost_percent": round(float(mean_atp_boost), 1),
            "gamma_synchrony_score": round(float(mean_microglial_activation), 3),
            "circuit_clearance_acceleration": round(float(mean_clearance_acceleration), 2),
            "circuits": circuit_results
        }
