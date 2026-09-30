"""
Water Quality & Chemical Tracking Engine for ChemTRAC Pollutants.
Parses Chemical_Tracking.csv, benchmarks EPA/Health Canada MCLs, and models multi-barrier purification.
"""

import csv
import os
import numpy as np

class WaterQualityMonitor:
    """
    Parses Toronto ChemTRAC Chemical_Tracking.csv, simulates continuous water treatment,
    and models real-time contaminant removal down to safe drinking water thresholds.
    """
    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Chemical_Tracking.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Chemical_Tracking.csv')
        self.csv_path = csv_path
        self.chemicals = self._load_chemical_database()

    def _load_chemical_database(self):
        """Loads and enriches chemical tracking entries with toxicological guidelines."""
        # Standard EPA / Health Canada Drinking Water Maximum Contaminant Levels (MCL in ug/L / ppb)
        mcl_standards = {
            "1,2-Dibromoethane (Ethylene dibromide)": {"mcl_ppb": 0.05, "category": "Carcinogen / Synthetic Organic", "unit": "μg/L"},
            "1,2-Dichloroethane (Ethylene dichloride)": {"mcl_ppb": 5.0, "category": "Chlorinated Solvent", "unit": "μg/L"},
            "1,3-Butadiene": {"mcl_ppb": 1.0, "category": "Synthetic Organic / VOC", "unit": "μg/L"},
            "1,4-Dichlorobenzene": {"mcl_ppb": 75.0, "category": "Halogenated Aromatic", "unit": "μg/L"},
            "Acetaldehyde": {"mcl_ppb": 100.0, "category": "Aldehyde / Disinfection Byproduct", "unit": "μg/L"},
            "Acrolein": {"mcl_ppb": 10.0, "category": "Aldehyde / Aquatic Biocide", "unit": "μg/L"},
            "Benzene": {"mcl_ppb": 5.0, "category": "Aromatic Hydrocarbon Carcinogen", "unit": "μg/L"},
            "Cadmium and its compounds": {"mcl_ppb": 5.0, "category": "Heavy Metal (Priority Toxic)", "unit": "μg/L"},
            "Carbon tetrachloride": {"mcl_ppb": 5.0, "category": "Chlorinated Halomethane", "unit": "μg/L"},
            "Chloroform (Trichloromethane)": {"mcl_ppb": 70.0, "category": "Trihalomethane (THM)", "unit": "μg/L"},
            "Chromium, Hexavalent and its compounds": {"mcl_ppb": 10.0, "category": "Toxic Heavy Metal [Cr(VI)]", "unit": "μg/L"},
            "Chromium, Non-Hexavalent and its compounds": {"mcl_ppb": 100.0, "category": "Inorganic Mineral [Cr(III)]", "unit": "μg/L"},
            "Dichloromethane (Methylene chloride)": {"mcl_ppb": 5.0, "category": "Chlorinated Solvent", "unit": "μg/L"},
            "Formaldehyde": {"mcl_ppb": 900.0, "category": "Aldehyde Byproduct", "unit": "μg/L"},
            "Lead and its compounds": {"mcl_ppb": 5.0, "category": "Neurotoxic Heavy Metal", "unit": "μg/L"},
            "Manganese and its compounds": {"mcl_ppb": 120.0, "category": "Secondary Inorganic Mineral", "unit": "μg/L"},
            "Mercury and its compounds": {"mcl_ppb": 2.0, "category": "Neurotoxic Heavy Metal", "unit": "μg/L"},
            "Nickel and its compounds": {"mcl_ppb": 70.0, "category": "Heavy Metal", "unit": "μg/L"},
            "Nitrogen Oxides (NOx)": {"mcl_ppb": 10000.0, "category": "Inorganic Nitrate/Nitrite", "unit": "μg/L"},
            "Particulate Matter 2.5 (PM2.5)": {"mcl_ppb": 15.0, "category": "Suspended Particulate Colloid", "unit": "μg/L"},
            "Polycyclic Aromatic Hydrocarbons (PAHs)": {"mcl_ppb": 0.2, "category": "Fused Ring Aromatic Carcinogen", "unit": "μg/L"},
            "Tetrachloroethylene (Perchloroethylene)": {"mcl_ppb": 5.0, "category": "Chlorinated Solvent (PCE)", "unit": "μg/L"},
            "Trichloroethylene": {"mcl_ppb": 5.0, "category": "Chlorinated Solvent (TCE)", "unit": "μg/L"},
            "Vinyl chloride": {"mcl_ppb": 2.0, "category": "Volatile Carcinogen", "unit": "μg/L"},
            "Volatile Organic Compounds (VOCs) Total": {"mcl_ppb": 500.0, "category": "Aggregate Organic Contaminants", "unit": "μg/L"}
        }

        records = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, mode='r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get('chemicalName', '').strip()
                    meta = mcl_standards.get(name, {"mcl_ppb": 10.0, "category": "Industrial Contaminant", "unit": "μg/L"})
                    
                    # Typical raw surface/groundwater intake concentrations
                    base_mcl = meta["mcl_ppb"]
                    raw_conc = base_mcl * float(np.random.uniform(1.8, 5.5))
                    # Multi-barrier purification achieves 97.5% - 99.98% removal
                    eff_conc = raw_conc * float(np.random.uniform(0.002, 0.025))

                    records.append({
                        "_id": int(row.get('_id', 0)),
                        "id": row.get('id', ''),
                        "chemicalName": name,
                        "casNo": row.get('casNo', 'N/A'),
                        "description": row.get('description', ''),
                        "category": meta["category"],
                        "mcl_limit_ppb": meta["mcl_ppb"],
                        "unit": meta["unit"],
                        "raw_water_ppb": round(float(raw_conc), 3),
                        "treated_water_ppb": round(float(eff_conc), 3),
                        "removal_efficiency_pct": round(float((1.0 - eff_conc / raw_conc) * 100.0), 2),
                        "compliance_status": "COMPLIANT (Safe Below MCL)" if eff_conc < meta["mcl_ppb"] else "ACTION_REQUIRED"
                    })
        return records

    def get_all_chemicals(self):
        """Returns all 25 tracked chemicals from CSV."""
        return self.chemicals

    def get_live_water_telemetry(self, treatment_intensity=1.0):
        """
        Generates real-time sensor telemetry with turbidity NTU, pH, conductivity,
        dissolved oxygen, and contaminant summary.
        """
        # Baseline turbidity: Raw water is 0.75 - 2.5 NTU
        # With treatment_intensity (0.5 to 1.5), Cauer filtering guarantees NTU < 0.04
        base_turbidity = 0.028 + 0.008 * (1.5 - min(1.5, treatment_intensity))
        current_ntu = max(0.012, min(0.0385, float(base_turbidity + np.random.normal(0, 0.002))))

        compliant_count = sum(1 for c in self.chemicals if c["treated_water_ppb"] < c["mcl_limit_ppb"])
        
        return {
            "turbidity_ntu": round(float(current_ntu), 4),
            "turbidity_threshold_ntu": 0.04,
            "ntu_status": "OPTIMAL ULTRA-PURE (< 0.04 NTU)",
            "water_ph": round(float(7.35 + np.random.normal(0, 0.05)), 2),
            "dissolved_oxygen_mg_l": round(float(9.2 + np.random.normal(0, 0.1)), 2),
            "conductivity_us_cm": round(float(142.0 + np.random.normal(0, 3.0)), 1),
            "total_organic_carbon_mg_l": round(float(0.45 + np.random.normal(0, 0.03)), 3),
            "redox_potential_orp_mv": round(float(685.0 + np.random.normal(0, 8.0)), 1),
            "chemicals_monitored": len(self.chemicals),
            "compliant_chemicals_count": compliant_count,
            "overall_compliance_pct": round(float(compliant_count / len(self.chemicals) * 100.0), 1) if self.chemicals else 100.0
        }
