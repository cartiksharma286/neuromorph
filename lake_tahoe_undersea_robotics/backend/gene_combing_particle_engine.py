"""
Single-Molecule Molecular Gene Combing & Particle Microstate Engine:
Simulates hydrodynamic meniscus-stretching of environmental DNA (eDNA),
fluorescence barcoding, and suspended particle microstates in Lake Tahoe's benthic waters.
"""

import numpy as np
import math
from typing import Dict, List, Any

class GeneCombingParticleEngine:
    """Simulates DNA molecular combing elongation, fluorescence barcodes, and particle states."""

    SPECIES_BARCODES = {
        'cyclotella_gordonensis': {
            'id': 'cyclotella_gordonensis',
            'species': 'Cyclotella gordonensis (Endemic Diatom)',
            'role': 'Primary biological indicator of Lake Tahoe ultra-oligotrophic water clarity',
            'genome_fragment_kb': 168.0,
            'combed_length_um': 84.0,  # At 2.0 kb/um
            'fluorophore': 'Cyan (488 nm)',
            'fluorophore_color': '#06b6d4',
            'marker_gene': 'rbcL Chloroplast Rubisco Large Subunit',
            'probe_intervals_kb': [12.4, 38.2, 71.0, 94.5, 126.8, 152.0]
        },
        'mysis_diluviana': {
            'id': 'mysis_diluviana',
            'species': 'Mysis diluviana (Relict Opossum Shrimp)',
            'role': 'Deep-water benthic zooplankton introduced in 1963; drives pelagic food web shifts',
            'genome_fragment_kb': 220.0,
            'combed_length_um': 110.0,
            'fluorophore': 'Emerald Green (515 nm)',
            'fluorophore_color': '#10b981',
            'marker_gene': '16S / COI Mitochondrial Barcode',
            'probe_intervals_kb': [18.0, 45.5, 82.0, 114.0, 158.2, 189.5, 210.0]
        },
        'anabaena_flos_aquae': {
            'id': 'anabaena_flos_aquae',
            'species': 'Anabaena flos-aquae (Littoral Cyanobacteria)',
            'role': 'Algal bloom / cyanotoxin monitoring in warm embayments (Emerald Bay)',
            'genome_fragment_kb': 140.0,
            'combed_length_um': 70.0,
            'fluorophore': 'Crimson Red (647 nm)',
            'fluorophore_color': '#ef4444',
            'marker_gene': 'mcyA Microcystin Synthetase Cluster',
            'probe_intervals_kb': [14.0, 32.5, 68.0, 99.4, 128.0]
        },
        'aquatic_bacteroidetes': {
            'id': 'aquatic_bacteroidetes',
            'species': 'Bacteroidetes (Urban Runoff Bioindicator)',
            'role': 'eDNA marker for anthropogenic watershed runoff and nearshore storm drains',
            'genome_fragment_kb': 180.0,
            'combed_length_um': 90.0,
            'fluorophore': 'Amber Yellow (570 nm)',
            'fluorophore_color': '#fbbf24',
            'marker_gene': '16S rRNA V4 Hypervariable Hyper-Amplicons',
            'probe_intervals_kb': [22.0, 54.0, 89.2, 134.0, 168.5]
        }
    }

    def __init__(self):
        np.random.seed(42)
        # Meniscus molecular combing stretching constants
        self.stretching_factor_kb_um = 2.0  # 2.0 kb per micron (linearized dsDNA)
        self.meniscus_speed_um_s = 300.0    # 300 um/s optimal recession velocity
        self.surface_tension_mn_m = 72.8    # Water-air interface at 20 C
        self.contact_angle_deg = 105.0      # Silanized hydrophobic glass substrate (e.g. vinyl-silane)

    def simulate_gene_combing(self, species_key: str, shear_rate_s_minus_1: float = 1200.0) -> Dict[str, Any]:
        """
        Simulates microfluidic DNA linearization and optical fluorophore barcode readout.
        """
        spec = self.SPECIES_BARCODES.get(species_key, self.SPECIES_BARCODES['cyclotella_gordonensis'])
        total_kb = spec['genome_fragment_kb']
        
        # Linear extension length L = total_kb / 2.0 (microns)
        # Hydrodynamic shear stretching ratio
        shear_ext_factor = min(1.05, max(0.92, 1.0 + 0.00004 * (shear_rate_s_minus_1 - 1200.0)))
        actual_length_um = (total_kb / self.stretching_factor_kb_um) * shear_ext_factor
        
        # Probe coordinates along combed strand (0 to actual_length_um)
        probe_coords = []
        for p_kb in spec['probe_intervals_kb']:
            pos_um = (p_kb / total_kb) * actual_length_um
            # Random fluorescence intensity (photons / pixel)
            intensity = float(np.random.normal(840.0, 65.0))
            probe_coords.append({
                'locus_kb': p_kb,
                'pos_um': round(float(pos_um), 2),
                'intensity_photons': round(intensity, 1)
            })
            
        # Particle microstates (mineral grains vs Particulate Organic Carbon POC)
        water_clarity_ntu = 0.12 if 'cyclotella' in species_key else 0.45
        poc_ug_l = float(np.random.normal(32.5, 4.2))
        secchi_depth_estimate_m = max(12.0, min(36.0, 31.0 - (water_clarity_ntu * 18.0)))
        
        # Suspended particle distribution (Stokes hydrodynamic drag)
        particle_bins = [
            {'size_um': '0.5 – 1.0', 'fraction_pct': 42.5, 'type': 'Ultra-fine granitic clay'},
            {'size_um': '1.0 – 5.0', 'fraction_pct': 38.0, 'type': 'Diatom frustules & organic flakes'},
            {'size_um': '5.0 – 20.0', 'fraction_pct': 15.5, 'type': 'Feldspar & quartz sand grains'},
            {'size_um': '> 20.0', 'fraction_pct': 4.0, 'type': 'Detrital organic aggregates'}
        ]
        
        return {
            'species_id': spec['id'],
            'species_name': spec['species'],
            'ecological_role': spec['role'],
            'marker_gene': spec['marker_gene'],
            'fluorophore': spec['fluorophore'],
            'fluorophore_color': spec['fluorophore_color'],
            'genome_fragment_kb': total_kb,
            'combed_length_um': round(actual_length_um, 2),
            'stretching_constant': f'{self.stretching_factor_kb_um:.1f} kb/µm (0.50 nm/bp)',
            'meniscus_velocity': f'{self.meniscus_speed_um_s} µm/s',
            'surface_contact_angle': f'{self.contact_angle_deg}° (Vinyl-silane substrate)',
            'probe_barcodes': probe_coords,
            'water_clarity_ntu': water_clarity_ntu,
            'secchi_depth_estimate_m': round(secchi_depth_estimate_m, 1),
            'particulate_organic_carbon_ug_l': round(poc_ug_l, 1),
            'particle_microstate_bins': particle_bins
        }
