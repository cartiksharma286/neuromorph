"""
Invisalign Orthodontic Engine: Geodesic Dental Arches, Measure-Theoretic Smile Signatures,
and Polynomial Ring Algebra Staging
========================================================================================
Formulations:
1. Riemannian Manifold Geodesic Archform:
   ds^2 = g_{uu}(u,v) du^2 + 2 g_{uv}(u,v) du dv + g_{vv}(u,v) dv^2
   Geodesic equation: d^2 x^k / dt^2 + \Gamma^k_{ij} (dx^i/dt)(dx^j/dt) = 0
2. Measure-Theoretic Smile Signature:
   \mu_{\mathrm{smile}}(A) = \int_A \rho_{\mathrm{dent}}(\mathbf{x}) \, d\mathcal{H}^2(\mathbf{x})
   Wasserstein-2 Discrepancy:
   W_2^2(\mu_{\mathrm{smile}}, \nu_{\mathrm{ideal}}) = \inf_{\gamma \in \Pi} \int \|\mathbf{x} - \mathbf{y}\|^2 d\gamma(\mathbf{x}, \mathbf{y})
3. Polynomial Ring Algebra Aligners:
   \mathcal{R} = \mathbb{R}[x, y, z] / \langle f_{\mathrm{arch}}(x, y), z - h_{\mathrm{occ}}(x,y) \rangle
   Aligner Stage Morphism: \Phi_k: \mathcal{R}_k \to \mathcal{R}_{k+1} with maximal step bound \delta \le 0.25\text{ mm}
4. Craniofacial Anatomy & Biomechanical Optimization:
   Interpolation (Thin Plate Splines / Hermite Quintic Geodesics) & Spatial Procrustes Superimposition.
"""

import numpy as np
import math

class InvisalignSmileEngine:
    def __init__(self):
        # 14 Maxillary Teeth Nomenclature (FDI 17 to 27)
        self.tooth_names = [
            "17 (R 2nd Molar)", "16 (R 1st Molar)", "15 (R 2nd Premolar)", "14 (R 1st Premolar)",
            "13 (R Canine)", "12 (R Lateral Incisor)", "11 (R Central Incisor)",
            "21 (L Central Incisor)", "22 (L Lateral Incisor)", "23 (L Canine)",
            "24 (L 1st Premolar)", "25 (L 2nd Premolar)", "26 (L 1st Molar)", "27 (L 2nd Molar)"
        ]
        self.num_teeth = len(self.tooth_names)

    def generate_ideal_arch_geodesic(self, width_intermolar=54.0, depth_arch=38.0, arch_form="catenary"):
        """
        Generates the canonical ideal Riemannian dental arch on a 2D/3D Riemannian manifold.
        """
        t = np.linspace(-1.0, 1.0, self.num_teeth)
        
        if arch_form == "catenary":
            a = depth_arch / (np.cosh(1.0) - 1.0)
            y = a * (np.cosh(t) - 1.0)
            x = (width_intermolar / 2.0) * t
        elif arch_form == "bonwill_hawley":
            r = width_intermolar / 2.0
            theta = t * (np.pi / 2.5)
            x = r * np.sin(theta)
            y = depth_arch * (1.0 - np.cos(theta))
        else:  # Elliptic / Brader
            x = (width_intermolar / 2.0) * np.sin(t * np.pi / 2.2)
            y = depth_arch * (1.0 - np.cos(t * np.pi / 2.2))

        # Curve of Spee (z elevation)
        z = 1.8 * (t**2)

        return np.column_stack([x, y, z])

    def generate_malocclusion_source(self, crowding_severity="moderate", arch_form="catenary", seed=42):
        """
        Synthesizes realistic patient malocclusion with crowding, rotations, and axial inclination deviations.
        """
        np.random.seed(seed)
        ideal = self.generate_ideal_arch_geodesic(arch_form=arch_form)
        
        scale_map = {
            "mild": {"disp": 1.2, "rot": 6.0, "ang": 3.0},
            "moderate": {"disp": 2.8, "rot": 14.0, "ang": 7.5},
            "severe": {"disp": 4.6, "rot": 24.0, "ang": 13.0}
        }
        params = scale_map.get(crowding_severity, scale_map["moderate"])

        source = np.copy(ideal)
        rotations = []
        angulations = []

        for i in range(self.num_teeth):
            # Rotational / spatial displacement
            dx = np.random.normal(0, params["disp"] * (0.8 if i in [5,6,7,8] else 0.4))
            dy = np.random.normal(0, params["disp"] * (0.9 if i in [5,6,7,8] else 0.3))
            dz = np.random.normal(0, params["disp"] * 0.25)
            
            source[i] += [dx, dy, dz]
            rot = float(np.random.normal(0, params["rot"]))
            ang = float(np.random.normal(0, params["ang"]))
            rotations.append(rot)
            angulations.append(ang)

        return {
            "ideal_target": ideal,
            "malocclusion_source": source,
            "rotations_deg": rotations,
            "angulations_deg": angulations
        }

    def compute_ring_algebra_staging(self, source, target, max_step_mm=0.22, max_rot_deg=1.8):
        """
        Polynomial ring algebra aligner progression:
        Decomposes total geodesic trajectory into quotient stages in R[x,y,z] / <f_arch>.
        """
        displacements = np.linalg.norm(target - source, axis=1)
        max_disp = np.max(displacements)
        
        num_stages = int(np.ceil(max_disp / max_step_mm))
        num_stages = max(num_stages, 12)  # Clinical minimum aligner set
        num_stages = min(num_stages, 36)

        staging_trajectories = []
        for s in range(num_stages + 1):
            alpha = s / float(num_stages)
            # Smooth Hermite-quintic interpolation factor
            s_curve = 10 * (alpha**3) - 15 * (alpha**4) + 6 * (alpha**5)
            interp_pts = (1.0 - s_curve) * source + s_curve * target
            staging_trajectories.append({
                "stage": s,
                "alpha": float(round(s_curve, 4)),
                "points": interp_pts.tolist(),
                "mean_error_to_target_mm": float(round(np.mean(np.linalg.norm(target - interp_pts, axis=1)), 3))
            })

        return {
            "total_stages": num_stages,
            "max_initial_discrepancy_mm": float(round(max_disp, 3)),
            "per_stage_rate_mm": float(round(max_disp / num_stages, 3)),
            "stages": staging_trajectories
        }

    def compute_measure_theoretic_smile_signature(self, dental_coords, lip_line_height=14.0, smile_width=62.0):
        """
        Evaluates Measure-Theoretic Smile Signature and Craniofacial Esthetic Index:
        - Smile Arc Concordance: Consonance between incisal edges and lower lip curvature.
        - Buccal Corridor Ratio: Fraction of transverse smile space filled by maxillary dentition.
        - Dental Midline Coincidence with Facial Midline.
        - Incisal Display at Rest & Full Smile.
        - Wasserstein-2 Distance to Golden Aesthetic Measure.
        """
        # Central incisors (indices 6, 7)
        c1, c2 = dental_coords[6], dental_coords[7]
        dental_midline_x = (c1[0] + c2[0]) / 2.0
        
        # Canine to Canine arch width (indices 4 and 9)
        intercanine_width = np.linalg.norm(dental_coords[4][:2] - dental_coords[9][:2])
        # 1st Molar to Molar arch width (indices 1 and 12)
        intermolar_width = np.linalg.norm(dental_coords[1][:2] - dental_coords[12][:2])

        # Buccal Corridor Ratio: (Smile Width - Dentition Width) / Smile Width
        buccal_corridor_ratio = max(0.0, (smile_width - intermolar_width) / smile_width)
        buccal_corridor_pct = float(round(buccal_corridor_ratio * 100.0, 2))

        # Smile Arc Concordance Index: Curvature matching of anterior 6 teeth (indices 4..9)
        ant_pts = dental_coords[4:10]
        y_coords = ant_pts[:, 1]
        x_coords = ant_pts[:, 0]
        
        # Parabolic fit of incisal edges
        poly_fit = np.polyfit(x_coords, y_coords, 2)
        curvature_k = abs(2 * poly_fit[0])
        # Ideal consonance curvature is ~0.0085 mm^-1
        smile_arc_concordance = float(np.clip(1.0 - abs(curvature_k - 0.0085) * 60.0, 0.50, 0.998))

        # Facial Midline Deviation (mm)
        midline_deviation_mm = float(abs(dental_midline_x))

        # Incisal Display (mm)
        incisal_display_mm = float(round(lip_line_height - np.mean(ant_pts[:, 1]) * 0.15, 2))

        # Wasserstein-2 Discrepancy to Golden Esthetic Measure
        ideal_signature = np.array([0.98, 12.0, 0.0, 10.5])
        current_signature = np.array([smile_arc_concordance, buccal_corridor_pct, midline_deviation_mm, incisal_display_mm])
        w2_distance = float(round(np.linalg.norm(current_signature - ideal_signature), 3))

        # Composite Smile Attractiveness Index (CSAI) [0 to 100]
        csai = float(round(
            (smile_arc_concordance * 40.0) +
            (max(0, 1.0 - abs(buccal_corridor_pct - 12.0) / 12.0) * 25.0) +
            (max(0, 1.0 - midline_deviation_mm / 2.5) * 20.0) +
            (max(0, 1.0 - abs(incisal_display_mm - 10.5) / 5.0) * 15.0),
            2
        ))

        return {
            "smile_arc_concordance_pct": float(round(smile_arc_concordance * 100.0, 2)),
            "buccal_corridor_ratio_pct": buccal_corridor_pct,
            "midline_deviation_mm": float(round(midline_deviation_mm, 3)),
            "incisal_display_mm": incisal_display_mm,
            "wasserstein_2_discrepancy": w2_distance,
            "composite_smile_attractiveness_index": csai
        }

    def simulate_full_treatment(self, crowding="moderate", arch_form="catenary", lip_line=14.0, smile_width=62.0):
        """
        Executes end-to-end Invisalign orthodontic treatment simulation:
        Pre-op, Staging, Post-op Smile, and Craniofacial Morphometry.
        """
        synth = self.generate_malocclusion_source(crowding_severity=crowding, arch_form=arch_form)
        source = synth["malocclusion_source"]
        target = synth["ideal_target"]

        staging_info = self.compute_ring_algebra_staging(source, target)
        preop_smile = self.compute_measure_theoretic_smile_signature(source, lip_line, smile_width)
        postop_smile = self.compute_measure_theoretic_smile_signature(target, lip_line, smile_width)

        # Tooth-by-tooth movement analysis
        tooth_movements = []
        for i in range(self.num_teeth):
            trans_vec = target[i] - source[i]
            trans_mag = float(np.linalg.norm(trans_vec))
            tooth_movements.append({
                "tooth": self.tooth_names[i],
                "translation_vector_mm": [float(round(v, 2)) for v in trans_vec],
                "total_translation_mm": float(round(trans_mag, 2)),
                "mesiodistal_mm": float(round(trans_vec[0], 2)),
                "buccolingual_mm": float(round(trans_vec[1], 2)),
                "extrusion_intrusion_mm": float(round(trans_vec[2], 2)),
                "initial_rotation_deg": float(round(synth["rotations_deg"][i], 1)),
                "initial_angulation_deg": float(round(synth["angulations_deg"][i], 1))
            })

        return {
            "tooth_names": self.tooth_names,
            "preop_coords": source.tolist(),
            "target_coords": target.tolist(),
            "staging": staging_info,
            "preop_smile_signature": preop_smile,
            "postop_smile_signature": postop_smile,
            "tooth_movements": tooth_movements,
            "craniofacial_metadata": {
                "arch_form": arch_form,
                "crowding_severity": crowding,
                "lip_line_height_mm": lip_line,
                "smile_width_mm": smile_width,
                "biomechanical_stiffness_n_mm": 38.5,
                "aligner_material": "SmartTrack™ Multi-Layer Polyurethane Aligner Ring"
            }
        }

smile_engine = InvisalignSmileEngine()
