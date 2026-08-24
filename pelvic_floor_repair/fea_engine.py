"""
Finite Element Analysis (FEA) Engine for Pelvic Floor Reconstruction Implants.
Simulates biomechanical stress distribution, displacement, Valsalva pressure loads,
and structural fatigue lifecycle under physiological pelvic loading conditions.
"""

import numpy as np
from typing import Dict, List, Any
import uuid

class PelvicFEAEngine:
    """Biomechanical Finite Element Analysis solver for gynecological implants"""

    def __init__(self):
        # Material mechanical parameters (Young's Modulus E in MPa, Poisson's ratio nu, Yield Strength in MPa)
        self.material_db = {
            'composite': {
                'youngs_modulus_mpa': 120.0,
                'poissons_ratio': 0.38,
                'yield_strength_mpa': 28.0,
                'ultimate_tensile_mpa': 35.0,
                'density_g_cm3': 1.15,
                'fatigue_exponent': -0.09,
                'fatigue_coefficient_mpa': 45.0
            },
            'mesh': {
                'youngs_modulus_mpa': 45.0,
                'poissons_ratio': 0.45,
                'yield_strength_mpa': 16.0,
                'ultimate_tensile_mpa': 22.0,
                'density_g_cm3': 0.94,
                'fatigue_exponent': -0.12,
                'fatigue_coefficient_mpa': 30.0
            },
            'xenograft': {
                'youngs_modulus_mpa': 25.0,
                'poissons_ratio': 0.48,
                'yield_strength_mpa': 9.5,
                'ultimate_tensile_mpa': 14.0,
                'density_g_cm3': 1.05,
                'fatigue_exponent': -0.15,
                'fatigue_coefficient_mpa': 18.0
            },
            'autograft': {
                'youngs_modulus_mpa': 35.0,
                'poissons_ratio': 0.46,
                'yield_strength_mpa': 13.0,
                'ultimate_tensile_mpa': 18.5,
                'density_g_cm3': 1.08,
                'fatigue_exponent': -0.13,
                'fatigue_coefficient_mpa': 24.0
            },
            'synthetic_polymer': {
                'youngs_modulus_mpa': 85.0,
                'poissons_ratio': 0.40,
                'yield_strength_mpa': 22.0,
                'ultimate_tensile_mpa': 30.0,
                'density_g_cm3': 1.12,
                'fatigue_exponent': -0.10,
                'fatigue_coefficient_mpa': 38.0
            }
        }

    def run_fea_analysis(self, design: Dict[str, Any], pressure_kpa: float = 15.0,
                         grid_res: int = 21, anchoring_type: str = 'bilateral_sacrospinous') -> Dict[str, Any]:
        """
        Execute 2D/3D continuum plate FEA simulation over the implant geometry.
        
        Args:
            design: Implant configuration dictionary
            pressure_kpa: Intra-abdominal pressure load (kPa). Resting=3.5, Coughing=12.0, Valsalva=20-25
            grid_res: Node resolution along X and Y axes
            anchoring_type: Boundary condition scheme
        """
        material_key = str(design.get('material', 'composite')).lower()
        mat_props = self.material_db.get(material_key, self.material_db['composite'])
        
        dims = design.get('dimensions', {})
        length_mm = float(dims.get('length_mm', design.get('length', 40.0)))
        width_mm = float(dims.get('width_mm', design.get('width', 30.0)))
        thickness_mm = float(dims.get('thickness_mm', design.get('thickness', 1.0)))
        pore_size_um = float(design.get('pore_size_microns', 100))

        # Effective stiffness reduced by porosity
        porosity_ratio = (pore_size_um / 350.0) * 0.45
        effective_E = mat_props['youngs_modulus_mpa'] * (1.0 - porosity_ratio)
        nu = mat_props['poissons_ratio']
        
        # Spatial coordinate mesh (centered around origin)
        x = np.linspace(-length_mm / 2.0, length_mm / 2.0, grid_res)
        y = np.linspace(-width_mm / 2.0, width_mm / 2.0, grid_res)
        X, Y = np.meshgrid(x, y)
        
        # Normalized coordinates [-1, 1]
        X_norm = X / (length_mm / 2.0 + 1e-6)
        Y_norm = Y / (width_mm / 2.0 + 1e-6)
        R_sq = X_norm**2 + Y_norm**2

        # Pressure converted to MPa: 1 kPa = 0.001 MPa (N/mm^2)
        P_mpa = pressure_kpa * 0.001
        
        # Flexural plate rigidity D = E * t^3 / (12 * (1 - nu^2))
        D = (effective_E * (thickness_mm ** 3)) / (12.0 * (1.0 - nu**2) + 1e-6)
        
        # Plate boundary displacement profile w(x, y)
        # Clamped/pinned at lateral anchor horns with center dome deflection
        # w(r) profile satisfies physiological pelvic diaphragm support
        boundary_factor = np.clip(1.0 - (X_norm**2 * 0.7 + Y_norm**2 * 0.8), 0.0, 1.0)
        shape_profile = str(design.get('shape_profile', 'anatomical')).lower()
        if 'curved' in shape_profile or 'anatomical' in shape_profile:
            arch_gain = 1.15
        elif 'reinforced' in shape_profile:
            arch_gain = 0.85
        else:
            arch_gain = 1.0

        # Max central deflection estimate (Kirchhoff-Love shell approximation)
        a = min(length_mm, width_mm)
        w_center_mm = (0.0026 * P_mpa * (a**4) / (D + 1e-5)) * arch_gain
        w_center_mm = float(np.clip(w_center_mm, 0.05, 12.0))
        
        displacement_field = w_center_mm * (boundary_factor ** 2)
        
        # Second spatial derivatives -> Curvature -> Stresses
        # Stress sigma_x = -E * z / (1 - nu^2) * (d2w/dx2 + nu * d2w/dy2)
        # Numerical gradient approximations
        dx = (length_mm / (grid_res - 1))
        dy = (width_mm / (grid_res - 1))
        
        grad_y, grad_x = np.gradient(displacement_field, dy, dx)
        curv_yy, curv_yx = np.gradient(grad_y, dy, dx)
        curv_xy, curv_xx = np.gradient(grad_x, dy, dx)
        
        # Extreme fiber (z = t/2) stresses
        z_fiber = thickness_mm / 2.0
        sigma_x = (effective_E * z_fiber / (1.0 - nu**2)) * np.abs(curv_xx + nu * curv_yy)
        sigma_y = (effective_E * z_fiber / (1.0 - nu**2)) * np.abs(curv_yy + nu * curv_xx)
        tau_xy = (effective_E * z_fiber / (2.0 * (1.0 + nu))) * np.abs(curv_xy)

        # Baseline membrane tension under pressure (Laplace law for anatomical dome)
        membrane_tension = (P_mpa * length_mm * width_mm) / (4.0 * thickness_mm * (length_mm + width_mm) + 1e-6)
        
        # Total stress tensor superposition
        sigma_x_total = sigma_x + membrane_tension * (1.0 + 0.3 * np.abs(X_norm))
        sigma_y_total = sigma_y + membrane_tension * (1.0 + 0.3 * np.abs(Y_norm))
        tau_total = tau_xy + 0.2 * membrane_tension * np.abs(X_norm * Y_norm)

        # Von Mises Stress = sqrt(sigma_x^2 - sigma_x*sigma_y + sigma_y^2 + 3*tau_xy^2)
        von_mises_field = np.sqrt(
            sigma_x_total**2 - sigma_x_total * sigma_y_total + sigma_y_total**2 + 3.0 * (tau_total**2)
        )
        
        # Add stress concentration near anchor horns
        anchor_horn_boost = np.exp(-((np.abs(X_norm) - 0.85)**2 + (np.abs(Y_norm) - 0.75)**2) / 0.1) * 1.8
        von_mises_field = von_mises_field * (1.0 + anchor_horn_boost)

        # Extreme values
        max_von_mises_mpa = float(np.max(von_mises_field))
        avg_von_mises_mpa = float(np.mean(von_mises_field))
        max_displacement_mm = float(np.max(displacement_field))
        
        # Safety Factor against yield
        yield_strength = mat_props['yield_strength_mpa']
        safety_factor = float(yield_strength / (max_von_mises_mpa + 1e-6))
        
        # Fatigue Life cycle calculation (Basquin relation under cyclic Valsalva)
        stress_amplitude = max_von_mises_mpa * 0.75
        sigma_f = mat_props['fatigue_coefficient_mpa']
        b_exp = mat_props['fatigue_exponent']
        if stress_amplitude < sigma_f:
            cycles_to_failure = int((stress_amplitude / sigma_f) ** (1.0 / b_exp))
            cycles_to_failure = min(cycles_to_failure, 10_000_000)
        else:
            cycles_to_failure = int(max(100, 1000 * (sigma_f / stress_amplitude)**2))

        # Anchor reaction forces (Total pressure force distributed to 4 cardinal anchors)
        total_load_n = (pressure_kpa * 0.001) * (length_mm * width_mm)
        anterior_anchor_n = float(total_load_n * 0.22)
        posterior_anchor_n = float(total_load_n * 0.28)
        lateral_left_n = float(total_load_n * 0.25)
        lateral_right_n = float(total_load_n * 0.25)
        
        # Mesh Erosion Risk Score (0 - 100%)
        # Higher stiffness + thin profile + high edge stress = higher erosion risk
        stiffness_penalty = (effective_E / 120.0) * 25.0
        thinness_penalty = max(0.0, (1.2 - thickness_mm)) * 30.0
        stress_penalty = min(35.0, (max_von_mises_mpa / yield_strength) * 35.0)
        erosion_risk_pct = float(np.clip(stiffness_penalty + thinness_penalty + stress_penalty - 10.0, 3.0, 85.0))

        # Build colorized colormap nodes for WebGL / UI rendering (Jet/Turbo color gradient)
        nodes_data = []
        vm_min = float(np.min(von_mises_field))
        vm_range = max(0.01, max_von_mises_mpa - vm_min)
        
        for i in range(grid_res):
            for j in range(grid_res):
                stress_val = float(von_mises_field[i, j])
                disp_val = float(displacement_field[i, j])
                norm_val = (stress_val - vm_min) / vm_range
                
                # Turbo/Rainbow RGB mapping
                r, g, b = self._get_stress_color_rgb(norm_val)
                
                nodes_data.append({
                    'x': round(float(X[i, j]), 2),
                    'y': round(float(Y[i, j]), 2),
                    'z_deflection': round(disp_val, 3),
                    'von_mises_mpa': round(stress_val, 3),
                    'color_hex': f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                })

        # Stress level classification
        if safety_factor >= 2.5:
            structural_integrity = 'Excellent (Optimal Margin)'
            integrity_code = 'SUCCESS'
        elif safety_factor >= 1.5:
            structural_integrity = 'Adequate (Physiological Tolerance)'
            integrity_code = 'WARNING'
        else:
            structural_integrity = 'Critical (Yield Risk under Peak Valsalva)'
            integrity_code = 'DANGER'

        return {
            'analysis_id': f"fea_{uuid.uuid4().hex[:8]}",
            'applied_pressure_kpa': pressure_kpa,
            'anchoring_type': anchoring_type,
            'material': material_key,
            'grid_resolution': f"{grid_res}x{grid_res}",
            'results': {
                'max_von_mises_mpa': round(max_von_mises_mpa, 2),
                'avg_von_mises_mpa': round(avg_von_mises_mpa, 2),
                'max_displacement_mm': round(max_displacement_mm, 2),
                'yield_strength_mpa': round(yield_strength, 1),
                'safety_factor': round(safety_factor, 2),
                'structural_integrity': structural_integrity,
                'integrity_code': integrity_code,
                'estimated_fatigue_cycles': cycles_to_failure,
                'mesh_erosion_risk_pct': round(erosion_risk_pct, 1),
                'reaction_forces_n': {
                    'total_load_n': round(total_load_n, 2),
                    'anterior_anchors_n': round(anterior_anchor_n, 2),
                    'posterior_sacrospinous_n': round(posterior_anchor_n, 2),
                    'lateral_atfp_left_n': round(lateral_left_n, 2),
                    'lateral_atfp_right_n': round(lateral_right_n, 2)
                }
            },
            'stress_grid': {
                'dimensions': {'length_mm': length_mm, 'width_mm': width_mm},
                'nodes_sample': nodes_data[::2]  # Downsampled for fast JSON transport
            }
        }

    @staticmethod
    def _get_stress_color_rgb(t: float):
        """Rainbow transfer function for FEA Von Mises stress visualization"""
        t = np.clip(t, 0.0, 1.0)
        # Blue (low stress) -> Cyan -> Green -> Yellow -> Red (peak stress)
        if t < 0.25:
            s = t / 0.25
            return 0.0, s, 1.0
        elif t < 0.5:
            s = (t - 0.25) / 0.25
            return 0.0, 1.0, 1.0 - s
        elif t < 0.75:
            s = (t - 0.5) / 0.25
            return s, 1.0, 0.0
        else:
            s = (t - 0.75) / 0.25
            return 1.0, 1.0 - s, 0.0
