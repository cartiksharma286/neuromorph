#!/usr/bin/env python3
"""
Nature Preprint Generator:
"Comparative Millimetric Tumour Ablation: Nonlinear Bioheat LITT, Stefan-Front
Cryoablation, and Ultrasound-Guided HIFU with Stieltjes-Continued-Fraction
SLAM Trajectory Optimization and Long-Horizon Clinical Outcome Trajectories"

Compiles a publication-grade Nature preprint PDF (written to l.pdf) detailing:
1. Nonlinear Pennes bioheat equation with temperature-dependent conductivity
   and vascular-stasis perfusion shutdown (LITT).
2. Stefan moving-boundary ice-front cryoablation physics.
3. Beer-Lambert HIFU volumetric heating with Stieltjes-impedance-refined
   millimetric target-registration precision.
4. Stieltjes-continued-fraction / Jacobi-elliptic SLAM insertion trajectory
   optimization.
5. Gompertz residual-tumour regrowth and Weibull local-control/survival
   long-horizon clinical outcome trajectories.
6. Empirical grayscale + heatmap ablation-zone visualizations and comparative
   clinical trajectory figures generated directly from the simulation engines.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from litt_cryo_ablation import (
    LITTAblationEngine,
    CryoAblationClinicalEngine,
    UltrasoundGuidedAblationPrecision,
    optimal_slam_trajectory,
    clinical_outcome_trajectories,
)


def build_pdf(output_path: str = None) -> str:
    """Builds the Nature LITT/Cryo/US ablation preprint and writes it to
    l.pdf (the literal output filename requested for this preprint)."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'l.pdf')

    # Run the real simulation engines so the figures reflect the same
    # physics/data exposed through the Flask API and UI tab.
    litt = LITTAblationEngine().simulate()
    cryo = CryoAblationClinicalEngine().simulate()
    us = UltrasoundGuidedAblationPrecision().simulate()
    slam = optimal_slam_trajectory()
    outcomes = clinical_outcome_trajectories()

    with PdfPages(output_path) as pdf:
        # =====================================================================
        # PAGE 1: TITLE, ABSTRACT & GOVERNING EQUATIONS
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11), dpi=300)
        fig.clf()

        fig.text(0.5, 0.965, 'Nature Medicine & Biomedical Engineering | Research Preprint',
                  fontsize=10.5, ha='center', style='italic', color='#1e3a8a', weight='bold')
        fig.text(0.5, 0.915,
                  'Comparative Millimetric Tumour Ablation: Nonlinear Bioheat LITT,\n'
                  'Stefan-Front Cryoablation, and Stieltjes-Refined Ultrasound-Guided HIFU',
                  fontsize=12.6, ha='center', weight='bold', color='#0f172a', linespacing=1.25)
        fig.text(0.5, 0.862,
                  'Cartik Sharma, PhD et al. | Neuromorph Surgical Robotics & Biophysical Modeling Group',
                  fontsize=9.2, ha='center', color='#475569')

        rect = FancyBboxPatch((0.08, 0.735), 0.84, 0.108, boxstyle="round,pad=0.015",
                               ec="#cbd5e1", fc="#f8fafc", linewidth=1.2)
        fig.gca().add_patch(rect)
        abstract_body = (
            "We present a unified finite-difference (bilinear-FEM-equivalent) comparative framework for three "
            "stereotactic tumour-ablation modalities: laser interstitial thermal therapy (LITT), cryoablation, and "
            "ultrasound-guided high-intensity focused ultrasound (HIFU). Nonlinear Pennes bioheat transport "
            "(temperature-dependent conductivity, vascular-stasis perfusion shutdown), Stefan moving-boundary "
            "ice-front kinetics, and Stieltjes-continued-fraction impedance/SLAM trajectory refinement jointly "
            f"deliver necrotic radii of {litt['necrotic_radius_mm']:.1f} mm (LITT), {cryo['ice_ball_radius_mm']:.1f} mm "
            f"(cryo), and {us['necrotic_radius_mm']:.1f} mm (HIFU) with a {us['target_registration_error_mm']:.2f} mm "
            f"ultrasound target-registration error and a {slam['final_target_registration_error_mm']:.1f} mm "
            "converged SLAM insertion-trajectory tracking error. Long-horizon Gompertz/Weibull clinical trajectories "
            "project local control, overall survival, and functional (KPS) recovery over a 60-month horizon."
        )
        fig.text(0.10, 0.833, 'ABSTRACT', fontsize=8.8, weight='bold', color='#1e3a8a')
        fig.text(0.10, 0.822, abstract_body, fontsize=7.5, va='top', wrap=True, linespacing=1.35, color='#1e293b')

        # Section 1: Nonlinear Pennes Bioheat Equation
        fig.text(0.08, 0.705, '1. Nonlinear Pennes Bioheat Transport (LITT)', fontsize=10.3, weight='bold', color='#0f172a')
        fig.text(0.08, 0.668,
                 "Temperature-dependent conductivity k(T) and vascular-stasis perfusion shutdown omega_b(T) above the "
                 "coagulation threshold (60 C) are the dominant nonlinearities governing thermal spread:",
                 fontsize=7.8, wrap=True, color='#334155', linespacing=1.25)
        eq1 = r"$\rho c \frac{\partial T}{\partial t} = \nabla\cdot\left(k(T)\nabla T\right) + \rho_b c_b\,\omega_b(T)\,(T_a - T) + Q_{\mathrm{laser}}(\mathbf{r})$"
        fig.text(0.5, 0.635, eq1, fontsize=8.8, ha='center', color='#1e3a8a')
        eq2 = r"$Q_{\mathrm{laser}}(r) = \frac{P\,\mu_{\mathrm{eff}}}{4\pi D r}\exp(-\mu_{\mathrm{eff}} r),\quad \mu_{\mathrm{eff}} = \sqrt{3\mu_a(\mu_a+\mu_s')},\ D = \frac{1}{3(\mu_a+\mu_s')}$"
        fig.text(0.5, 0.598, eq2, fontsize=8.4, ha='center', color='#1e3a8a')
        eq3 = r"$\mathrm{CEM43} = \sum_i R^{43-T_i}\Delta t_i,\ R=\{0.5\ (T{\geq}43^{\circ}\mathrm{C}),\ 0.25\},\quad \mathrm{Necrosis:}\ \mathrm{CEM43}\geq 240$"
        fig.text(0.5, 0.561, eq3, fontsize=8.4, ha='center', color='#1e3a8a')

        # Section 2: Stefan Moving Boundary (Cryo)
        fig.text(0.08, 0.523, '2. Stefan Moving-Boundary Cryoablation Ice-Front', fontsize=10.3, weight='bold', color='#0f172a')
        eq4 = r"$k_{\mathrm{frozen}}\left.\frac{\partial T}{\partial n}\right|_{f} - k_{\mathrm{thawed}}\left.\frac{\partial T}{\partial n}\right|_{t} = \rho L\, v_n,\quad$ lethal isotherm $= -40\ ^{\circ}\mathrm{C}$"
        fig.text(0.5, 0.490, eq4, fontsize=8.6, ha='center', color='#1e3a8a')

        # Section 3: HIFU / Stieltjes precision
        fig.text(0.08, 0.452, '3. Ultrasound-Guided HIFU & Stieltjes Impedance Precision', fontsize=10.3, weight='bold', color='#0f172a')
        eq5 = r"$Q_{\mathrm{HIFU}}(r) = Q_0\,e^{-2\alpha r}\,e^{-r^2/2\sigma_f^2},\quad f(z) = \frac{c_1 z}{1+\frac{c_2 z}{1+\frac{c_3 z}{1+\cdots}}}$"
        fig.text(0.5, 0.415, eq5, fontsize=8.6, ha='center', color='#1e3a8a')
        fig.text(0.08, 0.385,
                 "The Stieltjes convergent error bound refines the effective focal spot, tightening target-registration "
                 "error (TRE) to sub-millimetric bounds independent of raw beam divergence.",
                 fontsize=7.6, wrap=True, color='#334155', linespacing=1.25)

        # Section 4: SLAM / Elliptic manifold
        fig.text(0.08, 0.352, '4. Stieltjes/Jacobi-Elliptic SLAM Insertion Trajectory', fontsize=10.3, weight='bold', color='#0f172a')
        eq6 = r"$F(\phi, m) = \int_0^{\phi} \frac{d\theta}{\sqrt{1-m\sin^2\theta}},\quad \mathrm{TRE}_{\mathrm{SLAM}} = \left\| \mathbf{x}_{\mathrm{FK}} - \mathbf{x}_{\mathrm{target}} \right\| + 0.01\,\delta_{\mathrm{CF}}$"
        fig.text(0.5, 0.318, eq6, fontsize=8.4, ha='center', color='#1e3a8a')

        # Section 5: Long-horizon clinical outcomes
        fig.text(0.08, 0.280, '5. Gompertz Regrowth & Weibull Local-Control/Survival', fontsize=10.3, weight='bold', color='#0f172a')
        eq7 = r"$\frac{dV}{dt} = aV\ln\!\left(\frac{b}{V}\right)\ \Rightarrow\ V(t) = b\exp\!\left(\ln(V_0/b)\,e^{-at}\right),\quad S(t) = \exp\!\left(-(t/\lambda)^{k}\right)$"
        fig.text(0.5, 0.245, eq7, fontsize=8.6, ha='center', color='#1e3a8a')

        summary_text = (
            f"Simulation summary (default clinical parameters): LITT peak temperature {litt['max_temperature_c']:.1f} C, "
            f"CEM43 peak {litt['cem43_peak_min']:.0f} min; cryo minimum temperature {cryo['min_temperature_c']:.1f} C, "
            f"Stefan front velocity {cryo['stefan_front_velocity_um_s']:.2f} um/s; HIFU peak temperature "
            f"{us['max_temperature_c']:.1f} C with TRE {us['target_registration_error_mm']:.3f} mm; SLAM converged "
            f"insertion TRE {slam['final_target_registration_error_mm']:.2f} mm."
        )
        fig.text(0.08, 0.145, summary_text, fontsize=7.4, wrap=True, color='#475569', linespacing=1.3)

        fig.text(0.5, 0.035, 'Nature Medicine & Biomedical Engineering | Preprint Manuscript | Page 1 of 3',
                  fontsize=8, ha='center', color='#94a3b8')
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # PAGE 2: EMPIRICAL ABLATION-ZONE GRAYSCALE + HEATMAP FIGURES
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11), dpi=300)
        fig.clf()
        fig.text(0.5, 0.965, '6. Empirical Grayscale Anatomy & Thermal-Dose Heatmaps',
                  fontsize=12, ha='center', weight='bold', color='#0f172a')

        anatomy_extent_mm = 96 * 0.6

        def _panel(ax_anat, ax_heat, result, cmap_heat, title, unit_label):
            anatomy = np.array(result['anatomy_grid'], dtype=float)
            heat = np.array(result['heatmap_grid'], dtype=float)
            ax_anat.imshow(anatomy, cmap='gray', vmin=0, vmax=255,
                           extent=[0, anatomy_extent_mm, 0, anatomy_extent_mm])
            ax_anat.set_title(f'{title} — Anatomy (B-mode/MR gray)', fontsize=7.8, weight='bold')
            ax_anat.set_xlabel('mm', fontsize=6.8)
            ax_anat.set_ylabel('mm', fontsize=6.8)
            im = ax_heat.imshow(anatomy, cmap='gray', vmin=0, vmax=255,
                                 extent=[0, anatomy_extent_mm, 0, anatomy_extent_mm])
            im2 = ax_heat.imshow(heat, cmap=cmap_heat, vmin=0, vmax=255, alpha=0.55,
                                  extent=[0, anatomy_extent_mm, 0, anatomy_extent_mm])
            ax_heat.set_title(f'{title} — Ablation Heatmap ({unit_label})', fontsize=7.8, weight='bold')
            ax_heat.set_xlabel('mm', fontsize=6.8)
            cbar = plt.colorbar(im2, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)

        ax1 = fig.add_axes([0.08, 0.67, 0.38, 0.24])
        ax2 = fig.add_axes([0.54, 0.67, 0.38, 0.24])
        _panel(ax1, ax2, litt, 'inferno', 'LITT', 'CEM43 dose')

        ax3 = fig.add_axes([0.08, 0.39, 0.38, 0.24])
        ax4 = fig.add_axes([0.54, 0.39, 0.38, 0.24])
        _panel(ax3, ax4, cryo, 'cool', 'Cryoablation', 'ice fraction')

        ax5 = fig.add_axes([0.08, 0.11, 0.38, 0.24])
        ax6 = fig.add_axes([0.54, 0.11, 0.38, 0.24])
        _panel(ax5, ax6, us, 'magma', 'US-Guided HIFU', 'temperature')

        fig.text(0.5, 0.035, 'Nature Medicine & Biomedical Engineering | Preprint Manuscript | Page 2 of 3',
                  fontsize=8, ha='center', color='#94a3b8')
        pdf.savefig(fig)
        plt.close()

        # =====================================================================
        # PAGE 3: SLAM TRAJECTORY & LONG-HORIZON CLINICAL OUTCOMES
        # =====================================================================
        fig = plt.figure(figsize=(8.5, 11), dpi=300)
        fig.clf()
        fig.text(0.5, 0.965, '7. SLAM/Stieltjes Trajectory Convergence & Long-Horizon Outcomes',
                  fontsize=11.6, ha='center', weight='bold', color='#0f172a')

        ax1 = fig.add_axes([0.10, 0.70, 0.80, 0.20])
        wp = [w['waypoint'] for w in slam['waypoints']]
        tre = [w['slam_tracking_error_mm'] for w in slam['waypoints']]
        ax1.plot(wp, tre, marker='o', color='#1e3a8a', linewidth=1.8)
        ax1.axhline(5.0, color='#ef4444', linestyle='--', linewidth=1.0, label='Approach safety margin (5 mm)')
        ax1.set_title('(a) SLAM Insertion Trajectory Tracking Error Convergence', fontsize=8.5, weight='bold')
        ax1.set_xlabel('Waypoint', fontsize=7.5)
        ax1.set_ylabel('TRE (mm)', fontsize=7.5)
        ax1.legend(fontsize=6.5)
        ax1.grid(True, linestyle=':', alpha=0.5)

        months = np.array(outcomes['months'])
        colors = {'LITT': '#ef4444', 'Cryoablation': '#22d3ee', 'US-Guided HIFU': '#f59e0b'}

        ax2 = fig.add_axes([0.10, 0.40, 0.38, 0.22])
        for modality, color in colors.items():
            ax2.plot(months, outcomes[modality]['local_control_pct'], color=color, linewidth=1.8, label=modality)
        ax2.set_title('(b) Local Control (60-month horizon)', fontsize=8.2, weight='bold')
        ax2.set_xlabel('Months', fontsize=7.2)
        ax2.set_ylabel('Local Control (%)', fontsize=7.2)
        ax2.legend(fontsize=6.0)
        ax2.grid(True, linestyle=':', alpha=0.5)

        ax3 = fig.add_axes([0.55, 0.40, 0.38, 0.22])
        for modality, color in colors.items():
            ax3.plot(months, outcomes[modality]['overall_survival_pct'], color=color, linewidth=1.8, label=modality)
        ax3.set_title('(c) Overall Survival (60-month horizon)', fontsize=8.2, weight='bold')
        ax3.set_xlabel('Months', fontsize=7.2)
        ax3.set_ylabel('Survival (%)', fontsize=7.2)
        ax3.legend(fontsize=6.0)
        ax3.grid(True, linestyle=':', alpha=0.5)

        ax4 = fig.add_axes([0.10, 0.11, 0.38, 0.22])
        for modality, color in colors.items():
            ax4.plot(months, outcomes[modality]['kps_trajectory'], color=color, linewidth=1.8, label=modality)
        ax4.set_title('(d) Karnofsky Performance Status Recovery', fontsize=8.2, weight='bold')
        ax4.set_xlabel('Months', fontsize=7.2)
        ax4.set_ylabel('KPS', fontsize=7.2)
        ax4.legend(fontsize=6.0)
        ax4.grid(True, linestyle=':', alpha=0.5)

        ax5 = fig.add_axes([0.55, 0.11, 0.38, 0.22])
        for modality, color in colors.items():
            ax5.plot(months, outcomes[modality]['residual_tumor_volume_cm3'], color=color, linewidth=1.8, label=modality)
        ax5.set_title('(e) Gompertz Residual Tumour Volume', fontsize=8.2, weight='bold')
        ax5.set_xlabel('Months', fontsize=7.2)
        ax5.set_ylabel('Volume (cm$^3$)', fontsize=7.2)
        ax5.legend(fontsize=6.0)
        ax5.grid(True, linestyle=':', alpha=0.5)

        fig.text(0.5, 0.035, 'Nature Medicine & Biomedical Engineering | Preprint Manuscript | Page 3 of 3',
                  fontsize=8, ha='center', color='#94a3b8')
        pdf.savefig(fig)
        plt.close()

    print(f"[+] Successfully generated Nature LITT/Cryo/US Ablation Preprint: {output_path}")
    return output_path


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), 'l.pdf')
    build_pdf(out)
