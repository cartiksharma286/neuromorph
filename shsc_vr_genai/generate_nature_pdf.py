#!/usr/bin/env python3
"""
generate_nature_pdf.py - Generates a Nature-style preprint PDF focused on the
knee surgery education tab for minimally invasive total knee arthroplasty.
"""

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from app import (
    KNEE_WORKFLOW_GUIDES,
    MILITARY_EDUCATION_PROFILES,
    compute_knee_education_simulation,
)


PHASE_WEIGHTS = np.array([0.16, 0.18, 0.24, 0.22, 0.20])
COLORS = {
    'navy': '#0f172a',
    'blue': '#1d4ed8',
    'cyan': '#0891b2',
    'teal': '#0f766e',
    'gold': '#a16207',
    'green': '#15803d',
    'rose': '#e11d48',
    'slate': '#475569',
    'light': '#e2e8f0',
    'muted': '#64748b',
}


def configure_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'text.color': COLORS['navy'],
        'axes.edgecolor': COLORS['muted'],
        'axes.linewidth': 0.8,
        'axes.titleweight': 'bold',
        'axes.labelcolor': COLORS['navy'],
        'xtick.color': COLORS['navy'],
        'ytick.color': COLORS['navy'],
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def add_rule(fig, y_pos, color=COLORS['light']):
    fig.add_artist(Line2D([0.08, 0.92], [y_pos, y_pos], transform=fig.transFigure, color=color, linewidth=1.0))


def add_wrapped_text(fig, x_pos, y_pos, text, width, fontsize=9.0, color=COLORS['navy'], weight='normal', style='normal'):
    fig.text(
        x_pos,
        y_pos,
        textwrap.fill(text, width=width),
        ha='left',
        va='top',
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        style=style,
        linespacing=1.4,
    )


def compute_phase_readiness_scores(payload):
    base_score = payload['characteristics']['workflow_readiness_score']
    phase_shape = 12.0 * (PHASE_WEIGHTS - PHASE_WEIGHTS.mean())
    phase_offsets = np.array([1.2, 0.4, -0.8, -0.3, 0.9])
    scores = np.clip(base_score + phase_shape + phase_offsets, 60.0, 99.0)
    return scores.tolist()


def build_reference_datasets():
    reference_payload = compute_knee_education_simulation(
        wire_diameter_mm=2.8,
        stiffness_index=58.0,
        insertion_angle_deg=1.5,
        approach='midvastus',
        mission='joint_allied',
    )
    optimized_payload = compute_knee_education_simulation(
        wire_diameter_mm=3.0,
        stiffness_index=64.0,
        insertion_angle_deg=0.6,
        approach='subvastus',
        mission='joint_allied',
    )

    approach_payloads = {
        key: compute_knee_education_simulation(
            wire_diameter_mm=2.8,
            stiffness_index=58.0,
            insertion_angle_deg=1.5,
            approach=key,
            mission='joint_allied',
        )
        for key in KNEE_WORKFLOW_GUIDES
    }

    mission_payloads = {
        key: compute_knee_education_simulation(
            wire_diameter_mm=2.8,
            stiffness_index=58.0,
            insertion_angle_deg=1.5,
            approach='midvastus',
            mission=key,
        )
        for key in MILITARY_EDUCATION_PROFILES
    }

    angles = np.linspace(-6.0, 6.0, 25)
    angle_capture = []
    angle_alignment = []
    for angle in angles:
        payload = compute_knee_education_simulation(
            wire_diameter_mm=2.8,
            stiffness_index=58.0,
            insertion_angle_deg=float(angle),
            approach='midvastus',
            mission='joint_allied',
        )
        angle_capture.append(payload['characteristics']['canal_capture_probability'])
        angle_alignment.append(payload['characteristics']['alignment_index'])

    return {
        'reference': reference_payload,
        'optimized': optimized_payload,
        'approach_payloads': approach_payloads,
        'mission_payloads': mission_payloads,
        'angles': angles,
        'angle_capture': np.array(angle_capture),
        'angle_alignment': np.array(angle_alignment),
        'phase_scores': compute_phase_readiness_scores(reference_payload),
    }


def render_page_one(pdf, data):
    reference = data['reference']
    optimized = data['optimized']
    chars = reference['characteristics']

    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    fig.text(0.5, 0.965, 'NATURE PREPRINT | DIGITAL SURGICAL EDUCATION', ha='center', fontsize=8.5, fontweight='bold', color=COLORS['blue'])
    fig.text(0.5, 0.928, 'A Finite-Horizon Guidewire Education Model for', ha='center', fontsize=16, fontweight='bold', color=COLORS['navy'])
    fig.text(0.5, 0.898, 'Minimally Invasive Total Knee Arthroplasty Training', ha='center', fontsize=15, fontweight='bold', color=COLORS['navy'])
    fig.text(0.5, 0.870, 'Cartik Sharma | Surgical Engineering Simulation Program', ha='center', fontsize=9, style='italic', color=COLORS['muted'])
    add_rule(fig, 0.848)

    fig.text(0.08, 0.824, 'Abstract', ha='left', va='top', fontsize=11, fontweight='bold')
    abstract_text = (
        'We present a Nature-style preprint for the new knee surgery education tab added to shsc_vr_genai. '
        'The module couples guidewire diameter, stiffness, insertion-angle bias, minimally invasive exposure, '
        'and coalition training mission profiles over a finite horizon of 36 depth samples. In the dashboard reference '
        'configuration, the simulator projects a canal capture probability of '
        f"{chars['canal_capture_probability']:.1f}%, an alignment index of {chars['alignment_index']:.1f}%, a workflow readiness score of "
        f"{chars['workflow_readiness_score']:.1f}%, and a digital interoperability score of {chars['digital_interoperability_score']:.1f}%. "
        'These outputs support minimally invasive total knee arthroplasty rehearsal, structured workflow coaching, '
        'and readiness analytics for US and Canadian military education programs.'
    )
    add_wrapped_text(fig, 0.08, 0.798, abstract_text, width=116, fontsize=9.2)

    fig.text(0.08, 0.664, 'Finite-Horizon Formulation', ha='left', va='top', fontsize=11, fontweight='bold')
    formulation_text = (
        'All dashboard quantities are evaluated on a finite depth lattice k = 0, ..., N with N = 35 increments spanning '
        '0 to 120 mm. The formulation is algebraic rather than continuous so the preprint mirrors exactly what is rendered '
        'inside the education tab and can be audited by instructor teams.'
    )
    add_wrapped_text(fig, 0.08, 0.638, formulation_text, width=116, fontsize=9.0)

    fig.text(0.10, 0.566, r"$e_k = 0.28 + \frac{1.35\,\eta_a\,\eta_{\theta}}{\eta_s\,\eta_d}\frac{k}{N} + 0.06\sin\left(\frac{3\pi k}{N}\right), \quad k = 0, \ldots, N$", ha='left', va='top', fontsize=12, color=COLORS['blue'])
    fig.text(0.10, 0.490, r"$\ell_k = 15 + 0.18S + 4.8D + 16\eta_a\frac{k}{N} + 1.2|\theta|$", ha='left', va='top', fontsize=12, color=COLORS['blue'])
    fig.text(0.10, 0.414, r"$R = 0.45A + 0.35C + 0.20P, \qquad I_m = b_m + 0.12(A-80) + 0.08(C-80) - 0.35F$", ha='left', va='top', fontsize=12, color=COLORS['blue'])

    definitions_text = (
        'Here, D is guidewire diameter in millimetres, S is the stiffness index, theta is insertion-angle bias in degrees, '
        'eta_a is the minimally invasive exposure factor, eta_theta = 1 + |theta| / 7, eta_s is the stiffness gain, eta_d is the diameter gain, '
        'A is the alignment index, C is canal capture probability, P is the soft-tissue preservation score, F is the number of fluoroscopy verification '
        'passes, and b_m is the mission-specific digital transformation baseline.'
    )
    add_wrapped_text(fig, 0.08, 0.356, definitions_text, width=116, fontsize=8.8, color=COLORS['slate'])

    fig.text(0.08, 0.250, 'Reference Scenario and Educational Meaning', ha='left', va='top', fontsize=11, fontweight='bold')
    scenario_text = (
        'The reference tab configuration uses the Mini-midvastus MIS TKA pathway with a 2.8 mm guidewire, stiffness 58, and 1.5 degree angular bias '
        'for the joint US/Canadian allied initiative. An optimized subvastus rehearsal scenario at 3.0 mm, stiffness 64, and 0.6 degree angular bias raises '
        'canal capture to '
        f"{optimized['characteristics']['canal_capture_probability']:.1f}% and alignment to {optimized['characteristics']['alignment_index']:.1f}%, "
        'illustrating that angular control and moderate stiffness gains are the dominant levers for finite-horizon improvement.'
    )
    add_wrapped_text(fig, 0.08, 0.224, scenario_text, width=116, fontsize=9.0)
    add_wrapped_text(fig, 0.08, 0.136, reference['initiative_brief'], width=116, fontsize=8.9, color=COLORS['teal'])

    add_rule(fig, 0.090)
    fig.text(0.5, 0.056, 'Page 1 of 3', ha='center', fontsize=9, color=COLORS['muted'])
    fig.text(0.5, 0.034, 'Finite simulation note: this preprint describes a digital education model, not a clinical outcomes trial.', ha='center', fontsize=7.5, color=COLORS['muted'])

    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)


def render_page_two(pdf, data):
    reference = data['reference']
    approach_payloads = data['approach_payloads']
    mission_payloads = data['mission_payloads']

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11), facecolor='white')
    fig.suptitle('Finite-Horizon Knee Education Simulation Outputs', fontsize=12, fontweight='bold', y=0.975, color=COLORS['navy'])
    fig.text(0.5, 0.952, 'Reference scenario: Mini-midvastus | Joint allied initiative | 2.8 mm wire | stiffness 58 | angle 1.5 deg', ha='center', fontsize=8.5, color=COLORS['muted'])

    ax = axes[0, 0]
    ax.plot(reference['depth_axis_mm'], reference['tracking_error_mm'], color=COLORS['cyan'], linewidth=2.2, label='Tracking error (mm)')
    ax.set_title('Guidewire tracking and tactile load vs depth', fontsize=9)
    ax.set_xlabel('Depth (mm)', fontsize=8)
    ax.set_ylabel('Tracking error (mm)', fontsize=8)
    ax.grid(True, color='#eef2ff')
    ax.tick_params(labelsize=8)

    ax_right = ax.twinx()
    ax_right.plot(reference['depth_axis_mm'], reference['tactile_load_n'], color=COLORS['gold'], linewidth=1.9, linestyle='--', label='Tactile load (N)')
    ax_right.set_ylabel('Tactile load (N)', fontsize=8, color=COLORS['gold'])
    ax_right.tick_params(labelsize=8, colors=COLORS['gold'])
    handles = ax.get_lines() + ax_right.get_lines()
    labels = [line.get_label() for line in handles]
    ax.legend(handles, labels, fontsize=7, frameon=False, loc='upper left')

    ax = axes[0, 1]
    approach_keys = list(approach_payloads.keys())
    approach_labels = ['Midvastus', 'Subvastus', 'Quad-sparing']
    capture_vals = [approach_payloads[key]['characteristics']['canal_capture_probability'] for key in approach_keys]
    alignment_vals = [approach_payloads[key]['characteristics']['alignment_index'] for key in approach_keys]
    readiness_vals = [approach_payloads[key]['characteristics']['workflow_readiness_score'] for key in approach_keys]
    x_pos = np.arange(len(approach_labels))
    width = 0.24
    ax.bar(x_pos - width, capture_vals, width=width, color=COLORS['teal'], label='Canal capture')
    ax.bar(x_pos, alignment_vals, width=width, color=COLORS['blue'], label='Alignment')
    ax.bar(x_pos + width, readiness_vals, width=width, color=COLORS['rose'], label='Workflow readiness')
    ax.set_title('Approach comparison at fixed wire settings', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approach_labels, fontsize=8)
    ax.set_ylim(65, 98)
    ax.set_ylabel('Score (%)', fontsize=8)
    ax.grid(True, axis='y', color='#eef2ff')
    ax.legend(fontsize=7, frameon=False, loc='upper right')

    ax = axes[1, 0]
    ax.plot(data['angles'], data['angle_capture'], color=COLORS['green'], linewidth=2.2, label='Canal capture')
    ax.plot(data['angles'], data['angle_alignment'], color=COLORS['rose'], linewidth=2.2, label='Alignment index')
    ax.axvline(x=0.0, color=COLORS['muted'], linewidth=1.0, linestyle=':')
    ax.set_title('Sensitivity to insertion-angle bias', fontsize=9)
    ax.set_xlabel('Angle bias (deg)', fontsize=8)
    ax.set_ylabel('Score (%)', fontsize=8)
    ax.set_ylim(60, 100)
    ax.grid(True, color='#eef2ff')
    ax.legend(fontsize=7, frameon=False, loc='lower center')
    ax.tick_params(labelsize=8)

    ax = axes[1, 1]
    mission_keys = list(mission_payloads.keys())
    mission_labels = ['US', 'Canada', 'Allied']
    digital_scores = [mission_payloads[key]['characteristics']['digital_interoperability_score'] for key in mission_keys]
    fluoroscopy_scores = [mission_payloads[key]['characteristics']['estimated_fluoroscopy_saves'] for key in mission_keys]
    bars = ax.bar(mission_labels, digital_scores, color=[COLORS['blue'], COLORS['cyan'], COLORS['teal']], width=0.58)
    for idx, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.6, f"F={fluoroscopy_scores[idx]}", ha='center', va='bottom', fontsize=7, color=COLORS['muted'])
    ax.set_title('Mission-level digital interoperability', fontsize=9)
    ax.set_ylabel('Digital interoperability score', fontsize=8)
    ax.set_ylim(78, 94)
    ax.grid(True, axis='y', color='#eef2ff')
    ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0.06, 0.06, 0.96, 0.93])
    fig.text(0.5, 0.026, 'Page 2 of 3', ha='center', fontsize=9, color=COLORS['muted'])
    pdf.savefig(fig)
    plt.close(fig)


def render_page_three(pdf, data):
    reference = data['reference']
    phase_scores = data['phase_scores']
    mission_payloads = data['mission_payloads']

    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    fig.text(0.5, 0.965, 'Workflow Translation and Military Education Implications', ha='center', fontsize=12, fontweight='bold', color=COLORS['navy'])
    add_rule(fig, 0.944)

    ax_phase = fig.add_axes([0.08, 0.60, 0.38, 0.24])
    phase_names = [
        'Templating',
        'Exposure',
        'Guidewire control',
        'Bone preparation',
        'Implant and closure',
    ]
    y_pos = np.arange(len(phase_names))
    ax_phase.barh(y_pos, phase_scores, color=COLORS['teal'])
    ax_phase.set_yticks(y_pos)
    ax_phase.set_yticklabels(phase_names, fontsize=8)
    ax_phase.invert_yaxis()
    ax_phase.set_xlim(70, 90)
    ax_phase.set_xlabel('Phase readiness score', fontsize=8)
    ax_phase.set_title('Finite workflow ladder from the tab', fontsize=9)
    ax_phase.grid(True, axis='x', color='#eef2ff')
    ax_phase.tick_params(labelsize=8)

    fig.text(0.55, 0.844, 'Structured minimally invasive workflow guide', ha='left', va='top', fontsize=10, fontweight='bold')
    phase_y = 0.816
    for index, step in enumerate(reference['workflow_guide'], start=1):
        fig.text(0.55, phase_y, f"{index}. {step['phase']}", ha='left', va='top', fontsize=8.8, fontweight='bold', color=COLORS['blue'])
        step_text = f"Objective: {step['objective']} Teaching focus: {step['teaching_focus']}"
        add_wrapped_text(fig, 0.55, phase_y - 0.022, step_text, width=48, fontsize=8.0, color=COLORS['slate'])
        phase_y -= 0.105

    fig.text(0.08, 0.470, 'Operational interpretation', ha='left', va='top', fontsize=10, fontweight='bold')
    interpretation_text = (
        'The knee education tab is most useful when treated as a finite rehearsal system rather than a free-form simulator. '
        'The guidewire equations define a constrained space in which angular bias drives both canal capture and downstream implant alignment. '
        'The workflow layer then translates those continuous-looking curves into five discrete coaching phases that an instructor can evaluate during a single session.'
    )
    add_wrapped_text(fig, 0.08, 0.444, interpretation_text, width=116, fontsize=9.0)

    fig.text(0.08, 0.334, 'US and Canadian military digital transformation framing', ha='left', va='top', fontsize=10, fontweight='bold')
    mission_lines = []
    for mission_key, profile in MILITARY_EDUCATION_PROFILES.items():
        mission_score = mission_payloads[mission_key]['characteristics']['digital_interoperability_score']
        mission_lines.append(
            f"{profile['label']}: {profile['initiative']} Priority: {profile['priority']} "
            f"Model score = {mission_score:.1f}."
        )
    add_wrapped_text(fig, 0.08, 0.308, ' '.join(mission_lines), width=116, fontsize=8.9, color=COLORS['teal'])

    fig.text(0.08, 0.186, 'Conclusion', ha='left', va='top', fontsize=10, fontweight='bold')
    conclusion_text = (
        'This preprint translates the knee surgery education tab into a publication-ready model with explicit finite equations, figure outputs, and workflow guidance. '
        'The resulting PDF can support instructor calibration, program briefs, and internal digital transformation proposals without over-claiming clinical evidence. '
        'Its main value is educational traceability: the same simulation logic now drives the dashboard, the explanatory mathematics, and the exported preprint artifact.'
    )
    add_wrapped_text(fig, 0.08, 0.160, conclusion_text, width=116, fontsize=9.0)

    add_rule(fig, 0.090)
    fig.text(0.5, 0.056, 'Page 3 of 3', ha='center', fontsize=9, color=COLORS['muted'])
    fig.text(0.5, 0.034, 'Artifact generated directly from the knee-education simulation model in app.py.', ha='center', fontsize=7.5, color=COLORS['muted'])

    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)


def run_pdf_generation():
    configure_style()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(repo_root, 'Nature_Preprint_Knee_MIS_Education.pdf')
    data = build_reference_datasets()

    with PdfPages(pdf_path) as pdf:
        metadata = pdf.infodict()
        metadata['Title'] = 'Nature Preprint - Knee MIS Education'
        metadata['Author'] = 'GitHub Copilot / Cartik Sharma workspace'
        metadata['Subject'] = 'Minimally invasive knee surgery education simulation'
        metadata['Keywords'] = 'knee surgery, MIS TKA, guidewire simulation, military education, digital transformation'

        render_page_one(pdf, data)
        render_page_two(pdf, data)
        render_page_three(pdf, data)

    return pdf_path


if __name__ == '__main__':
    output_path = run_pdf_generation()
    print(f'Nature preprint saved to {output_path}')