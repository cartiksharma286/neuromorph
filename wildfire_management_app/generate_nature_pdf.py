#!/usr/bin/env python3
"""
generate_nature_pdf.py - Generates a 3-page Nature-style preprint PDF focused
on Ontario wildfire operations, containment strategies with ecological restoration,
and East Coast smoke propagation. Integrates finite algebra equations.
"""

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from app import (
    EAST_COAST_CITIES,
    MONTH_LABELS,
    SUMMER_INDICES,
    compute_containment_strategy,
    compute_smoke_propagation,
    compute_wildfire_command,
)


COLORS = {
    'navy': '#0b1622',
    'forest': '#1e3f20',
    'teal': '#0f766e',
    'ember': '#c2410c',
    'gold': '#a16207',
    'sky': '#0369a1',
    'muted': '#71717a',
    'light': '#e4e4e7',
    'white': '#ffffff',
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


def add_wrapped_text(fig, x_pos, y_pos, text, width, fontsize=9.2, color=COLORS['navy'], weight='normal', style='normal'):
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


def render_page_one(pdf, state):
    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    fig.text(0.5, 0.965, 'NATURE PREPRINT | STATISTICAL QUANTUM MACHINE LEARNING', ha='center', fontsize=8.5, fontweight='bold', color=COLORS['ember'])
    fig.text(0.5, 0.928, 'A Finite-Horizon Quantum Machine Learning and', ha='center', fontsize=15.5, fontweight='bold', color=COLORS['navy'])
    fig.text(0.5, 0.898, 'Ecological Land Restoration Model for Ontario Wildfires', ha='center', fontsize=15.5, fontweight='bold', color=COLORS['navy'])
    fig.text(0.5, 0.870, 'Cartik Sharma | Department of Climate and Computational Physics', ha='center', fontsize=9, style='italic', color=COLORS['muted'])
    add_rule(fig, 0.848)

    fig.text(0.08, 0.824, 'Abstract', ha='left', va='top', fontsize=11, fontweight='bold')
    abstract_text = (
        'Here we present a comprehensive preprint document for the new Ontario and Northern Ontario wildfire '
        'management and air quality simulator. This model incorporates direct attack, firebreak creation, prescribed burning, '
        'ecological land restoration, and smoke propagation eastward to the US East Coast. By linking quantum machine learning '
        'alignment weights and meteorological drivers on a finite-horizon lattice spanning 24 months, the simulator projects '
        'mitigation convergence and downstream plume transport under varying policy interventions. For Northern Ontario as a baseline, '
        'the simulator resolves a summer PM2.5 reduction and accelerates ecological land recovery, establishing a clear link '
        'between active fire mitigation and regional air quality safeguards.'
    )
    add_wrapped_text(fig, 0.08, 0.798, abstract_text, width=116, fontsize=9.2)

    fig.text(0.08, 0.664, 'Finite-Horizon Formulation', ha='left', va='top', fontsize=11, fontweight='bold')
    formulation_text = (
        'The containment and transport models are formulated as discrete algebraic structures evaluated for months k = 0, ..., 23 '
        'covering the May-to-May 24-month horizon. This finite math mirrors the backend logic in app.py precisely, allowing direct audit '
        'and validation of the dashboard characteristics.'
    )
    add_wrapped_text(fig, 0.08, 0.638, formulation_text, width=116, fontsize=9.0)

    fig.text(0.10, 0.566, r"$C_k = 30 + 56\left(1 - e^{-\alpha (k+1) K_k}\right) - 8.5(\Psi_k - 0.8) + 4\eta_p + 2\eta_f$", ha='left', va='top', fontsize=12, color=COLORS['teal'])
    fig.text(0.10, 0.490, r"$\Delta_k = A_k \cdot \Psi_k \cdot e^{-0.05k} \cdot \left(1.18 - \frac{S_k}{100}\right)$", ha='left', va='top', fontsize=12, color=COLORS['teal'])
    fig.text(0.10, 0.414, r"$P_{\text{city}, k} = S_{\text{load}, k} \cdot (0.84 + 0.012 F) \cdot e^{-\lambda_d d_{\text{city}} / 900} \cdot (1.0 + 0.12\Psi_k)$", ha='left', va='top', fontsize=12, color=COLORS['teal'])

    definitions_text = (
        'where C_k is mitigation convergence percentage, K_k is the QML alignment kernel, alpha is the convergence rate, '
        'Psi_k is the seasonal wildfire pressure, eta_p is the prescribed burn capacity, eta_f is the firebreak length, '
        'Delta_k is the active residual risk area, S_k is suppression efficiency, P_city,k is the transported PM2.5 at a given US city, '
        'F is the easterly wind corridor flow, lambda_d is the dissipation coefficient, and d_city is the city distance in kilometers.'
    )
    add_wrapped_text(fig, 0.08, 0.356, definitions_text, width=116, fontsize=8.8, color=COLORS['muted'])

    fig.text(0.08, 0.250, 'Baseline Ontario Scenario Analysis', ha='left', va='top', fontsize=11, fontweight='bold')
    scenario_text = (
        f"For Northern Ontario Boreal Belt, the reference containment strategy with {state['crews']} crews, 145 km of firebreaks, "
        f"{state['restoration']}% ecological restoration, and {state['burn']}% prescribed burn capacity converges toward a final "
        f"mitigation score of {state['characteristics']['final_convergence_pct']:.1f}% by month {state['characteristics']['convergence_month']}. "
        f"This strategy results in a cumulative {state['characteristics']['restored_hectares']:.0f} restored hectares across the region "
        f"and reduces summer PM2.5 concentrations by {state['characteristics']['summer_pm25_reduction']:.1f} ug/m3, showing the "
        'significant leverage generated by linking containment with active ecological restoration.'
    )
    add_wrapped_text(fig, 0.08, 0.224, scenario_text, width=116, fontsize=9.0)

    add_rule(fig, 0.090)
    fig.text(0.5, 0.056, 'Page 1 of 3', ha='center', fontsize=9, color=COLORS['muted'])
    fig.text(0.5, 0.034, 'Preprint generated directly from wildfire simulation models in app.py.', ha='center', fontsize=7.5, color=COLORS['muted'])

    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)


def render_page_two(pdf, state):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11), facecolor='white')
    fig.suptitle('Ontario Wildfire Finite-Horizon Simulation Metrics', fontsize=12, fontweight='bold', y=0.975, color=COLORS['navy'])
    fig.text(0.5, 0.952, f"Active Source Area: {state['region_label']} | QML Mitigation Constraint: {state['qml']}", ha='center', fontsize=8.5, color=COLORS['muted'])

    ax = axes[0, 0]
    ax.plot(MONTH_LABELS, state['mitigation_convergence_pct'], color=COLORS['forest'], linewidth=2.2, label='Mitigation convergence %')
    ax.set_title('Mitigation convergence vs residual risk area', fontsize=9)
    ax.set_xlabel('Horizon Month', fontsize=8)
    ax.set_ylabel('Convergence (%)', fontsize=8)
    ax.grid(True, color='#f1f5f9')
    ax.tick_params(labelsize=7, labelrotation=30)

    ax_right = ax.twinx()
    ax_right.plot(MONTH_LABELS, state['active_risk_hectares'], color=COLORS['ember'], linewidth=1.8, linestyle='--', label='Risk area (ha)')
    ax_right.set_ylabel('Residual fire area (ha)', fontsize=8, color=COLORS['ember'])
    ax_right.tick_params(labelsize=7, colors=COLORS['ember'])
    ax_right.tick_params(axis='x', labelrotation=30)
    handles = ax.get_lines() + ax_right.get_lines()
    labels = [line.get_label() for line in handles]
    ax.legend(handles, labels, fontsize=7, frameon=False, loc='upper left')

    ax = axes[0, 1]
    ax.plot(MONTH_LABELS, state['cumulative_restored_hectares'], color=COLORS['teal'], linewidth=2.2, label='Cumulative restored ha')
    ax.set_title('Ecological restoration and land recovery index', fontsize=9)
    ax.set_xlabel('Horizon Month', fontsize=8)
    ax.set_ylabel('Restored hectares', fontsize=8)
    ax.grid(True, color='#f1f5f9')
    ax.tick_params(labelsize=7, labelrotation=30)

    ax_right = ax.twinx()
    ax_right.plot(MONTH_LABELS, state['land_recovery_index'], color=COLORS['sky'], linewidth=1.8, linestyle=':', label='Recovery score')
    ax_right.set_ylabel('Recovery score', fontsize=8, color=COLORS['sky'])
    ax_right.tick_params(labelsize=7, colors=COLORS['sky'])
    ax_right.tick_params(axis='x', labelrotation=30)
    handles = ax.get_lines() + ax_right.get_lines()
    labels = [line.get_label() for line in handles]
    ax.legend(handles, labels, fontsize=7, frameon=False, loc='upper left')

    ax = axes[1, 0]
    ax.plot(MONTH_LABELS, state['regional_pm25'], color=COLORS['ember'], linewidth=2.0, label='Ontario PM2.5 (ug/m3)')
    ax.plot(MONTH_LABELS, state['smoke']['transported_pm25'], color=COLORS['sky'], linewidth=1.8, linestyle='--', label='East Coast PM2.5')
    ax.set_title('Source emission vs East Coast transported PM2.5', fontsize=9)
    ax.set_xlabel('Horizon Month', fontsize=8)
    ax.set_ylabel('PM2.5 concentration', fontsize=8)
    ax.grid(True, color='#f1f5f9')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.tick_params(labelsize=7, labelrotation=30)

    ax = axes[1, 1]
    city_pm = [city['pm25'] for city in state['smoke']['city_snapshot']]
    city_aqi = [city['aqi'] for city in state['smoke']['city_snapshot']]
    city_names = [city['city'] for city in state['smoke']['city_snapshot']]
    x_pos = np.arange(len(city_names))
    width = 0.35
    ax.bar(x_pos - width/2.0, city_pm, width=width, color=COLORS['ember'], label='Summer PM2.5')
    ax.set_title('US East Coast city-by-city summer burden', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(city_names, fontsize=7, rotation=15)
    ax.set_ylabel('PM2.5 (ug/m3)', fontsize=8)
    ax.set_xlabel('US City', fontsize=8)
    ax.grid(True, axis='y', color='#f1f5f9')
    
    ax_right = ax.twinx()
    ax_right.scatter(x_pos + width/2.0, city_aqi, color=COLORS['navy'], marker='d', s=24, label='Summer AQI')
    ax_right.set_ylabel('AQI index', fontsize=8)
    ax_right.tick_params(labelsize=7)
    handles = [axes[1, 1].patches[0], ax_right.collections[0]]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, fontsize=7, frameon=False, loc='upper right')

    plt.tight_layout(rect=[0.06, 0.06, 0.96, 0.93])
    fig.text(0.5, 0.026, 'Page 2 of 3', ha='center', fontsize=9, color=COLORS['muted'])
    pdf.savefig(fig)
    plt.close(fig)


def render_page_three(pdf, state):
    reference = state['characteristics']
    smoke_chars = state['smoke']['characteristics']

    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    fig.text(0.5, 0.965, 'Downstream Air Quality Safeguards & Propagation Physics', ha='center', fontsize=12, fontweight='bold', color=COLORS['navy'])
    add_rule(fig, 0.944)

    fig.text(0.08, 0.900, 'Downstream US East Coast Propagation Analytics', ha='left', va='top', fontsize=11, fontweight='bold')
    add_wrapped_text(fig, 0.08, 0.874, state['smoke']['narrative'], width=116, fontsize=9.0)

    fig.text(0.08, 0.740, 'Ecological restoration of land and watershed recovery', ha='left', va='top', fontsize=11, fontweight='bold')
    eco_text = (
        'A main feature of the containment strategy tab is the link between mechanical firefighting '
        'and ecological restoration. Under a high-capacity scenario, active restoration suppresses '
        'secondary ignition hotspots, speeds up forest biome recovery, and protects local watersheds. This links '
        'short-term wildfire defense directly to long-term carbon capture goals and forest health.'
    )
    add_wrapped_text(fig, 0.08, 0.714, eco_text, width=116, fontsize=9.2)

    fig.text(0.08, 0.584, 'Statistical quantum machine learning dissipation model', ha='left', va='top', fontsize=11, fontweight='bold')
    qml_text = (
        f"The quantum machine learning alignment weight (set at {state['qml']:.2f}) acts as a policy dissipation "
        'regularizer during regional operations. By projecting wildfire containment steps as a sequence of quantum '
        'rotation states, the optimizer finds containment corridors that are robust to shifting winds. This regularized '
        f"pathway yields a containment stability score of {reference['convergence_stability_score']:.1f}% "
        f"and achieves an East Coast AQI safeguard index of {smoke_chars['east_coast_aqi_safeguard']:.1f}/100, "
        'safeguarding major populated East Coast targets.'
    )
    add_wrapped_text(fig, 0.08, 0.558, qml_text, width=116, fontsize=9.2)

    fig.text(0.08, 0.404, 'Conclusion and future policy outlook', ha='left', va='top', fontsize=11, fontweight='bold')
    conclusion_text = (
        'This nature preprint confirms that coupling machine learning with ecological restoration produces '
        'a robust containment framework for Northern Ontario forests. By mapping the full cascade of wildfire '
        'containment parameters through to downstream US East Coast air quality safeguards, the model supports '
        'cooperative trans-boundary climate and air quality policy development.'
    )
    add_wrapped_text(fig, 0.08, 0.378, conclusion_text, width=116, fontsize=9.2)

    fig.text(0.08, 0.234, 'Model metadata and verification markers', ha='left', va='top', fontsize=10, fontweight='bold')
    metadata_lines = [
        f"Source Region Profile: {state['region_label']}",
        f"Crews Deployed: {state['crews']} | Target Firebreaks: {state['firebreak']} km",
        f"Ecological Restoration Weight: {state['restoration']}% | Prescribed Burning: {state['burn']}%",
        f"Mitigation Convergence Target: M{reference['convergence_month']} | Half-Life: {reference['mitigation_half_life_month']} months",
        f"US East Coast Safeguard Score: {smoke_chars['east_coast_aqi_safeguard']:.1f}/100",
        f"Highest Risk East Coast City Corridor: {smoke_chars['highest_risk_city']} | Plume Half-Life: {smoke_chars['average_plume_half_life_hours']:.1f} hours",
    ]
    metadata_text = ' '.join(metadata_lines)
    add_wrapped_text(fig, 0.08, 0.208, metadata_text, width=116, fontsize=8.8, color=COLORS['teal'])

    add_rule(fig, 0.090)
    fig.text(0.5, 0.056, 'Page 3 of 3', ha='center', fontsize=9, color=COLORS['muted'])
    fig.text(0.5, 0.034, 'End of preprint artifact.', ha='center', fontsize=7.5, color=COLORS['muted'])

    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)


def run_pdf_generation():
    configure_style()

    state = compute_containment_strategy(
        region='northern_ontario',
        crews=118,
        firebreak_km=145,
        restoration=70,
        prescribed_burn=54,
        qml_weight=0.66,
    )
    state['crews'] = 118
    state['firebreak'] = 145
    state['restoration'] = 70
    state['burn'] = 54
    state['qml'] = 0.66

    state['smoke'] = compute_smoke_propagation(
        region='northern_ontario',
        source_intensity=1.15,
        easterly_flow=24,
        humidity_scrub=58,
        qml_dissipation=0.64,
    )

    repo_root = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(repo_root, 'Nature_Preprint_Wildfire_QML_Restoration.pdf')

    with PdfPages(pdf_path) as pdf:
        metadata = pdf.infodict()
        metadata['Title'] = 'Wildfire QML and Ecological Restoration Preprint'
        metadata['Author'] = 'Cartik Sharma Dept of Climate and Computational Physics'
        metadata['Subject'] = 'Ontario wildfire containment and East Coast smoke propagation'
        metadata['Keywords'] = 'wildfire, qml, ecological restoration, smoke propagation, east coast aqi'

        render_page_one(pdf, state)
        render_page_two(pdf, state)
        render_page_three(pdf, state)

    return pdf_path


if __name__ == '__main__':
    output_path = run_pdf_generation()
    print(f"Preprint successfully compiled and saved to {output_path}")
