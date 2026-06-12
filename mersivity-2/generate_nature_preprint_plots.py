#!/usr/bin/env python3
"""
Generate publication-quality scientific plots for the Nature Preprint.
Produces a 2x2 grid plot saved as 'nature_plots_grid.png'.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    # Setup styles
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Dataset matching the frontend Registration Characteristics
    algorithms = [
        'GMM', 'qLoRA', 'Feynman', 'Cont. Fraction', 
        'Quantum ML', 'MRI-to-CT QML', 'MRI-to-STL QML+Fey', 'Stat+Comb'
    ]
    tre_vals = [0.142, 0.134, 0.148, 0.118, 0.095, 0.086, 0.076, 0.068]
    speeds = [1.25, 0.85, 0.92, 2.10, 3.45, 3.82, 4.56, 1.84]
    var_vals = [0.245, 0.224, 0.238, 0.194, 0.145, 0.134, 0.112, 0.098]
    cvar_vals = [0.312, 0.285, 0.298, 0.246, 0.186, 0.168, 0.136, 0.124]
    resilience_scores = [65, 70, 72, 80, 88, 90, 94, 98]

    # Nature-inspired palette
    colors = ['#818cf8', '#a78bfa', '#f472b6', '#fb7185', '#60a5fa', '#06b6d4', '#2dd4bf', '#34d399']
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), facecolor='#ffffff')
    
    # 1. Target Registration Error (TRE)
    axs[0, 0].bar(algorithms, tre_vals, color=colors, edgecolor='#e2e8f0', width=0.6)
    axs[0, 0].set_title('a  Target Registration Error (TRE)', fontsize=11, fontweight='bold', loc='left', pad=10)
    axs[0, 0].set_ylabel('TRE (mm)', fontsize=9)
    axs[0, 0].tick_params(axis='x', rotation=25, labelsize=8)
    axs[0, 0].tick_params(axis='y', labelsize=8)
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.3)
    
    # 2. Computation / Convergence Time
    axs[0, 1].bar(algorithms, speeds, color=colors, edgecolor='#e2e8f0', width=0.6)
    axs[0, 1].set_title('b  Computation / Convergence Time', fontsize=11, fontweight='bold', loc='left', pad=10)
    axs[0, 1].set_ylabel('Time (seconds)', fontsize=9)
    axs[0, 1].tick_params(axis='x', rotation=25, labelsize=8)
    axs[0, 1].tick_params(axis='y', labelsize=8)
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.3)
    
    # 3. Outlier Risk Telemetry (VaR & CVaR 95%)
    x = np.arange(len(algorithms))
    width = 0.35
    axs[1, 0].bar(x - width/2, var_vals, width, label='VaR (95%)', color='#fb923c', edgecolor='#e2e8f0')
    axs[1, 0].bar(x + width/2, cvar_vals, width, label='CVaR (95%)', color='#ef4444', edgecolor='#e2e8f0')
    axs[1, 0].set_title('c  Outlier Risk Telemetry (95% Conf.)', fontsize=11, fontweight='bold', loc='left', pad=10)
    axs[1, 0].set_ylabel('Residual Error Bounds (mm)', fontsize=9)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(algorithms)
    axs[1, 0].tick_params(axis='x', rotation=25, labelsize=8)
    axs[1, 0].tick_params(axis='y', labelsize=8)
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.3)
    axs[1, 0].legend(fontsize=8, framealpha=0.8)
    
    # 4. Outlier Noise Resilience Index
    axs[1, 1].plot(algorithms, resilience_scores, marker='o', color='#10b981', linewidth=2, markersize=6)
    axs[1, 1].set_title('d  Outlier Noise Resilience Index', fontsize=11, fontweight='bold', loc='left', pad=10)
    axs[1, 1].set_ylabel('Resilience Score (%)', fontsize=9)
    axs[1, 1].tick_params(axis='x', rotation=25, labelsize=8)
    axs[1, 1].tick_params(axis='y', labelsize=8)
    axs[1, 1].grid(axis='both', linestyle='--', alpha=0.3)
    axs[1, 1].set_ylim(60, 102)

    plt.tight_layout()
    plt.savefig('nature_plots_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Nature preprint plots saved to: nature_plots_grid.png")

if __name__ == '__main__':
    generate_plots()
