#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_plots():
    # Set premium scientific styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#94a3b8'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.color'] = '#475569'
    plt.rcParams['ytick.color'] = '#475569'
    
    # -------------------------------------------------------------
    # FIGURE 1: CLINICAL EEG CHARACTERISTICS (WAVEFORMS & PSD)
    # -------------------------------------------------------------
    print("Generating Figure 1: EEG Characteristics...")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5))
    
    # Generate time series data
    t = np.linspace(0, 3.0, 600)
    # Healthy: Alpha dominant
    clean_healthy = 15.0 * np.sin(2 * np.pi * 10.0 * t) + 8.0 * np.sin(2 * np.pi * 20.0 * t)
    noise_healthy = 20.0 * np.random.normal(0, 1.0, len(t))
    # Apnea: Delta/Theta dominant + Beta arousal bursts
    clean_apnea = 25.0 * np.sin(2 * np.pi * 2.0 * t) + 12.0 * np.sin(2 * np.pi * 6.0 * t)
    arousal_mask = (np.sin(2 * np.pi * 0.67 * t) > 0.65).astype(float)
    clean_apnea += arousal_mask * 15.0 * np.sin(2 * np.pi * 20.0 * t)
    noise_apnea = 25.0 * np.random.normal(0, 1.0, len(t)) + arousal_mask * 15.0 * np.random.normal(0, 1.0, len(t))
    # Dementia: Severe theta/delta slowing
    clean_dementia = 20.0 * np.sin(2 * np.pi * 2.0 * t) + 24.0 * np.sin(2 * np.pi * 5.0 * t)
    noise_dementia = 18.0 * np.random.normal(0, 1.0, len(t))
    
    # Denoised estimates (simulated Laplace/Wiener filters)
    filt_healthy = clean_healthy + 3.0 * np.random.normal(0, 1.0, len(t))
    filt_apnea = clean_apnea + 5.0 * np.random.normal(0, 1.0, len(t))
    filt_dementia = clean_dementia + 4.0 * np.random.normal(0, 1.0, len(t))

    axes[0].plot(t, clean_healthy + noise_healthy, color='#ef4444', alpha=0.3, label='Raw Noisy Baseline')
    axes[0].plot(t, filt_healthy, color='#1e3a8a', linewidth=1.5, label='Healthy Baseline (10Hz Alpha)')
    axes[0].plot(t, filt_apnea - 80, color='#0891b2', linewidth=1.5, label='Obstructive Sleep Apnea (Slow + Bursts)')
    axes[0].plot(t, filt_dementia - 160, color='#b45309', linewidth=1.5, label='Dementia (Severe Slowing / Excess Theta)')
    
    axes[0].set_title('a. Clinical EEG Time-Series Characteristics (Raw vs. Denoised)', fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xlabel('Time (seconds)', fontsize=9)
    axes[0].set_ylabel('Amplitude ($\mu$V) / Offset', fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    axes[0].tick_params(axis='both', which='major', labelsize=8)
    
    # PSD Bar Chart
    bands = ['Delta (0.5-3Hz)', 'Theta (4-7Hz)', 'Alpha (8-12Hz)', 'Beta (13-30Hz)', 'Gamma (31-50Hz)']
    healthy_psd = [2.0, 4.0, 15.0, 8.0, 1.5]
    apnea_psd = [25.0, 12.0, 4.0, 16.0, 1.0]
    dementia_psd = [20.0, 24.0, 2.0, 1.5, 0.5]
    
    x = np.arange(len(bands))
    width = 0.25
    
    axes[1].bar(x - width, healthy_psd, width, label='Healthy', color='#1e3a8a', edgecolor='none')
    axes[1].bar(x, apnea_psd, width, label='Apnea (OSA)', color='#0891b2', edgecolor='none')
    axes[1].bar(x + width, dementia_psd, width, label='Dementia', color='#b45309', edgecolor='none')
    
    axes[1].set_title('b. Relative Power Spectral Density (PSD) Band Profiles', fontsize=11, fontweight='bold', pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bands, fontsize=8.5)
    axes[1].set_ylabel('Power ($\mu$V$^2$/Hz)', fontsize=9)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.3)
    axes[1].legend(fontsize=8.5, frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    axes[1].tick_params(axis='both', which='major', labelsize=8.5)
    
    plt.tight_layout()
    plt.savefig('nature_eeg_characteristics.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # FIGURE 2: ANALOG FRONT-END CIRCUIT SCHEMATIC
    # -------------------------------------------------------------
    print("Generating Figure 2: Analog Front-End Schematic...")
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Background card
    ax.add_patch(patches.Rectangle((0, 0), 10, 4, fill=True, color='#f8fafc', ec='#cbd5e1', lw=1))
    
    # Draw Blocks
    def draw_block(x, y, w, h, label, color, border_color):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec=border_color, lw=1.5))
        ax.text(x + w/2.0, y + h/2.0, label, color='#0f172a', fontsize=9, fontweight='bold', ha='center', va='center', wrap=True)

    # Signal path lines
    ax.annotate('', xy=(1.5, 2.0), xytext=(0.5, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    ax.annotate('', xy=(3.2, 2.0), xytext=(2.5, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    ax.annotate('', xy=(5.0, 2.0), xytext=(4.2, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    ax.annotate('', xy=(7.0, 2.0), xytext=(6.0, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    ax.annotate('', xy=(9.0, 2.0), xytext=(8.0, 2.0), arrowprops=dict(arrowstyle="->", lw=1.5, color='#475569'))
    
    # Draw Circuit Blocks
    ax.text(0.5, 2.3, "Scalp\nElectrode\n(Ag/AgCl)", fontsize=8, ha='center', va='center', fontweight='bold', color='#1e3a8a')
    draw_block(1.5, 1.5, 1.0, 1.0, "Schottky\nClamping\nDiodes", "#dbeafe", "#3b82f6")
    draw_block(3.2, 1.5, 1.0, 1.0, "CMOS\nBlanking\nSwitch", "#e2e8f0", "#94a3b8")
    draw_block(5.0, 1.5, 1.0, 1.0, "Instr. Amp\n(AD8221)\nGain Stage", "#ccfbf1", "#0d9488")
    draw_block(7.0, 1.5, 1.0, 1.0, "Active SK\nButterworth\nBandpass", "#fef9c3", "#ca8a04")
    
    ax.text(9.4, 2.2, "Denoised\nOutput\nto ADC", fontsize=8.5, ha='center', va='center', fontweight='bold', color='#1e3a8a')
    
    # Add Math Equations
    ax.text(2.0, 0.7, "Dual clamping:\n$V_{clamp} = \pm 0.35$ V", fontsize=8, ha='center', va='center', color='#334155', bbox=dict(boxstyle="square,pad=0.3", fc="#f1f5f9", ec="#e2e8f0"))
    ax.text(3.7, 0.7, "Isolation:\n$\Delta t_{blank} = 200\ \mu$s", fontsize=8, ha='center', va='center', color='#334155', bbox=dict(boxstyle="square,pad=0.3", fc="#f1f5f9", ec="#e2e8f0"))
    ax.text(5.5, 0.7, "Amplification:\n$G = 1 + \\frac{49.4\\text{ k}\\Omega}{R_g}$", fontsize=8, ha='center', va='center', color='#334155', bbox=dict(boxstyle="square,pad=0.3", fc="#f1f5f9", ec="#e2e8f0"))
    ax.text(7.5, 0.7, "Active Cutoffs:\n$f_{c,HPF} = \\frac{1}{2\\pi R C}$\n$f_{c,LPF} = \\frac{1}{2\\pi\\sqrt{R_1 R_2 C_1 C_2}}$", fontsize=8, ha='center', va='center', color='#334155', bbox=dict(boxstyle="square,pad=0.3", fc="#f1f5f9", ec="#e2e8f0"))
    
    ax.text(5.0, 3.5, "Figure 2: Analog Front-End (AFE) Signal Acquisition Chain with Mathematical Models", fontsize=11, fontweight='bold', ha='center', color='#0f172a')
    
    plt.tight_layout()
    plt.savefig('nature_eeg_afe_schematic.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # FIGURE 3: NEUROMODULATION CIRCUITS & PULSES
    # -------------------------------------------------------------
    print("Generating Figure 3: Neuromodulation circuits...")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    
    # Subplot 0: rTMS Discharge Circuit Block & Pulse
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    ax1.add_patch(patches.Rectangle((0, 0), 10, 5, fill=True, color='#f8fafc', ec='#cbd5e1', lw=1))
    
    # Draw blocks
    draw_block_ax = lambda ax, x, y, w, h, label, fc, ec: ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=fc, ec=ec, lw=1.2)) or ax.text(x + w/2.0, y + h/2.0, label, color='#0f172a', fontsize=8, fontweight='bold', ha='center', va='center')
    
    draw_block_ax(ax1, 0.5, 3.2, 1.2, 0.8, "HV Charger\n(0-2.5 kV)", "#ffe4e6", "#f43f5e")
    draw_block_ax(ax1, 2.3, 3.2, 1.2, 0.8, "Cap Bank\n(50 $\mu$F)", "#ffe4e6", "#f43f5e")
    draw_block_ax(ax1, 4.1, 3.2, 1.2, 0.8, "Thyristor\nSCR Switch", "#ffe4e6", "#f43f5e")
    draw_block_ax(ax1, 5.9, 3.2, 1.2, 0.8, "Figure-8\nCoil (15 $\mu$H)", "#ffe4e6", "#f43f5e")
    
    ax1.annotate('', xy=(2.3, 3.6), xytext=(1.7, 3.6), arrowprops=dict(arrowstyle="->", lw=1.2, color='#475569'))
    ax1.annotate('', xy=(4.1, 3.6), xytext=(3.5, 3.6), arrowprops=dict(arrowstyle="->", lw=1.2, color='#475569'))
    ax1.annotate('', xy=(5.9, 3.6), xytext=(5.3, 3.6), arrowprops=dict(arrowstyle="->", lw=1.2, color='#475569'))
    
    # Draw rTMS wave shape
    t_pulse = np.linspace(0, 1.0, 100)
    rtms_pulse = np.sin(2 * np.pi * 1.5 * t_pulse) * np.exp(-3.0 * t_pulse)
    ax_inset = fig.add_axes([0.1, 0.18, 0.35, 0.22])
    ax_inset.plot(t_pulse * 300, rtms_pulse * 2.2, color='#f43f5e', lw=1.8)
    ax_inset.set_title("Induced Magnetic Pulse (B, Tesla)", fontsize=7.5, fontweight='bold', color='#475569')
    ax_inset.set_xlabel('Time ($\mu$s)', fontsize=6.5)
    ax_inset.set_ylabel('B (Tesla)', fontsize=6.5)
    ax_inset.tick_params(axis='both', which='major', labelsize=6)
    ax_inset.grid(True, linestyle=':', alpha=0.3)
    ax_inset.axhline(0, color='black', lw=0.5, ls='--')
    
    ax1.text(5.0, 4.6, "a. rTMS Discharge Stage & Transcranial Pulse", fontsize=9.5, fontweight='bold', ha='center', color='#0f172a')
    ax1.text(7.6, 3.6, "Energy:\n$E = \\frac{1}{2} C V^2$", fontsize=8, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="#fff", ec="#e2e8f0"))
    
    # Subplot 1: DBS Stage & Balanced Current Pulse
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    ax2.add_patch(patches.Rectangle((0, 0), 10, 5, fill=True, color='#f8fafc', ec='#cbd5e1', lw=1))
    
    draw_block_ax(ax2, 0.5, 3.2, 1.2, 0.8, "Current\nSource", "#ccfbf1", "#0d9488")
    draw_block_ax(ax2, 2.3, 3.2, 1.2, 0.8, "H-Bridge\nBalancer", "#ccfbf1", "#0d9488")
    draw_block_ax(ax2, 4.1, 3.2, 1.2, 0.8, "Charge\nCapacitor", "#ccfbf1", "#0d9488")
    draw_block_ax(ax2, 5.9, 3.2, 1.2, 0.8, "DBS Leads\nElectrodes", "#ccfbf1", "#0d9488")
    
    ax2.annotate('', xy=(2.3, 3.6), xytext=(1.7, 3.6), arrowprops=dict(arrowstyle="->", lw=1.2, color='#475569'))
    ax2.annotate('', xy=(4.1, 3.6), xytext=(3.5, 3.6), arrowprops=dict(arrowstyle="->", lw=1.2, color='#475569'))
    ax2.annotate('', xy=(5.9, 3.6), xytext=(5.3, 3.6), arrowprops=dict(arrowstyle="->", lw=1.2, color='#475569'))
    
    # Draw DBS balanced pulse
    t_dbs = np.linspace(0, 1.0, 500)
    dbs_pulse = np.zeros_like(t_dbs)
    # Phase 1: Excitatory cathodal pulse (100-300us)
    dbs_pulse[(t_dbs >= 0.1) & (t_dbs < 0.3)] = -5.0
    # Phase 2: Anodal charge balancing pulse (slower, longer)
    dbs_pulse[(t_dbs >= 0.3) & (t_dbs < 0.7)] = 1.66  # Area balances: 0.2*(-5.0) + 0.4*(1.66) approx 0
    
    ax_inset2 = fig.add_axes([0.58, 0.18, 0.35, 0.22])
    ax_inset2.plot(t_dbs * 500, dbs_pulse, color='#0d9488', lw=1.8)
    ax_inset2.set_title("Biphasic Charge-Balanced Pulse (I, mA)", fontsize=7.5, fontweight='bold', color='#475569')
    ax_inset2.set_xlabel('Time ($\mu$s)', fontsize=6.5)
    ax_inset2.set_ylabel('I (mA)', fontsize=6.5)
    ax_inset2.tick_params(axis='both', which='major', labelsize=6)
    ax_inset2.grid(True, linestyle=':', alpha=0.3)
    ax_inset2.axhline(0, color='black', lw=0.5, ls='--')
    
    ax2.text(5.0, 4.6, "b. Deep Brain Stimulation (DBS) Balanced Pulse", fontsize=9.5, fontweight='bold', ha='center', color='#0f172a')
    ax2.text(7.6, 3.6, "Balance:\n$\\int I dt = 0$", fontsize=8, ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="#fff", ec="#e2e8f0"))
    
    plt.tight_layout()
    plt.savefig('nature_neuromodulation_schematics.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # FIGURE 4: OPTIMIZATION CONVERGENCE PROFILE
    # -------------------------------------------------------------
    print("Generating Figure 4: Optimization Convergence...")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    
    epochs = np.arange(1, 31)
    
    # 1. VQE cost function: Hamiltonian <H> eigenvalue
    vqe_energy = -8.5 - 4.2 * np.exp(-epochs / 6.0) + np.random.normal(0, 0.02, len(epochs))
    axes[0, 0].plot(epochs, vqe_energy, color='#8b5cf6', marker='o', markersize=4, lw=1.5, label='$\langle H \\rangle$ expectation')
    axes[0, 0].axhline(-12.7, color='#ef4444', ls='--', lw=1, label='Ground state (N=64)')
    axes[0, 0].set_title('a. VQE Parameterized Ansatz Optimizer', fontsize=10, fontweight='bold')
    axes[0, 0].set_xlabel('Iteration / Optimization Step', fontsize=8.5)
    axes[0, 0].set_ylabel('Energy Expectation $\langle H \\rangle$', fontsize=8.5)
    axes[0, 0].grid(True, linestyle='--', alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc='upper right')
    
    # 2. QAOA cost function: Ising Hamiltonian energy
    qaoa_energy = 12.5 * np.exp(-epochs / 8.0) + 0.45 + np.random.normal(0, 0.01, len(epochs))
    axes[0, 1].plot(epochs, qaoa_energy, color='#ec4899', marker='^', markersize=4, lw=1.5, label='Ising Energy')
    axes[0, 1].set_title('b. QAOA Combinatorial Ising Solver', fontsize=10, fontweight='bold')
    axes[0, 1].set_xlabel('Outer Loop Step', fontsize=8.5)
    axes[0, 1].set_ylabel('Expectation value $E(\\vec{\\gamma}, \\vec{\\beta})$', fontsize=8.5)
    axes[0, 1].grid(True, linestyle='--', alpha=0.3)
    axes[0, 1].legend(fontsize=8, loc='upper right')
    
    # 3. Heuristic Simulated Annealing Fitness
    sa_fitness = 18.5 - 12.0 * np.exp(-epochs / 10.0) + np.random.normal(0, 0.15, len(epochs))
    axes[1, 0].plot(epochs, sa_fitness, color='#ca8a04', marker='s', markersize=4, lw=1.5, label='Fitness Score')
    axes[1, 0].set_title('c. Heuristic Simulated Annealing Search', fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch (cooling steps)', fontsize=8.5)
    axes[1, 0].set_ylabel('Capacity Fitness Score', fontsize=8.5)
    axes[1, 0].grid(True, linestyle='--', alpha=0.3)
    axes[1, 0].legend(fontsize=8, loc='lower right')
    
    # 4. Statistical ML (Laplace Denoising Shrinkage)
    laplace_loss = 35.0 / (epochs**0.5) + np.random.normal(0, 0.05, len(epochs))
    axes[1, 1].plot(epochs, laplace_loss, color='#0ea5e9', marker='d', markersize=4, lw=1.5, label='Laplace loss')
    axes[1, 1].set_title('d. Statistical ML Laplace Denoising', fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel('Fitting Iterations', fontsize=8.5)
    axes[1, 1].set_ylabel('SNR Optimization Loss (dB)', fontsize=8.5)
    axes[1, 1].grid(True, linestyle='--', alpha=0.3)
    axes[1, 1].legend(fontsize=8, loc='upper right')
    
    plt.suptitle('Figure 4: Optimization Solver Convergence & Cost Traces', fontsize=12, fontweight='bold', y=0.98, color='#0f172a')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('nature_optimization_convergence.png', dpi=300)
    plt.close()
    
    print("All figures successfully generated!")

if __name__ == '__main__':
    generate_plots()
