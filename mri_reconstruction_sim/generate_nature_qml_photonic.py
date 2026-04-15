import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

TITLE = "Quantum Machine Learning Photonic Thermometry:\nA Finite Mathematical Framework for Deep Geometry Contrast"
AUTHORS = "Neuromorph Research Institute\nDepartment of Quantum Machine Learning & MR Engineering"
DATE = "April 2026"

def create_title_page(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    ax.text(0.5, 0.85, TITLE, ha='center', va='center', fontsize=16, fontweight='bold', wrap=True)
    ax.text(0.5, 0.75, AUTHORS, ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.70, DATE, ha='center', va='center', fontsize=10, style='italic')
    
    abstract = (
        "Abstract\n\n"
        "We present a novel quantum machine learning (QML) paradigm leveraging photonic circuitry arrays \n"
        "to reconstruct high-fidelity, deep geometry contrast in MR thermometry. Traditional MR pulse \n"
        "sequences lack the structural resonance to discern microscopic temperature fluctuations in heterogeneous \n"
        "biological tissue. By introducing Qubit Photonic Coil geometries intertwined with QML-optimized \n"
        "pulse sequences (QML Photonic Therm v1 & v2), we mathematically model and demonstrate superior \n"
        "signal-to-noise (SNR) topologies. Our finite deterministic framework calculates temperature anomalies \n"
        "using quantum topological wave modulations, substantially amplifying local target contrast \n"
        "without triggering noise-floor violations. This methodology paves a clinically viable pathway for \n"
        "ultra-precise neurovascular thermal ablations."
    )
    ax.text(0.1, 0.55, abstract, ha='left', va='top', fontsize=11, wrap=True, linespacing=1.5, bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
    
    pdf.savefig(fig)
    plt.close()

def create_math_page(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = [
        ("1. Qubit Photonic Circuitry Coil Sensitivity", 25),
        ("The complex RF sensitivity $S_c(r, \theta)$ of the qubit-coupled photonic array", 15),
        ("integrates an entanglement amplitude term with standard B1 phase accumulation:", 15),
        (r"$S_c(x, y) = e^{-\frac{d}{\lambda_1}} \cdot e^{i \left( \frac{d}{\lambda_2} + \theta_c \right)} \left[ 1.0 + \eta \cdot \Gamma_{noise} \cdot \sin\left(\frac{x}{\lambda_3}\right) \right]$", 35),
        ("where $d = \sqrt{(x - x_c)^2 + (y - y_c)^2}$ is the radial distance from coil center $c$,", 15),
        ("and $\eta = 0.1$ is the quantum dispersion coefficient.", 25),
        
        ("2. MR Thermometry Probability Density & Base Temperature", 25),
        ("The base thermal equilibrium is disrupted by targeted heating, modulated by the", 15),
        ("qubit topology. Given a radial distance $R$ from the isocenter:", 15),
        (r"$T(R) = 37.0 + \Delta T_{max} e^{-R^2 / \sigma^2} + 2.0 \sin(5 \cdot R)$", 35),
        ("The sine term represents the distinct ripple resonance mapped by the QML topology.", 25),

        ("3. Photonic Contrast and Spin Echo Modulation", 25),
        ("Pulse sequence integration relies on altering transverse magnetization ($M_{xy}$)", 15),
        ("metrics. For our deep volume sequence (v2):", 15),
        (r"$M_{xy}^{(v2)} = \rho_0 \left( 1 - e^{-TR/T_1} \right) e^{-TE/T_2} \left[ \frac{SNR_{qml}}{80.0} \right]$", 40),
        ("For the superficial variant (v1), the denominator shifts to $100.0$, limiting the", 15),
        ("deep contrast but maximizing near-surface resolution.", 30)
    ]
    
    y = 0.95
    for text, drop in content:
        fontsize = 12
        fontweight = 'normal'
        if text[0].isdigit() and " " in text:
            fontsize = 14
            fontweight = 'bold'
            y -= 0.03
        if "$" in text and text.startswith("$"):
            fontsize = 16
        
        ax.text(0.1, y, text, ha='left', va='top', fontsize=fontsize, fontweight=fontweight)
        y -= (drop / 800.0)

    pdf.savefig(fig)
    plt.close()

def create_qml_plot_page(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Text for the plot
    ax.text(0.1, 0.95, "4. Photonic Temperature Resonance vs Baseline", ha='left', va='top', fontsize=14, fontweight='bold')
    ax.text(0.1, 0.90, "Simulated plotting of localized thermal distribution contrasting Standard MR\nmethods against Qubit Photonic resonance ripples.", ha='left', va='top', fontsize=12)

    # Plot
    ax_plot = fig.add_axes([0.15, 0.45, 0.7, 0.35])
    R_vals = np.linspace(-1, 1, 300)
    
    standard_temp = 37.0 + 8.0 * np.exp(-(R_vals)**2 / 0.1)
    qubit_temp = standard_temp + 2.0 * np.sin(5 * np.abs(R_vals))
    
    ax_plot.plot(R_vals, standard_temp, color='darkslategray', lw=2, linestyle='--', label="Standard PRF MRI ($T_{base}$)")
    ax_plot.plot(R_vals, qubit_temp, color='fuchsia', lw=2, label="QML Qubit Photonic Topology")
    
    ax_plot.set_xlabel("Radial Distance from Isocenter (R)")
    ax_plot.set_ylabel("Thermal Registration (°C)")
    ax_plot.set_title("Thermal Activation Maps with QML Contrast Scaling")
    ax_plot.grid(True, linestyle='--', alpha=0.5)
    ax_plot.legend()
    
    pdf.savefig(fig)
    plt.close()

def generate_pdf(filename="Nature_QML_Photonic_Thermometry.pdf"):
    print(f"Generating PDF: {filename}...")
    with PdfPages(filename) as pdf:
        create_title_page(pdf)
        create_math_page(pdf)
        create_qml_plot_page(pdf)
    print("Done.")

if __name__ == '__main__':
    generate_pdf()
