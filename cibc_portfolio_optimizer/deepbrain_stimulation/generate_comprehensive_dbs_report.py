
"""
Generates a Comprehensive Finite Math Technical Report for the Deep Brain Stimulation App.
Includes modules: FEA, OCD, Depression, ASD, and Dementia.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.stats as stats

# --- Aesthetics ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

# --- Content Definitions ---
TITLE = "Comprehensive Finite Mathematical Framework\nfor Deep Brain Stimulation Systems"
SUBTITLE = "Technical Report: Equations, Models, and Dynamics"
AUTHORS = "Deep Brain Stimulation Scientific Advisory Board"
DATE = "January 2026"

ABSTRACT = (
    "This technical report formalizes the mathematical underpinnings of the Deep Brain Stimulation (DBS) "
    "application. It covers the Finite Element Analysis (FEA) of electric fields, the coupled differential "
    "equations for OCD CSTC loops, the neurotransmitter dynamics in Depression, the quantum surface integrals "
    "and continued fraction metrics for ASD neural repair, and the neural field equations for Dementia pathology. "
    "Each module is grounded in stochastic physics and statistical mechanics."
)

# --- SECTION 1: FEA ---
H1 = "1. Finite Element Analysis (FEA) of Electric Fields"
T1 = "The fundamental physics of DBS involves the distribution of electric potential Phi in the brain tissue, governed by the quasi-static Poisson equation:"
E1 = r"$\nabla \cdot (\sigma(\mathbf{x}) \nabla \Phi(\mathbf{x})) = -\nabla \cdot \mathbf{J}_{source}$"
T1b = "where sigma(x) represents the anisotropic conductivity tensor. The Volume of Tissue Activated (VTA) is the manifold where the second spatial derivative of the potential exceeds the axonal threshold:"
E1b = r"$VTA = \{ \mathbf{x} \in \Omega \mid \Delta^2 \Phi(\mathbf{x}) > \theta_{activation} \}$"

# --- SECTION 2: OCD (CSTC Loop) ---
H2 = "2. OCD Module: CSTC Loop Dynamics"
T2 = "OCD pathology is modeled as a hyperactive gain cycle in the Cortico-Striato-Thalamo-Cortical loop. The state vectors for Orbitofrontal Cortex (OFC), Caudate (C), and Thalamus (T) evolve according to coupled nonlinear ODEs:"
E2a = r"$\tau \frac{dC}{dt} = -C + \mathcal{S}(w_{OFC \to C} \cdot OFC + I_{DBS})$"
E2b = r"$\tau \frac{dT}{dt} = -T + \mathcal{S}(w_{C \to T} \cdot C - w_{GPi \to T} \cdot GPi)$"
E2c = r"$\tau \frac{dOFC}{dt} = -OFC + \mathcal{S}(w_{T \to OFC} \cdot T + I_{ext})$"
T2b = "where S(x) is the sigmoid activation function. The 'Obsession' state corresponds to a high-energy attractor in this dynamical system."

# --- SECTION 3: DEPRESSION ---
H3 = "3. Depression Module: Neurotransmitter Dynamics"
T3 = "The Depression model simulates the restoration of bioamines (Serotonin 5-HT, Dopamine DA) and the regulation of Glutamate (Glu). The dynamics under formulation are:"
E3a = r"$\frac{dS_{5HT}}{dt} = \alpha \cdot E_{eff}(f) \cdot (1 - S_{5HT}) - \gamma (S_{5HT} - S_{base})$"
E3b = r"$\frac{dD_{DA}}{dt} = \beta \cdot E_{eff}(f) \cdot (1 - D_{DA}) - \gamma (D_{DA} - D_{base})$"
T3b = "The stimulation efficacy E_eff is strictly frequency-dependent, modeled as a Gaussian resonance around 130 Hz:"
E3c = r"$E_{eff}(f) = A \cdot \exp\left( - \frac{(f - 130)^2}{2\sigma^2} \right) \cdot \frac{V_{activation}}{V_{ref}}$"
T3c = "Executive Function metrics (M) are derived as linear combinations of the state vector:"
E3d = r"$M_{decision} = w_1 S_{5HT} + w_2 D_{DA} + b$"

# --- SECTION 4: ASD (Quantum Repair) ---
H4 = "4. ASD Module: Quantum Neural Repair"
T4 = "ASD repair focuses on restoring connectivity. We utilize a Quantum Surface Integral formalism where the neural connectivity is treated as a wavefunction Psi:"
E4a = r"$\Phi_{repair} = \oint_{S} \Psi^* \nabla \Psi \cdot \hat{\mathbf{n}} \, dS$"
T4b = "The repair trajectory is modeled using Continued Fraction sequences, representing the hierarchical restructuring of neural pathways:"
E4b = r"$C(t) = a_0 + \frac{1}{a_1 + \frac{1}{a_2 + \dots}}$"
T4c = "Coherence of the repaired state is measured via Von Neumann Entropy:"
E4c = r"$S_{vonNeumann} = - \mathrm{Tr}(\rho \ln \rho)$"

# --- SECTION 5: DEMENTIA ---
H5 = "5. Dementia Module: Neural Decay & Memory"
T5 = "Dementia pathology involves time-dependent atrophy and amyloid burden (Beta). The activity A of a memory region i is governed by:"
E5a = r"$\tau \frac{dA_i}{dt} = -A_i + \sigma\left( \sum_j w_{ij} A_j + C_{chol} - P_{\beta} \right)$"
T5b = "Cognitive scores (MMSE) scale with the integral of hippocampal activity and inverse pathology load:"
E5b = r"$MMSE \propto \int_{T} A_{hippocampus}(t) \cdot (1 - P_{\beta}(t)) \, dt$"
T5c = "DBS effectiveness at theta frequencies (4-8 Hz) enhances memory encoding via a logarithmic sensitivity function:"
E5c = r"$E_{stim} \propto Q_{charge} \cdot \frac{\ln(f+1)}{\ln(f_{cut})}$"

# --- PLOTTING ---

def plot_depression_dynamics(ax):
    """Visualizes Treatment Efficacy vs Frequency for Depression."""
    f = np.linspace(0, 200, 200)
    # Gaussian centered at 130 Hz
    eff = np.exp(-((f - 130)**2) / (2 * 40**2))
    
    ax.plot(f, eff, 'b-', linewidth=2, label=r'$E_{eff}(f)$')
    ax.fill_between(f, eff, alpha=0.1, color='blue')
    ax.axvline(130, color='r', linestyle='--', label='130 Hz Optimum')
    
    ax.set_title("Depression Treatment Efficacy Profile", fontsize=10)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Efficacy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_asd_continued_fraction(ax):
    """Visualizes convergence of continued fractions."""
    t = np.arange(1, 21)
    # Simulated convergence
    y = 1.618 - 0.5 * np.exp(-t/5) + 0.05 * np.cos(t)
    
    ax.plot(t, y, 'g-o', markersize=4, label='Convergence C(t)')
    ax.axhline(1.618, color='k', linestyle='--', alpha=0.5, label='Golden Ratio (Target)')
    
    ax.set_title("ASD Repair: Continued Fraction Convergence", fontsize=10)
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Metric Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_dementia_decay(ax):
    """Visualizes Neural Activity Decay vs DBS Rescue."""
    t = np.linspace(0, 10, 100)
    decay = np.exp(-0.3 * t)
    rescue = np.exp(-0.3 * t) + 0.4 * (1 - np.exp(-(t-4)/2)) * (t>4)
    
    ax.plot(t, decay, 'r--', label='Natural Decay')
    ax.plot(t, rescue, 'b-', label='With DBS (t=4)')
    
    ax.set_title("Dementia: Hippocampal Activity Rescue", fontsize=10)
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Normalized Activity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# --- LAYOUT HELPER ---
def draw_content(ax, items):
    y = 0.90
    
    line_h_text = 0.03
    spacing_para = 0.04
    spacing_eq = 0.08
    
    for kind, text_content in items:
        if kind == 'head':
            y -= 0.04
            ax.text(0.00, y, text_content, weight='bold', fontsize=12, color='#222')
            y -= 0.03
        elif kind == 'text':
            ax.text(0.02, y, text_content, fontsize=10, wrap=True, ha='left', va='top', color='#333')
            # Estimate lines
            num_lines = len(text_content) // 95 + 1
            y -= num_lines * line_h_text
            y -= spacing_para
        elif kind == 'eq':
            y -= 0.01
            for eq in text_content if isinstance(text_content, list) else [text_content]:
                ax.text(0.5, y, eq, fontsize=12, ha='center', va='top', color='#003366')
                y -= 0.05
            y -= spacing_para

# --- MAIN GENERATOR ---
def create_comprehensive_report():
    pdf_name = "Comprehensive_DBS_Technical_Report.pdf"
    print(f"Generating {pdf_name}...")
    
    with PdfPages(pdf_name) as pdf:
        
        # --- PAGE 1: Intro + FEA + OCD ---
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        
        # Header
        ax.text(0.5, 0.96, TITLE, ha='center', fontsize=14, weight='bold')
        ax.text(0.5, 0.93, SUBTITLE, ha='center', fontsize=10, style='italic')
        ax.text(0.5, 0.90, DATE, ha='center', fontsize=9)
        
        # Divider
        ax.plot([0, 1], [0.88, 0.88], 'k-', lw=1.5)
        
        items_p1 = [
            ('text', ABSTRACT),
            ('head', H1), ('text', T1), ('eq', E1), ('text', T1b), ('eq', E1b),
            ('head', H2), ('text', T2), ('eq', [E2a, E2b, E2c]), ('text', T2b)
        ]
        
        # Custom draw for layout
        y = 0.85
        for kind, content in items_p1:
            if kind == 'head':
                y -= 0.03
                ax.text(0, y, content, weight='bold', fontsize=11); y-=0.02
            elif kind == 'text':
                ax.text(0, y, content, fontsize=9, wrap=True, va='top'); 
                y -= (len(content)//100 + 1)*0.025 + 0.015
            elif kind == 'eq':
                eqs = content if isinstance(content, list) else [content]
                for e in eqs:
                    ax.text(0.5, y, e, ha='center', fontsize=11, color='darkblue'); y-=0.04
        
        pdf.savefig(); plt.close()
        
        # --- PAGE 2: Depression + Plot ---
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        
        items_p2 = [
            ('head', H3), ('text', T3), ('eq', [E3a, E3b]), 
            ('text', T3b), ('eq', E3c), ('text', T3c), ('eq', E3d)
        ]
        
        draw_content(ax, items_p2)
        
        # Plot
        ax_plot = fig.add_axes([0.2, 0.15, 0.6, 0.25])
        plot_depression_dynamics(ax_plot)
        
        pdf.savefig(); plt.close()
        
        # --- PAGE 3: ASD (Quantum) + Dementia ---
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        
        items_p3 = [
            ('head', H4), ('text', T4), ('eq', E4a), 
            ('text', T4b), ('eq', E4b), ('text', T4c), ('eq', E4c),
            ('head', H5), ('text', T5), ('eq', E5a), 
            ('text', T5b), ('eq', E5b), ('text', T5c), ('eq', E5c)
        ]
        
        # Manual layout for this dense page
        y = 0.95
        for kind, content in items_p3:
            if kind == 'head':
                y -= 0.04
                ax.text(0, y, content, weight='bold', fontsize=11); y-=0.02
            elif kind == 'text':
                ax.text(0, y, content, fontsize=9, wrap=True, va='top'); 
                y -= (len(content)//100 + 1)*0.025 + 0.01
            elif kind == 'eq':
                ax.text(0.5, y, content, ha='center', fontsize=10, color='darkblue'); y-=0.045

        # Plots at bottom
        ax_asd = fig.add_axes([0.1, 0.05, 0.35, 0.2])
        plot_asd_continued_fraction(ax_asd)
        
        ax_dem = fig.add_axes([0.55, 0.05, 0.35, 0.2])
        plot_dementia_decay(ax_dem)
        
        pdf.savefig(); plt.close()
        
    print("Done.")

if __name__ == "__main__":
    create_comprehensive_report()
