
"""
Generates a Finite Math Report for the Deep Brain Stimulation App.
Includes FEA, CSTC Loop Dynamics, Quantum Surface Integrals, and Clinical Trial Stats.
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
TITLE = "Finite Mathematical Framework for\nDeep Brain Stimulation & OCD Treatment"
AUTHORS = "Scientific & Medical Advisory Board\nDeep Brain Stimulation System"
DATE = "January 2026"

ABSTRACT = r"This report details the underlying finite mathematical models governing the Deep Brain Stimulation (DBS) application. We present the Poisson formulation for Finite Element Analysis (FEA) of electric fields, the coupled differential equations modeling the Cortico-Striato-Thalamo-Cortical (CSTC) loop dynamics in OCD, the novel Quantum Surface Integral optimization for attractor state transitions, and the statistical mechanics underpinning clinical trial simulations."

# Section 1: FEA
H1 = "1. Finite Element Analysis (FEA) of Electric Fields"
T1 = "The electric potential Phi within the brain tissue is governed by the quasi-static Poisson equation for volume conduction:"
E1 = r"$\nabla \cdot (\sigma(x) \nabla \Phi(x)) = -\nabla \cdot J_{source}$"
T1b = "where sigma(x) is the conductivity tensor of the tissue (Gray Matter, White Matter, CSF). The Volume of Tissue Activated (VTA) is defined by the set of points where the second spatial difference of the potential exceeds the axonal activation threshold:"
E1b = r"$VTA = \{ x \in \Omega \mid \Delta^2 \Phi(x) > \theta_{activation} \}$"

# Section 2: Neural Dynamics (CSTC Loop)
H2 = "2. CSTC Loop Dynamics & OCD Attractors"
T2 = "We model the OCD pathology as a hyperactive gain cycle in the CSTC loop. The firing rates of the Orbitofrontal Cortex (OFC), Caudate (C), and Thalamus (T) are governed by coupled nonlinear differential equations:"
E2 = r"$\tau \frac{dC}{dt} = -C + S(w_{OFC \to C} \cdot OFC + I_{DBS})$"
E2b = r"$\tau \frac{dT}{dt} = -T + S(w_{C \to T} \cdot C - w_{GPi \to T} \cdot GPi)$"
E2c = r"$\tau \frac{dOFC}{dt} = -OFC + S(w_{T \to OFC} \cdot T + I_{ext})$"
T2b = "where S(x) is the sigmoid activation function. In OCD, the loop gain Gamma > 1, creating a stable high-energy 'obsession' attractor."

# Section 3: Quantum Surface Integrals
H3 = "3. Quantum Surface Integral Optimization"
T3 = "To optimize treatment, we calculate the probability of tunneling out of the pathological attractor. We define the 'Quantum Surface' S as the energy manifold of the obsession state. The transition probability P is derived from the E-field flux integrated over this surface:"
E3 = r"$H_{delivered} = \oint_{S} \left( \frac{1}{2}\epsilon |\mathbf{E}|^2 \right) \cdot d\mathbf{A}$"
T3b = "Using the WKB approximation for tunneling, the transition probability P is:"
E3b = r"$P_{transition} \approx \exp\left( - \frac{V_{barrier} - H_{delivered}}{k_B T_{eff}} \right)$"
T3c = "The optimizer maximizes P while minimizing total energy deposition (safety)."

# Section 4: Statistical Clinical Trials
H4 = "4. Statistical Mechanics of Clinical Trials"
T4 = "We simulate clinical trials using a population distribution model. The pre- and post-treatment YBOCS scores are modeled as distributions shifted by the treatment efficacy E:"
E4 = r"$P_{post}(s) = \int P_{pre}(s') \cdot \mathcal{N}(s' - E(s'), \sigma_{var}) \, ds'$"
T4b = "Significance is determined via a paired t-test on the finite sample N:"
E4b = r"$t = \frac{\bar{X}_{diff}}{s_{diff} / \sqrt{N}}, \quad p = 2(1 - F_{t_{N-1}}(|t|))$"

# --- Plotting Functions ---

def plot_attractor_landscape(ax):
    """Visualizes the attractor landscape."""
    x = np.linspace(-2, 2, 100)
    # Double well potential
    y = x**4 - 2*x**2 
    ax.plot(x, y, 'r-', linewidth=2, label='Energy Landscape')
    ax.plot(1, -1, 'bo', markersize=10, label='Obsession State')
    ax.plot(-1, -1, 'go', markersize=10, label='Healthy State')
    
    # Arrow for DBS
    ax.arrow(0.8, -0.5, -1.0, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    ax.text(0, 0.5, "DBS Energy\nInjection", ha='center', color='blue')
    
    ax.set_title("Tunneling out of OCD Attractor", fontsize=10)
    ax.set_xlabel("Neural State")
    ax.set_ylabel("Potential V(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_fea_mesh(ax):
    """Visualizes a finite element mesh."""
    # Simplified grid
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    ax.plot(x, y, 'k-', lw=0.5, alpha=0.5)
    ax.plot(y, x, 'k-', lw=0.5, alpha=0.5)
    
    # E-field heat map
    z = np.exp(-((x-0.5)**2 + (y-0.5)**2)/0.1)
    c = ax.contourf(x, y, z, cmap='viridis', alpha=0.8)
    # fig.colorbar(c, ax=ax)
    
    ax.set_title("FEA Mesh & E-Field", fontsize=10)
    ax.axis('off')

# --- Layout Function ---
def draw_content(ax, items):
    y = 0.88
    # Header
    ax.text(0.5, 0.96, TITLE, ha='center', fontsize=12, weight='bold')
    ax.text(0.5, 0.92, AUTHORS, ha='center', fontsize=9, style='italic')
    
    line_h_text = 0.032
    spacing_para = 0.025
    spacing_eq = 0.07
    
    for kind, text_content in items:
        if kind == 'head':
            y -= 0.04
            ax.text(0.05, y, text_content, weight='bold', fontsize=11, color='#222')
            y -= 0.02
        elif kind == 'text':
            ax.text(0.05, y, text_content, fontsize=9.5, wrap=True, ha='left', va='top', color='#333')
            num_lines = len(text_content) // 90 + 1
            y -= num_lines * line_h_text
            y -= spacing_para
        elif kind == 'eq':
            y -= 0.015
            ax.text(0.5, y, text_content, fontsize=12, ha='center', va='top', color='#003366')
            y -= spacing_eq
        elif kind == 'list':
            y -= 0.01
            for item in text_content:
                ax.text(0.08, y, "- " + item, fontsize=9.5, wrap=True, color='#444')
                y -= 0.035
            y -= spacing_para
            
        if y < 0.05:
            # Simple overflow handling
            pass

def create_report():
    pdf_name = "DBS_Finite_Math_Report.pdf"
    print(f"Generating optimized {pdf_name}...")
    
    with PdfPages(pdf_name) as pdf:
        # Page 1: Abstract + FEA + Dynamics
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        
        items_p1 = [
            ('text', ABSTRACT),
            ('head', H1), ('text', T1), ('eq', E1), ('text', T1b), ('eq', E1b),
            ('head', H2), ('text', T2), ('eq', E2), ('eq', E2b), ('eq', E2c), ('text', T2b)
        ]
        draw_content(ax, items_p1)
        
        # Add small FEA plot
        ax_plot = fig.add_axes([0.65, 0.65, 0.25, 0.15])
        plot_fea_mesh(ax_plot)
        
        pdf.savefig(); plt.close()
        
        # Page 2: Quantum + Stats
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        
        items_p2 = [
            ('head', H3), ('text', T3), ('eq', E3), ('text', T3b), ('eq', E3b), ('text', T3c),
            ('head', H4), ('text', T4), ('eq', E4), ('text', T4b), ('eq', E4b)
        ]
        draw_content(ax, items_p2)
        
        # Add Attractor plot
        ax_plot2 = fig.add_axes([0.2, 0.45, 0.6, 0.2])
        plot_attractor_landscape(ax_plot2)
        
        pdf.savefig(); plt.close()
        
    print("Done.")

if __name__ == "__main__":
    create_report()
