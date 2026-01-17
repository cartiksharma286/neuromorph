
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# --- Aesthetics ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

# --- Content Definitions ---
TITLE = "The Feynman-Neurogenomic Propagator:\nQuantum Field Theory of Cognitive Repair"
AUTHORS = "Cartik Sharma, Chief Scientist\nNeuromorph Quantum Systems"
AFFILIATION = "Department of Cognitive Physics"
DATE = "January 2026"

ABSTRACT = r"We derive the 'Cognitive Propagator' using Feynman Path Integrals, treating the connectome as a quantum field evolving under a Neurogenomic Action S. We demonstrate that psychiatric disorders correspond to trapped vacuum states, and that the 'God Repair' protocol acts as a topological instanton, enabling tunnel transitions to the optimized state."

# Section 1: Action Principle
H1 = "1. The Cognitive Action Principle"
T1 = "We define the state of the mind as a wavefunction Psi evolving on the neural manifold. The dynamics are governed by the Principle of Least Action, where action S is:"
E1 = r"$S[\Psi] = \int dt \left( \frac{1}{2} |\dot{\Psi}|^2 - V_{eff}(\Psi) \right)$"
T1b = "The effective potential V_eff includes the Neurogenomic constraint G_ij. The brain naturally seeks to minimize this Action."

# Section 2: Feynman Path Integral
H2 = "2. The Feynman Path Integral Formulation"
T2 = "The probability amplitude for the brain to transition from a Demented state (A) to a Healed state (B) is the sum over all possible synaptic rewiring histories:"
E2 = r"$K(B, A) = \int_{A}^{B} \mathcal{D}\Psi(t) \, \exp\left( \frac{i}{\hbar} S[\Psi] \right)$"
T2b = "Most random paths cancel out due to phase interference. The 'Classical Path' (healthy state) is the stationary phase solution delta S = 0."

# Section 3: Instantons and Tunneling
H3 = "3. Quantum Tunneling out of Pathology"
T3 = "Disorders like Depression represent deep local minima. We introduce a 'Neurogenomic Instanton'—a solution in imaginary time (tau = it):"
E3 = r"$\Gamma \propto \exp\left( - \frac{1}{\hbar} \int d\tau \, \sqrt{2m(V - E)} \right)$"
T3b = "By modulating genomic weights, we lower the barrier, exponentially increasing the tunneling rate Gamma to the healthy ground state."

# Section 4: The God Repair Operator
H4 = "4. The God Repair Operator"
T4 = "Combining Perelman's Ricci Flow with the Feynman Propagator, we derive the Operator for Total Reconstruction:"
E4 = r"$\hat{O}_{God} = T \left\{ \exp\left( - \int_0^t dt' \, \hat{H}_{Gen}(t') \right) \right\}$"
T4b = "Where T is Time-Ordering. This operator restores information from its genetic blueprint, effectively reversing cognitive entropy."

# --- Layout Function ---
def draw_content(ax, items):
    y = 0.88
    # Header
    ax.text(0.5, 0.96, TITLE, ha='center', fontsize=12, weight='bold') # Reduced font
    ax.text(0.5, 0.92, AUTHORS, ha='center', fontsize=9, style='italic') # Reduced font
    
    # Config
    line_h_text = 0.03
    spacing_para = 0.02
    spacing_eq = 0.06
    
    for kind, text in items:
        if kind == 'head':
            y -= 0.04
            ax.text(0.05, y, text, weight='bold', fontsize=11, color='#222')
            y -= 0.02
        elif kind == 'text':
            t = ax.text(0.05, y, text, fontsize=9, wrap=True, ha='left', va='top', color='#333')
            num_lines = len(text) // 100 + 1
            y -= num_lines * line_h_text
            y -= spacing_para
        elif kind == 'eq':
            y -= 0.01
            ax.text(0.5, y, text, fontsize=12, ha='center', va='top', color='#003366')
            y -= spacing_eq
            
        if y < 0.05:
            print("Page overflow warning")

def create_report():
    pdf_name = "Neurogenomics_Feynman_Report.pdf"
    print(f"Generating optimized {pdf_name}...")
    
    with PdfPages(pdf_name) as pdf:
        # Page 1
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        
        items_p1 = [
            ('text', ABSTRACT),
            ('head', H1), ('text', T1), ('eq', E1), ('text', T1b),
            ('head', H2), ('text', T2), ('eq', E2), ('text', T2b)
        ]
        draw_content(ax, items_p1)
        pdf.savefig(); plt.close()
        
        # Page 2
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        
        items_p2 = [
            ('head', H3), ('text', T3), ('eq', E3), ('text', T3b),
            ('head', H4), ('text', T4), ('eq', E4), ('text', T4b),
            ('head', "5. Conclusion"),
            ('text', "The Feynman-Neurogenomic framework provides a rigorous basis for 'God Repair'. We prove that optimization is inevitable under correct Genomic constraints.")
        ]
        draw_content(ax, items_p2)
        pdf.savefig(); plt.close()
        
    print("Done.")

if __name__ == "__main__":
    create_report()
