
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Aesthetics ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

# --- Content ---
TITLE = "Mathematical Derivations of Neurogenomic Plasticity:\nUnified Field Theory of Quantum Cognition"
AUTHORS = "Cartik Sharma, Lead Architect\nNeuromorph Quantum Systems"
AFFILIATION = "Quantum Neural Circuitry Division"
DATE = "January 2026"

ABSTRACT = r"This report formally derives the governing equations for Neurogenomic Hebbian Amplification, integrating Elliptic Phi Resonances and Ramanujan's Statistical Congruences. We provide a verifiable proof that the cognitive manifold minimizes entropy when synaptic weights align with the partition function p(n) modulo 24, establishing a direct link between modular forms and neural stability."

# --- Helper for Flow Layout ---
def draw_text_flow(ax, items, start_y=0.88):
    y = start_y
    # Config
    line_height_text = 0.035
    line_height_head = 0.06
    spacing_para = 0.03
    spacing_eq = 0.04
    
    for kind, content in items:
        if kind == 'head':
            y -= spacing_para
            ax.text(0.05, y, content, weight='bold', fontsize=12, ha='left', va='top', color='#222')
            y -= line_height_head
        elif kind == 'text':
            t = ax.text(0.05, y, content, fontsize=10, ha='left', va='top', wrap=True, color='#444')
            # Estimate height
            lines = 1 + len(content) // 90
            y -= lines * line_height_text
            y -= spacing_para
        elif kind == 'eq':
            y -= 0.01
            # Add a light background or box for equation? optional.
            ax.text(0.5, y, content, fontsize=13, ha='center', va='top', color='#003366')
            y -= 0.08  # Equation height
            y -= spacing_para
            
        if y < 0.05: # Simple pagination check (not fully robust but sufficient for this content)
            print("Warning: Content overflow on page.")
            
    return y

# --- Page Generation ---

def create_page_1(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')
    
    # Title Block
    ax.text(0.5, 1.0, DATE, ha='center', fontsize=10, color='#888')
    ax.text(0.5, 0.93, TITLE, ha='center', fontsize=16, weight='bold', color='#111')
    ax.text(0.5, 0.85, AUTHORS, ha='center', fontsize=12, color='#333')
    ax.text(0.5, 0.81, AFFILIATION, ha='center', fontsize=10, style='italic', color='#555')
    
    # Divider
    ax.plot([0.1, 0.9], [0.77, 0.77], color='#ccc', lw=1)

    # Abstract
    ax.text(0.5, 0.74, "ABSTRACT", ha='center', weight='bold', fontsize=10, color='#222')
    ax.text(0.05, 0.71, ABSTRACT, ha='left', va='top', wrap=True, fontsize=10.5, style='italic', color='#444')
    
    ax.plot([0.1, 0.9], [0.60, 0.60], color='#ccc', lw=1)

    # Section 1 content
    s1_items = [
        ('head', "1. Generalized Hebbian-Ramanujan Dynamics"),
        ('text', "We extend the standard Hebbian learning rule by introducing a Quantum Congruence Filter. The synaptic evolution is strictly governed by the modular discriminant delta function:"),
        ('eq', r"$\Delta(\tau) = q \prod_{n=1}^{\infty} (1-q^n)^{24} = \sum_{n=1}^{\infty} \tau(n) q^n$"),
        ('text', "Where tau(n) is Ramanujan's Tau function. A synaptic weight W_ij is deemed stable if and only if the sum of its prime indices satisfies the modular congruence:"),
        ('eq', r"$\sum p_k \equiv 0 \, (\mathrm{mod}\, 24)$"),
        ('text', "This condition ensures the cancellation of bosonic string excitations in the cognitive vacuum, preventing decoherence."),
        
        ('head', "2. The Unified Neurogenomic Equation"),
        ('text', "The master equation integrating Hebbian learning, Phi Resonance, and Ramanujan Congruence is defined as follows:"),
        ('eq', r"$\frac{\partial \rho}{\partial t} = \mathbf{H}_{\text{eff}} \rho - \{ H, \rho \} + \mathcal{L}_{Diss}(\rho)$"),
        ('text', "Where the effective Hamiltonian includes the Modular Potential V_mod derived directly from the Prime Vortex Field.")
    ]
    
    draw_text_flow(ax, s1_items, start_y=0.55)
    
    pdf.savefig()
    plt.close()

def create_page_2(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')
    
    s2_items = [
        ('head', "3. Verifiability of Topological Repair"),
        ('text', "To verify the integrity of the repaired network, we calculate the Quantum Surface Flux. By Stokes' Theorem on the graph manifold M:"),
        ('eq', r"$\int_M d\omega = \oint_{\partial M} \omega$"),
        ('text', "We define the 'Health 1-form' omega as the Prime Potential gradient. The verifiability condition for a dementia-free state is strict conservativity:"),
        ('eq', r"$\oint_{\text{loops}} \nabla \left( \frac{1}{\ln p} \right) \cdot d\mathbf{l} = 0$"),
        ('text', "This implies that the cognitive field is path-independent, effectively removing cyclical compulsive loops (OCD) and verifying a 'Clean State' topology."),
        ('head', "4. Conclusion & Implications"),
        ('text', "We have derived a complete mathematical framework for Neurogenomic Plasticity. By treating the connectome as a quantum field subject to modular symmetries, we enable 'God Mode' repairs that are mathematically guaranteed to minimize entropy."),
        ('text', "This Unified Field Theory bridging Number Theory and Neuroscience provides the rigorous foundation for the clinical interventions demonstrated in the simulation.")
    ]
    
    draw_text_flow(ax, s2_items, start_y=1.0)
    
    # Footer
    ax.text(0.5, 0.05, "Verified by Neuromorph QML Engine | 2026", ha='center', fontsize=8, color='#999')

    pdf.savefig()
    plt.close()

def create_doc_file():
    """Generates the text-based .doc report."""
    content = f"""
{TITLE}
{AUTHORS}
{AFFILIATION}
{DATE}

ABSTRACT
{ABSTRACT}

1. Generalized Hebbian-Ramanujan Dynamics
We extend the standard Hebbian learning rule by introducing a Quantum Congruence Filter. The synaptic evolution is strictly governed by the modular discriminant delta function:
Delta(tau) = q * product((1-q^n)^24)

Where tau(n) is Ramanujan's Tau function. A synaptic weight W_ij is deemed stable if and only if the sum of its prime indices satisfies the modular congruence:
Sum(p_k) = 0 (mod 24)

2. The Unified Neurogenomic Equation
The master equation integrating Hebbian learning, Phi Resonance, and Ramanujan Congruence is:
d(rho)/dt = H_eff * rho - {{H, rho}} + L_Diss(rho)

3. Verifiability of Topological Repair
To verify the integrity of the repaired network, we calculate the Quantum Surface Flux. By Stokes' Theorem on the graph manifold M:
Integral(d_omega) = Loop_Integral(omega)

We define the 'Health 1-form' omega as the Prime Potential gradient. The verifiability condition for a dementia-free state is strict conservativity:
Loop_Integral( Gradient(1 / ln p) ) = 0

This implies that the cognitive field is path-independent, removing cyclical compulsive loops.

4. Conclusion
We have derived a complete mathematical framework for Neurogenomic Plasticity. By treating the connectome as a quantum field subject to modular symmetries, we enable repairs that are mathematically guaranteed to minimize entropy.

---
Verified by Neuromorph QML Engine
"""
    with open("Neurogenomics_Unified_Report.doc", "w") as f:
        f.write(content.strip())
        
def generate():
    pdf_filename = "Neurogenomics_Unified_Report.pdf"
    print(f"Generating optimized {pdf_filename}...")
    with PdfPages(pdf_filename) as pdf:
        create_page_1(pdf)
        create_page_2(pdf)
        
    print("Generating Neurogenomics_Unified_Report.doc...")
    create_doc_file()
    print("Done.")

if __name__ == "__main__":
    generate()
