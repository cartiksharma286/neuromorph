
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# --- Aesthetics ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

# --- Report Constants ---
TITLE = "Finite Mathematical Foundations of Neurogenomics:\nDiscrete Topology and Number-Theoretic Optimizations"
AUTHORS = "Cartik Sharma\nLead Architect, Neuromorph Quantum Systems"
AFFILIATION = "Institute for Cognitive Geometry & Quantum Neuroscience"
DATE = "January 2026"

# --- Content Generators ---

def gen_intro():
    return [
        ('section', '1. Introduction to Finite Neurogenomics'),
        ('text', "While the Ricci Flow treats the brain as a continuous manifold, the biological reality is discrete: finite neurons, finite synapses, and quantized vesicles. This report derives the governing equations of 'God Repair' from the perspective of Finite Mathematics and Discrete Topology."),
        ('subsection', '1.1 The Connectome as a Finite Graph'),
        ('text', "We define the Neural Graph G = (V, E) where |V| ~ 86 billion. The state of the system is a vector in a finite field space F_p^N, not a Hilbert Hilbert space. This discretization allows for exact combinatorial optimization."),
        ('eq', r"State \in \mathbb{F}_p^N, \quad N = |V|")
    ]

def gen_prime_density():
    return [
        ('section', '2. Prime Number Derivations'),
        ('subsection', '2.1 Synaptic Density and the Prime Number Theorem'),
        ('text', "The optimal distribution of connections in a 'God Repair' network follows the Prime Number Theorem. The information density I(x) at cortical depth x must scale inversely with the prime density:"),
        ('eq', r"I(x) \sim \pi(x) \approx \frac{x}{\ln x}"),
        ('text', "This implies that the 'energy cost' of a memory of size x is exactly ln(x), minimizing metabolic overhead."),
        ('subsection', '2.2 Prime Gap Statistics'),
        ('text', "We derive the probability P(g) of a 'synaptic gap' g occurring between connected nodes. For a stabilized network, this must follow the Cramer-Granville heuristic:"),
        ('eq', r"P(g > \epsilon) \approx e^{-\epsilon}")
    ]

def gen_modular_math():
    return [
        ('section', '3. Modular Arithmetic and Resonances'),
        ('subsection', '3.1 The Modulo 24 Congruence'),
        ('text', "Why Modulo 24? In string theory (and by extension, high-dimensional neural topology), the number of transverse directions is 24. We derive the condition for 'Bosonic Calmness' (zero vacuum energy) using Ramanujan's tau function sum:"),
        ('eq', r"\sum_{n=1}^{\infty} \frac{\tau(n)}{n^s} = \prod_{p} (1 - \tau(p)p^{-s} + p^{11-2s})^{-1}"),
        ('text', "For a neural circuit to be stable (non-decaying), the sum of prime indices p_i and p_j of any active edge must satisfy:"),
        ('eq', r"p_i + p_j \equiv 0 \, (\mathrm{mod} \, 24)"),
        ('text', "This works because 24 is the only integer n such that sum(k^2) = n^2 implies the Leech Lattice packing efficiency.")
    ]

def gen_combinatorics():
    return [
        ('section', '4. Combinatorial Topology'),
        ('subsection', '4.1 Euler Characteristic of the Healed Mind'),
        ('text', "We model the 'hole' caused by Dementia as a topological puncture. The Euler characteristic X is V - E + F. A damaged brain has X < 1 (high genus, many holes)."),
        ('text', "The 'God Repair' algorithm maximizes X by adding prime-weighted edges until the manifold is simply connected (sphere-like):"),
        ('eq', r"\chi_{\mathrm{repaired}} = 2 \quad (\text{Sphere Topology})"),
        ('subsection', '4.2 The Hamiltonian Path Problem'),
        ('text', "The problem of retrieving a memory is equivalent to finding a Hamiltonian Path. By restructuring the graph G using the G_ij tensor, we transform this NP-hard problem into a P-time traversal on a Prime-Weighted graph.")
    ]

def gen_discrete_hamiltonian():
    return [
        ('section', '5. The Finite God Repair Operator'),
        ('text', "Combining discrete prime mechanics and modular arithmetic, the update rule for a single synapse w_ij in discrete time t is:"),
        ('eq', r"w_{ij}^{(t+1)} = w_{ij}^{(t)} + \delta_{p_i+p_j, 24k} \cdot \frac{\phi}{\ln(p_i p_j)}"),
        ('text', "Where delta is the Kronecker delta (1 if sum is multiple of 24, 0 otherwise) and Phi is the Golden Ratio. This finite difference equation generates the exact 'Golden Network' required for optimal cognition.")
    ]

# --- Helper: Content Rendering (Reused) ---
def draw_page_header(ax, page_num):
    ax.text(0.5, 0.96, TITLE.split('\n')[0], ha='center', fontsize=9, color='#888', style='italic')
    ax.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=9, color='#666')

def draw_text_flow(ax, items, start_y=0.9):
    y = start_y
    line_h = 0.035 
    spacing_para = 0.03
    spacing_eq = 0.08
    
    for kind, content in items:
        if kind == 'section':
            y -= 0.06
            ax.text(0.05, y, content, weight='bold', fontsize=14, color='#111')
            ax.plot([0.05, 0.95], [y-0.015, y-0.015], color='#111', lw=1)
            y -= 0.05
        elif kind == 'subsection':
            y -= 0.04
            ax.text(0.05, y, content, weight='bold', fontsize=11, color='#333')
            y -= 0.03
        elif kind == 'text':
            t = ax.text(0.05, y, content, fontsize=10, wrap=True, ha='left', va='top', color='#333')
            lines = len(content) // 90 + 1
            y -= lines * line_h
            y -= spacing_para
        elif kind == 'eq':
            y -= 0.015
            content_display = r"$" + content.strip('$') + r"$"
            # Using slightly smaller font for equations to fit finite math notation
            ax.text(0.5, y, content_display, fontsize=12, ha='center', va='top', color='#003366')
            y -= spacing_eq
            
    return y

def generate_report():
    print("Generating Finite_Math_Neurogenomics_Report.pdf...")
    pdf_filename = "Finite_Math_Neurogenomics_Report.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        # Title Page
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0,0,1,1]); ax.axis('off')
        ax.text(0.5, 0.6, TITLE, ha='center', fontsize=20, weight='bold', color='#222')
        ax.text(0.5, 0.5, AUTHORS, ha='center', fontsize=14)
        ax.text(0.5, 0.45, AFFILIATION, ha='center', fontsize=12, style='italic')
        pdf.savefig(); plt.close()

        # Page 1: Intro + Primes
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        draw_page_header(ax, 1)
        items1 = gen_intro() + gen_prime_density()
        draw_text_flow(ax, items1)
        pdf.savefig(); plt.close()
        
        # Page 2: Modular + Combinatorics
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        draw_page_header(ax, 2)
        items2 = gen_modular_math() + gen_combinatorics()
        draw_text_flow(ax, items2)
        pdf.savefig(); plt.close()

        # Page 3: Hamiltonian
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        draw_page_header(ax, 3)
        items3 = gen_discrete_hamiltonian()
        draw_text_flow(ax, items3)
        pdf.savefig(); plt.close()

    print("Done.")

if __name__ == "__main__":
    generate_report()
