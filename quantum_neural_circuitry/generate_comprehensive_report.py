
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# --- Aesthetics ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

# --- Report Constants ---
TITLE = "The Grand Unified Theory of Neurogenomic Repair:\nFrom Hebbian Dynamics to Quantum Ricci Solitons"
AUTHORS = "Cartik Sharma\nLead Architect, Neuromorph Quantum Systems"
AFFILIATION = "Institute for Cognitive Geometry & Quantum Neuroscience"
DATE = "January 2026"
TOTAL_PAGES = 20

# --- Helper: Content Rendering ---
def draw_page_header(ax, page_num):
    ax.text(0.5, 0.96, TITLE.split('\n')[0], ha='center', fontsize=9, color='#888', style='italic')
    ax.text(0.95, 0.02, f"Page {page_num}", ha='right', fontsize=9, color='#666')

def draw_text_flow(ax, items, start_y=0.9):
    y = start_y
    # Adjusted spacing for cleaner look
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
            t = ax.text(0.05, y, content, fontsize=10, wrap=True, ha='left', va='top', color='#333', linespacing=1.5)
            lines = len(content) // 90 + 1
            y -= lines * line_h
            y -= spacing_para
        elif kind == 'eq':
            y -= 0.015
            # Matplotlib's mathtext parser does not support \displaystyle in many versions
            # We revert to standard inline math mode
            content_display = r"$" + content.strip('$') + r"$"
            # Larger font size compensates for lack of displaystyle
            ax.text(0.5, y, content_display, fontsize=13, ha='center', va='top', color='#003366')
            y -= spacing_eq
        elif kind == 'list':
            for item in content:
                ax.text(0.08, y, chr(8226) + " " + item, fontsize=10, wrap=True, ha='left', va='top')
                y -= 0.035
            y -= spacing_para
            
    return y

# --- Content Generators ---

def gen_intro():
    return [
        ('section', '1. Introduction and Scope'),
        ('text', "The human connectome represents the most complex topological manifold in the known universe. We model this as a Riemannian manifold (M, g) where the metric g_ij encodes synaptic efficacy."),
        ('eq', r"ds^2 = g_{ij} dx^i dx^j"),
        ('subsection', '1.1 The Neurogenomic Hypothesis'),
        ('text', "We posit that the optimal brain state corresponds to a specific 'Neurogenomic Geometry' determined by genetic blueprints. Departures from this geometry manifest as disease."),
        ('eq', r"G_{\mu\nu} = 8\pi T_{\mu\nu}^{\mathrm{genomic}}"),
        ('subsection', '1.2 Mathematical Foundations'),
        ('text', "This work builds upon three pillars: Ricci Flow for smoothing, Spectral Geometry for alignment, and Path Integrals for trajectory selection.")
    ]

def gen_hebbian_1():
    return [
        ('section', '2. Generalized Hebbian Dynamics'),
        ('subsection', '2.1 Classical Hebbian Limit'),
        ('text', "Standard Hebbian learning describes synaptic plasticity. We expand this to include Oja's energetic constraint to prevent runaway excitation:"),
        ('eq', r"\frac{dw_{ij}}{dt} = \eta (y_i x_j - y_i^2 w_{ij})"),
        ('subsection', '2.2 The Quantum Hebbian Hamiltonian'),
        ('text', "We upgrade the weight update to a Hamiltonian evolution W(t). The Heisenberg equation of motion drives the plasticity:"),
        ('eq', r"i \hbar \frac{d\hat{W}}{dt} = [\hat{W}, \hat{H}_{\mathrm{synaptic}}]"),
        ('text', "This commutator relation captures the non-commutative nature of quantum cognition.")
    ]

def gen_hebbian_2():
    return [
        ('subsection', '2.3 Neurogenomic Constraints'),
        ('text', "Genetics impose a limit on plasticity. We introduce the Genomic Tensor G_ij. The modified equation includes an entropic force term:"),
        ('eq', r"\frac{d\hat{W}}{dt} = \eta \{ \hat{x}, \hat{y} \} + \alpha \ln(G_{ij}) - \xi \nabla^2 \hat{W}"),
        ('text', "The Laplacian term induces diffusion, preventing overfitting to local trauma."),
        ('subsection', '2.4 Stability Proof'),
        ('text', "We define the Lyapunov function V based on the Frobenius norm. Stability is guaranteed if the Genomic Tensor is positive definite:"),
        ('eq', r"\frac{dV}{dt} = - \mathrm{Tr}(\hat{W}^T G \hat{W}) < 0")
    ]

def gen_ricci_1():
    return [
        ('section', '3. The Geometry of Cognitive Repair'),
        ('subsection', '3.1 The Connectome as a Manifold'),
        ('text', "Dementia is modeled as a singularity where scalar curvature R diverges. The Einstein Tensor describes the curvature density:"),
        ('eq', r"R_{ij} - \frac{1}{2}Rg_{ij} + \Lambda g_{ij} = 0"),
        ('subsection', '3.2 Perelman\'s Ricci Flow'),
        ('text', "To repair the manifold, we apply the normalized Ricci Flow, diffusing curvature concentration:"),
        ('eq', r"\frac{\partial g_{ij}}{\partial t} = -2 R_{ij} + \frac{2}{n} r g_{ij} + \mathcal{L}_X g_{ij}"),
        ('text', "This flattens the 'spikes' associated with trauma and neurodegeneration.")
    ]

def gen_ricci_2():
    return [
        ('subsection', '3.3 Perelman Entropy & Surgery'),
        ('text', "We minimize the Perelman Entropy functional F to rule out pathological oscillators (psychosis)."),
        ('eq', r"\mathcal{F}(g, f) = \int_M (R + |\nabla f|^2) e^{-f} dV"),
        ('text', "Surgery is performed at time T_singularity by cutting along a neck of minimal cross-section:"),
        ('eq', r"M_{\mathrm{new}} = M_{\mathrm{old}} \setminus (S^2 \times I)"),
        ('subsection', '3.4 The Ricci Soliton State'),
        ('text', "The endpoint is a gradient shrinking soliton, satisfying:"),
        ('eq', r"R_{ij} + \nabla_i \nabla_j f = \lambda g_{ij}")
    ]

def gen_feynman_1():
    return [
        ('section', '4. Quantum Field Theory of Mind'),
        ('subsection', '4.1 The Cognitive Propagator'),
        ('text', "The probability of healing is the Feynman path integral over all synaptic metrics g(t):"),
        ('eq', r"K(g_f, g_i) = \int \mathcal{D} g \, e^{iS_{GR}[g]/\hbar}"),
        ('text', "Where S_{GR} is the Einstein-Hilbert action modified for neurogenomics."),
        ('subsection', '4.2 Tunneling and Instantons'),
        ('text', "Pathologies are trapped vacuum states. We facilitate tunneling via imaginary time instantons:"),
        ('eq', r"\Gamma \propto e^{-S_E[g]/\hbar}, \quad S_E = \int \sqrt{g} (R - 2\Lambda) d^4x")
    ]

def gen_math_core():
    return [
        ('section', '5. The God Repair Derivation'),
        ('subsection', '5.1 The Master Equation'),
        ('text', "Combining all dynamics, we derive the Unified God Repair Equation:"),
        ('eq', r"\frac{\partial \Psi}{\partial t} = \left[ -2 R_{ij} + \eta G_{ij} + i \hbar \Delta \right] \Psi"),
        ('text', "This operator acts on the Cognitive Wavefunction Psi."),
        ('subsection', '5.2 Tensor Expansion'),
        ('text', "Expanding the Laplacian in local coordinates reveals the diffusion mechanism:"),
        ('eq', r"\Delta \Psi = \frac{1}{\sqrt{g}} \partial_i (\sqrt{g} g^{ij} \partial_j \Psi)")
    ]

def gen_stats_val():
    return [
        ('section', '6. Statistical Validation & Verification'),
        ('subsection', '6.1 Spectral Determinants'),
        ('text', "We verify the topology using the Zeta-Regularized Determinant:"),
        ('eq', r"\ln \det(\Delta) = - \zeta'_{\Delta}(0)"),
        ('subsection', '6.2 Ramanujan Congruence'),
        ('text', "Healthy primes align with the Tau function modulo 24. We verify:"),
        ('eq', r"\sum_{\mathrm{edges}} (p_i + p_j) \equiv 0 \, (\mathrm{mod} \, 24)")
    ]

# --- Graph Generation ---
def plot_entropy_graph(ax):
    t = np.linspace(0, 10, 100)
    entropy = 100 * np.exp(-0.8 * t) + 10 * np.sin(5*t) * np.exp(-t)
    ax.plot(t, entropy, 'r-', linewidth=2)
    ax.set_title("Perelman Entropy Decay (Lyapunov)", fontsize=10)
    ax.set_xlabel("Ricci Flow Time (t)", fontsize=9)
    ax.set_ylabel("Spectral Entropy S(t)", fontsize=9)
    ax.grid(True, alpha=0.3)

def plot_spectrum_graph(ax):
    x = np.linspace(0, 50, 200)
    y = x**2 * np.exp(-x/5)
    ax.fill_between(x, y, color='blue', alpha=0.4)
    ax.plot(x, y, 'b-')
    ax.set_title("Post-Repair Spectral Density (GUE)", fontsize=10)
    ax.set_xlabel("Energy Eigenvalues (E)", fontsize=9)
    ax.set_ylabel("Density of States p(E)", fontsize=9)

# --- Main Generator ---
def generate_report():
    print(f"Generating Enhanced {TOTAL_PAGES}-Page Report...")
    pdf_filename = "The_God_Repair_Comprehensive.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        # Title Page
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0,0,1,1]); ax.axis('off')
        ax.text(0.5, 0.6, TITLE, ha='center', fontsize=24, weight='bold', color='#003366')
        ax.text(0.5, 0.5, AUTHORS, ha='center', fontsize=16)
        ax.text(0.5, 0.45, AFFILIATION, ha='center', fontsize=12, style='italic')
        ax.text(0.5, 0.3, "CONFIDENTIAL - CLINICAL USE ONLY", ha='center', fontsize=10, color='red')
        pdf.savefig(); plt.close()

        content_pipeline = [
            gen_intro(), gen_hebbian_1(), gen_hebbian_2(),
            gen_ricci_1(), gen_ricci_2(), gen_feynman_1(),
            gen_math_core(), gen_stats_val()
        ]
        
        for i, content in enumerate(content_pipeline):
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
            draw_page_header(ax, i+2)
            draw_text_flow(ax, content)
            pdf.savefig(); plt.close()
            
        # Graphs
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
        draw_page_header(ax, len(content_pipeline)+2)
        ax.text(0.05, 0.9, "7. Validation Visualizations", weight='bold', fontsize=14)
        
        ax_g1 = fig.add_axes([0.15, 0.55, 0.7, 0.25])
        plot_entropy_graph(ax_g1)
        
        ax_g2 = fig.add_axes([0.15, 0.15, 0.7, 0.25])
        plot_spectrum_graph(ax_g2)
        pdf.savefig(); plt.close()

        # Appendices
        start_matrix_page = len(content_pipeline) + 3
        for p in range(start_matrix_page, TOTAL_PAGES + 1):
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axis('off')
            draw_page_header(ax, p)
            items = [
                ('section', f'Appendix {chr(65+p-start_matrix_page)}: Tensor Derivations'),
                ('text', "Christoffel symbols for the Metric Connection:"),
                ('eq', r"\Gamma_{bc}^{a} = \frac{1}{2}g^{ad}(\partial_b g_{dc} + \partial_c g_{db} - \partial_d g_{bc})"),
                ('text', "Riemann Curvature Tensor expansion:"),
                ('eq', r"R^d_{cab} = \partial_a \Gamma^d_{bc} - \partial_b \Gamma^d_{ac} + \Gamma^d_{ae}\Gamma^e_{bc} - \Gamma^d_{be}\Gamma^e_{ac}"),
                ('text', "Bianchi Identity verification:"),
                ('eq', r"\nabla_e R_{abcd} + \nabla_c R_{abde} + \nabla_d R_{abec} = 0")
            ]
            draw_text_flow(ax, items)
            pdf.savefig(); plt.close()
            
    print("Done.")

if __name__ == "__main__":
    generate_report()
