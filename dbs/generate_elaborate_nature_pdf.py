import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = "Nature_Report_Elaborate_QML_DBS.pdf"

with PdfPages(pdf_path) as pdf:
    # Page 1: Title and Abstract
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.5, 0.95, "Elaborate Quantum Machine Learning Frameworks", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.91, "and Adaptive Feynman Path Integrals for", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.86, "Demyelinating Disease Mitigation in DBS", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.80, "A Theoretical Nature Preprint", horizontalalignment='center', fontsize=14, family='serif', style='italic')
    
    plt.text(0.1, 0.70, "Abstract:", fontsize=12, fontweight='bold', family='serif')
    abstract_text = (
        "We introduce a rigorously elaborated computational framework combining Quantum\n"
        "Machine Learning (QML) and finite mathematical models to calculate and maximize\n"
        "neural recovery in Multiple Sclerosis (MS), alongside an adaptive Deep Brain\n"
        "Stimulation (DBS) target protocol designed to ablate Rosenthal fibers in\n"
        "Alexander's Disease. Employing higher-order finite difference formulations on\n"
        "astrocytic and myelin manifolds, we optimize plaque dissipation via a dynamically\n"
        "tunable cost function. Adaptive ablation is mathematically bounded by discretized\n"
        "Feynman path integrals mapping thermal-electric energy trajectories, avoiding\n"
        "collateral necrosis. This elaborate formulation bridges quantum neural networks\n"
        "with clinical neuro-electrodynamics."
    )
    plt.text(0.1, 0.50, abstract_text, fontsize=11, family='serif', linespacing=1.5)
    pdf.savefig()
    plt.close()

    # Page 2: MS Plaque Mitigation - Finite Math Elaborated
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.1, 0.95, "1. Elaborate Finite Mathematics for QML MS Plaque Mitigation", fontsize=14, fontweight='bold', family='serif')
    
    p2_text1 = (
        "Volumetric plaque density $P_n$ at discrete time step $n$ under DBS electromagnetic\n"
        "field influence is modeled using a second-order Crank-Nicolson finite difference scheme.\n"
        "The QML optimization relies on minimizing a finite state cost function $C(\\theta)$:"
    )
    plt.text(0.1, 0.82, p2_text1, fontsize=11, family='serif')
    
    plt.text(0.2, 0.70, r"$P_{n+1} = P_n - \Delta t \left[ \alpha \sum_{k=1}^K \nabla_{\theta_k} C(\theta) \cdot f(E_n, B_n) + \frac{1}{2} D \nabla^2 P_n \right]$", fontsize=14)
    
    p2_text2 = (
        "where $D$ represents the spatial diffusion tensor for remyelination agents.\n"
        "The neural recovery index $R_n$ iterates via a coupled non-linear finite difference\n"
        "equation, mapping recovery as an inverse logistic saturation of plaque reduction:"
    )
    plt.text(0.1, 0.52, p2_text2, fontsize=11, family='serif')
    
    plt.text(0.2, 0.40, r"$R_{n+1} = R_n + \frac{\gamma \Delta t}{1 + e^{\lambda (P_{n+1} - P_{crit})}} - \beta R_n$", fontsize=14)
    
    p2_text3 = (
        "This discrete mathematical model forms the predictive basis for the MS mitigation tab,\n"
        "accelerating recovery metrics through exact quantum state descent bounding."
    )
    plt.text(0.1, 0.28, p2_text3, fontsize=11, family='serif')

    pdf.savefig()
    plt.close()

    # Page 3: Alexander's Disease - Feynman Integrals
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.1, 0.95, "2. Elaborate Feynman Path Integrals in Adaptive Ablation", fontsize=14, fontweight='bold', family='serif')
    
    p3_text1 = (
        "To trace the precise electromagnetic trajectory for ablating Rosenthal fibers,\n"
        "we construct an adaptive grid. The quantum energy propagation amplitude\n"
        "$K(x_f, t_f; x_i, t_i)$ across astrocytic finite volumes is mapped using a\n"
        "discretized Feynman Path Integral sum over all possible conduction paths $x_k$:"
    )
    plt.text(0.1, 0.78, p3_text1, fontsize=11, family='serif')
    
    plt.text(0.15, 0.60, r"$K \approx \prod_{j=1}^{N-1} \int dx_j \left(\frac{m}{2\pi i \hbar \epsilon}\right)^{N/2} \exp\left( \frac{i}{\hbar} \sum_{j=1}^{N} \left[ \frac{m(x_j - x_{j-1})^2}{2\epsilon} - V(x_j)\epsilon \right] \right)$", fontsize=12)
    
    p3_text2 = (
        "where $\epsilon = \Delta t$. The potential $V(x_j)$ maps the impedance density of mutant\n"
        "GFAP aggregates. As the path sum interferes destructively outside the Rosenthal fiber,\n"
        "ablation is localized. The cumulative ablation density $A_n$ follows:"
    )
    plt.text(0.1, 0.40, p3_text2, fontsize=11, family='serif')
    
    plt.text(0.25, 0.25, r"$A_{n+1} = A_n + \kappa \int_{\Omega} |K|^2 \cdot \left[1 - \rho_{RF}(x)\right] d^3x$", fontsize=14)

    pdf.savefig()
    plt.close()

print(f"Generated elaborate Nature preprint: {pdf_path}")
