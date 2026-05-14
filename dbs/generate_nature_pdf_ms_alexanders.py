import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = "Nature_Report_Multiple_Sclerosis_Alexander_QML_DBS.pdf"

with PdfPages(pdf_path) as pdf:
    # Page 1: Abstract and Title
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.5, 0.95, "Quantum Machine Learning and Adaptive Feynman Path", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.91, "Integrals for Demyelinating Disease Mitigation", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.86, "DBS Framework for Multiple Sclerosis & Alexander's Disease", horizontalalignment='center', fontsize=14, family='serif', style='italic')
    
    # Abstract
    plt.text(0.1, 0.78, "Abstract:", fontsize=12, fontweight='bold', family='serif')
    abstract_text = (
        "We present a novel Deep Brain Stimulation (DBS) computational framework leveraging\n"
        "Quantum Machine Learning (QML) and finite mathematical models to probabilistically\n"
        "model neural recovery in Multiple Sclerosis (MS) and adaptively ablate Rosenthal\n"
        "fibers in Alexander's Disease. We dynamically map MS plaque density mitigation\n"
        "using finite QML gradient descent cost functions on discrete networks. Furthermore,\n"
        "we utilize a discretized formulation of Feynman path integrals over astrocytic\n"
        "networks to probabilistically guide adaptive electromagnetic ablation. The models\n"
        "illustrate highly accelerated theoretical plaque dissipation and localized\n"
        "ablation stability for white matter preservation."
    )
    plt.text(0.1, 0.62, abstract_text, fontsize=11, family='serif', linespacing=1.5)

    pdf.savefig()
    plt.close()

    # Page 2: Multiple Sclerosis QML Finite Math
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.1, 0.95, "1. Finite Mathematics for QML MS Plaque Mitigation", fontsize=14, fontweight='bold', family='serif')
    
    p2_text1 = (
        "We approach the Multiple Sclerosis demyelination target by constructing a discrete\n"
        "quantum neural network (QNN). Let $P_n$ represent the volumetric plaque density\n"
        "at discrete time step $n$. The QML optimization minimizes a finite cost function\n"
        "$C(\\theta)$ over the parameter space $\\theta$, capturing the DBS protocol:"
    )
    plt.text(0.1, 0.82, p2_text1, fontsize=11, family='serif')
    
    # Equation 1
    plt.text(0.2, 0.72, r"$P_{n+1} = P_n - \alpha \sum_{k=1}^K \nabla_{\theta_k} C(\theta) \cdot f(E_n, B_n)$", fontsize=14)
    
    p2_text2 = (
        "where $f(E_n, B_n)$ formulates the biophysical coupling of the applied electric ($E_n$)\n"
        "and magnetic ($B_n$) stimulation fields across the finite element boundaries.\n"
        "The neural recovery index $R_n$ is subsequently defined by a first-order finite\n"
        "difference equation balancing plaque reduction against the natural remyelination rate $\\gamma$:"
    )
    plt.text(0.1, 0.55, p2_text2, fontsize=11, family='serif')
    
    # Equation 2
    plt.text(0.3, 0.45, r"$\Delta R_n = R_{n+1} - R_n = \gamma(100 - P_{n+1}) - \beta R_n$", fontsize=14)

    pdf.savefig()
    plt.close()

    # Page 3: Alexander's Disease Feynman Path Integrals
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.1, 0.95, "2. Adaptive Rosenthal Fiber Ablation & Feynman Path Integrals", fontsize=14, fontweight='bold', family='serif')
    
    p3_text1 = (
        "In Alexander's Disease, mutant GFAP aggregates coalesce into Rosenthal fibers within\n"
        "the astrocytes. We deploy an adaptive DBS ablation strategy constrained by a finite\n"
        "approximation of Feynman Path Integrals to trace the optimal electromagnetic\n"
        "energy dissipation path $\\Gamma$ without necrosing adjacent functional tissue.\n"
        "The propagation amplitude $K(x_f, t_f; x_i, t_i)$ across finite volumetric nodes is:"
    )
    plt.text(0.1, 0.78, p3_text1, fontsize=11, family='serif')
    
    # Equation 3
    plt.text(0.15, 0.65, r"$K(x_f, t_f; x_i, t_i) \approx \sum_{\text{discrete paths } x_k} \exp\left( \frac{i}{\hbar} \sum_{j=1}^{N} S(x_j, x_{j-1}) \right)$", fontsize=14)
    
    p3_text2 = (
        "Here, the discrete action variable $S(x_j, x_{j-1})$ integrates the thermal-electric\n"
        "gradient. This path-integral mapping localizes the trajectory of ablation, ensuring\n"
        "energy decays strictly over aggregate coordinates. The resulting ablation stability\n"
        "profile $A_n$ converges asymptotically based on the interference sum:"
    )
    plt.text(0.1, 0.48, p3_text2, fontsize=11, family='serif')
    
    # Equation 4
    plt.text(0.25, 0.35, r"$A_{n+1} = A_n + \kappa \sum_{m} |K_m|^2 \cdot \left[1 - \rho_{RF}(x_m)\right]$", fontsize=14)
    
    plt.text(0.1, 0.2, "Conclusion:", fontsize=12, fontweight='bold', family='serif')
    conclusion = (
        "Applying finite QML cost minimization algorithms alongside discretized Feynman\n"
        "path mapping generates a robust theoretical foundation for precision DBS therapeutics\n"
        "combatting severe demyelinating and leukodystrophic pathologies."
    )
    plt.text(0.1, 0.1, conclusion, fontsize=11, family='serif')

    pdf.savefig()
    plt.close()

print(f"Generated Nature preprint: {pdf_path}")
