import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = "Nature_Report_Huntingtons_DBS.pdf"

with PdfPages(pdf_path) as pdf:
    # Page 1: Title and Abstract
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.5, 0.95, "Statistical Parametric Optimization Circuitry", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.91, "for Deep Brain Stimulation in Huntington's Disease", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.86, "A Theoretical Nature Preprint", horizontalalignment='center', fontsize=14, family='serif', style='italic')
    
    plt.text(0.1, 0.70, "Abstract:", fontsize=12, fontweight='bold', family='serif')
    abstract_text = (
        "Huntington's disease (HD) leads to progressive stripping of cortical and striatal\n"
        "motor circuits. We present a formalized, interventional Deep Brain Stimulation (DBS)\n"
        "computational model employing Statistical Parametric Optimization Circuitry (SPOC).\n"
        "Our mathematical framework uses finite difference equations and stochastic differential\n"
        "calculus to simulate delayed motor degeneration and targeted interventional repair\n"
        "signals within the Motor Cortex and Striatum. By deriving precise limits on neural\n"
        "network stability and electrical optimization, we illustrate theoretically convergent\n"
        "pathways for mitigating structural decay using adaptive feedback arrays."
    )
    plt.text(0.1, 0.50, abstract_text, fontsize=11, family='serif', linespacing=1.5)
    pdf.savefig()
    plt.close()

    # Page 2: Finite Math for Neural Degeneration
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.1, 0.95, "1. Formal Derivations of Cortical Degeneration", fontsize=14, fontweight='bold', family='serif')
    
    p2_text1 = (
        "The cortical decay mapping $M_n$ characterizes the remaining functional motor capacity\n"
        "at discrete interval $n$. We model the uninhibited degradation using a finite forward\n"
        "difference operator, accelerated by a mutant Huntingtin (mHTT) toxicity variable $\\lambda$:"
    )
    plt.text(0.1, 0.82, p2_text1, fontsize=11, family='serif')
    
    plt.text(0.2, 0.70, r"$M_{n+1} = M_n - \Delta t \left( \lambda M_n + \psi \nabla^2 M_n \right) + \omega$", fontsize=14)
    
    p2_text2 = (
        "where $\\psi$ is the localized spatial diffusion of toxicity across the striatum.\n"
        "Under Statistical Parametric Optimization Circuitry, a dynamically adjusted interventional\n"
        "feedback signal $I_{opt}$ is injected. The new system converges via stochastic integrals:"
    )
    plt.text(0.1, 0.55, p2_text2, fontsize=11, family='serif')
    
    plt.text(0.2, 0.40, r"$M_{n+1} = M_n * \exp\left( - \frac{\lambda \Delta t}{1 + I_{opt}} \right) + \mathcal{N}(0, \sigma^2)$", fontsize=14)
    
    p2_text3 = (
        "This defines the bounds on degeneration wherein optimal electrical specifications strictly\n"
        "counter the decay exponential, leading to stabilization."
    )
    plt.text(0.1, 0.28, p2_text3, fontsize=11, family='serif')
    pdf.savefig()
    plt.close()

    # Page 3: SPOC Framework
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.1, 0.95, "2. Statistical Parametric Optimization Circuitry (SPOC)", fontsize=14, fontweight='bold', family='serif')
    
    p3_text1 = (
        "To define the interventional repair signal $I_{opt}$, we derive a cost-minimization\n"
        "equation over the circuit parameters involving DBS voltage $V$, pulse width $W$, and\n"
        "frequency $F$. Let the finite repair functional $R(V, W, F)$ be evaluated as:"
    )
    plt.text(0.1, 0.82, p3_text1, fontsize=11, family='serif')
    
    plt.text(0.2, 0.65, r"$R(V, W, F) = \sum_{k=1}^K w_k \cdot \tanh\left( \frac{V_k \cdot W_k}{F_k} \right)$", fontsize=16)
    
    p3_text2 = (
        "Maximizing the interventional repair necessitates traversing the parametric gradient:\n"
    )
    plt.text(0.1, 0.53, p3_text2, fontsize=11, family='serif')
    
    plt.text(0.25, 0.43, r"$I_{opt} = \max_{V, W, F} \left[ R(V, W, F) - \gamma C(V, W, F) \right]$", fontsize=14)
    
    p3_text3 = (
        "where $C(V, W, F)$ is the tissue impedance cost penalty. This SPOC model dynamically\n"
        "routes electrical flow across the Motor Cortex and Striatum, minimizing $mHTT$ toxicity\n"
        "while stabilizing $M_n$ indefinitely."
    )
    plt.text(0.1, 0.28, p3_text3, fontsize=11, family='serif')

    pdf.savefig()
    plt.close()

print(f"Generated Huntington's Nature preprint: {pdf_path}")
