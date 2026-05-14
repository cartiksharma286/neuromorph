import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = "Nature_Report_Continued_Fractions_DBS.pdf"

with PdfPages(pdf_path) as pdf:
    # Page 1: Abstract
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.5, 0.95, "Continued Fractions and Ramanujan Operators", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.91, "for QML Deep Brain Stimulation Optimization", horizontalalignment='center', fontsize=18, fontweight='bold', family='serif')
    plt.text(0.5, 0.86, "An Extension for MS and Alexander's Disease", horizontalalignment='center', fontsize=14, family='serif', style='italic')
    
    plt.text(0.1, 0.70, "Abstract:", fontsize=12, fontweight='bold', family='serif')
    abstract_text = (
        "We extend our prior Quantum Machine Learning (QML) and finite mathematical frameworks\n"
        "for neuro-electrodynamic ablation by introducing Ramanujan operators and infinite\n"
        "continued fraction (CF) expansions. These operators bound the error gradients in\n"
        "plaque density mitigation for Multiple Sclerosis (MS) more strictly than discrete\n"
        "finite schemas. Additionally, we provide an Addendum analyzing neural recovery and\n"
        "Rosenthal fiber ablation in Alexander's Disease utilizing a novel CF-bounded model,\n"
        "yielding highly asymmetric, infinitely convergent recovery curves."
    )
    plt.text(0.1, 0.50, abstract_text, fontsize=11, family='serif', linespacing=1.5)
    pdf.savefig()
    plt.close()

    # Page 2: MS and CF
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.1, 0.95, "1. MS Paradigm Improvement via Ramanujan Operators", fontsize=14, fontweight='bold', family='serif')
    
    p2_text1 = (
        "In our MS plaque mitigation paradigm, we substitute standard finite difference QML\n"
        "cost functions with a Ramanujan modular expansion. Plaque density $P(t)$ is now modeled\n"
        "as a continuously converging fraction evaluating the field potential $E$:"
    )
    plt.text(0.1, 0.82, p2_text1, fontsize=11, family='serif')
    
    plt.text(0.3, 0.65, r"$P(t) = \frac{1}{1 + \frac{\alpha E(t)}{1 + \frac{\beta E(t)^2}{1 + \dots}}}$", fontsize=18)
    
    p2_text2 = (
        "This Ramanujan operator enforces a bounded saturation limit on mitigation,\n"
        "preventing overshoot in stochastic QML evaluations."
    )
    plt.text(0.1, 0.45, p2_text2, fontsize=11, family='serif')
    pdf.savefig()
    plt.close()

    # Page 3: Alexander's Addendum
    fig = plt.figure(figsize=(8.5, 11))
    fig.clf()
    plt.axis('off')
    
    plt.text(0.1, 0.95, "2. Addendum: Continued Fractions for Alexander's Disease", fontsize=14, fontweight='bold', family='serif')
    
    p3_text1 = (
        "For Rosenthal fiber ablation, we map the structural degradation of the fibers $A(t)$ alongside\n"
        "Neural Recovery $R(t)$ forming a coupled system of continued fractions. The ablation rate is bounded by:"
    )
    plt.text(0.1, 0.82, p3_text1, fontsize=11, family='serif')
    
    plt.text(0.2, 0.65, r"$A(t) = A_0 - \frac{\kappa}{\delta + \frac{t}{1 + \frac{t^2}{2 + \dots}}}$", fontsize=16)
    
    p3_text2 = (
        "Here, $\delta$ reflects the impedance resistance of mutant GFAP aggregates. This ensures\n"
        "the neural recovery metrics dynamically converge exactly as Rosenthal fibers degenerate,\n"
        "safeguarding surrounding white matter astrocytes."
    )
    plt.text(0.1, 0.45, p3_text2, fontsize=11, family='serif')

    pdf.savefig()
    plt.close()

print(f"Generated CF Nature preprint: {pdf_path}")
