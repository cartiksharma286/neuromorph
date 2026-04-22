import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_nature_pdf(filename):
    with PdfPages(filename) as pdf:
        # Page 1: Title and Abstract
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.5, 0.9, 'Nature Machine Intelligence', fontsize=12, ha='center', style='italic')
        fig.text(0.5, 0.85, 'A Combinatorial Pigeonhole Manifold Operator\nfor Anastomotic Surfboard Registration', 
                 fontsize=16, ha='center', weight='bold')
        fig.text(0.5, 0.78, 'Authors: Cartik Sharma et al.', fontsize=12, ha='center')
        
        abstract = (
            "Abstract\n\n"
            "Building upon stochastic Wishart deformations, we present a novel combinatorial\n"
            "topological framework leveraging a continuous analog of the Pigeonhole Principle. \n"
            "By partitioning the target surgical manifold into discrete 'holes' (attractor nodes),\n"
            "source 'pigeons' (excision vertices) are optimally mapped to minimize structural\n"
            "geometric residuals. This reduces the problem to a non-linear Dirichlet energy\n"
            "relaxation gradient, offering deterministically bounded computation time for\n"
            "complex plastic surgery applications and graft modeling."
        )
        fig.text(0.1, 0.6, abstract, fontsize=11, va='top', wrap=True)
        pdf.savefig(fig)
        plt.close()

        # Page 2: Mathematical Framework - Pigeonhole & Dirichlet
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.9, '1. Pigeonhole Combinatorial Map & Dirichlet Energy', fontsize=14, weight='bold')
        
        text1 = (
            "For a discrete target manifold $Y$, we define a subset of attractor 'holes' $\mathcal{H}$.\n"
            "Each source vertex $x_i$ ('pigeon') is assigned to the nearest hole minimizing the $L_2$ norm:"
        )
        fig.text(0.1, 0.8, text1, fontsize=12)
        
        # Pigeonhole Assignment Equation
        eq1 = r"$h^*(x_i) = \arg\min_{h \in \mathcal{H}} \| x_i - h \|_{2}$"
        fig.text(0.25, 0.7, eq1, fontsize=16)
        
        text2 = (
            "The manifold undergoes a geometric relaxation guided by a discretized Dirichlet energy:\n"
        )
        fig.text(0.1, 0.6, text2, fontsize=12)

        # Dirichlet Energy Equation
        eq2 = r"$E(x) = \frac{1}{2} \sum_{i} \| x_i - h^*(x_i) \|^{2} + \lambda \| x_i - y_i \|^{2}$"
        fig.text(0.20, 0.5, eq2, fontsize=16)

        text3 = (
            "The iterative update limits asymptotic divergence by linearly blending local combinatorial\n"
            "relaxation with a global target drift vector."
        )
        fig.text(0.1, 0.4, text3, fontsize=12)
        
        # Gradient Update Equation
        eq3 = r"$x_i^{(t+1)} = x_i^{(t)} + \alpha (h^*(x_i^{(t)}) - x_i^{(t)}) + \beta (y_i - x_i^{(t)})$"
        fig.text(0.15, 0.28, eq3, fontsize=16)

        conclusion = (
            "Conclusion\n\n"
            "The Pigeonhole Manifold operator provides a robust combinatorial bound to \n"
            "otherwise unconstrained non-rigid deformations, highly optimizing surgical alignments.\n\n"
            "(c) 2026 Cartik Sharma via Neuromorph Framework."
        )
        fig.text(0.1, 0.12, conclusion, fontsize=11)
        pdf.savefig(fig)
        plt.close()

if __name__ == '__main__':
    output_pdf = "/Users/cartiksharma/Downloads/neuromorph-main-10/anastomosis_surgery/Nature_Anastomosis_Pigeonhole_Math.pdf"
    create_nature_pdf(output_pdf)
    print(f"PDF generated successfully at {output_pdf}")