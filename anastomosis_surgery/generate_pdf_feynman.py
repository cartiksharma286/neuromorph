import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_nature_pdf(filename):
    with PdfPages(filename) as pdf:
        # Page 1: Title and Abstract
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.5, 0.9, 'Nature Machine Intelligence', fontsize=12, ha='center', style='italic')
        fig.text(0.5, 0.85, 'A Feynman Path Integral Framework for Adaptive Anastomotic\nPatterning via Finite Mathematics', 
                 fontsize=16, ha='center', weight='bold')
        fig.text(0.5, 0.78, 'Authors: Cartik Sharma et al.', fontsize=12, ha='center')
        
        abstract = (
            "Abstract\n\n"
            "This study introduces a novel non-rigid registration paradigm governing the \n"
            "surgicobotanical application of 'surfboard' excisions in complex anastomosis. \n"
            "We extend traditional Coherent Point Drift by formulating the deformation field \n"
            "as a quantum-like system governed by Feynman Path Integrals. The action \n"
            "functional is mapped to finite properties of numbers and prime algebraic\n"
            "structures, effectively regularizing structural alignment over multiple \n"
            "simulated geometrical histories to achieve an optimal manifold mapping."
        )
        fig.text(0.1, 0.6, abstract, fontsize=11, va='top', wrap=True)
        pdf.savefig(fig)
        plt.close()

        # Page 2: Feynman Path Integrals in Registration
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.9, '1. Feynman Path Integrals (Sum Over Histories)', fontsize=14, weight='bold')
        
        text1 = (
            "The non-rigid deformation from the source lattice $X$ to the target $Y$ is \n"
            "evaluated across all possible stochastic trajectories $x(t)$. The transition \n"
            "amplitude $K$ is the integral over all paths, weighted by the action $S[x(t)]$:"
        )
        fig.text(0.1, 0.8, text1, fontsize=12)
        
        # Path Integral Equation
        eq1 = r"$K(X_f, t_f; X_i, t_i) = \int \mathcal{D}[x(t)] e^{-S[x(t)] / \gamma}$"
        fig.text(0.2, 0.7, eq1, fontsize=16)
        
        text2 = (
            "Here, $\gamma$ is a regularization parameter analogous to Planck's constant, \n"
            "and $\mathcal{D}[x(t)]$ represents the measure over the space of deformations. \n"
            "The action functional $S[x(t)]$ encodes the geometric residual error:"
        )
        fig.text(0.1, 0.55, text2, fontsize=12)
        
        # Action Equation
        eq2 = r"$S[x(t)] = \int_{t_i}^{t_f} \left( \frac{1}{2} \alpha \|\dot{x}(t)\|^2 + V(x(t), Y) \right) dt$"
        fig.text(0.2, 0.45, eq2, fontsize=14)
        pdf.savefig(fig)
        plt.close()

        # Page 3: Finite Math and Residual Differential Equations
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.9, '2. Finite Math Coupling for Residual Optimization', fontsize=14, weight='bold')
        
        text3 = (
            "To constrain the functional space to biological equivalences, the action $S$ \n"
            "is partitioned into finite fields using equivalence properties of prime residuals. \n"
            "The residual equation governing the potential $V$ becomes:"
        )
        fig.text(0.1, 0.8, text3, fontsize=12)
        
        # Finite Math Equation
        eq3 = r"$V(x_k, Y) \equiv \min_{y \in Y} \| x_k - y \|_W^2 \ (\mathrm{mod}\ p_m)$"
        fig.text(0.25, 0.68, eq3, fontsize=16)
        
        text4 = (
            "where $p_m$ iterates over a set of anatomical characteristic primes $\mathbb{P}$. \n"
            "The differential updating rule combines the Feynman weight $w_p = e^{-S_p}$ \n"
            "with a numerical phase factor bridging discrete geometries:"
        )
        fig.text(0.1, 0.55, text4, fontsize=12)
        
        # Updating Equation
        eq4 = r"$x^{(i+1)} = \frac{\sum_{paths} x^{(i)}_{path} \cdot e^{-S_{path}} \cos(2\pi / p_m)}{\sum_{paths} e^{-S_{path}}}$"
        fig.text(0.15, 0.45, eq4, fontsize=14)

        conclusion = (
            "Conclusion\n\n"
            "By mapping the continuous Feynman Path Integral to finite number fields, we establish\n"
            "a highly robust and globally optimal alignment algorithm for surgical anastomosis.\n\n"
            "(c) 2026 Cartik Sharma via Neuromorph Framework."
        )
        fig.text(0.1, 0.25, conclusion, fontsize=11)
        pdf.savefig(fig)
        plt.close()

if __name__ == '__main__':
    output_pdf = "/Users/cartiksharma/Downloads/neuromorph-main-10/anastomosis_surgery/Nature_Anastomosis_Feynman_Math.pdf"
    create_nature_pdf(output_pdf)
    print(f"PDF generated successfully at {output_pdf}")