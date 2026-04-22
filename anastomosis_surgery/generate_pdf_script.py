import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_nature_pdf(filename):
    with PdfPages(filename) as pdf:
        # Page 1: Title and Abstract
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.5, 0.9, 'Nature Machine Intelligence', fontsize=12, ha='center', style='italic')
        fig.text(0.5, 0.85, 'A Generalizable Topological Framework for Adaptive Anastomotic\nPatterning via Residual Prime Manifolds', 
                 fontsize=16, ha='center', weight='bold')
        fig.text(0.5, 0.78, 'Authors: Cartik Sharma et al.', fontsize=12, ha='center')
        
        abstract = (
            "Abstract\n\n"
            "This study introduces a novel non-rigid registration paradigm governing the \n"
            "surgicobotanical application of 'surfboard' excisions in complex anastomosis. \n"
            "By modeling stochastic deformation fields as Wishart distributions interacting \n"
            "with biological elliptic properties characterized by generalized continued fractions, \n"
            "we constrain the Coherent Point Drift (CPD). The error surfaces are parameterized \n"
            "using residual differential equations grounded in finite properties of prime \n"
            "numbers to guarantee boundary continuity during skin or vascular grafting."
        )
        fig.text(0.1, 0.6, abstract, fontsize=11, va='top', wrap=True)
        pdf.savefig(fig)
        plt.close()

        # Page 2: Mathematical Framework - CPD & Wishart
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.9, '1. Coherent Point Drift (CPD) & Wishart Noise', fontsize=14, weight='bold')
        
        text1 = (
            "We model the moving lattice $Y$ transitioning towards the target dermal lattice $X$ \n"
            "using a Gaussian Mixture Model (GMM) augmented by Wishart distributed noise \n"
            "simulating biological elasticity."
        )
        fig.text(0.1, 0.8, text1, fontsize=12)
        
        # CPD Equation
        eq1 = r"$P(\mathbf{x}_i | \mathbf{Y}) = \sum_{j=1}^{M} \frac{1}{M \cdot (2\pi\sigma^2)^{D/2}} \exp\left( - \frac{\| \mathbf{x}_i - \mathcal{T}(\mathbf{y}_j, \mathbf{\theta}) \|^2}{2\sigma^2} \right)$"
        fig.text(0.15, 0.7, eq1, fontsize=14)
        
        # Wishart Matrix Equation
        eq2 = r"$W(\Sigma | S, \nu) = \frac{|\Sigma|^{(\nu - D - 1)/2}}{2^{\nu D / 2} |S|^{\nu/2} \Gamma_D(\frac{\nu}{2})} \exp\left( -\frac{1}{2} \text{Tr}(S^{-1}\Sigma) \right)$"
        fig.text(0.15, 0.55, eq2, fontsize=14)
        
        text2 = (
            "Where $S$ denotes the target spatial covariance structure dictating the allowable \n"
            "deformation limits of the viscoelastic tissue in the surfboard anastomosis."
        )
        fig.text(0.1, 0.45, text2, fontsize=12)
        pdf.savefig(fig)
        plt.close()

        # Page 3: Continued Fractions & Prime Residuals
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.9, '2. Elliptic Continuation & Prime Residual Diff. Eq.', fontsize=14, weight='bold')
        
        text3 = (
            "To resolve pathological singularities at the incision vertices, we apply a continued \n"
            "fraction representation to the elliptic integrals governing the geometry:"
        )
        fig.text(0.1, 0.8, text3, fontsize=12)
        
        # Continued Fraction
        eq3 = r"$\mathcal{E}(\phi, k) = \frac{a_1}{b_1 + \frac{a_2}{b_2 + \frac{a_3}{b_3 + \dots}}}$"
        fig.text(0.25, 0.7, eq3, fontsize=16)
        
        text4 = (
            "The residual tracking of the displacement field $u(t)$ across modular prime properties \n"
            "is defined as:"
        )
        fig.text(0.1, 0.58, text4, fontsize=12)
        
        # Prime Equation
        eq4 = r"$\frac{\partial u}{\partial t}(x) \equiv \sum_{p \in \mathbb{P}} \left( \nabla^2 u + p \cdot \mathcal{F}(x) \right) \ (\mathrm{mod}\ p)$"
        fig.text(0.2, 0.45, eq4, fontsize=16)

        conclusion = (
            "Conclusion\n\n"
            "These models define a generalized paradigm for simulated physiological anastomosis,\n"
            "verifying spatial coherence through highly stochastic topological limitations.\n\n"
            "(c) 2026 Cartik Sharma via Neuromorph Framework."
        )
        fig.text(0.1, 0.25, conclusion, fontsize=11)
        pdf.savefig(fig)
        plt.close()

if __name__ == '__main__':
    output_pdf = "/Users/cartiksharma/Downloads/neuromorph-main-10/anastomosis_surgery/Nature_Anastomosis_Finite_Math.pdf"
    create_nature_pdf(output_pdf)
    print(f"PDF generated successfully at {output_pdf}")
