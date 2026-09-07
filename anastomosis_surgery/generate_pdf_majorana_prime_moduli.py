import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def create_nature_majorana_prime_moduli_pdf(filename):
    with PdfPages(filename) as pdf:
        # Page 1: Title and Abstract
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.5, 0.92, 'Nature Machine Intelligence - Preprint', fontsize=12, ha='center', style='italic', color='#1a365d')
        fig.text(0.5, 0.85, 'Topological Anastomotic Registration via Majorana Zero Modes\nand Galois Prime Moduli Field Theory', 
                 fontsize=15, ha='center', weight='bold')
        fig.text(0.5, 0.77, 'Authors: Cartik Sharma et al.\nNeuromorph Surgical Intelligence & Quantum Topology Laboratory', fontsize=11, ha='center', color='#333333')
        
        abstract = (
            "Abstract\n\n"
            "Accurate non-rigid registration of microvascular surfboard anastomoses in reconstructive and plastic \n"
            "surgery requires topological robustness against non-isometric deformations, tissue shearing, and local \n"
            "extrema. Here we establish a mathematical registration framework integrating non-Abelian Majorana \n"
            "Zero Modes (MZMs) with finite Galois prime moduli field theory $\mathbb{Z}_p$. By representing surgical \n"
            "excision boundaries as quasiparticle lattices governed by Clifford algebra self-adjoint operators \n"
            "$\gamma_i = \gamma_i^\dagger$, non-rigid displacement vectors are braided in protected topological sub-spaces. \n"
            "Spatial congruences are strictly evaluated over finite prime rings $\mathbb{Z}/p\mathbb{Z}$, bounding \n"
            "Dirichlet deformation residuals and preserving tissue manifold homeomorphisms under extreme intraoperative strain."
        )
        fig.text(0.1, 0.48, abstract, fontsize=10.5, va='top', wrap=True)
        
        keywords = "Keywords: Majorana Zero Modes, Non-Abelian Anyons, Prime Moduli, Finite Fields, Anastomosis, Topological Registration"
        fig.text(0.1, 0.22, keywords, fontsize=9.5, style='italic', color='#6b21a8')
        pdf.savefig(fig)
        plt.close()

        # Page 2: Majorana Zero-Mode Operators & Braiding
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.92, '1. Majorana Operator Algebra & Non-Abelian Braiding', fontsize=13, weight='bold')
        
        text1 = (
            "Let $\{\gamma_1, \gamma_2, \dots, \gamma_{2N}\}$ be a discrete set of self-conjugate Majorana operators \n"
            "assigned to the discrete vertices of the surfboard excision boundary, satisfying the canonical Clifford anticommutation:"
        )
        fig.text(0.1, 0.82, text1, fontsize=10.5)
        
        # Majorana Clifford Equation
        eq1 = r"$\gamma_i^\dagger = \gamma_i, \quad \{\gamma_i, \gamma_j\} = \gamma_i \gamma_j + \gamma_j \gamma_i = 2 \delta_{ij} \mathbf{I}$"
        fig.text(0.2, 0.73, eq1, fontsize=13)
        
        text2 = (
            "Exchanging adjacent boundary keypoints corresponds to the adiabatic non-Abelian braiding unitary operator \n"
            "$\mathcal{B}_{ij} \in SO(2N)$, operating on the degenerate topological ground-state space:"
        )
        fig.text(0.1, 0.65, text2, fontsize=10.5)
        
        # Braiding Operator Equation
        eq2 = r"$\mathcal{B}_{ij} = \exp\left( \frac{\pi}{4} \gamma_i \gamma_j \right) = \frac{1}{\sqrt{2}} \left( \mathbf{I} + \gamma_i \gamma_j \right)$"
        fig.text(0.22, 0.56, eq2, fontsize=13)
        
        text3 = (
            "The total fermion parity operator $(-1)^F = \prod_{k=1}^N (i \gamma_{2k-1} \gamma_{2k})$ establishes a global topological \n"
            "invariant that prevents self-intersecting boundary inversions during surgical approximation."
        )
        fig.text(0.1, 0.46, text3, fontsize=10.5)
        pdf.savefig(fig)
        plt.close()

        # Page 3: Finite Prime Moduli Congruences & Convergence
        fig = plt.figure(figsize=(8.5, 11))
        fig.clf()
        fig.text(0.1, 0.92, '2. Galois Prime Moduli Field Equations for Manifold Drift', fontsize=13, weight='bold')
        
        text4 = (
            "To eliminate continuous error accumulation, the residual deformation field $u(x, t)$ is \n"
            "projected onto an anatomical sequence of prime ideals $\mathfrak{p}_m \subset \mathcal{O}_K$ within Galois fields $\mathbb{F}_p$:"
        )
        fig.text(0.1, 0.82, text4, fontsize=10.5)
        
        # Finite Math Prime Moduli Equation
        eq3 = r"$\mathcal{R}_p(\mathbf{x}_i) \equiv \sum_{j=1}^{M} \left( \|\mathbf{x}_i - \mathcal{T}(\mathbf{y}_j)\|^2 + \left( \mathbf{y}_j^T \mathbf{A} \mathbf{x}_i \right) \right) \ (\mathrm{mod}\ p_m)$"
        fig.text(0.12, 0.71, eq3, fontsize=12)
        
        text5 = (
            "The regularized velocity field update for registration along the surfboard perimeter is governed by the \n"
            "braided prime differential equation:"
        )
        fig.text(0.1, 0.61, text5, fontsize=10.5)
        
        # Braided Update Equation
        eq4 = r"$\frac{d \mathbf{x}_i}{dt} = \sum_{j} \mathcal{B}_{ij} \left( \mathbf{y}_j - \mathbf{x}_i \right) \cdot \left( \alpha + \beta \frac{(i \cdot p_m) \ (\mathrm{mod}\ p_{m+1})}{p_{m+1}} \right)$"
        fig.text(0.10, 0.50, eq4, fontsize=12)
        
        text6 = (
            "Under the Chinese Remainder Theorem $\mathbb{Z}/N\mathbb{Z} \cong \prod_m \mathbb{Z}/p_m \mathbb{Z}$, the composite registration trajectory \n"
            "uniquely minimizes the global Riemannian distortion energy without stochastic divergence."
        )
        fig.text(0.1, 0.40, text6, fontsize=10.5)
        
        eq5 = r"$\Phi_{\mathrm{total}}(\mathbf{X}, \mathbf{Y}) \equiv \bigoplus_{m=1}^{K} \left( \frac{1}{2} \text{Tr}\left( \mathbf{X}^T \mathcal{L}_{\mathrm{Majorana}} \mathbf{X} \right) \right) \ (\mathrm{mod}\ p_m)$"
        fig.text(0.12, 0.30, eq5, fontsize=12)

        conclusion = (
            "3. Conclusion\n\n"
            "Majorana Zero-Mode non-Abelian braiding coupled with prime moduli arithmetic delivers unconditional \n"
            "topological protection and exact manifold registration for delicate vascular micro-anastomoses.\n\n"
            "(c) 2026 Cartik Sharma via Neuromorph Framework."
        )
        fig.text(0.1, 0.12, conclusion, fontsize=9.5)
        pdf.savefig(fig)
        plt.close()

if __name__ == '__main__':
    output_pdf = "/Users/cartiksharma/Downloads/neuromorph-main-10/anastomosis_surgery/Nature_Anastomosis_Majorana_Prime_Moduli_Finite_Math.pdf"
    create_nature_majorana_prime_moduli_pdf(output_pdf)
    print(f"PDF generated successfully at {output_pdf}")
