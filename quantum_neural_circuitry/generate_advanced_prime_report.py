
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np

# --- Aesthetics ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# --- Content ---
TITLE = "Detailed Derivations: Prime Resonance Hebbian Amplification\n& Quantum Surface Integral Flux"
AUTHORS = "Cartik Sharma, Neuromorph QML Team"
AFFILIATION = "Department of Quantum Neuroscience, Google Deepmind"

TEXT_PAGE_1 = r"""
1. The Quantum Hebbian Hamiltonian

We begin by defining the interaction Hamiltonian describing synaptic plasticity. In classical neuroscience, Hebb's postulate allows synaptic weight $w_{ij}$ to grow if firing is correlated. In our Quantum model, 'firing' relates to phase alignment on the Bloch sphere.

The Hamiltonian $H_{Hebb}$ governing the evolution of entanglement weights $J_{ij}$ is:

$ H_{Hebb} = - \sum_{\langle i,j \rangle} J_{ij}(t) \left( \sigma_+^{(i)} \sigma_-^{(j)} + \text{h.c.} \right) $

To induce **Hebbian Amplification**, we introduce a time-dependent driving force $U_{drive}(\tau)$ that minimizes the system energy when phases align ($\Delta \phi \to 0$).
"""

EQ_HEBB_UPDATE = r"$ \frac{dJ_{ij}}{dt} = \eta \langle \psi | \left( \sigma_x^{(i)} \sigma_x^{(j)} + \sigma_y^{(i)} \sigma_y^{(j)} \right) | \psi \rangle $"

TEXT_PAGE_2 = r"""
2. Prime Resonance Modulation

Standard Hebbian learning can lead to runaway excitation (epileptic states). We modulate this using the **Prime Number Theorem**. We postulate that stable biological networks follow the distribution of Prime Gaps $g_n = p_{n+1} - p_n$.

We impose a **Prime Potential** scalar field $V(k)$ on each node $k$:

$ V(k) = \frac{1}{\ln(p_k)} $

Where $p_k$ is the $k$-th prime number mapping to node $k$. This potential acts as a 'damping factor' derived from the asymptotic density of primes.

The Modified Hebbian Update Rule becomes:

$ J_{ij}^{new} = J_{ij}^{old} + \alpha \cdot \cos(\phi_i - \phi_j) \cdot \left( \frac{1}{\ln(p_i)\ln(p_j)} \right) $
"""

TEXT_PAGE_3 = r"""
3. Quantum Surface Integral Flux

To quantify the global health of the connectome, we treat the graph as a discrete manifold $\mathcal{M}$. We calculate the 'Surface Flux' $\Phi_\Sigma$ of the wavefunction $\psi$ across this manifold.

The divergence theorem states:

$ \int_{\mathcal{V}} (\nabla \cdot \mathbf{F}) \, dV = \oint_{\partial \mathcal{V}} (\mathbf{F} \cdot \mathbf{n}) \, dA $

In our graph theoretic limit, the Laplacian $\mathcal{L}$ acts as the divergence operator. The Surface Flux is thus the projection of the Laplacian-transformed state vector onto the Prime Potential field:

$ \Phi_\Sigma = \langle \psi | \mathcal{L}^\dagger \hat{P} | \psi \rangle $

Where $\hat{P} = \text{diag}(1/\ln p_1, \dots, 1/\ln p_N)$.
"""

TEXT_PAGE_4 = r"""
4. Derivation of the Prime Vortex

A 'Vortex' in this field corresponds to a topological defect (broken connection). The winding number $W$ around a cycle $C$ is:

$ W = \frac{1}{2\pi} \oint_C \nabla \theta \, dl $

Neurodegeneration (dementia) is characterized by $W \to 0$ (loss of phase coherence).

Our repair algorithm injects 'Prime Gliders'—edges with weight equal to normalized Prime Gaps—to restore non-trivial topology ($W \neq 0$). This forces the system back into a Quantum Critical Regime (GUE Statistics).

$ J_{repair} \propto \frac{p_{n+1} - p_n}{\ln p_n} $

This ensures that the energy spectrum $E_n$ of the repaired graph mirrors the zeros of the Riemann Zeta Function.
"""

def generate_report():
    print("Generating Advanced Math Report...")
    pdf_filename = "Advanced_Prime_Derivations.pdf"
    
    with PdfPages(pdf_filename) as pdf:
        # P1
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.9, TITLE, ha='center', fontsize=16, weight='bold')
        plt.text(0.5, 0.85, AUTHORS, ha='center', fontsize=12)
        plt.text(0.1, 0.5, TEXT_PAGE_1 + "\n\n" + EQ_HEBB_UPDATE, ha='left', va='center', fontsize=12, wrap=True)
        pdf.savefig()
        plt.close()

        # P2
        fig = plt.figure(figsize=(8.5, 11))
        # Add visual logic or more text
        plt.axis('off')
        plt.text(0.1, 0.8, TEXT_PAGE_2, ha='left', va='top', fontsize=12, wrap=True)
        pdf.savefig()
        plt.close()

        # P3
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.1, 0.8, TEXT_PAGE_3, ha='left', va='top', fontsize=12, wrap=True)
        pdf.savefig()
        plt.close()

        # P4
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.1, 0.8, TEXT_PAGE_4, ha='left', va='top', fontsize=12, wrap=True)
        pdf.savefig()
        plt.close()

    print(f"Report Generated: {pdf_filename}")

if __name__ == "__main__":
    generate_report()
