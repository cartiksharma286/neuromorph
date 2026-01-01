
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# --- Configuration ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# --- Output Filename ---
FILENAME = "Final_Derivations_Report.pdf"

def draw_page(pdf, title, content_blocks):
    """
    Draws a page with a title and a list of content blocks.
    Each block is a tuple: (type, content)
    type: 'text' or 'equation'
    """
    fig = plt.figure(figsize=(8.5, 11))
    
    # Page Title
    plt.text(0.5, 0.92, title, ha='center', fontsize=16, weight='bold')
    
    current_y = 0.85
    
    for c_type, content in content_blocks:
        if c_type == 'text':
            # split slightly to avoid huge blocks
            plt.text(0.1, current_y, content.strip(), ha='left', va='top', fontsize=11, wrap=True)
            # Estimate drop roughly (this is a simple heuristic)
            lines = content.count('\n') + (len(content) // 90)
            current_y -= (lines * 0.025 + 0.05)
            
        elif c_type == 'equation':
            plt.text(0.5, current_y, content, ha='center', va='top', fontsize=14, color='#003366')
            current_y -= 0.08
            
    # Footer
    plt.text(0.5, 0.05, "Neuromorph Systems Integration - Confidential", ha='center', fontsize=8, color='gray')
    
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

def generate_report():
    print("Generating Refined Derivations Report...")
    
    with PdfPages(FILENAME) as pdf:
        
        # --- Page 1 ---
        P1_BLOCKS = [
            ('text', "1. The Hamiltonian Formulation\n\nWe model the neural system as a graph G(V, E) where each node (neuron/column) is a qubit state |psi>. The total system Hamiltonian is:"),
            ('equation', r"$ H_{sys} = H_{local} + H_{int} $"),
            ('text', "1.1 Local Term:\nFluctuations in local cognitive potential are modeled as Pauli-Z rotations:"),
            ('equation', r"$ H_{local} = \sum_{i} \hbar \omega_i \sigma_z^{(i)} $"),
            ('text', "1.2 Interaction Term (Synaptic Coupling):\nThe edges E represent entangled states. We use an XY-interaction model for Hebbian exchange:"),
            ('equation', r"$ H_{int} = - \sum_{\langle i,j \rangle} J_{ij}(t) \left( \sigma_+^{(i)} \sigma_-^{(j)} + \text{h.c.} \right) $"),
            ('text', "Step 1 Derivation:\nInformation transfer probability is proportional to the square of the transition amplitude:"),
            ('equation', r"$ P_{i \to j}(t) \approx \sin^2(J_{ij} t / \hbar) $"),
            ('text', "Thus, maximizing J_ij directly maximizes the coherent information flux.")
        ]
        draw_page(pdf, "1. The Quantum Neural Hamiltonian", P1_BLOCKS)
        
        # --- Page 2 ---
        P2_BLOCKS = [
            ('text', "2. Deriving the Hebbian Operator\n\nHebbian plasticity states: 'Neurons that fire together, wire together.' In Quantum terms: 'Qubits that phase-lock maximize mutual inductance.'\n\nWe define a Plasticity Operator K that acts on the couplings J:"),
            ('equation', r"$ \frac{d J_{ij}}{dt} = \eta \langle \Psi | \hat{K}_{ij} | \Psi \rangle $"),
            ('text', "Step-by-Step Derivation:\n1. Let the system seek the ground state of an 'Optimization Hamiltonian' where aligned spins have lower energy:"),
            ('equation', r"$ H_{opt} = - \sum \sigma_x^{(i)} \sigma_x^{(j)} $"),
            ('text', "2. The 'force' driving plasticity is the gradient of the energy expectation:"),
            ('equation', r"$ F_{ij} = - \nabla_{J_{ij}} \langle H_{opt} \rangle $"),
            ('text', "3. Substituting the correlations (<sigma_x sigma_x> ~ cos Delta phi):"),
            ('equation', r"$ J_{ij}(t+1) = J_{ij}(t) + \alpha \cos(\phi_i - \phi_j) $")
        ]
        draw_page(pdf, "2. Hebbian Plasticity First Principles", P2_BLOCKS)

        # --- Page 3 ---
        P3_BLOCKS = [
            ('text', "3. Prime Resonance Field Theory\n\nWe postulate that critical stability in neural networks follows the Montgomery-Odlyzko Law, relating quantum energy levels to the zeros of the Riemann Zeta function.\n\n3.1 The Prime Potential:\nWe define a scalar potential V_p on the discretized manifold:"),
            ('equation', r"$ V_p(x) = \sum_k \delta(x - x_k) (\ln p_k)^{-1} $"),
            ('text', "3.2 Modified Hebbian Update:\nTo prevent runaway excitation, we damp the Hebbian term with the Prime Potential density. This acts as a regularization term R:"),
            ('equation', r"$ \mathcal{R}_{ij} = \frac{1}{\ln p_i \ln p_j} $"),
            ('text', "3.3 Final Update Equation:"),
            ('equation', r"$ \Delta J_{ij} = \alpha \cos(\Delta \phi_{ij}) \cdot (\ln p_i \ln p_j)^{-1} $")
        ]
        draw_page(pdf, "3. Prime Resonance Regularization", P3_BLOCKS)

        # --- Page 4 ---
        P4_BLOCKS = [
            ('text', "4. Quantum Surface Integral Flux\n\nTo categorize the global state, we calculate the flux of the wavefunction across the topology.\n\nDefinition:\nLet L be the Graph Laplacian. We define the 'Neuro-Flux' Phi as:"),
            ('equation', r"$ \Phi = \oint_{\partial \mathcal{G}} (\psi^\dagger \nabla \psi) \cdot d\mathbf{S} $"),
            ('text', "Discretization:\n1. The gradient maps to L|psi>.\n2. The surface element dS is weighted by the Prime Metric g_kk."),
            ('equation', r"$ \Phi \approx \psi^\dagger \cdot (\mathcal{L} \cdot \mathbf{g}) \cdot \psi $"),
            ('text', "Interpretation:\n- High Flux: Critical State (Healthy)\n- Low Flux: Disconnected (Dementia)\n\nOur repair algorithm injects edges specifically to maximize this integral.")
        ]
        draw_page(pdf, "4. Quantum Surface Integral Flux", P4_BLOCKS)

    print(f"Report Generated: {FILENAME}")

if __name__ == "__main__":
    generate_report()
