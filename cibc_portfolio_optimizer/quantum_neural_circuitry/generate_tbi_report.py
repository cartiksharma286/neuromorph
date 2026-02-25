
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# --- Configuration ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

FILENAME = "TBI_Quantum_Derivations.pdf"

def draw_page(pdf, title, content_blocks):
    fig = plt.figure(figsize=(8.5, 11))
    plt.text(0.5, 0.92, title, ha='center', fontsize=16, weight='bold')
    
    current_y = 0.85
    
    for c_type, content in content_blocks:
        if c_type == 'text':
            plt.text(0.1, current_y, content.strip(), ha='left', va='top', fontsize=11, wrap=True)
            lines = content.count('\n') + (len(content) // 90)
            current_y -= (lines * 0.025 + 0.05)
            
        elif c_type == 'equation':
            plt.text(0.5, current_y, content, ha='center', va='top', fontsize=14, color='#003366')
            current_y -= 0.08
            
    plt.text(0.5, 0.05, "TBI Research Division - Neuromorph Systems", ha='center', fontsize=8, color='gray')
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()

def generate_report():
    print("Generating TBI (Traumatic Brain Injury) Derivations Report...")
    
    with PdfPages(FILENAME) as pdf:
        
        # --- Page 1: The TBI Impact Operator ---
        P1 = [
            ('text', "1. Modeling Traumatic Brain Injury (TBI) as a Quantum Shock\n\nTBI differs from dementia (chronic decay) by being an acute, localized disruption of the wavefunction. We model the physical impact as a non-unitary 'Measurement Shock' operator acting on a specific region Omega (the lesion site)."),
            ('text', "The Impact Operator T_shock forces a sudden collapse of coherence (decoherence) in the affected qubits:"),
            ('equation', r"$ \hat{T}_{shock} = \exp\left( - \gamma \sum_{j \in \Omega} \hat{\sigma}_z^{(j)} \Delta t \right) $"),
            ('text', "This operator exponentially damps off-diagonal density matrix elements (coherence) in the impact zone, creating a 'Quantum Void' or topological hole in the connectome.")
        ]
        draw_page(pdf, "1. The TBI Impact Operator", P1)
        
        # --- Page 2: Axonal Regrowth via Tunneling ---
        P2 = [
            ('text', "2. Axonal Regrowth as Quantum Tunneling\n\nRepairing TBI requires bridging the lesion gap. Classically, this is Axonal Regeneration. In our model, we treat this as Quantum Tunneling through the potential barrier created by the injury."),
            ('text', "The probability P of an axon (qubit link) reconnecting across the void depends on the 'Prime Barrier' height V_p and the Hebbian driving force E:"),
            ('equation', r"$ P_{tunnel} \propto \exp\left( - \frac{2}{\hbar} \int_{x_1}^{x_2} \sqrt{2m(V_p(x) - E)} \, dx \right) $"),
            ('text', "Our 'Prime Resonance' treatment effectively lowers the barrier V_p(x) by aligning the void geometry with Prime Geodesics, exponentially increasing the tunneling (healing) probability.")
        ]
        draw_page(pdf, "2. Healing via Quantum Tunneling", P2)
        
        # --- Page 3: The Lagrangian of Neurogenesis ---
        P3 = [
            ('text', "3. The Lagrangian of Neurogenesis\n\nThe optimal path for neural reconstruction minimizes the Action S. We define a 'Neuro-Lagrangian' L that balances the metabolic cost of growth (Kinetic Energy) against the gain in connectivity (Potential Energy)."),
            ('equation', r"$ \mathcal{L} = \frac{1}{2} \sum_{i,j} \dot{J}_{ij}^2 - \lambda \sum_{i} (\mathcal{C}_i - \mathcal{C}_{target})^2 $"),
            ('text', "Where dot{J} is the rate of synaptic weight change and C is the local clustering coefficient. The Euler-Lagrange equations give us the optimal growth trajectory:"),
            ('equation', r"$ \frac{d}{dt} \frac{\partial \mathcal{L}}{\partial \dot{J}_{ij}} = \frac{\partial \mathcal{L}}{\partial J}_{ij} \Rightarrow \ddot{J}_{ij} = -2\lambda (\mathcal{C}_i - \mathcal{C}_{target}) \frac{\partial \mathcal{C}_i}{\partial J_{ij}} $")
        ]
        draw_page(pdf, "3. The Lagrangian of Neurogenesis", P3)

    print(f"Report Generated: {FILENAME}")

if __name__ == "__main__":
    generate_report()
