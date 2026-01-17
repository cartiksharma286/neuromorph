
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

# --- Settings ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

TITLE = "NEURAL CIRCUITRY PARADIGM REPORT:\nSTATISTICAL CONGRUENCES & TRADE WARS"
AUTHORS = "Neuromorph Quantum Systems"
DATE = "January 2026"

def draw_header(ax, pagenum):
    ax.text(0.5, 0.97, "Paradigm Report: Game Theory V Quantum Neural Circuitry", 
            ha='center', fontsize=9, color='#777', style='italic')
    ax.text(0.95, 0.02, f"Page {pagenum}", ha='right', fontsize=9, color='#777')

def generate_pdf():
    print("Generating Neural_Paradigm_Report.pdf...")
    with PdfPages('Neural_Paradigm_Report.pdf') as pdf:
        
        # --- PAGE 1: Intro & Math 1 ---
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
        ax.axis('off')
        draw_header(ax, 1)
        
        y = 0.9
        ax.text(0.5, y, TITLE, ha='center', fontsize=18, weight='bold', color='#111')
        y -= 0.08
        ax.text(0.5, y, AUTHORS + " | " + DATE, ha='center', fontsize=12, style='italic', color='#444')
        y -= 0.06
        
        ax.plot([0, 1], [y, y], color='black', lw=2)
        y -= 0.05
        
        # Executive Summary
        ax.text(0, y, "EXECUTIVE SUMMARY", fontsize=12, weight='bold')
        y -= 0.03
        text_summary = ("This report details the integration of Paradigm Models for neural circuitry, "
                        "contrasting Game Theoretic stability with Quantum Surface Integral optimization. "
                        "We exhibit the system's performance under 'Trade War' conditions—high competitive "
                        "pressure and metabolic scaling constraints.")
        ax.text(0, y, text_summary, fontsize=10, wrap=True, va='top', ha='left')
        y -= 0.12
        
        res_text = ("Key Result: The application of Measure Theoretic Pruning enhances Neural Plasticity "
                    "significantly during peak Trade War intensity by identifying essential supports.")
        ax.text(0, y, res_text, fontsize=10, weight='bold', wrap=True, va='top')
        y -= 0.08
        
        # Finite Math 1
        ax.text(0, y, "1. FINITE MATH DERIVATIONS", fontsize=12, weight='bold')
        y -= 0.04
        ax.text(0, y, "1.1 The Quantum Surface Integral", fontsize=11, weight='bold', color='#333')
        y -= 0.03
        t1 = "The measure of coherence flux $\Phi$ across the neural manifold is derived as a discrete surface integral over the graph Laplacian spectrum:"
        ax.text(0, y, t1, fontsize=10, wrap=True, va='top')
        y -= 0.06
        
        eq1 = r"$\Phi = \oint_{\partial \Sigma} \psi \cdot (\nabla \psi) dA \approx \sum_{i} \psi_i (L \psi)_i \frac{1}{\ln(p_i)}$"
        ax.text(0.5, y, eq1, fontsize=14, ha='center', color='#003366')
        y -= 0.08
        
        notes1 = ("Where $\psi$ is the vector state, $L$ is the Laplacian, and $p_i$ is the Prime Metric mapping. "
                  "This integral defines the 'Resonance' of the topology.")
        ax.text(0, y, notes1, fontsize=10, wrap=True, va='top')
        
        pdf.savefig()
        plt.close()
        
        # --- PAGE 2: Math 2 & Nash ---
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
        ax.axis('off')
        draw_header(ax, 2)
        y = 0.95
        
        # Radom-Nikodym
        ax.text(0, y, "1.2 Radon-Nikodym Derivative for Information Density", fontsize=11, weight='bold', color='#333')
        y -= 0.03
        t2 = "We define structural measure $\mu$ and information measure $\\nu$. The derivative identifies hotspots:"
        ax.text(0, y, t2, fontsize=10, va='top')
        y -= 0.05
        
        eq2 = r"$\frac{d\nu}{d\mu}(x) = \lim_{\epsilon \to 0} \frac{\nu(B_\epsilon(x))}{\mu(B_\epsilon(x))} \approx \frac{\sum w_{ij} (1 + |\psi_i|^2)}{\deg(i)}$"
        ax.text(0.5, y, eq2, fontsize=14, ha='center', color='#003366')
        y -= 0.06
        ax.text(0, y, "Nodes where $D_i < \epsilon$ are Sets of Measure Zero and are pruned.", fontsize=10)
        y -= 0.08
        
        # Nash Eq
        ax.text(0, y, "1.3 Nash Equilibrium in Metabolic Trade Wars", fontsize=11, weight='bold', color='#333')
        y -= 0.03
        t3 = "Synapses compete for metabolic substrates ($M$). The payoff function implies:"
        ax.text(0, y, t3, fontsize=10, va='top')
        y -= 0.05
        
        eq3 = r"$w^*_{uv} = \frac{\alpha C_{uv} (1 - \lambda M)}{2 \beta}$"
        ax.text(0.5, y, eq3, fontsize=14, ha='center', color='#003366')
        y -= 0.06
        t4 = "As $\lambda M \\to 1$ (Trade War), $w^* \\to 0$. Measure theory prevents this collapse by shrinking the domain."
        ax.text(0, y, t4, fontsize=10, wrap=True, va='top')
        
        pdf.savefig()
        plt.close()
        
        # --- PAGE 3: Plots & Results ---
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
        ax.axis('off')
        draw_header(ax, 3)
        y = 0.95
        
        ax.text(0, y, "2. CHARACTERIZATIONS & RESULTS", fontsize=12, weight='bold')
        y -= 0.04
        
        # Plot 1
        ax.text(0, y, "2.1 Neural Plasticity at Trade Wars", fontsize=11, weight='bold')
        y -= 0.35
        try:
            img = mpimg.imread('trade_wars_plasticity.png')
            ax_img = fig.add_axes([0.15, y, 0.7, 0.35])
            ax_img.imshow(img)
            ax_img.axis('off')
        except Exception as e:
            ax.text(0.5, y+0.15, "[Image Placeholder: trade_wars_plasticity.png]", ha='center')
        
        y -= 0.08
        ax_text_bbox = fig.add_axes([0.1, y-0.1, 0.8, 0.1])
        ax_text_bbox.axis('off')
        caption1 = ("Figure 1: Comparison of Plasticity. The Quantum Model (Blue) maintains adaptability "
                    "even as Metabolic Cost increases, whereas the Classical Model (Red) freezes.")
        ax_text_bbox.text(0, 1, caption1, fontsize=9, wrap=True, style='italic')
        
        # Plot 2
        y -= 0.05
        ax.text(0, y, "2.2 Topology Optimization", fontsize=11, weight='bold')
        y -= 0.35
        try:
            img2 = mpimg.imread('topology_optimization.png')
            ax_img2 = fig.add_axes([0.15, y, 0.7, 0.30])
            ax_img2.imshow(img2)
            ax_img2.axis('off')
        except:
            ax.text(0.5, y+0.15, "[Image Placeholder: topology_optimization.png]", ha='center')
            
        pdf.savefig()
        plt.close()
        
        # --- PAGE 4: Conclusion ---
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
        ax.axis('off')
        draw_header(ax, 4)
        y = 0.95
        
        ax.text(0, y, "3. IMPROVEMENTS APPLIED", fontsize=12, weight='bold')
        y -= 0.04
        impr = ("1. Statistical Congruences: Ramanujan's Tau-function constraints.\n"
                "2. Hyper-Criticality: Prime Gap distribution tuning.\n"
                "3. Ghost-Pruning: Radon-Nikodym signal-to-noise optimization.")
        ax.text(0.02, y, impr, fontsize=10, va='top', linespacing=1.8)
        y -= 0.15
        
        ax.text(0, y, "4. CONCLUSION", fontsize=12, weight='bold')
        y -= 0.04
        conc = ("The synthesis of Game Theory and Quantum Measure Theory provides a robust "
                "framework for AGI neural circuits in adversarial environments. Maximizing the "
                "Quantum Surface Integral on a reduced measure space yields a Pareto-Optimal "
                "Nash Equilibrium that is resilient to metabolic trade wars.")
        ax.text(0, y, conc, fontsize=10, wrap=True, va='top')
        
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    generate_pdf()
