import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fig, ax = plt.subplots(figsize=(8.5, 11))
ax.axis('off')

y_pos = 0.95

def add_text(x, y, text, fontsize=10, weight='normal', ha='left', va='top'):
    ax.text(x, y, text, fontsize=fontsize, fontweight=weight, ha=ha, va=va, wrap=True)

add_text(0.5, y_pos, "Optimal Configurations in Optical Routing Networks:\nA Finite Mathematical Framework", fontsize=16, weight='bold', ha='center')
y_pos -= 0.08
add_text(0.5, y_pos, "Department of Photonic Computing", fontsize=12, ha='center')
y_pos -= 0.08

add_text(0.1, y_pos, "Abstract", fontsize=12, weight='bold')
y_pos -= 0.03
abs_text = ("This report introduces a finite mathematical approach to optimizing optical router\n"
            "configurations. By representing photonic switching states as finite Markov processes\n"
            "and employing linear programming, we establish optimal pathing limits, minimizing\n"
            "signal degradation while maximizing network throughput.")
add_text(0.1, y_pos, abs_text, fontsize=10)
y_pos -= 0.1

add_text(0.1, y_pos, "1. Introduction", fontsize=12, weight='bold')
y_pos -= 0.03
intro_text = ("Optical routers form the core of ultra-high-bandwidth photonic networks. However,\n"
              "the combinatorial explosion of routing states in multi-port architectures poses\n"
              "significant computational challenges. We apply finite mathematics to constrain\n"
              "the configurational space and derive optimal optical switching topologies.")
add_text(0.1, y_pos, intro_text, fontsize=10)
y_pos -= 0.12

add_text(0.1, y_pos, "2. Finite State Automata in Network Topologies", fontsize=12, weight='bold')
y_pos -= 0.03
topo_text = ("The router switching matrix is defined as a finite state space. The discrete\n"
             "transition matrix R dictates the probability of signal traversal between ports:")
add_text(0.1, y_pos, topo_text, fontsize=10)
y_pos -= 0.06
ax.text(0.3, y_pos, r'$R = [r_{ij}]_{N \times N}, \quad \sum_{j=1}^{N} r_{ij} = 1$', fontsize=12)
y_pos -= 0.06
ax.text(0.3, y_pos, r'$A_{ij} \in \{0, 1\}$', fontsize=12)
y_pos -= 0.08

add_text(0.1, y_pos, "3. Combinatorial Path Optimization", fontsize=12, weight='bold')
y_pos -= 0.03
combo_text = ("Evaluating redundant non-interfering routes requires combinatorial analysis\n"
              "over the finite photonic grid of k layers and n ports:")
add_text(0.1, y_pos, combo_text, fontsize=10)
y_pos -= 0.06
ax.text(0.3, y_pos, r'$P_{valid} = \sum_{k=1}^{K} \binom{n}{k} \cdot k!$', fontsize=12)
y_pos -= 0.08

add_text(0.1, y_pos, "4. Linear Programming for Bandwidth Allocation", fontsize=12, weight='bold')
y_pos -= 0.03
lp_text = ("We formulate the optimal bandwidth assignment as a finite linear program,\n"
           "maximizing total throughput Z subject to spectral capacity C:")
add_text(0.1, y_pos, lp_text, fontsize=10)
y_pos -= 0.06
ax.text(0.3, y_pos, r'$\max Z = \sum_{i=1}^{M} b_i x_i$', fontsize=12)
y_pos -= 0.04
ax.text(0.3, y_pos, r'subject to $\sum_{i=1}^{M} w_i x_i \leq C$', fontsize=12)
y_pos -= 0.04
ax.text(0.3, y_pos, r'and $\mathbf{x} \geq 0$', fontsize=12)
y_pos -= 0.08

add_text(0.1, y_pos, "5. Conclusion", fontsize=12, weight='bold')
y_pos -= 0.03
concl_text = ("By restricting the continuous optical domain into finite mathematical structures,\n"
              "we achieve deterministic, highly optimized router configurations that scale\n"
              "efficiently alongside next-generation network demands.")
add_text(0.1, y_pos, concl_text, fontsize=10)

pdf_path = "Nature_Optical_Router_Configurations_Finite_Math.pdf"
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)

print(f"Publication successfully saved as {pdf_path}")
