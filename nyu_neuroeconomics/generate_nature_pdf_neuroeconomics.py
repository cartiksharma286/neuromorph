import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set up the figure for a US Letter page
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.axis('off')

# Text layout
y_pos = 0.95

def add_text(x, y, text, fontsize=10, weight='normal', ha='left', va='top'):
    ax.text(x, y, text, fontsize=fontsize, fontweight=weight, ha=ha, va=va, wrap=True)

# Title
add_text(0.5, y_pos, "Quantum Neuroeconomics: Finite Mathematical Optimization\nin Decision Neurobiology", fontsize=16, weight='bold', ha='center')
y_pos -= 0.08

# Author and Affiliation
add_text(0.5, y_pos, "Department of Neuroeconomics, Autonomous Research System", fontsize=12, ha='center')
y_pos -= 0.08

# Abstract
add_text(0.1, y_pos, "Abstract", fontsize=12, weight='bold')
y_pos -= 0.03
abs_text = ("This article presents a framework integrating finite mathematics and quantum\n"
            "machine learning to optimize neuroeconomic decision-making processes. By\n"
            "modeling human portfolio selections as finite Markov decision processes,\n"
            "we reveal the computational foundations of choice under uncertainty.")
add_text(0.1, y_pos, abs_text, fontsize=10)
y_pos -= 0.1

# Introduction
add_text(0.1, y_pos, "1. Introduction", fontsize=12, weight='bold')
y_pos -= 0.03
intro_text = ("Neuroeconomics seeks to understand how the brain computes value and drives\n"
              "behavior. Classical models often fail to capture the discrete, finite nature\n"
              "of synaptic computations. Here, we apply combinatorial optimization and finite\n"
              "probability to simulate and predict portfolio choices in dynamic markets.")
add_text(0.1, y_pos, intro_text, fontsize=10)
y_pos -= 0.12

# Finite Math Optimization
add_text(0.1, y_pos, "2. Finite Mathematics in Choice Computations", fontsize=12, weight='bold')
y_pos -= 0.03
fd_text = ("A subject's choice among discrete alternatives forms a finite state space, S.\n"
           "Expected utility is maximized over a predefined combinatorial set:")
add_text(0.1, y_pos, fd_text, fontsize=10)
y_pos -= 0.06
ax.text(0.3, y_pos, r'$E[U] = \sum_{s=1}^{S} P(s) U(x_s)$', fontsize=12)
y_pos -= 0.06
fd_text2 = ("Linear constraints limit the resources (e.g., cognitive capacity, capital):")
add_text(0.1, y_pos, fd_text2, fontsize=10)
y_pos -= 0.03
ax.text(0.3, y_pos, r'$\sum_{i=1}^{n} c_i x_i \leq C, \quad x_i \in \{0, 1\}$', fontsize=12)
y_pos -= 0.08

# QML
add_text(0.1, y_pos, "3. Quantum Machine Learning for Neuro-Portfolios", fontsize=12, weight='bold')
y_pos -= 0.03
qml_text = ("We model firing rates as probability amplitudes in a quantum circuit. Superposition\n"
            "represents the deliberation phase before state collapse (choice execution):")
add_text(0.1, y_pos, qml_text, fontsize=10)
y_pos -= 0.06
ax.text(0.3, y_pos, r'$|\Psi\rangle = \alpha |Buy\rangle + \beta |Sell\rangle$', fontsize=12)
y_pos -= 0.08

# Conclusion
add_text(0.1, y_pos, "4. Conclusion", fontsize=12, weight='bold')
y_pos -= 0.03
concl_text = ("Coupling quantum algorithms with finite mathematics elucidates the mechanics\n"
              "of neuronal decision-making, offering scalable telemetry for financial modeling.")
add_text(0.1, y_pos, concl_text, fontsize=10)

# Save to PDF
pdf_path = "Nature_Neuroeconomics_Finite_Math.pdf"
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)

print(f"Publication successfully saved as {pdf_path}")
