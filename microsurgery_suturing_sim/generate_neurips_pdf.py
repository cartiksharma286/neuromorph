import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set up the figure for a US Letter page
fig, ax = plt.subplots(figsize=(8.5, 11))
ax.axis('off')

# Text layout
y_pos = 0.95

def add_text(x, y, text, fontsize=10, weight='normal', ha='left'):
    ax.text(x, y, text, fontsize=fontsize, fontweight=weight, ha=ha, va='top', wrap=True)

# Title
add_text(0.5, y_pos, "Optimizational Strategies in Microsurgery Suturing App:\n"
                     "A Finite Mathematics Approach", fontsize=16, weight='bold', ha='center')
y_pos -= 0.08

# Author and Affiliation
add_text(0.5, y_pos, "Autonomous Research System\nDepartment of Computational Surgery", 
         fontsize=12, ha='center')
y_pos -= 0.08

# Abstract
add_text(0.1, y_pos, "Abstract", fontsize=12, weight='bold')
y_pos -= 0.03
abs_text = ("This paper presents the foundational finite mathematics utilized in our novel\n"
            "microsurgery suturing simulator application. We articulate how discrete structures,\n"
            "transition matrices, and linear optimization provide the computational framework\n"
            "necessary for simulating high-precision microvascular anastomoses.")
add_text(0.1, y_pos, abs_text, fontsize=10)
y_pos -= 0.1

# Introduction
add_text(0.1, y_pos, "1. Introduction", fontsize=12, weight='bold')
y_pos -= 0.03
intro_text = ("Microsurgery requires precise manipulation of tissues at an extreme scale. The\n"
              "development of a virtual reality and robotic-assisted suturing app necessitates\n"
              "robust mathematical modeling of tissue behavior and instrument kinematics. In\n"
              "this paper, we leverage finite mathematics—specifically Markov chains, \n"
              "combinatorial optimization, and linear programming—to model suture point\n"
              "selection and risk mitigation.")
add_text(0.1, y_pos, intro_text, fontsize=10)
y_pos -= 0.15

# Finite Mathematics
add_text(0.1, y_pos, "2. Finite Mathematics in Suturing Kinematics", fontsize=12, weight='bold')
y_pos -= 0.04

add_text(0.1, y_pos, "2.1 Markov Decision Processes for State Transitions", fontsize=11, weight='bold')
y_pos -= 0.03
mdp_text = ("The process of suturing can be modeled as a finite set of states S = {s1, s2, s3, s4},\n"
            "where s1 represents tissue penetration, s2 needle driving, s3 thread pulling, and \n"
            "s4 knot tying. The transition probability matrix P is defined as:")
add_text(0.1, y_pos, mdp_text, fontsize=10)
y_pos -= 0.08
# Simple matrix representation that matplotlib mathtext handles safely
ax.text(0.3, y_pos, r'$P = [p_{ij}]_{4 \times 4}, \quad \sum_{j} p_{ij} = 1$', fontsize=12)
y_pos -= 0.1

add_text(0.1, y_pos, "2.2 Combinatorial Optimization for Suture Points", fontsize=11, weight='bold')
y_pos -= 0.03
combo_text = ("To optimize the structural integrity of the anastomosis, we must select k suture \n"
              "points from a finite set of n possible loci around the vessel circumference. The \n"
              "number of possible configurations is given by the binomial coefficient:")
add_text(0.1, y_pos, combo_text, fontsize=10)
y_pos -= 0.08
ax.text(0.3, y_pos, r'$C(n, k) = \frac{n!}{k!(n-k)!}$', fontsize=12)
y_pos -= 0.07

add_text(0.1, y_pos, "2.3 Linear Programming for Force Distribution", fontsize=11, weight='bold')
y_pos -= 0.03
lp_text = ("We define an objective function Z to minimize the tension T across the repaired \n"
           "vessel, subject to finite linear constraints of cellular tensile strength. \n"
           "Let x_i be the force vector at point i:")
add_text(0.1, y_pos, lp_text, fontsize=10)
y_pos -= 0.08
ax.text(0.3, y_pos, r'$\min Z = \sum_{i=1}^{k} c_i x_i$', fontsize=12)
y_pos -= 0.04
ax.text(0.3, y_pos, r'subject to $A x \leq b$', fontsize=12)
y_pos -= 0.04
ax.text(0.3, y_pos, r'and $x \geq 0$', fontsize=12)

# Save to PDF
pdf_path = "Microsurgery_Suturing_App_Publication.pdf"
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)

print(f"Publication successfully saved as {pdf_path}")
