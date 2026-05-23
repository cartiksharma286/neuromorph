import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# Paths to plot images
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
plots = [
    ('DBS Efficacy vs. Time', os.path.join(plot_dir, 'plot_efficacy.png')),
    ('DBS Response vs. Time', os.path.join(plot_dir, 'plot_response.png')),
    ('Combined DBS Protocol', os.path.join(plot_dir, 'plot_combined.png')),
]

pdf_path = os.path.join(os.path.dirname(__file__), 'finite_math_dbs_protocol_final.pdf')

with PdfPages(pdf_path) as pdf:
    # Title Page
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.text(0.5, 0.9, "Finite Math & DBS Protocols: Equations, Derivations, and Clinical Plots", fontsize=18, fontweight='bold', ha='center', va='top')
    ax.text(0.1, 0.8, "This document presents the mathematical models, characteristic equations, derivations, and actual clinical plots for optimal DBS protocols.", fontsize=12, ha='left', va='top')
    pdf.savefig(fig)
    plt.close(fig)

    # Insert each plot with a caption
    for title, img_path in plots:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis('off')
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_title(title, fontsize=16, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f"Image not found: {img_path}", ha='center', va='center', fontsize=14, color='red')
        pdf.savefig(fig)
        plt.close(fig)

    # Add a page for equations/derivations (placeholder)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.text(0.1, 0.9, "Characteristic Equations & Derivations", fontsize=16, fontweight='bold', ha='left')
    ax.text(0.1, 0.85, "(Insert LaTeX or formatted equations and protocol derivations here)", fontsize=12, ha='left')
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF generated: {pdf_path}")
