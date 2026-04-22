import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

pdf_path = "Dementia_DBS_Technical_Report.pdf"

# Initialize PDF
with PdfPages(pdf_path) as pdf:
    # Title Page
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.text(0.5, 0.9, "Nature Technical Report:\nOptimization of Deep Brain Stimulation\nfor Dementia and Smart Aging Paradigms", fontsize=18, fontweight='bold', ha='center', va='top')
    
    abstract = ("This report documents the architectural specifications and mathematical manifolds\n"
                "governing the advanced Deep Brain Stimulation (DBS) framework for dementia deceleration.")
    ax.text(0.1, 0.75, abstract, fontsize=12, ha='left', va='top', style='italic')

    # Electrical Specs
    ax.text(0.1, 0.6, "1. Comprehensive Electrical Specifications", fontsize=14, fontweight='bold', ha='left')
    specs = (
        "- Voltage Range: 0.0 V - 10.5 V\n"
        "- Current Range: 0.0 mA - 25.5 mA\n"
        "- Rate (Frequency): 2 Hz - 250 Hz\n"
        "- Pulse Width: 60 \u03bcs - 450 \u03bcs\n"
        "- Interfacial Range: 500 \u03a9 - 1500 \u03a9\n"
        "- Telemetry Bands: MICS (402-405 MHz), ISM (2.4 GHz), mmWave\n"
        "- Power: Inductive Link (~125 kHz) or Resonant (~6.78 MHz)\n"
        "- Constraint: Sub-mW limitations strict tissue thermal protection."
    )
    ax.text(0.1, 0.55, specs, fontsize=12, ha='left', va='top')
    pdf.savefig(fig)
    plt.close()

    # Math Page
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    ax.text(0.1, 0.95, "2. Cortical Manifolds & Finite Element Modeling", fontsize=14, fontweight='bold', ha='left')
    
    math_desc = (
        "The cortical Volume of Tissue Activated (VTA) is modeled as a Riemannian manifold.\n"
        "We deploy the finite math mapping for residual constraints on the electrode field:"
    )
    ax.text(0.1, 0.85, math_desc, fontsize=12, ha='left', va='top')

    eq1 = r"$H_{mod}(E) = \int_{\mathcal{M}} \nabla \cdot (\sigma(r) \nabla V) \ d\Omega \ (\mathrm{mod}\ p_k)$"
    ax.text(0.1, 0.70, eq1, fontsize=16, ha='left')

    text_sub = (
        r"Where $\mathcal{M}$ represents the structural cortical manifold, and $\sigma(r)$" "\n"
        r"denotes the conductivity tensor bounded by a prime modulus $p_k$." "\n"
        "The application of Feynman Path Integrals maps sub-mW thermal dispersions."
    )
    ax.text(0.1, 0.55, text_sub, fontsize=12, ha='left', va='top')

    eq2 = r"$K(x, y) = \lim_{N\to\infty} \sum_{paths} \exp\left(\frac{iS[x(t)]}{\hbar}\right) \equiv F_R(V_{stim})$"
    ax.text(0.1, 0.4, eq2, fontsize=16, ha='left')

    pdf.savefig(fig)
    plt.close()

print(f"Successfully generated {pdf_path}")
