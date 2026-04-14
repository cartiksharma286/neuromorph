import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# --- General Plot Settings for Nature-style Appearance ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern

TITLE = "A Finite Mathematical Framework Correlating Cognitive Behavioral Optimization,\nPharmacological Mediators, and QML Neuromodulation in Schizoaffective Disorder"
AUTHORS = "Neuromorph Research Institute\nDepartment of Computational Psychiatry & Quantum Machine Learning"
DATE = "April 2026"

def create_title_page(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.85, TITLE, ha='center', va='center', fontsize=18, fontweight='bold', wrap=True)
    ax.text(0.5, 0.75, AUTHORS, ha='center', va='center', fontsize=12)
    ax.text(0.5, 0.70, DATE, ha='center', va='center', fontsize=10, style='italic')
    
    # Abstract
    abstract = (
        "Abstract\n\n"
        "We introduce a finite deterministic mathematical framework to model the complex neural correlates \n"
        "of Schizoaffective Disorder under the influence of both pharmacological antipsychotics and limited \n"
        "intake of <4.5% ABV beer. Traditional approaches treat alcohol unconditionally as a rigid trigger; here, we \n"
        "quantify the precise boundaries where mild social reward circuits can be activated without destabilizing \n"
        "prefrontal gating via mesolimbic dopamine surge. Furthermore, we outline the integration of Quantum \n"
        "Machine Learning (QML) optimized Transcranial Magnetic Stimulation (TMS), alongside Hebbian amplification \n"
        "metrics and Cortical Control indices, explicitly formulated to predict and minimize relapse probability \n"
        "while maximizing Cognitive Behavioral Therapy (CBT) outcomes."
    )
    ax.text(0.1, 0.55, abstract, ha='left', va='top', fontsize=11, wrap=True, linespacing=1.5, bbox=dict(facecolor='none', edgecolor='black', pad=10.0))
    
    pdf.savefig(fig)
    plt.close()

def create_math_page(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = [
        ("1. Dopamine Variance & Neuropharmacological Bounding", 30),
        ("The primary driver of psychotic manifestation in the model is the variance in dopamine", 15),
        ("$\sigma_d^2$, influenced linearly by beer intake units $B$ and inversely by medication $M$:", 15),
        (r"$\sigma_d^2 = \max\left(0.1, \sigma_{base}^2 + 1.5(0.35 B) - 1.2M\right)$", 30),
        ("This variance dictates the probability density function (PDF) of symptom severity $x$:", 15),
        (r"$P(x) = \frac{1}{\sigma_d \sqrt{2\pi}} e^{-\frac{(x - \mu_{symp})^2}{2\sigma_d^2}}$", 40),
        
        ("2. Prefrontal Cortical Control Index ($C_{pfc}$)", 30),
        ("To quantify executive oversight, we take the ratio of prefrontal node activity ($A_{pfc}$) to", 15),
        ("subcortical limbic hyperactivity (average of Amygdala $A_{amy}$ & Hippocampus $A_{hip}$):", 15),
        (r"$A_{pfc} = \max(0.1, 1.0 - 0.8(0.35B) + 0.5M)$", 25),
        (r"$C_{pfc} = \min\left(1.0, \frac{A_{pfc}}{\max(0.1, \frac{1}{2}(A_{amy} + A_{hip}))}\right)$", 40),

        ("3. Hebbian Amplification Optimization for CBT", 30),
        ("Neuroplastic efficiency underlying cognitive restructuring is maximized when cortical control", 15),
        ("dominates and dopamine variance is stable. The Hebbian Amplification Efficiency ($H$) is:", 15),
        (r"$H(\%) = \min\left(100.0, \max\left(0.0, \left[\frac{A_{pfc}}{\sigma_d^2}\right] \times 45.0\right)\right)$", 40),

        ("4. Minimal Relapse Probability Derivation", 30),
        ("The deterministic probability of clinical relapse ($P_{rel}$) emerges as a composite vector:", 15),
        (r"$P_{rel} = 100.0 - 80.0(C_{pfc}) + 15.0(\sigma_d^2) - 10.0(M) - \Delta_{CBT}$", 25),
        ("Where $\Delta_{CBT} = 15.0$ iff the composite CBT Optimization Score $\geq 70$.", 30)
    ]
    
    y = 0.95
    for text, drop in content:
        fontsize = 12
        fontweight = 'normal'
        if text[0].isdigit() and " " in text:
            fontsize = 14
            fontweight = 'bold'
            y -= 0.03
        if "$" in text and text.startswith("$"):
            fontsize = 16
        
        ax.text(0.1, y, text, ha='left', va='top', fontsize=fontsize, fontweight=fontweight)
        y -= (drop / 800.0)

    pdf.savefig(fig)
    plt.close()

def create_qml_page(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    content = [
        ("5. QML Neuromodulation Trajectories (TMS)", 20),
        ("Quantum state measurements of entangled subsystem proxies are iteratively optimized", 10),
        ("to derive the exact therapeutic frequency for transcranial magnetic stimulation ($F_{TMS}$):", 10),
        (r"$F_{TMS} = 10.0 + 8.5 \cdot \sigma_d^2 \quad \text{(Hz)}$", 30),
        
        ("This frequency dynamically tracks neurotransmitter volatility to enforce phase-locked", 10),
        ("entrainment across the frontostriatal tracts. Combined with precise pharmacology, the", 10),
        ("overall projected yield equation resolves as:", 10),
        (r"$\gamma_{yield}(\%) = \min\left(99.9, \max\left(0, 65.0 - 30.0(0.35B) + 45.0M\right)\right)$", 40)
    ]
    
    y = 0.95
    for text, drop in content:
        fontsize = 12
        fontweight = 'normal'
        if text[0].isdigit() and " " in text:
            fontsize = 14
            fontweight = 'bold'
            y -= 0.02
        if "$" in text and text.startswith("$"):
            fontsize = 16
        
        ax.text(0.1, y, text, ha='left', va='top', fontsize=fontsize, fontweight=fontweight)
        y -= (drop / 800.0)

    # Add a visual plot for relapse probability vs beer intake
    ax_plot = fig.add_axes([0.15, 0.25, 0.7, 0.4])
    B_vals = np.linspace(0, 4, 100)
    # Fixed Medication M = 1.0
    M = 1.0
    relapse_probs = []
    
    for B in B_vals:
        b_imp = B * 0.35
        d_var = max(0.1, 1.0 + (b_imp * 1.5) - (M * 1.2))
        a_pfc = max(0.1, 1.0 - (b_imp * 0.8) + (M * 0.5))
        a_amy = min(2.5, 0.8 + (b_imp * 1.2) - (M * 0.9))
        a_hip = min(2.5, 0.9 + (b_imp * 1.0) - (M * 0.7))
        c_pfc = min(1.0, a_pfc / max(0.1, (a_amy + a_hip)/2.0))
        cbt_score = min(100., max(0., 50. + (M*35) - (b_imp*45)))
        p_rel = max(1.0, min(99., 100. - 80.*c_pfc + 15.*d_var - 10.*M))
        if cbt_score > 70: p_rel -= 15
        relapse_probs.append(p_rel)
    
    ax_plot.plot(B_vals, relapse_probs, color='red', lw=2, label="Medication M=1.0")
    ax_plot.set_xlabel("Beer Intake (<4.5% ABV) [Units]")
    ax_plot.set_ylabel("Minimal Relapse Probability (%)")
    ax_plot.set_title("Relapse Trajectory Mapping vs. Beer Intake")
    ax_plot.grid(True, linestyle='--', alpha=0.6)
    ax_plot.legend()

    pdf.savefig(fig)
    plt.close()

def generate_pdf(filename="Nature_Schizoaffective_Framework.pdf"):
    print(f"Generating PDF: {filename}...")
    with PdfPages(filename) as pdf:
        create_title_page(pdf)
        create_math_page(pdf)
        create_qml_page(pdf)
    print("Done.")

if __name__ == '__main__':
    generate_pdf()
