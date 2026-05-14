import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


PDF_PATH = "Nature_Report_FASD_Cure_Quantum_Health_Economics.pdf"

YEARS = np.arange(2024, 2037)
REGIONS = ["North America", "South Asia", "South East Asia"]
REGION_COLORS = {
    "North America": "#355070",
    "South Asia": "#b56576",
    "South East Asia": "#6d597a",
}

BASE_MARKET = {
    "North America": 2.85,
    "South Asia": 1.10,
    "South East Asia": 0.96,
}

CAGR = {
    "North America": 0.124,
    "South Asia": 0.171,
    "South East Asia": 0.162,
}

PATIENTS_M = {
    "North America": 1.42,
    "South Asia": 2.85,
    "South East Asia": 2.10,
}

INFRA_INDEX = {
    "North America": 0.97,
    "South Asia": 0.71,
    "South East Asia": 0.76,
}

TECH_INDEX = {
    "North America": 0.96,
    "South Asia": 0.69,
    "South East Asia": 0.73,
}


def market_projection(base, cagr, years):
    return base * (1 + cagr) ** (years - years[0])


def adoption_gradient(region):
    return INFRA_INDEX[region] * TECH_INDEX[region]


def complete_elliptic_k(m):
    m = np.clip(np.asarray(m, dtype=float), 0.0, 0.999999)
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    for _ in range(10):
        an = 0.5 * (a + b)
        bn = np.sqrt(a * b)
        a, b = an, bn
    return np.pi / (2.0 * a)


def continued_fraction_convergents(value, depth):
    coeffs = []
    x = float(value)
    for _ in range(depth):
        a = math.floor(x)
        coeffs.append(a)
        frac = x - a
        if abs(frac) < 1e-10:
            break
        x = 1.0 / frac

    convergents = []
    p_minus_2, p_minus_1 = 0, 1
    q_minus_2, q_minus_1 = 1, 0
    for a in coeffs:
        p = a * p_minus_1 + p_minus_2
        q = a * q_minus_1 + q_minus_2
        convergents.append((p, q, p / q))
        p_minus_2, p_minus_1 = p_minus_1, p
        q_minus_2, q_minus_1 = q_minus_1, q
    return coeffs, convergents


def quantum_stats_curve(time_grid):
    mu = 0.35 * np.exp(-0.09 * time_grid)
    sigma = 0.08 + 0.015 * np.cos(0.8 * time_grid)
    z = (time_grid - 6.0) / 2.3
    wave = np.exp(-0.5 * z**2) * np.cos(2.1 * time_grid)
    posterior = 0.55 + 0.30 * np.tanh(1.7 * wave + 0.8 * (0.20 - mu))
    return mu, sigma, np.clip(posterior, 0.0, 1.0)


def qaly_gain_curve(years):
    year0 = years - years[0]
    return 0.22 + 0.11 * np.log1p(year0) + 0.015 * np.sqrt(year0)


def icer_curve(years):
    year0 = years - years[0]
    return 98000 * np.exp(-0.13 * year0) + 26000


def add_page_title(fig, title, subtitle=""):
    fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=15, fontweight="bold", family="serif", color="#243b53")
    if subtitle:
        fig.text(0.5, 0.952, subtitle, ha="center", va="top", fontsize=9, family="serif", color="#486581", style="italic")


def text_panel(ax, title, body, face="#f8f9fb", edge="#bcccdc"):
    ax.set_facecolor(face)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(edge)
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.95, title, ha="left", va="top", fontsize=11, fontweight="bold", family="serif", color="#102a43", transform=ax.transAxes)
    ax.text(0.02, 0.88, body, ha="left", va="top", fontsize=9, family="serif", color="#243b53", linespacing=1.45, transform=ax.transAxes)


with PdfPages(PDF_PATH) as pdf:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.patch.set_facecolor("#eef2f6")
    ax.set_facecolor("#eef2f6")
    ax.text(0.5, 0.88, "Nature Technical Report:\nFASD Cure Architecture with Elliptic Integrals,\nContinued Fractions, and Quantum Theoretic Statistics", ha="center", va="center", fontsize=21, fontweight="bold", family="serif", color="#102a43", linespacing=1.4)
    ax.text(0.5, 0.76, "Health Economics, Market Projections, and Finite Mathematical Derivations", ha="center", va="center", fontsize=13, family="serif", color="#7c3aed", style="italic")
    ax.text(0.5, 0.71, "A theoretical Nature-style preprint for Fetal Alcohol Spectrum Disorder intervention planning", ha="center", va="center", fontsize=11, family="serif", color="#486581")
    abstract = (
        "Abstract\n\n"
        "This report develops a theoretical cure-program framework for Fetal Alcohol Spectrum Disorder (FASD) using a finite mathematical\n"
        "stack that couples continued-fraction pulse convergence, elliptic-integral field regularization, and quantum-theoretic posterior\n"
        "statistics for response forecasting. A dedicated clinical section formalizes a Deep Brain Stimulation (DBS) care pathway targeting\n"
        "fronto-striatal circuits impaired by prenatal alcohol exposure, including target-site selection, programming parameters, closed-loop\n"
        "adaptation, and pediatric-safety considerations. The economic layer integrates demographic prevalence, hospital-system readiness,\n"
        "payer savings, quality-adjusted life-year (QALY) gains, and net market value projections through 2036."
    )
    ax.text(0.10, 0.58, abstract, ha="left", va="top", fontsize=10.2, family="serif", color="#243b53", linespacing=1.55)
    ax.text(0.10, 0.18, "Core ingredients: DBS clinical care pathway · elliptic integrals · continued fractions · quantum Bayes posteriors · ICER/QALY modeling · regional TAM estimates", ha="left", va="center", fontsize=10, family="serif", color="#334e68")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(8.5, 11))
    add_page_title(fig, "Section I: Finite Mathematical Derivations", "Elliptic-integral field geometry, continued-fraction convergence, and quantum posterior statistics")
    gs = gridspec.GridSpec(3, 1, figure=fig, left=0.08, right=0.92, top=0.91, bottom=0.07, hspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    body1 = (
        "1. Elliptic-Integral Field Regularization\n\n"
        "Let E(r, t) denote the cortical repair field. A regularized conformal coupling over an anisotropic cortical sheet is written as\n"
        "E(r, t) = E0 exp(-lambda t) K(m(r)) / K(m0), where K(m) is the complete elliptic integral of the first kind.\n\n"
        "We define the finite update:\n"
        "E_{n+1} = E_n - Delta t (alpha E_n - beta K(m_n) + gamma I_n).\n\n"
        "The elliptic modulus m_n encodes spatial anisotropy of impaired fronto-striatal tracts."
    )
    text_panel(ax1, "Elliptic Integral Model", body1)
    ax2 = fig.add_subplot(gs[1, 0])
    body2 = (
        "2. Continued-Fraction Neuromodulation Tuning\n\n"
        "For the stimulation ratio rho = f*/f_ref, define the continued fraction\n"
        "rho = a0 + 1/(a1 + 1/(a2 + ... )).\n\n"
        "Its k-th convergent p_k / q_k produces bounded pulse schedules with integer duty structure.\n"
        "The optimization objective is\n"
        "J_k = |rho - p_k/q_k| + eta ||u_k||^2,\n"
        "where u_k is the finite pulse vector. Convergents supply robust schedules under battery and thermal constraints."
    )
    text_panel(ax2, "Continued Fraction Control", body2)
    ax3 = fig.add_subplot(gs[2, 0])
    body3 = (
        "3. Quantum-Theoretic Statistics\n\n"
        "Response amplitude A is treated as a latent random variable with prior psi(A). After observing biomarker vector y,\n"
        "the posterior obeys\n"
        "P(A | y) proportional to |<y | U_theta | A>|^2 psi(A).\n\n"
        "The finite posterior recursion is\n"
        "pi_{n+1}(A) = Z_n^{-1} pi_n(A) exp(-0.5 (y_n - H A)^T Sigma^{-1} (y_n - H A)).\n\n"
        "The treatment threshold is selected by the posterior cure-probability functional\n"
        "Q_n = Integral_{A > A*} pi_n(A) dA."
    )
    text_panel(ax3, "Quantum Posterior Statistics", body3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # ── DBS Clinical Care section ──────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    add_page_title(
        fig,
        "Section II: Deep Brain Stimulation Clinical Care for FASD",
        "Fronto-striatal targeting, programming parameters, closed-loop adaptation, and pediatric safety",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.07, right=0.95, top=0.89, bottom=0.08, hspace=0.32, wspace=0.26)

    # Panel A – text: rationale and target anatomy
    ax = fig.add_subplot(gs[0, :])
    text_panel(
        ax,
        "A. Rationale and Neuroanatomical Targets",
        "Prenatal alcohol exposure disrupts myelination and GABAergic signaling within the fronto-striatal-thalamic loop, producing the executive-\n"
        "function, impulse-control, and adaptive-behaviour deficits that define FASD. DBS is proposed as a neuromodulatory rescue strategy that\n"
        "restores tonic inhibition and synaptic homeostasis in three primary targets:\n"
        "  - Nucleus Accumbens (NAc, shell region): regulates reward-learning and impulsivity. Bilateral lead placement at AC-PC +2 mm.\n"
        "  - Anterior Limb of Internal Capsule (ALIC): modulates prefrontal drive to striatum; targets obsessive and hyperactive phenotypes.\n"
        "  - Subthalamic Nucleus (STN, dorsomedial border): improves response inhibition; co-stimulated with NAc for comorbid ADHD profiles.",
        face="#f0f4ff",
        edge="#7c9fcf",
    )

    # Panel B – text: clinical protocol
    ax = fig.add_subplot(gs[1, 0])
    text_panel(
        ax,
        "B. Clinical Protocol",
        "Patient selection:\n"
        "  FASD confirmed (ND-PAE/FAS/ARND), age >= 16,\n"
        "  failed >= 2 pharmacological trials,\n"
        "  MRI-compatible, informed consent.\n\n"
        "Surgical approach:\n"
        "  Frameless stereotaxy under GA, MER confirmation,\n"
        "  bilateral quadripolar leads (contacts 0-3),\n"
        "  implantable pulse generator subclavicular.\n\n"
        "Initial programming (NAc target):\n"
        "  Frequency: 130 Hz  Pulse width: 90 us\n"
        "  Amplitude: 2.0-4.5 V (titrated over 8 weeks).",
        face="#f4f0ff",
        edge="#9b7fd4",
    )

    # Panel C – text: closed-loop and safety
    ax = fig.add_subplot(gs[1, 1])
    text_panel(
        ax,
        "C. Closed-Loop Adaptation and Safety",
        "Closed-loop control:\n"
        "  Local field potential (LFP) beta-band power\n"
        "  (13-30 Hz) drives adaptive amplitude modulation.\n"
        "  Target: beta suppression > 40% from baseline.\n\n"
        "Paediatric safety considerations:\n"
        "  Cranial growth monitoring q-6-months until 25 y.\n"
        "  Lead extension assessed annually.\n"
        "  Threshold for battery replacement: imp > 1500 Ohm.\n\n"
        "Adverse event monitoring:\n"
        "  Mood, cognition, seizure, hardware quarterly.",
        face="#f0fff4",
        edge="#7ccfa0",
    )

    # Panel D – stimulation parameter space heatmap
    ax = fig.add_subplot(gs[2, 0])
    freq = np.linspace(60, 185, 60)
    amp = np.linspace(1.0, 6.0, 50)
    F, A = np.meshgrid(freq, amp)
    efficacy = np.exp(-((F - 130) ** 2) / 900.0 - ((A - 3.2) ** 2) / 1.8)
    cmap = ax.contourf(F, A, efficacy, levels=14, cmap="RdYlGn")
    fig.colorbar(cmap, ax=ax, label="Theoretical efficacy index")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("DBS Parameter Space: NAc Target", fontsize=10, fontweight="bold")
    ax.plot(130, 3.2, "w*", markersize=14, label="Optimal operating point")
    ax.legend(fontsize=8, loc="upper right")
    ax.axvline(100, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(160, color="white", linewidth=0.8, linestyle="--", alpha=0.6)

    # Panel E – clinical timeline
    ax = fig.add_subplot(gs[2, 1])
    timeline_months = np.arange(0, 25)
    beta_power = 100 * np.exp(-0.18 * timeline_months) + 12
    adaptive_score = 45 + 38 * (1 - np.exp(-0.22 * timeline_months))
    ax2b = ax.twinx()
    ax.plot(timeline_months, beta_power, color="#e63946", linewidth=2.2, label="Beta-band power (%)")
    ax2b.plot(timeline_months, adaptive_score, color="#2a9d8f", linewidth=2.2, linestyle="--", label="Adaptive behaviour score")
    ax.set_xlabel("Months post-implant")
    ax.set_ylabel("LFP beta power (% baseline)", color="#e63946")
    ax2b.set_ylabel("Adaptive score (0-100)", color="#2a9d8f")
    ax.set_title("Longitudinal Clinical Response", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    # ── Mathematical Figures section ───────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    add_page_title(fig, "Section III: Mathematical Figures", "Elliptic integral curve, continued-fraction convergence, and quantum posterior probability")
    gs = gridspec.GridSpec(3, 1, figure=fig, left=0.10, right=0.92, top=0.90, bottom=0.08, hspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    m = np.linspace(0.0, 0.97, 240)
    ax.plot(m, complete_elliptic_k(m), color="#5a189a", linewidth=2.6)
    ax.set_title("Complete Elliptic Integral K(m)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Modulus m")
    ax.set_ylabel("K(m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.text(0.02, 0.90, r"$K(m) = \int_0^{\pi/2} (1 - m \sin^2 \theta)^{-1/2} d\theta$", transform=ax.transAxes, fontsize=11, family="serif")

    ax = fig.add_subplot(gs[1, 0])
    coeffs, convergents = continued_fraction_convergents((1 + np.sqrt(5)) / 2, 8)
    depths = np.arange(1, len(convergents) + 1)
    errors = [abs(((1 + np.sqrt(5)) / 2) - conv[2]) for conv in convergents]
    ax.plot(depths, errors, marker="o", color="#ef476f", linewidth=2.3)
    ax.set_title("Continued-Fraction Convergence Error", fontsize=11, fontweight="bold")
    ax.set_xlabel("Convergent depth k")
    ax.set_ylabel("Absolute error")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.5)
    coeff_text = ", ".join(str(c) for c in coeffs[:6])
    ax.text(0.02, 0.88, f"coefficients: [{coeff_text}, ...]", transform=ax.transAxes, fontsize=9, family="monospace")

    ax = fig.add_subplot(gs[2, 0])
    t = np.linspace(0, 12, 240)
    mu, sigma, posterior = quantum_stats_curve(t)
    ax.plot(t, posterior, color="#118ab2", linewidth=2.4, label="Posterior cure probability")
    ax.fill_between(t, np.clip(posterior - sigma, 0, 1), np.clip(posterior + sigma, 0, 1), color="#118ab2", alpha=0.18, label="Quantum uncertainty band")
    ax.set_title("Quantum-Theoretic Posterior Cure Probability", fontsize=11, fontweight="bold")
    ax.set_xlabel("Treatment time (months)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(8.5, 11))
    add_page_title(fig, "Section IV: Market Projections", "Regional total addressable market for a theoretical FASD cure program")
    gs = gridspec.GridSpec(2, 1, figure=fig, left=0.10, right=0.92, top=0.90, bottom=0.10, hspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    projected = {}
    for region in REGIONS:
        projected[region] = market_projection(BASE_MARKET[region], CAGR[region], YEARS) * adoption_gradient(region)
        ax.plot(YEARS, projected[region], linewidth=2.4, color=REGION_COLORS[region], marker="o", markersize=3, label=region)
        ax.fill_between(YEARS, projected[region] * 0.91, projected[region] * 1.09, color=REGION_COLORS[region], alpha=0.10)
    ax.set_title("Net Market Value Projection (USD Billions)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("USD Billions")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    year0 = np.arange(len(REGIONS))
    y2024 = [projected[r][0] for r in REGIONS]
    y2036 = [projected[r][-1] for r in REGIONS]
    width = 0.34
    ax.bar(year0 - width / 2, y2024, width, color="#94a3b8", label="2024")
    ax.bar(year0 + width / 2, y2036, width, color=[REGION_COLORS[r] for r in REGIONS], label="2036")
    ax.set_xticks(year0)
    ax.set_xticklabels(["N. America", "S. Asia", "SE Asia"])
    ax.set_ylabel("USD Billions")
    ax.set_title("Base Year vs 2036 Market Comparison", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    summary = (
        f"2036 totals: North America USD {y2036[0]:.2f}B | South Asia USD {y2036[1]:.2f}B | "
        f"South East Asia USD {y2036[2]:.2f}B"
    )
    ax.text(0.02, 0.93, summary, transform=ax.transAxes, fontsize=9, family="monospace", color="#243b53")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(8.5, 11))
    add_page_title(fig, "Section V: Health Economics", "QALY, ICER, payer savings, and hospital-system economic return")
    gs = gridspec.GridSpec(2, 2, figure=fig, left=0.08, right=0.94, top=0.89, bottom=0.09, hspace=0.34, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    qaly = qaly_gain_curve(YEARS)
    ax.plot(YEARS, qaly, color="#2a9d8f", linewidth=2.5)
    ax.set_title("Incremental QALY Gain per Treated Child", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("QALYs")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = fig.add_subplot(gs[0, 1])
    icer = icer_curve(YEARS)
    ax.plot(YEARS, icer, color="#e76f51", linewidth=2.5)
    ax.axhline(50000, color="#6b7280", linestyle="--", linewidth=1.2, label="WTP lower bound")
    ax.axhline(100000, color="#6b7280", linestyle=":", linewidth=1.2, label="WTP upper bound")
    ax.set_title("ICER Trajectory", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("USD per QALY")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 0])
    savings = np.array([0.42, 0.83, 1.28, 1.90, 2.55, 3.20, 3.84, 4.55, 5.31, 6.15, 6.94, 7.80, 8.62])
    ax.bar(YEARS, savings, color="#577590")
    ax.set_title("Annual Payer Savings from Reduced Lifetime Support Burden", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("USD Billions")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    ax = fig.add_subplot(gs[1, 1])
    roi = 1.2 + 0.18 * np.sqrt(YEARS - YEARS[0])
    hospital_margin = 0.35 + 0.05 * np.log1p(YEARS - YEARS[0])
    ax.plot(YEARS, roi, color="#7209b7", linewidth=2.3, label="Provider ROI multiple")
    ax.plot(YEARS, hospital_margin, color="#4d908e", linewidth=2.3, label="Operating margin uplift")
    ax.set_title("Hospital-System Economic Performance", fontsize=10, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index / multiple")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(8.5, 11))
    add_page_title(fig, "Section VI: Formal Economic Statements", "Finite health-economic equations and summary conclusions")
    gs = gridspec.GridSpec(3, 1, figure=fig, left=0.08, right=0.92, top=0.90, bottom=0.08, hspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    text_panel(
        ax1,
        "Health Economic Equations",
        "Expected net monetary benefit is defined as\n"
        "NMB = lambda_WTP Delta QALY - Delta Cost.\n\n"
        "The incremental cost-effectiveness ratio is\n"
        "ICER = Delta Cost / Delta QALY.\n\n"
        "Regional market recursion is\n"
        "M_{n+1}^{(r)} = M_n^{(r)} (1 + g_r) + beta_r P_n^{(r)} alpha_r,\n"
        "where alpha_r is the adoption gradient, P_n^{(r)} is eligible prevalence, and g_r is regional CAGR.")
    ax2 = fig.add_subplot(gs[1, 0])
    text_panel(
        ax2,
        "Interpretation",
        "The mathematical program suggests that FASD intervention economics improve if early neurodevelopmental rescue shifts lifelong support intensity,\n"
        "school-remediation cost, juvenile-justice burden, and adult disability expenditure. South Asian and South East Asian markets grow faster because\n"
        "of underpenetrated pediatric-neurodevelopment infrastructure, while North America retains the largest absolute reimbursable market.\n\n"
        "Quantum-theoretic statistics are used here as a probabilistic decision layer rather than as a literal clinical mechanism."
    )
    ax3 = fig.add_subplot(gs[2, 0])
    total_2036 = sum(projected[r][-1] for r in REGIONS)
    conclusion = (
        f"2036 combined theoretical TAM: USD {total_2036:.2f}B.\n"
        f"2024 to 2036 CAGR-adjusted adoption strongest in South Asia ({CAGR['South Asia'] * 100:.1f}%) and South East Asia ({CAGR['South East Asia'] * 100:.1f}%).\n"
        f"ICER falls below USD 50k/QALY late in the horizon under the modeled QALY curve, suggesting eventual cost-effectiveness in optimistic scenarios.\n\n"
        "This document is theoretical and not a medical recommendation."
    )
    text_panel(ax3, "Conclusion", conclusion, face="#eef6f6", edge="#7cc6bf")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()

print(f"PDF generated: {PDF_PATH}")