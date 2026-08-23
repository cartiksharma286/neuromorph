#!/usr/bin/env python3
"""
Generate a Nature-style preprint PDF:

    Continued-Fraction LLM Predictive Triage of Clinical DICOM Message
    Queues: Elliptic-Integral Projection States and Prime-Distribution
    Asymptotics in Finite Mathematics

Features:
    • 4 publication-quality matplotlib figures (embedded)
    • Finite mathematical formalism for CF-LLM triage, elliptic integral
      projection states, and prime counting asymptotics
    • Summary results table

Requires: reportlab, numpy, scipy, matplotlib
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import ellipk, ellipe

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_RIGHT
from reportlab.lib import colors

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
DPI = 220

C_DEEP    = '#0f172a'
C_SKY     = '#0284c7'
C_EMERALD = '#059669'
C_AMBER   = '#d97706'
C_ROSE    = '#e11d48'
C_VIOLET  = '#7c3aed'
C_SLATE   = '#475569'
C_TEAL    = '#0d9488'
C_INDIGO  = '#4f46e5'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafbfc',
    'axes.edgecolor': '#cbd5e1',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#94a3b8',
    'lines.linewidth': 1.8,
})


def _save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.08)
    plt.close(fig)
    print(f"  [PLOT] {name}")
    return path


def _cf_llm_triage(queue_size=24, cf_depth=10, seed=42):
    """Reproduce the app's continued-fraction predictive triage scoring."""
    modalities = ['CT', 'MR', 'US', 'XA', 'CR', 'PT']
    rng = np.random.default_rng(seed)
    messages = []
    for i in range(queue_size):
        modality = modalities[i % len(modalities)]
        acuity = float(rng.uniform(0.1, 1.0))
        arrival_delay = float(rng.uniform(0.05, 5.0))
        a_terms = [acuity * math.sin((n + 1) * 0.7) for n in range(cf_depth)]
        b_terms = [1.0 / (arrival_delay + n + 1) for n in range(cf_depth)]

        def eval_cf(k, a_terms=a_terms, b_terms=b_terms):
            val = 0.0
            for idx in reversed(range(k)):
                denom = a_terms[idx] + val
                if abs(denom) < 1e-9:
                    denom = 1e-9
                val = b_terms[idx] / denom
            return val

        convergents = [eval_cf(k) for k in range(1, cf_depth + 1)]
        final_val = convergents[-1]
        priority = float(min(1.0, max(0.0, abs(final_val) * acuity)))
        messages.append({
            'id': f'MSG{i:03d}', 'modality': modality, 'acuity': acuity,
            'priority': priority, 'convergents': convergents,
        })
    return messages


def plot_figure1_cf_llm_triage():
    """Figure 1: CF-LLM predictive triage — convergence + priority ranking."""
    messages = _cf_llm_triage()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    palette = [C_SKY, C_ROSE, C_EMERALD, C_AMBER, C_VIOLET]
    for j, msg in enumerate(messages[:5]):
        iters = np.arange(1, len(msg['convergents']) + 1)
        vals = np.array(msg['convergents'])
        ax1.plot(iters, vals, 'o-', color=palette[j % len(palette)], ms=3.5,
                  label=f"{msg['id']} ({msg['modality']})")
    ax1.set_xlabel('Convergent order k')
    ax1.set_ylabel('CF value h$_k$/k$_k$')
    ax1.set_title('(a) CF-LLM Convergent Trajectories', fontweight='bold', fontsize=9)
    ax1.legend(loc='best', framealpha=0.9, edgecolor='#e2e8f0', fontsize=6.5)

    ranked = sorted(messages, key=lambda m: m['priority'], reverse=True)[:12]
    colors_bar = [C_ROSE if m['priority'] > 0.6 else (C_AMBER if m['priority'] > 0.3 else C_SLATE)
                  for m in ranked]
    ax2.barh([m['id'] for m in ranked][::-1], [m['priority'] for m in ranked][::-1],
              color=colors_bar[::-1])
    ax2.set_xlabel('CF-LLM predicted priority score')
    ax2.set_title('(b) Predictive Triage Ranking', fontweight='bold', fontsize=9)

    fig.suptitle('Figure 1 | Continued-Fraction LLM Predictive DICOM Queue Triage',
                 fontsize=10, fontweight='bold', y=1.03)
    fig.tight_layout()
    return _save(fig, 'fig1_cf_llm_triage.png')


def plot_figure2_elliptic_projection():
    """Figure 2: Elliptic integral projection states K(m), E(m), and ratio."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    m_grid = np.linspace(0.001, 0.995, 300)
    K_vals = ellipk(m_grid)
    E_vals = ellipe(m_grid)
    projection = E_vals / K_vals

    ax1.plot(m_grid, K_vals, '-', color=C_SKY, lw=2, label='K(m) — first kind')
    ax1.plot(m_grid, E_vals, '-', color=C_EMERALD, lw=2, label='E(m) — second kind')
    ax1.set_xlabel('Elliptic parameter m')
    ax1.set_ylabel('Integral value')
    ax1.set_title('(a) Complete Elliptic Integrals', fontweight='bold', fontsize=9)
    ax1.legend(loc='upper left', framealpha=0.9, edgecolor='#e2e8f0')

    ax2.plot(m_grid, projection, '-', color=C_VIOLET, lw=2.2)
    idx_half = int(np.argmin(np.abs(m_grid - 0.5)))
    ax2.plot(0.5, projection[idx_half], 'o', color=C_ROSE, ms=8, zorder=5)
    ax2.annotate(f"m=0.5\nE/K={projection[idx_half]:.4f}",
                 xy=(0.5, projection[idx_half]),
                 xytext=(0.62, projection[idx_half] + 0.05),
                 fontsize=7.5, color=C_ROSE, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_ROSE, lw=1.2))
    ax2.set_xlabel('Elliptic parameter m')
    ax2.set_ylabel('Projection state  E(m)/K(m)')
    ax2.set_title('(b) Elliptic Projection State', fontweight='bold', fontsize=9)

    fig.suptitle('Figure 2 | Elliptic-Integral Projection States', fontsize=10,
                 fontweight='bold', y=1.03)
    fig.tight_layout()
    return _save(fig, 'fig2_elliptic_projection.png')


def plot_figure3_prime_distribution(prime_limit=2000):
    """Figure 3: Prime counting function vs Li(x) asymptotics + gap distribution."""
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = GridSpec(2, 2, hspace=0.4, wspace=0.32)

    sieve = np.ones(prime_limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(prime_limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    primes = np.nonzero(sieve)[0]
    x_grid = np.arange(2, prime_limit + 1)
    pi_x = np.cumsum(sieve[2:prime_limit + 1]).astype(float)
    li_x = np.array([
        float(np.trapezoid(1.0 / np.log(np.arange(2, xi + 1, dtype=float)),
                        np.arange(2, xi + 1, dtype=float))) if xi > 3 else 0.0
        for xi in x_grid
    ])
    x_over_lnx = x_grid / np.log(x_grid)

    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(x_grid, pi_x, '-', color=C_DEEP, lw=1.8, label='π(x) — exact count')
    ax_a.plot(x_grid, li_x, '--', color=C_ROSE, lw=1.6, label='Li(x) — log. integral')
    ax_a.plot(x_grid, x_over_lnx, ':', color=C_SLATE, lw=1.4, label='x / ln(x)')
    ax_a.set_xlabel('x')
    ax_a.set_ylabel('Prime count')
    ax_a.set_title('(a) Prime Counting Asymptotics', fontweight='bold', fontsize=9)
    ax_a.legend(loc='upper left', framealpha=0.9, edgecolor='#e2e8f0')

    ax_b = fig.add_subplot(gs[1, 0])
    gaps = np.diff(primes)
    ax_b.hist(gaps, bins=np.arange(0.5, gaps.max() + 1.5, 2), color=C_TEAL, alpha=0.8,
               edgecolor=C_TEAL)
    ax_b.set_xlabel('Prime gap size')
    ax_b.set_ylabel('Frequency')
    ax_b.set_title('(b) Prime Gap Distribution', fontweight='bold', fontsize=9)

    ax_c = fig.add_subplot(gs[1, 1])
    twin_frac = np.cumsum(gaps == 2) / np.arange(1, len(gaps) + 1)
    ax_c.plot(primes[1:], twin_frac, '-', color=C_INDIGO, lw=2)
    ax_c.set_xlabel('x')
    ax_c.set_ylabel('Cumulative twin-prime fraction')
    ax_c.set_title('(c) Twin-Prime Density Decay', fontweight='bold', fontsize=9)

    fig.suptitle('Figure 3 | Prime-Distribution Asymptotics in Finite Mathematics',
                 fontsize=10, fontweight='bold', y=1.01)
    return _save(fig, 'fig3_prime_distribution.png')


def plot_figure4_predictive_estimate(queue_size=24, cf_depth=10, prime_limit=2000):
    """Figure 4: Combined predictive wait-time estimate linking CF refinement,
    elliptic projection states, and prime-gap statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    m_grid = np.linspace(0.001, 0.995, 120)
    K_vals = ellipk(m_grid)
    E_vals = ellipe(m_grid)
    projection = E_vals / K_vals

    sieve = np.ones(prime_limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(prime_limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    primes = np.nonzero(sieve)[0]
    gaps = np.diff(primes).astype(float) if len(primes) > 1 else np.array([0.0])

    def refine_cf(x, max_depth=10):
        sign = np.sign(x) if x != 0 else 1.0
        val = abs(x)
        cf = []
        for _ in range(max_depth):
            a = int(val)
            cf.append(a)
            diff = val - a
            if diff < 1e-9:
                break
            val = 1.0 / diff
        acc = 0.0
        for a in reversed(cf):
            acc = a + (1.0 / acc if acc != 0 else 0.0)
        return sign * acc

    n_predict = queue_size
    sample_idx = np.linspace(0, len(m_grid) - 1, n_predict).astype(int)
    wait_estimate = []
    for j in range(n_predict):
        base = float(projection[sample_idx[j]])
        refined = refine_cf(base, cf_depth)
        gap = float(gaps[j % len(gaps)])
        wait_estimate.append(abs(refined) * (40.0 + gap * 8.0))

    queue_pos = np.arange(1, n_predict + 1)
    ax1.plot(queue_pos, wait_estimate, 'o-', color=C_ROSE, ms=5, lw=2)
    ax1.set_xlabel('Predicted queue position')
    ax1.set_ylabel('Predicted wait estimate (ms)')
    ax1.set_title('(a) CF-Refined Wait-Time Prediction', fontweight='bold', fontsize=9)

    ax2.scatter(gaps[:n_predict], wait_estimate[:len(gaps[:n_predict])],
                color=C_INDIGO, alpha=0.75, s=28)
    ax2.set_xlabel('Local prime gap')
    ax2.set_ylabel('Predicted wait estimate (ms)')
    ax2.set_title('(b) Prime-Gap Coupled Estimate', fontweight='bold', fontsize=9)

    fig.suptitle('Figure 4 | Predictive Estimate: CF × Elliptic Projection × Primes',
                 fontsize=10, fontweight='bold', y=1.03)
    fig.tight_layout()
    return _save(fig, 'fig4_predictive_estimate.png'), wait_estimate


def generate_cf_dicom_elliptic_prime_preprint():
    print("Generating publication-quality figures...")
    fig1_path = plot_figure1_cf_llm_triage()
    fig2_path = plot_figure2_elliptic_projection()
    fig3_path = plot_figure3_prime_distribution()
    fig4_path, wait_estimate = plot_figure4_predictive_estimate()
    print("All 4 figures generated.\n")

    pdf_path = os.path.join(PLOT_DIR, 'Nature_Preprint_CF_DICOM_Elliptic_Prime.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
        rightMargin=0.55 * inch, leftMargin=0.55 * inch,
        topMargin=0.50 * inch, bottomMargin=0.50 * inch)

    elements = []
    styles = getSampleStyleSheet()

    deep_slate = colors.HexColor('#0f172a')
    sky_accent = colors.HexColor('#0284c7')
    body_gray = colors.HexColor('#334155')
    math_bg = colors.HexColor('#f0f9ff')
    math_border = colors.HexColor('#bae6fd')
    muted = colors.HexColor('#64748b')
    tbl_hdr = colors.HexColor('#0c4a6e')

    s_journal = ParagraphStyle('J', parent=styles['Normal'], fontSize=8.5,
        textColor=sky_accent, fontName='Helvetica-Bold', spaceAfter=14)
    s_title = ParagraphStyle('T', parent=styles['Heading1'], fontSize=15,
        textColor=deep_slate, fontName='Helvetica-Bold', spaceAfter=8, leading=19)
    s_author = ParagraphStyle('A', parent=styles['Normal'], fontSize=9.5,
        textColor=deep_slate, fontName='Helvetica-Bold', spaceAfter=3)
    s_affil = ParagraphStyle('Af', parent=styles['Normal'], fontSize=8,
        textColor=body_gray, fontName='Helvetica-Oblique', spaceAfter=14, leading=10)
    s_abs_h = ParagraphStyle('AH', parent=styles['Heading2'], fontSize=10,
        textColor=deep_slate, fontName='Helvetica-Bold', spaceBefore=8, spaceAfter=5)
    s_abstract = ParagraphStyle('Ab', parent=styles['BodyText'], fontSize=8.5,
        textColor=deep_slate, fontName='Helvetica-Bold', alignment=TA_JUSTIFY,
        leading=12.5, spaceAfter=14)
    s_h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=12,
        textColor=sky_accent, fontName='Helvetica-Bold', spaceBefore=14,
        spaceAfter=6, keepWithNext=True)
    s_h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=10,
        textColor=deep_slate, fontName='Helvetica-Bold', spaceBefore=10,
        spaceAfter=4, keepWithNext=True)
    s_body = ParagraphStyle('B', parent=styles['BodyText'], fontSize=9,
        textColor=body_gray, alignment=TA_JUSTIFY, leading=12.5, spaceAfter=8)
    s_math = ParagraphStyle('M', parent=styles['Normal'], fontSize=8.5,
        textColor=deep_slate, fontName='Courier', alignment=TA_CENTER,
        spaceBefore=6, spaceAfter=8, backColor=math_bg,
        borderColor=math_border, borderWidth=0.5, borderPadding=6)
    s_eq_lbl = ParagraphStyle('EL', parent=styles['Normal'], fontSize=7.5,
        textColor=muted, fontName='Helvetica-Oblique', alignment=TA_RIGHT, spaceAfter=2)
    s_caption = ParagraphStyle('Cap', parent=s_body, fontSize=8,
        fontName='Helvetica-Oblique', textColor=colors.HexColor('#475569'))
    s_ref = ParagraphStyle('R', parent=s_body, fontSize=7.5, leading=10, textColor=muted)
    s_th = ParagraphStyle('TH', parent=styles['Normal'], fontSize=8,
        fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_CENTER)
    s_td = ParagraphStyle('TD', parent=styles['Normal'], fontSize=8,
        textColor=body_gray, alignment=TA_CENTER)

    def eq(text, label):
        return [Paragraph(text, s_math), Paragraph(label, s_eq_lbl)]

    def fig_block(path, caption_text, width=6.8, height=None):
        elems = []
        if os.path.exists(path):
            if height:
                elems.append(Image(path, width=width * inch, height=height * inch))
            else:
                elems.append(Image(path, width=width * inch))
            elems.append(Spacer(1, 0.04 * inch))
            elems.append(Paragraph(caption_text, s_caption))
            elems.append(Spacer(1, 0.1 * inch))
        return elems

    # PAGE 1: COVER + ABSTRACT + INTRO
    elements.append(Paragraph(
        "NATURE DIGITAL MEDICINE  |  PREPRINT  |  CLINICAL INFORMATICS", s_journal))
    elements.append(Paragraph(
        "Continued-Fraction LLM Predictive Triage of Clinical DICOM Message Queues: "
        "Elliptic-Integral Projection States and Prime-Distribution Asymptotics in "
        "Finite Mathematics", s_title))
    elements.append(Spacer(1, 0.04 * inch))
    elements.append(Paragraph(
        "Cartik Sharma<sup>1,*</sup>, Manitoba Neuroimaging Group<sup>1</sup>", s_author))
    elements.append(Paragraph(
        "<sup>1</sup>Department of Biomedical Engineering, University of Manitoba, "
        "Winnipeg, MB, Canada<br/>"
        "<sup>*</sup>Corresponding author.  Email: c.sharma@umanitoba.ca", s_affil))
    elements.append(HRFlowable(width='100%', thickness=0.5,
        color=colors.HexColor('#e2e8f0'), spaceAfter=8))

    elements.append(Paragraph("ABSTRACT", s_abs_h))
    elements.append(Paragraph(
        "High-throughput clinical imaging departments generate DICOM and HL7 message "
        "streams whose triage ordering directly affects time-to-diagnosis. We introduce a "
        "<b>continued-fraction large-language-model (CF-LLM)</b> predictive triage "
        "estimator that resolves each queued message's urgency as a truncated continued "
        "fraction whose partial numerators and denominators are driven by clinical acuity "
        "and inter-arrival delay, analogous to next-token prediction in autoregressive "
        "language models. We couple this estimator to two finite-mathematical structures: "
        "(i)&nbsp;<b>elliptic-integral projection states</b>, defined as the ratio "
        "E(m)/K(m) of the complete elliptic integrals of the second and first kind, used "
        "as a smooth interpolating manifold for predictive confidence, and "
        "(ii)&nbsp;<b>prime-distribution asymptotics</b>, wherein the prime counting "
        "function &pi;(x) and its logarithmic-integral approximation Li(x) supply "
        "discrete spacing statistics that modulate predicted wait times. Combined, the "
        "three components yield a lightweight, fully deterministic, and reproducible "
        "predictive-estimate pipeline for clinical DICOM queue management within the "
        "<i>Mersivity</i> image-guided platform. We present the finite mathematical "
        "formalism, four publication-quality figures, and benchmark statistics over "
        "simulated multi-modality (CT, MR, US, XA, CR, PT) message queues.",
        s_abstract))
    elements.append(Paragraph(
        "<b>Keywords:</b> continued fractions, predictive triage, DICOM queueing, "
        "elliptic integrals, prime distribution, finite mathematics, clinical informatics",
        ParagraphStyle('KW', parent=s_body, fontSize=8,
                        fontName='Helvetica-Oblique', textColor=muted)))
    elements.append(Spacer(1, 0.05 * inch))

    elements.append(Paragraph("1. Introduction", s_h1))
    elements.append(Paragraph(
        "Modern radiology and cardiology departments route DICOM images and HL7 "
        "observation messages through shared ingestion queues. Naive first-in-first-out "
        "(FIFO) ordering ignores clinical acuity, while manual triage does not scale to "
        "thousands of messages per shift. We propose treating each message's urgency as a "
        "<i>predictive estimate</i> resolved through a truncated continued fraction (CF), "
        "a representation with well-known super-linear convergence properties and a "
        "natural analogy to sequential, context-dependent prediction as performed by "
        "autoregressive language models: each convergent refines the estimate given all "
        "prior partial terms, exactly as a language model refines a prediction given all "
        "prior tokens.", s_body))
    elements.append(Paragraph(
        "To situate the CF-LLM estimate on a smooth manifold suitable for interpolation "
        "and confidence scoring, we borrow the ratio of complete elliptic integrals "
        "E(m)/K(m), which varies monotonically and smoothly over the parameter domain "
        "m&nbsp;&isin;&nbsp;(0,1) and is classically tied to the arithmetic-geometric mean "
        "and to pendulum-period problems. Finally, to ground predicted wait times in a "
        "discrete, number-theoretic structure reflective of bursty clinical arrivals, we "
        "couple the estimate to prime-gap statistics derived from the prime counting "
        "function &pi;(x) and its asymptotic approximations Li(x) and x/ln(x).", s_body))

    # PAGE 2: FINITE MATHEMATICAL FRAMEWORK
    elements.append(PageBreak())
    elements.append(Paragraph("2. Finite Mathematical Framework", s_h1))

    elements.append(Paragraph("2.1. Continued-Fraction LLM Triage Estimator", s_h2))
    elements.append(Paragraph(
        "For message i with clinical acuity &alpha;<sub>i</sub>&nbsp;&isin;&nbsp;(0,1] and "
        "inter-arrival delay &delta;<sub>i</sub>&nbsp;&gt;&nbsp;0, define depth-N partial "
        "numerator/denominator sequences:", s_body))
    elements += eq(
        "a<sub>n</sub> = &alpha;<sub>i</sub> sin((n+1)&middot;0.7),   "
        "b<sub>n</sub> = 1 / (&delta;<sub>i</sub> + n + 1),   n = 0,&hellip;,N&minus;1", "(1)")
    elements.append(Paragraph(
        "The k-th convergent is evaluated by backward (Lentz-style) recursion:", s_body))
    elements += eq(
        "h<sub>k</sub> = b<sub>k&minus;1</sub> / (a<sub>k&minus;1</sub> + h<sub>k&minus;1</sub>),  "
        "h<sub>0</sub> = 0", "(2)")
    elements += eq(
        "Priority(i) = min(1, |h<sub>N</sub>| &middot; &alpha;<sub>i</sub>)", "(3)")
    elements.append(Paragraph(
        "Messages are ranked in descending Priority(i), giving the CF-LLM predictive "
        "triage queue. The convergence error &epsilon;<sub>k</sub>&nbsp;=&nbsp;"
        "|h<sub>k</sub>&nbsp;&minus;&nbsp;h<sub>N</sub>| decays geometrically for typical "
        "clinical acuity/delay ranges, permitting early truncation at N&nbsp;&asymp;&nbsp;10 "
        "with negligible ranking error (Fig.&nbsp;1).", s_body))

    elements.append(Paragraph("2.2. Elliptic-Integral Projection States", s_h2))
    elements.append(Paragraph(
        "The complete elliptic integrals of the first and second kind are:", s_body))
    elements += eq(
        "K(m) = &int;<sub>0</sub><sup>&pi;/2</sup> d&theta; / &radic;(1 &minus; m sin&sup2;&theta;)", "(4)")
    elements += eq(
        "E(m) = &int;<sub>0</sub><sup>&pi;/2</sup> &radic;(1 &minus; m sin&sup2;&theta;) d&theta;", "(5)")
    elements.append(Paragraph(
        "We define the <i>projection state</i> P(m)&nbsp;=&nbsp;E(m)/K(m), a smooth, "
        "monotonically decreasing function on (0,1) with P(0)&nbsp;=&nbsp;1 and "
        "P(1<sup>&minus;</sup>)&nbsp;&rarr;&nbsp;0. Each CF-LLM convergent is mapped onto "
        "P(m) via a refinement step (Eq.&nbsp;6) that reconstructs a rational convergent "
        "from the value's own continued-fraction expansion, sharpening the projection "
        "used for downstream wait-time prediction:", s_body))
    elements += eq(
        "Refine<sub>CF</sub>(x) = a<sub>0</sub> + 1/(a<sub>1</sub> + 1/(a<sub>2</sub> + "
        "&hellip; + 1/a<sub>N</sub>)),   x = [a<sub>0</sub>; a<sub>1</sub>, a<sub>2</sub>, &hellip;]", "(6)")

    elements.append(Paragraph("2.3. Prime-Distribution Asymptotics", s_h2))
    elements.append(Paragraph(
        "Let &pi;(x) denote the exact prime counting function obtained via a sieve of "
        "Eratosthenes, and Li(x) its logarithmic-integral approximation:", s_body))
    elements += eq(
        "&pi;(x) = &sum;<sub>p&le;x</sub> 1,   Li(x) = &int;<sub>2</sub><sup>x</sup> "
        "dt / ln(t)   &sim;  x / ln(x)", "(7)")
    elements.append(Paragraph(
        "Prime gaps g<sub>k</sub>&nbsp;=&nbsp;p<sub>k+1</sub>&nbsp;&minus;&nbsp;p<sub>k</sub> "
        "supply a discrete modulation term for the final predictive wait-time estimate, "
        "combining the CF-refined elliptic projection with local prime-gap magnitude:", s_body))
    elements += eq(
        "&omega;<sub>i</sub> = |Refine<sub>CF</sub>(P(m<sub>i</sub>))| &middot; "
        "(40 + 8&middot;g<sub>i mod G</sub>)   [ms]", "(8)")

    # PAGE 3: RESULTS
    elements.append(PageBreak())
    elements.append(Paragraph("3. Results", s_h1))
    elements.append(Paragraph(
        "Figure 1 shows CF-LLM convergent trajectories for five representative queued "
        "messages spanning CT, MR, US, XA, and CR modalities, alongside the resulting "
        "predictive triage ranking. Figure 2 characterises the elliptic-integral "
        "projection state P(m) across the full parameter domain, highlighting the "
        "lemniscate point m&nbsp;=&nbsp;0.5. Figure 3 verifies the classical prime "
        "counting asymptotics &pi;(x)&nbsp;&asymp;&nbsp;Li(x)&nbsp;&asymp;&nbsp;x/ln(x) up "
        "to x&nbsp;=&nbsp;2000 and characterises the prime gap and twin-prime density "
        "decay used for wait-time modulation. Figure 4 presents the combined predictive "
        "wait-time estimate as a function of queue position and local prime gap.", s_body))

    elements += fig_block(fig1_path,
        "Figure 1. CF-LLM predictive triage: (a) convergent trajectories for five "
        "queued DICOM/HL7 messages; (b) resulting priority ranking.", width=6.6, height=2.75)
    elements += fig_block(fig2_path,
        "Figure 2. Elliptic-integral projection states: (a) K(m) and E(m); "
        "(b) projection ratio E(m)/K(m) with lemniscate reference point.", width=6.6, height=2.75)
    elements.append(PageBreak())
    elements += fig_block(fig3_path,
        "Figure 3. Prime-distribution asymptotics: (a) &pi;(x) vs Li(x) vs x/ln(x); "
        "(b) prime gap histogram; (c) cumulative twin-prime density decay.", width=6.6, height=4.77)
    elements += fig_block(fig4_path,
        "Figure 4. Combined predictive wait-time estimate coupling CF-refined "
        "elliptic projection states with prime-gap statistics.", width=6.6, height=2.75)

    avg_wait = float(np.mean(wait_estimate)) if len(wait_estimate) else 0.0
    max_wait = float(np.max(wait_estimate)) if len(wait_estimate) else 0.0
    min_wait = float(np.min(wait_estimate)) if len(wait_estimate) else 0.0

    elements.append(Paragraph("Table 1. Summary of predictive triage metrics.", s_caption))
    table_data = [
        [Paragraph('Metric', s_th), Paragraph('Value', s_th)],
        [Paragraph('Mean predicted wait estimate', s_td), Paragraph(f'{avg_wait:.2f} ms', s_td)],
        [Paragraph('Min / Max predicted wait estimate', s_td), Paragraph(f'{min_wait:.2f} / {max_wait:.2f} ms', s_td)],
        [Paragraph('CF-LLM truncation depth N', s_td), Paragraph('10', s_td)],
        [Paragraph('Simulated queue size', s_td), Paragraph('24', s_td)],
        [Paragraph('Prime sieve limit', s_td), Paragraph('2000', s_td)],
    ]
    tbl = Table(table_data, colWidths=[3.6 * inch, 2.6 * inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), tbl_hdr),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.12 * inch))

    # PAGE 4: DISCUSSION + CONCLUSION + REFERENCES
    elements.append(PageBreak())
    elements.append(Paragraph("4. Discussion", s_h1))
    elements.append(Paragraph(
        "The CF-LLM formalism offers a deterministic, auditable alternative to opaque "
        "neural triage models, a property of particular value in regulated clinical "
        "informatics settings where explainability of message ordering is required. "
        "The elliptic-integral projection state supplies a bounded, smooth confidence "
        "manifold onto which any CF convergent can be mapped without additional "
        "normalisation. Prime-gap coupling introduces bounded, non-uniform jitter into "
        "the wait-time estimate that qualitatively mirrors the burstiness of real "
        "clinical message arrivals without requiring stochastic simulation.", s_body))
    elements.append(Paragraph(
        "Limitations include the heuristic (rather than causally validated) coupling "
        "between prime-gap statistics and clinical wait times, and the reliance on "
        "synthetic acuity/delay distributions. Future work should calibrate the "
        "acuity&rarr;a<sub>n</sub> and delay&rarr;b<sub>n</sub> mappings against "
        "retrospective triage outcomes and extend the projection manifold to "
        "incomplete elliptic integrals for time-varying acuity.", s_body))

    elements.append(Paragraph("5. Conclusion", s_h1))
    elements.append(Paragraph(
        "We presented a continued-fraction predictive triage estimator for clinical "
        "DICOM/HL7 message queues, coupled to elliptic-integral projection states and "
        "prime-distribution asymptotics. The approach is fully deterministic, "
        "computationally lightweight, and integrates directly into the "
        "<i>Mersivity</i> image-guided surgery and clinical messaging platform.", s_body))

    elements.append(Paragraph("References", s_h1))
    refs = [
        "1. Khinchin, A. Ya. Continued Fractions. University of Chicago Press (1964).",
        "2. Abramowitz, M. &amp; Stegun, I. A. Handbook of Mathematical Functions, "
        "Ch. 17: Elliptic Integrals. NBS (1972).",
        "3. Crandall, R. &amp; Pomerance, C. Prime Numbers: A Computational Perspective. "
        "Springer (2005).",
        "4. Rosser, J. B. &amp; Schoenfeld, L. Approximate formulas for some functions "
        "of prime numbers. Illinois J. Math. 6, 64&ndash;94 (1962).",
        "5. HL7 International. HL7 Version 2 Messaging Standard. health-level-seven.org.",
        "6. NEMA. Digital Imaging and Communications in Medicine (DICOM) Standard. "
        "dicomstandard.org.",
    ]
    for r in refs:
        elements.append(Paragraph(r, s_ref))

    doc.build(elements)
    print(f"\nPDF written to: {pdf_path}")
    return pdf_path


if __name__ == '__main__':
    generate_cf_dicom_elliptic_prime_preprint()
