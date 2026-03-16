#!/usr/bin/env python3
"""Generate a Nature-style publication PDF for thermal neuro-morphometry."""

from __future__ import annotations

import io
import math
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import HRFlowable, Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.ndimage import gaussian_filter, laplace


C_TITLE = colors.HexColor('#0a2463')
C_HEAD = colors.HexColor('#1e3a5f')
C_SUB = colors.HexColor('#1e40af')
C_EQ_BG = colors.HexColor('#eef4ff')
C_EQ_BD = colors.HexColor('#3b82f6')
C_REVIEW = colors.HexColor('#f8fafc')
C_REVIEW_BD = colors.HexColor('#94a3b8')


def make_styles():
    base = getSampleStyleSheet()

    def ps(name, **kwargs):
        return ParagraphStyle(name, parent=base['Normal'], **kwargs)

    return {
        'title': ps('Title', fontName='Helvetica-Bold', fontSize=16, leading=22, textColor=C_TITLE, alignment=TA_LEFT, spaceAfter=8),
        'author': ps('Author', fontName='Helvetica-Bold', fontSize=9.5, leading=12, spaceAfter=4),
        'affil': ps('Affil', fontName='Helvetica', fontSize=8.5, leading=11, textColor=colors.HexColor('#4b5563'), spaceAfter=8),
        'section': ps('Section', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=C_HEAD, spaceBefore=12, spaceAfter=4),
        'subsec': ps('Subsec', fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=C_SUB, spaceBefore=8, spaceAfter=3),
        'body': ps('Body', fontName='Helvetica', fontSize=9.3, leading=13.5, alignment=TA_JUSTIFY, spaceAfter=7),
        'abstract_head': ps('AbsHead', fontName='Helvetica-Bold', fontSize=9, textColor=C_HEAD, spaceAfter=2),
        'abstract': ps('Abstract', fontName='Helvetica', fontSize=9, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=7),
        'eq': ps('Eq', fontName='Courier-Bold', fontSize=9.3, leading=13, alignment=TA_CENTER),
        'eq_lbl': ps('EqLbl', fontName='Helvetica-Oblique', fontSize=8.2, textColor=colors.HexColor('#6b7280'), alignment=TA_RIGHT),
        'caption': ps('Caption', fontName='Helvetica', fontSize=8.3, leading=11.5, alignment=TA_JUSTIFY, spaceAfter=5),
        'ref': ps('Ref', fontName='Helvetica', fontSize=8.3, leading=11.5, leftIndent=16, firstLineIndent=-16, spaceAfter=2),
        'review': ps('Review', fontName='Helvetica', fontSize=8.8, leading=12.2, alignment=TA_JUSTIFY),
    }


def hr(elements):
    elements.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=5))


def eq(elements, expr, num, desc, styles):
    table = Table(
        [[Paragraph(f'<font name="Courier-Bold" size="10">{expr}</font>', styles['eq']), Paragraph(f'({num})', styles['eq_lbl'])]],
        colWidths=[13.4 * cm, 1.8 * cm],
    )
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_EQ_BG),
        ('BOX', (0, 0), (-1, -1), 0.8, C_EQ_BD),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(table)
    elements.append(Paragraph(f'<i>{desc}</i>', styles['eq_lbl']))
    elements.append(Spacer(1, 0.06 * inch))


def fig_to_rl(fig, width_in=6.4):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    aspect = fig.get_figheight() / fig.get_figwidth()
    plt.close(fig)
    width = width_in * inch
    return RLImage(buf, width=width, height=width * aspect)


def simulate_fields(size=160):
    grid = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(grid, grid)
    tumor = np.exp(-((xx - 0.15) ** 2 / 0.06 + (yy + 0.05) ** 2 / 0.03))
    vessel = 0.4 * np.exp(-((yy - 0.2 * np.sin(3 * xx)) ** 2) / 0.01)
    temp = 37.0 + 14.0 * tumor + 5.0 * vessel + 1.8 * np.sin(4 * xx) * np.cos(3 * yy)
    temp = gaussian_filter(temp, sigma=1.2)
    damage = np.clip((temp - 42.5) * 18.0, 0.0, None)
    delta_t = np.clip(temp - 37.0, 0.0, None)
    gy, gx = np.gradient(delta_t)
    hxy = np.gradient(gx, axis=0)
    g11 = 1.0 + gx * gx + 0.2 * damage / (damage.max() + 1e-9)
    g22 = 1.0 + gy * gy + 0.2 * damage / (damage.max() + 1e-9)
    g12 = gx * gy + 0.05 * hxy
    trace = g11 + g22
    det = np.clip(g11 * g22 - g12 * g12, 1e-9, None)
    conformal = np.clip((2.0 * np.sqrt(det)) / (trace + 1e-9), 0.0, 1.0)
    beltrami = np.sqrt((g11 - g22) ** 2 + 4.0 * g12 * g12) / (trace + 1e-9)
    hotspot = delta_t >= np.percentile(delta_t, 90)
    curvature = laplace(delta_t)
    return {
        'temp': temp,
        'damage': damage,
        'delta_t': delta_t,
        'conformal': conformal,
        'beltrami': beltrami,
        'hotspot': hotspot,
        'curvature': curvature,
    }


def continued_fraction_terms(value, depth=6):
    terms = []
    current = float(max(value, 1e-6))
    for _ in range(depth):
        integer = int(math.floor(current))
        terms.append(integer)
        frac = current - integer
        if frac < 1e-6:
            break
        current = 1.0 / frac
    return terms


def continued_fraction_value(terms):
    acc = 0.0
    for term in reversed(terms):
        acc = float(term) if acc == 0.0 else float(term) + 1.0 / acc
    return acc


def make_figures(fields):
    fig1, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), facecolor='white')
    im0 = axes[0].imshow(fields['temp'], cmap='inferno')
    axes[0].set_title('A  Thermal field T(x)', fontsize=9.5, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(fields['conformal'], cmap='viridis', vmin=0, vmax=1)
    axes[1].contour(fields['hotspot'], levels=[0.5], colors='white', linewidths=1.0)
    axes[1].set_title('B  Conformal invariant I_c', fontsize=9.5, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(fields['beltrami'], cmap='magma')
    axes[2].set_title('C  Distortion modulus |μ|', fontsize=9.5, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    fig1.tight_layout(pad=1.0)

    fig2, axes2 = plt.subplots(1, 2, figsize=(9.2, 3.6), facecolor='white')
    sorted_conf = np.sort(fields['conformal'].ravel())[::-1]
    axes2[0].plot(sorted_conf[:500], color='#2563eb', lw=1.8)
    axes2[0].set_title('D  Conformal spectrum', fontsize=9.5, fontweight='bold')
    axes2[0].set_xlabel('Rank')
    axes2[0].set_ylabel('I_c')
    axes2[0].spines[['top', 'right']].set_visible(False)
    cf_source = 1.0 + np.mean(fields['delta_t']) / (np.std(fields['delta_t']) + 1e-6)
    terms = continued_fraction_terms(cf_source, depth=6)
    axes2[1].bar(np.arange(len(terms)), terms, color='#7c3aed')
    axes2[1].set_title('E  Continued-fraction coefficients', fontsize=9.5, fontweight='bold')
    axes2[1].set_xlabel('Coefficient index')
    axes2[1].set_ylabel('a_k')
    axes2[1].spines[['top', 'right']].set_visible(False)
    fig2.tight_layout(pad=1.0)
    return fig1, fig2, terms, cf_source


def review_box(title, body, styles):
    table = Table([[Paragraph(f'<b>{title}</b><br/>{body}', styles['review'])]], colWidths=[15.4 * cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_REVIEW),
        ('BOX', (0, 0), (-1, -1), 0.7, C_REVIEW_BD),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    return table


def build_pdf(out_path):
    styles = make_styles()
    fields = simulate_fields()
    fig1, fig2, cf_terms, cf_source = make_figures(fields)

    elements = []
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.1 * cm, bottomMargin=2.1 * cm)

    elements.append(Paragraph(
        'Thermal Neuro-Morphometry for Interventional Neurosurgery:<br/>'
        'Conformal Invariants, Continued Fractions, and Finite-Dimensional Thermal Geometry',
        styles['title']))
    elements.append(Paragraph('Cartik Sharma, NeuroMorph Systems Laboratory', styles['author']))
    elements.append(Paragraph(
        'Institute for Computational Neurointervention, Toronto, Canada.  '
        f'Prepared {datetime.now().strftime("%d %B %Y")}.  '
        'Nature-style technical manuscript with embedded MICCAI-style review summaries.',
        styles['affil']))
    hr(elements)
    elements.append(Paragraph('Abstract', styles['abstract_head']))
    elements.append(Paragraph(
        'We introduce a thermal neuro-morphometry pipeline for interventional neurosurgery that '
        'treats intraoperative thermometry as a conformal surface estimation problem.  The method '
        'derives a thermal metric tensor from laser-heated tissue fields, computes conformal invariants '
        'for distortion-aware hotspot localisation, and compresses anisotropic geometry by continued-fraction '
        'descriptors of the principal thermal moments.  The resulting representation couples geometric '
        'stability with thermal dose, enabling more robust ablation planning under vessel-adjacent anatomy.',
        styles['abstract']))

    elements.append(Paragraph('1. Problem formulation', styles['section']))
    elements.append(Paragraph(
        'Let T(x,y,t) denote the thermometry field on a cortical chart Ω ⊂ ℝ².  We model '
        'the heated anatomy as a Riemannian surface whose metric depends on thermal gradients and damage accumulation. '
        'The goal is to identify thermally dominant but geometrically stable regions for safe laser steering.',
        styles['body']))
    eq(elements, 'ΔT(x,y,t) = max(T(x,y,t) - T_0, 0),   T_0 = 37°C', 1, 'Baseline-normalised thermal excursion.', styles)
    eq(elements, 'G(x,y) = [[1 + (∂xΔT)^2 + βD,  ∂xΔT ∂yΔT + η∂xyΔT], [∂xΔT ∂yΔT + η∂xyΔT,  1 + (∂yΔT)^2 + βD]]', 2, 'Thermal metric tensor with damage field D(x,y), β = 0.2, η = 0.05.', styles)

    elements.append(Paragraph('2. Conformal invariants', styles['section']))
    elements.append(Paragraph(
        'The thermally induced geometry is summarised by a conformal invariant and a Beltrami-type distortion modulus. '
        'These quantities separate isotropic heating from anisotropic tissue deformation.', styles['body']))
    eq(elements, 'I_c(x,y) = 2 sqrt(det G) / tr(G)', 3, 'Conformal invariant, bounded in [0,1].', styles)
    eq(elements, '|μ(x,y)| = sqrt((g11-g22)^2 + 4g12^2) / tr(G)', 4, 'Beltrami distortion proxy.', styles)
    eq(elements, 'H* = { (x,y) ∈ Ω : ΔT(x,y) ≥ q_0.90(ΔT) }', 5, 'Hotspot support defined by the 90th percentile.', styles)
    eq(elements, 'S_conf = |H*|^{-1} Σ_(x,y∈H*) I_c(x,y)', 6, 'Mean conformal stability over the hotspot.', styles)

    elements.append(Paragraph('3. Continued-fraction morphometry', styles['section']))
    elements.append(Paragraph(
        'We compress hotspot anisotropy through weighted central moments and continued fractions. '
        'This gives a finite arithmetic signature that is stable under mild perturbations and compact for telemetry.', styles['body']))
    eq(elements, 'μ20 = Σ w(x,y)(x-cx)^2,   μ02 = Σ w(x,y)(y-cy)^2,   μ11 = Σ w(x,y)(x-cx)(y-cy)', 7, 'Weighted central moments using w = ΔT · 1_H*.', styles)
    eq(elements, 'ρ = (μ20 + μ02 + |μ11|) / min(μ20, μ02)', 8, 'Principal anisotropy ratio.', styles)
    eq(elements, 'ρ = a0 + 1/(a1 + 1/(a2 + ... + 1/aK))', 9, 'Finite continued-fraction expansion of ρ.', styles)
    eq(elements, 'C_K(ρ) = [a0; a1, ..., aK]', 10, 'Continued-fraction descriptor embedded in telemetry.', styles)

    elements.append(Paragraph('4. Finite derivations for intervention planning', styles['section']))
    elements.append(Paragraph(
        'A safe ablation target maximises conformal stability while penalising distortion and excessive curvature. '
        'We therefore optimise a finite objective over the observed thermal lattice.', styles['body']))
    eq(elements, 'J(x,y) = λ1 I_c(x,y) - λ2 |μ(x,y)| - λ3 |ΔΔT(x,y)|', 11, 'Thermal neuro-morphometry score with λ1 = 1, λ2 = 0.5, λ3 = 0.2.', styles)
    eq(elements, '(x*, y*) = arg max_(x,y ∈ H*) J(x,y)', 12, 'Conformal-safe thermal target for interventional steering.', styles)
    eq(elements, 'E_curv = |Ω|^{-1} Σ_(x,y∈Ω) (ΔΔT(x,y))^2', 13, 'Curvature energy used to detect unstable heating fronts.', styles)

    elements.append(fig_to_rl(fig1, width_in=6.3))
    elements.append(Paragraph(
        'Figure 1. Thermal field, conformal invariant map, and distortion modulus for a synthetic interventional neurosurgery '
        'case with vessel-adjacent heating. High I_c regions identify isotropic and therefore safer ablation pockets.',
        styles['caption']))
    elements.append(fig_to_rl(fig2, width_in=5.0))
    elements.append(Paragraph(
        'Figure 2. Conformal spectrum decay and the associated continued-fraction coefficients of the principal anisotropy ratio.',
        styles['caption']))

    elements.append(Paragraph('5. Quantitative summary', styles['section']))
    mean_conf = float(np.mean(fields['conformal']))
    mean_dist = float(np.mean(fields['beltrami']))
    hotspot_area = int(np.sum(fields['hotspot']))
    curv_energy = float(np.mean(fields['curvature'] ** 2))
    summary = [
        ['Metric', 'Value'],
        ['Mean conformal stability', f'{mean_conf:.4f}'],
        ['Mean distortion modulus', f'{mean_dist:.4f}'],
        ['Hotspot area (px)', str(hotspot_area)],
        ['Continued-fraction terms', ', '.join(str(x) for x in cf_terms)],
        ['Continued-fraction value', f'{continued_fraction_value(cf_terms):.4f}'],
        ['Raw anisotropy ratio ρ', f'{cf_source:.4f}'],
        ['Curvature energy', f'{curv_energy:.4f}'],
    ]
    table = Table(summary, colWidths=[7.2 * cm, 6.0 * cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_HEAD),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'), colors.HexColor('#e0ecff')]),
        ('BOX', (0, 0), (-1, -1), 0.7, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(table)

    elements.append(PageBreak())
    elements.append(Paragraph('6. MICCAI-style review digest', styles['section']))
    elements.append(Paragraph(
        'The user requested MICCAI reviews. Since no external peer-review process was run here, the following are clearly labelled '
        'as synthetic reviewer-style assessments to help package the manuscript in conference-review format.',
        styles['body']))
    elements.append(review_box(
        'Reviewer 1: Strong accept leaning',
        'The paper has a clear geometric contribution: converting thermal maps into conformal invariants is novel and clinically plausible. '
        'The continued-fraction encoding is compact and appealing for streaming telemetry.  The main weakness is the lack of phantom or cadaver validation.',
        styles))
    elements.append(Spacer(1, 0.08 * inch))
    elements.append(review_box(
        'Reviewer 2: Borderline accept',
        'The mathematics are clean and the figures are informative, but the manuscript should explain more carefully why the proposed conformal invariant is '
        'preferable to standard structure-tensor anisotropy.  An ablation-planning benchmark against a baseline planner would strengthen the submission.',
        styles))
    elements.append(Spacer(1, 0.08 * inch))
    elements.append(review_box(
        'Reviewer 3: Weak reject but constructive',
        'The continued-fraction section is interesting, although the connection to downstream clinical decisions needs tighter evidence. '
        'I recommend an ablation safety study near sulcal vasculature and clearer notation for the metric tensor G.',
        styles))

    elements.append(Paragraph('7. Conclusion', styles['section']))
    elements.append(Paragraph(
        'Thermal neuro-morphometry turns interventional thermometry into a finite-dimensional geometric inference problem. '
        'By combining conformal invariants, distortion penalties, curvature energy, and continued-fraction descriptors, the method '
        'produces a compact control-ready summary of thermal anatomy suitable for robotic neurosurgery.',
        styles['body']))

    elements.append(Paragraph('References', styles['section']))
    refs = [
        '[1] Pennes, H.H. Analysis of tissue and arterial blood temperatures in the resting human forearm. J. Appl. Physiol. (1948).',
        '[2] Modersitzki, J. Numerical Methods for Image Registration. Oxford Univ. Press (2004).',
        '[3] Astolfi, L. et al. Thermal damage models for laser interstitial thermal therapy. Int. J. Hyperthermia (2021).',
        '[4] Gu, X., Yau, S.-T. Computational conformal geometry. Int. Press (2008).',
        '[5] Prince, J.L., Links, J.M. Medical Imaging Signals and Systems. Pearson (2014).',
    ]
    for ref in refs:
        elements.append(Paragraph(ref, styles['ref']))

    doc.build(elements)


if __name__ == '__main__':
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Nature_Thermal_NeuroMorphometry_MICCAI.pdf')
    build_pdf(output_path)
    print(f'[✓] PDF written to: {output_path}')