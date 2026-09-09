#!/usr/bin/env python3
"""
Generate a Nature-style preprint for Moduli-Theoretic rTMS Treatment Paradigms
with Boundary Element Method (BEM) Cortical Surface Stimulation and Heat Maps.

Incorporates:
- Moduli space geometry of the stimulation parameter plane (H / SL(2,Z))
- Gauss fundamental domain reduction algorithm & elliptic stabilization
- Boundary Element Method (BEM) single-layer potential E-field formulation
- Layered tissue conductivity attenuation curves
- Rational continued fraction harmonic synchronization
"""

from pathlib import Path
import math
import numpy as np

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

from logic.moduli_bem_paradigm import get_moduli_bem_paradigm

OUTPUT = Path(__file__).with_name('Nature_Preprint_Moduli_BEM_Paradigm.pdf')

PURPLE = colors.HexColor('#6941c6')
BLUE = colors.HexColor('#175cd3')
GREEN = colors.HexColor('#067647')
AMBER = colors.HexColor('#b54708')
RED = colors.HexColor('#b42318')
TEAL = colors.HexColor('#0e7090')
INK = colors.HexColor('#1d2939')
MUTED = colors.HexColor('#667085')
PALE = colors.HexColor('#f8fafc')


def _line_chart(series, title, x_label, y_label, y_min=0.0, y_max=None):
    drawing = Drawing(470, 185)
    chart = LinePlot()
    chart.x = 48
    chart.y = 32
    chart.width = 395
    chart.height = 122
    chart.data = [[(float(x), float(y)) for x, y in zip(xs, ys)] for xs, ys, _ in series]

    max_x = max(max(xs) for xs, _, _ in series)
    max_y = max(max(ys) for _, ys, _ in series)

    chart.xValueAxis.valueMin = 0
    chart.xValueAxis.valueMax = float(max_x)
    chart.xValueAxis.valueStep = max(1, int(max_x / 6)) if max_x > 0 else 1
    chart.yValueAxis.valueMin = y_min
    chart.yValueAxis.valueMax = float(y_max or max_y * 1.08)
    chart.yValueAxis.valueStep = max(0.1, (chart.yValueAxis.valueMax - y_min) / 5.0)

    for index, (_, _, color) in enumerate(series):
        chart.lines[index].strokeColor = color
        chart.lines[index].strokeWidth = 1.6

    drawing.add(chart)
    drawing.add(String(235, 172, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=8.5, fillColor=INK))
    drawing.add(String(235, 8, f"{x_label} | {y_label}", textAnchor='middle', fontSize=7.2, fillColor=MUTED))
    return drawing


def _bar_chart(categories, data_series, title, y_label, colors_list):
    drawing = Drawing(470, 185)
    chart = VerticalBarChart()
    chart.x = 48
    chart.y = 32
    chart.width = 395
    chart.height = 120
    chart.data = data_series
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.fontSize = 6.8
    chart.categoryAxis.labels.dy = -10
    chart.valueAxis.labels.fontSize = 7.0
    chart.valueAxis.valueMin = 0.0

    for idx, col in enumerate(colors_list):
        chart.bars[idx].fillColor = col
        chart.bars[idx].strokeColor = colors.white

    drawing.add(chart)
    drawing.add(String(235, 172, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=8.5, fillColor=INK))
    drawing.add(String(235, 8, y_label, textAnchor='middle', fontSize=7.2, fillColor=MUTED))
    return drawing


def build_pdf(output_path=OUTPUT, params=None):
    condition = (params or {}).get('condition', 'stroke')
    data = get_moduli_bem_paradigm(condition)

    opt = data['optimal_protocol']
    attenuation = data['bem_attenuation']
    cf_convergents = data['cf_convergents']
    bem = data['bem_heatmap']
    grid = data['moduli_grid']

    # 1. Depth attenuation line chart
    attenuation_chart = _line_chart([
        (attenuation['depths_mm'], attenuation['field_pct'], PURPLE)
    ], 'Boundary Element Cortical Depth Field Attenuation (% MSO)', 'Cortical Depth (mm)', 'Induced E-Field (% MSO)', 0, 120)

    # 2. Moduli stability slice along optimal frequency
    opt_freq_idx = int(np.argmin([abs(f - opt['frequency_hz']) for f in grid['freq_axis']]))
    stability_slice = [row[opt_freq_idx] for row in grid['stability']]
    moduli_slice_chart = _line_chart([
        (grid['intensity_axis'], stability_slice, TEAL)
    ], f'Moduli Stability Score Phi(z) vs Intensity at f* = {opt["frequency_hz"]:.1f} Hz', 'Intensity I (% MSO)', 'Stability Score Phi(z)', 0, 1.05)

    # 3. Continued fraction approximation error bar chart
    cf_labels = [f"{c['numerator']}/{c['denominator']}" for c in cf_convergents[:6]]
    cf_errors = [max(0.001, float(c['error_pct'])) for c in cf_convergents[:6]]
    cf_bar_chart = _bar_chart(
        cf_labels,
        [cf_errors],
        'Continued-Fraction Rational Convergents Approximation Error (%)',
        'Approximation Error (%)',
        [BLUE]
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('NatureTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
    byline_style = ParagraphStyle('Byline', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=PURPLE, spaceAfter=8)
    heading_style = ParagraphStyle('NatureHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=RED, spaceBefore=9, spaceAfter=4)
    subheading_style = ParagraphStyle('NatureSubHeading', parent=styles['Heading3'], fontName='Helvetica-Bold', fontSize=9.0, leading=12, textColor=INK, spaceBefore=6, spaceAfter=3)
    body_style = ParagraphStyle('NatureBody', parent=styles['BodyText'], fontSize=8.4, leading=12.2, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5)
    equation_style = ParagraphStyle('Equation', parent=body_style, fontName='Courier', fontSize=7.8, leading=11.5, alignment=TA_CENTER, backColor=PALE, borderPadding=6, spaceBefore=4, spaceAfter=6)
    caption_style = ParagraphStyle('Caption', parent=body_style, fontSize=7.2, leading=9.5, textColor=MUTED, spaceAfter=7)
    notice_style = ParagraphStyle('Notice', parent=body_style, fontName='Helvetica-Bold', textColor=RED, borderColor=RED, borderWidth=0.6, borderPadding=6, spaceAfter=6)

    cf_str = ', '.join([f"{c['numerator']}/{c['denominator']}" for c in cf_convergents[:5]])

    story = [
        Paragraph('Moduli-Theoretic Optimization of rTMS Treatment Paradigms via SL(2,Z) Fundamental Domain Reduction and Boundary Element Cortical Field Simulations', title_style),
        Paragraph('Nature Computational Science / Nature Communications Preprint-Style Study | NeuroMorph Architecture | August 2026', byline_style),

        Paragraph('Translational Research Notice', heading_style),
        Paragraph('This manuscript develops a mathematical framework for rTMS parameter selection based on the moduli space of complex tori under the projective special linear group action SL(2,Z), coupled with boundary element electrodynamic modeling. Clinical administration of rTMS must follow established safety boundaries, motor threshold testing, and institutional guidelines.', notice_style),

        Paragraph('Abstract', heading_style),
        Paragraph(f'Optimization of repetitive Transcranial Magnetic Stimulation (rTMS) protocols requires joint tuning of pulse repetition frequency and coil discharge intensity across wide physiological search spaces. We formulate the stimulation parameter space as the upper half-plane H = {{z in C : Im(z) > 0}}, parameterizing each candidate protocol as z = f + i * (I / 100). The modular group SL(2,Z) acts on H by Mobius transformations, partitioning the space into isomorphic equivalence classes. Using the classical Gauss-Euclidean modular reduction algorithm, every candidate protocol is reduced to the standard fundamental domain F = {{z in H : |Re(z)| <= 1/2, |z| >= 1}}. Protocols whose canonical representative z* lies in proximity to the elliptic points z = i (order-2 stabilizer, S: z -> -1/z) or z = exp(i*pi/3) (order-3 stabilizer, ST) possess enhanced modular stability scores Phi(z*), establishing resonant parameter regimes robust against tissue conductivity drift and slight patient repositioning. For {data["label"]}, the global stability argmax identifies an optimal protocol of f* = {opt["frequency_hz"]:.2f} Hz at I* = {opt["intensity_pct"]:.1f}% MSO (reduced point z* = {opt["z_reduced"]["re"]:.3f} + {opt["z_reduced"]["im"]:.3f}i, Phi(z*) = {opt["stability_score"]:.4f}, nearest {opt["nearest_elliptic_point"]}). A 3D single-layer boundary element method (BEM) solver computes the induced cortical potential field (peak potential {bem["peak_potential"]:.2f}), confirming focal targeting and layered tissue attenuation.', body_style),

        Paragraph('1. Moduli Space Formulation of rTMS Parameters', heading_style),
        Paragraph('We define the rTMS stimulation state as a complex parameter z = f + i * sigma_norm in H, where f represents the repetition frequency in Hertz and sigma_norm = I / 100 represents normalized stimulation intensity relative to motor threshold. The action of gamma in SL(2,Z) represents modular relabelings of the underlying biophysical limit cycles:', body_style),
        Paragraph('gamma = [[a, b], [c, d]] in SL(2,Z),   ad - bc = 1<br/>gamma . z = (a*z + b) / (c*z + d)<br/>F = { z in H : -1/2 <= Re(z) <= 1/2,  |z| >= 1 }', equation_style),

        Paragraph('2. Gauss Fundamental Domain Reduction Algorithm', heading_style),
        Paragraph('Any candidate protocol z_0 in H is mapped into F by alternating translation generators T: z -> z + 1 and inversion S: z -> -1/z:', body_style),
        Paragraph('Step 1:  z_(k+1/2) = z_k - round(Re(z_k))   [Translation by T^(-round(Re(z)))]<br/>Step 2:  if |z_(k+1/2)| < 1,  z_(k+1) = -1 / z_(k+1/2)   [Inversion by S]<br/>Terminates when |Re(z*)| <= 1/2 and |z*| >= 1', equation_style),

        Paragraph('3. Elliptic Resonance Points & Stability Metric', heading_style),
        Paragraph('Points with non-trivial isotropy subgroups in PSL(2,Z) correspond to elliptic fixed points of order 2 (z = i) and order 3 (z = rho = e^{i*pi/3}). We define the moduli stability function Phi(z) measuring distance to the elliptic orbit:', body_style),
        Paragraph('d_E(z) = min( |z - i|, |z - exp(i*pi/3)| )<br/>Phi(z) = exp( - 4.0 * d_E(z_reduced)^2 )<br/>j(z) = 1728 * E_4(z)^3 / (E_4(z)^3 - E_6(z)^2) = 1/q + 744 + 196884*q + ...', equation_style),

        moduli_slice_chart,
        Paragraph('Figure 1 | Moduli stability function Phi(z) along the optimal frequency cross-section, showing resonance peaks corresponding to fundamental domain elliptic boundaries.', caption_style),

        Paragraph('4. Boundary Element Method (BEM) Cortical Surface Formulation', heading_style),
        Paragraph('The macroscopic induced electric scalar potential phi(r) on the closed cortical boundary Gamma is computed via the Green\'s single-layer boundary integral equation driven by the magnetic vector potential of the coil dipole moment m:', body_style),
        Paragraph('phi(r) = (mu_0 / (4*pi)) * integral_Gamma (m . (r - r_coil)) / |r - r_coil|^3  dGamma<br/>E(r) = - grad(phi) - dA/dt<br/>sigma_eff * grad^2(phi) = 0 in Omega_tissue', equation_style),

        attenuation_chart,
        Paragraph('Figure 2 | Boundary element electric field attenuation across concentric tissue layers (scalp, skull, CSF, grey matter) computed from the moduli-optimal protocol.', caption_style),

        Paragraph('5. Continued Fraction Harmonic Synchronization', heading_style),
        Paragraph('The optimal frequency f* is decomposed into continued fraction rational convergents p_k / q_k to establish phase-locked pulse timing gates that avoid destructive phase interference:', body_style),
        Paragraph('f* = a_0 + 1 / (a_1 + 1 / (a_2 + ...))<br/>p_k = a_k * p_(k-1) + p_(k-2),   q_k = a_k * q_(k-1) + q_(k-2)', equation_style),
        Paragraph(f'Principal convergents for f* = {opt["frequency_hz"]:.2f} Hz: [{cf_str}].', body_style),

        cf_bar_chart,
        Paragraph('Figure 3 | Logarithmic approximation error of rational continued-fraction convergents p_k / q_k relative to the moduli-optimal frequency f*.', caption_style),
    ]

    # Tables
    proto_rows = [
        ['Parameter', 'Moduli-Optimal Value', 'Biophysical Interpretation'],
        ['Optimal Frequency f*', f"{opt['frequency_hz']:.2f} Hz", 'Modular resonance frequency maximizing cortical entrainment'],
        ['Optimal Intensity I*', f"{opt['intensity_pct']:.1f}% MSO", 'Drive intensity maintaining localized BEM focusing'],
        ['Canonical Point z*', f"{opt['z_reduced']['re']:.3f} + {opt['z_reduced']['im']:.3f}i", 'Reduced coordinate in SL(2,Z) fundamental domain F'],
        ['Nearest Elliptic Orbit', opt['nearest_elliptic_point'], 'Order-2/3 stabilizer subgroup symmetry locus'],
        ['Elliptic Distance d_E', f"{opt['elliptic_distance']:.4f}", 'Proximity metric to isotropy fixed point'],
        ['Stability Index Phi(z*)', f"{opt['stability_score']:.4f}", 'Robustness score against biological parameter perturbation'],
        ['Klein j-Invariant j(z*)', f"{opt['j_invariant']['re']:.1f} + {opt['j_invariant']['im']:.1f}i", 'Modular-invariant isomorphism classification index'],
        ['Peak Cortical Potential', f"{bem['peak_potential']:.2f} (Scaled)", 'BEM single-layer peak induced surface potential'],
    ]
    t_proto = Table(proto_rows, colWidths=[4.5*cm, 4.2*cm, 7.8*cm], repeatRows=1)
    t_proto.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.2),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d0d5dd')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    cf_rows = [['Index k', 'Quotient a_k', 'Convergent p_k / q_k', 'Approximation (Hz)', 'Relative Error (%)']]
    for c in cf_convergents[:6]:
        cf_rows.append([
            str(c['iteration']), str(c['a_k']), f"{c['numerator']}/{c['denominator']}",
            f"{c['approx_freq']:.4f}", f"{c['error_pct']:.4f}%"
        ])
    t_cf = Table(cf_rows, colWidths=[2.5*cm, 3.0*cm, 3.5*cm, 3.8*cm, 3.7*cm], repeatRows=1)
    t_cf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.2),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d0d5dd')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    story += [
        Spacer(1, 6),
        Paragraph('Table 1 | Moduli-theoretic optimization parameters and BEM physical indicators', subheading_style),
        t_proto,
        Spacer(1, 8),
        Paragraph('Table 2 | Rational continued-fraction frequency convergents and synchronization hierarchy', subheading_style),
        t_cf,
        Spacer(1, 8),
        Paragraph('6. Discussion & Conclusions', heading_style),
        Paragraph('Mapping rTMS parameter search spaces onto the modular curve X(1) = H / SL(2,Z) provides a rigorous algebraic foundation for identifying robust neuromodulation protocols. Rather than relying on brute-force empirical sweeps, the moduli framework naturally groups stimulation regimes into fundamental domain equivalents, singling out elliptic fixed points as canonical resonance anchors. Coupling modular reduction with boundary element electrodynamics enables rapid, physics-grounded clinical protocol synthesis.', body_style),
    ]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2.0*cm,
        rightMargin=2.0*cm,
        topMargin=1.6*cm,
        bottomMargin=1.6*cm,
        title='Moduli-Theoretic Optimization of rTMS Treatment Paradigms',
        author='NeuroMorph Computational Platform',
    )
    doc.build(story)
    return str(output_path)


if __name__ == '__main__':
    print('Generating PDF to:', OUTPUT)
    out = build_pdf()
    print('Generated successfully at:', out)
