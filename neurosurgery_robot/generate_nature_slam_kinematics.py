#!/usr/bin/env python3
"""
generate_nature_slam_kinematics.py
─────────────────────────────────────────────────────────────────────────────
Generates a Nature-style journal article PDF for SLAM Correspondence Kinematics.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors

C_TITLE   = colors.HexColor('#0a2463')
C_HEAD    = colors.HexColor('#1e3a5f')
C_SUB     = colors.HexColor('#1e40af')
C_EQ_BG   = colors.HexColor('#f0f4ff')
C_EQ_BD   = colors.HexColor('#3b82f6')
C_CAPTION = colors.HexColor('#374151')

def _make_styles() -> dict:
    ST = getSampleStyleSheet()
    def _ps(name, **kw):
        return ParagraphStyle(name, parent=ST['Normal'], **kw)

    return dict(
        title   = _ps('NTitle',  fontSize=16, fontName='Helvetica-Bold', textColor=C_TITLE, spaceAfter=8, leading=22),
        author  = _ps('NAuthor', fontSize=9.5,fontName='Helvetica-Bold', textColor=colors.HexColor('#111827'), spaceAfter=3),
        affil   = _ps('NAffil',  fontSize=8.5,fontName='Helvetica', textColor=colors.HexColor('#4b5563'), spaceAfter=10, leading=12),
        abs_h   = _ps('NAbsH',   fontSize=9,  fontName='Helvetica-Bold', textColor=C_HEAD, spaceAfter=3),
        abstract= _ps('NAbst',   fontSize=9,  fontName='Helvetica', textColor=colors.HexColor('#1f2937'), leading=13.5, alignment=TA_JUSTIFY, spaceAfter=8),
        section = _ps('NSect',   fontSize=11, fontName='Helvetica-Bold', textColor=C_HEAD, spaceBefore=14, spaceAfter=4, leading=14),
        subsec  = _ps('NSSec',   fontSize=10, fontName='Helvetica-Bold', textColor=C_SUB, spaceBefore=8, spaceAfter=3, leading=13),
        body    = _ps('NBody',   fontSize=9.5,fontName='Helvetica', textColor=colors.HexColor('#111827'), leading=14, alignment=TA_JUSTIFY, spaceAfter=8),
        eq      = _ps('NEq',   fontSize=9.5,fontName='Courier-Bold', textColor=colors.HexColor('#1e3a5f'), alignment=TA_CENTER, spaceAfter=2, spaceBefore=2, leading=14),
        eq_lbl  = _ps('NEqL',  fontSize=8.5,fontName='Helvetica-Oblique', textColor=colors.HexColor('#6b7280'), alignment=TA_RIGHT, spaceAfter=6),
        kw      = _ps('NKw',   fontSize=8.5,fontName='Helvetica-Oblique', textColor=colors.HexColor('#374151'), spaceAfter=10, leading=12),
    )

def HR(EL):
    EL.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#d1d5db'), spaceAfter=5, spaceBefore=2))

def _eq(EL, expr: str, eqno: int, desc: str, S: dict, W_cm=13.5):
    cell = Paragraph(f'<font name="Courier-Bold" size="10">{expr}</font>', S['eq'])
    lbl = Paragraph(f'({eqno})', S['eq_lbl'])
    t = Table([[cell, lbl]], colWidths=[W_cm * cm, 1.8 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), C_EQ_BG),
        ('BOX',           (0, 0), (-1, -1), 0.75, C_EQ_BD),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    EL.append(t)
    if desc:
        EL.append(Paragraph(f'<i>{desc}</i>', S['eq_lbl']))
    EL.append(Spacer(1, 0.07 * inch))

def build_pdf(out_path: str):
    doc = SimpleDocTemplate(out_path, pagesize=A4, rightMargin=2.0*cm, leftMargin=2.0*cm, topMargin=2.2*cm, bottomMargin=2.2*cm)
    S = _make_styles()
    EL = []

    EL.append(Paragraph('Optimal Pin-Point Positioning via SLAM Correspondence: Continued Fractions and Elliptic Integrals on Non-Euclidean Manifolds', S['title']))
    EL.append(Spacer(1, 0.10 * inch))
    EL.append(Paragraph('<i>Cartik Sharma</i><sup>1,2*</sup>, <i>NeuroMorph Quantum Systems Division</i><sup>1</sup>', S['author']))
    EL.append(Paragraph('<sup>1</sup>Institute for Computational Neuroimaging &amp; Robotic Surgery<br/><sup>2</sup>Centre for Advanced Surgical Robotics', S['affil']))
    HR(EL)

    EL.append(Paragraph('Abstract', S['abs_h']))
    EL.append(Paragraph('Simultaneous Localization and Mapping (SLAM) for surgical robotics demands exceptional sub-millimeter positioning accuracy. Modern approaches mapping sensor coordinates across varying magnetic flux and environmental perturbations operate on non-Euclidean manifolds. We demonstrate the injection of Jacobi elliptic integrals of the first kind alongside Stern-Brocot continued fraction expansions to govern SLAM correspondence, overcoming Euclidean error drifts.', S['abstract']))
    HR(EL)

    EL.append(Paragraph('1. Introduction to Manifold Kinematics', S['section']))
    EL.append(Paragraph('Traditional Extended Kalman Filters and position mapping rely on linear estimations. Surgical environments present high-curvature metric distortions. The correspondence between SLAM anchor points and sensor readings dictates tracking accuracy.', S['body']))

    EL.append(Paragraph('2. Elliptic Integral Mapping', S['section']))
    EL.append(Paragraph('The non-Euclidean deformation of the spatial metric is governed by the Jacobi incomplete elliptic integral of the first kind F(φ, k):', S['body']))
    _eq(EL, 'F(φ, k) = ∫₀^φ dθ / √(1 - k² sin²θ)', 1, 'Elliptic amplitude function binding spatial distortions', S)
    _eq(EL, 'φ = arcsin( z / ||p|| ),  k = ||p|| / 2', 2, 'Tracking amplitude and modulus', S)

    EL.append(Paragraph('3. Continued Fraction Approximation', S['section']))
    EL.append(Paragraph('Discrete jump resonances occur when aligning raw localized bounds to precise sub-millimeter targets. Using continued fractions provides rational approximations bounding the irrational spatial gap δ.', S['body']))
    _eq(EL, 'z_residual = a₀ + 1 / (a₁ + 1 / (a₂ + ...))', 3, 'SLAM residual continued fraction', S)
    _eq(EL, 'δ_correction = | z_residual - pₙ / qₙ |', 4, 'Fractal residual offset correction', S)
    
    EL.append(Paragraph('4. Optimal Pin-Point Positioning Law', S['section']))
    EL.append(Paragraph('Combining the non-Euclidean bound tracking from elliptic integrals and discrete SLAM correspondence anchors bounded by fractal correction gives our target tracking law:', S['body']))
    _eq(EL, 'P_err = ||p_true - p_target|| + 0.01 × δ_correction', 5, 'Adjusted positioning loss function', S)

    doc.build(EL)
    print(f"[Done] Generated PDF at: {out_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, 'Nature_SLAM_Kinematics_Correspondence.pdf')
    build_pdf(pdf_path)
