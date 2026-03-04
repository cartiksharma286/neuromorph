#!/usr/bin/env python3
"""
Technical Nature Report Generator for Quantum MR Thermometry
============================================================
Generates an elaborate .pdf report with finite math equations natively formatted 
using Preformatted text for clean typography without image rendering artifacts.
"""
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

def generate_report():
    print("Generating Nature-style Technical PDF Report with formatted equations...")
    pdf_path = "Nature_Report_Quantum_Thermometry.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            leftMargin=1.0*inch, rightMargin=1.0*inch,
                            topMargin=1.0*inch, bottomMargin=1.0*inch)
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(
        name='NatureTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='NatureAbstract',
        parent=styles['BodyText'],
        fontSize=10,
        leading=13,
        fontName='Helvetica-Bold',
        alignment=TA_JUSTIFY,
        leftIndent=0.5*inch,
        rightIndent=0.5*inch,
        spaceBefore=12,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='Equation',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        leftIndent=0.6*inch,
        rightIndent=0.6*inch,
        spaceBefore=8,
        spaceAfter=8,
        fontName='Courier',
        textColor=colors.HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        name='EqNum',
        parent=styles['BodyText'],
        fontSize=9,
        textColor=colors.HexColor('#888888'),
        alignment=TA_LEFT,
        spaceAfter=12
    ))

    story = []

    def EQ(text, label=""):
        story.append(Preformatted(f"  {text}", styles['Equation']))
        if label:
            story.append(Paragraph(f"<i>({label})</i>", styles['EqNum']))

    # TITLE & ABSTRACT
    story.append(Paragraph("Nature Technical Report: Next-Generation MR Thermometry", styles['NatureTitle']))
    story.append(Paragraph("Quantum Restricted Boltzmann Machines, Bayesian Distributions, and Qubit Photonic Coil Topology",
        ParagraphStyle(name='Sub', parent=styles['BodyText'], alignment=TA_CENTER, fontName='Helvetica-Oblique', textColor=colors.HexColor('#4a4a4a'), spaceAfter=18)))
    
    abstract = (
        "This report outlines the theoretical foundation and finite mathematical derivations of two novel "
        "Magnetic Resonance (MR) Thermometry sequences: Quantum Restricted Boltzmann Machine (RBM) Thermometry "
        "and Statistical Bayesian Distribution Thermometry. Furthermore, we formalize the topological routing "
        "pathways for two discrete neurovascular imaging elements, termed 'Qubit Photonic Head Coils', "
        "which leverage deterministic phase interleaving (bit-shuffling) to attain high SNR in continuous "
        "data acquisition routines."
    )
    story.append(Paragraph("ABSTRACT", styles['Heading3']))
    story.append(Paragraph(abstract, styles['NatureAbstract']))
    
    # SECTION 2
    story.append(Paragraph("1. Quantum RBM Thermometry", styles['Heading2']))
    story.append(Paragraph(
        "MR Thermometry using Proton Resonance Frequency (PRF) Shift depends heavily on phase mapping "
        "susceptibilities. The Quantum Restricted Boltzmann Machine (RBM) attempts to denoise PRF inferences "
        "through simulated energy minimization of the MR signal manifold.", styles['BodyText']))
    
    story.append(Paragraph("1.1 Finite Energy Function", styles['Heading3']))
    story.append(Paragraph(
        "The visible units v_i correlate to the finite Cartesian-sampled k-space signal, "
        "and h_j the sub-voxel latent thermal characteristics.", styles['BodyText']))

    EQ("E(v, h) = - Σ_i a_i v_i - Σ_j b_j h_j - Σ_{i,j} v_i W_{ij} h_j", "Eq. 1 — RBM Energy")

    story.append(Paragraph("1.2 Phase Shift Modulator", styles['Heading3']))
    story.append(Paragraph(
        "A proxy for real-world MR interaction is established. Finite approximation yields local phase "
        "alterations on reconstructed magnitude matrices.", styles['BodyText']))

    EQ("ΔΦ_thermal ≈ γ_H · α_prf · B₀ · TE · ∇T", "Eq. 2 — PRF Phase Shift")

    # SECTION 3
    story.append(Paragraph("2. Statistical Bayesian Thermometry", styles['Heading2']))
    story.append(Paragraph(
        "By enforcing a Gaussian prior on finite structural edges, we obtain localized spatial homogeneity priors "
        "combined with the posterior thermal likelihoods.", styles['BodyText']))

    story.append(Paragraph("2.1 Gradient Probability Map", styles['Heading3']))
    story.append(Paragraph(
        "Thus creating a spatial temperature estimator heavily penalized for noisy thermal distributions outside "
        "physical boundaries.", styles['BodyText']))

    EQ("P(Θ|M) ∝ exp[ - ||∇M(x,y)||² / (2σ_edge²) ] · Π_{x,y} N(T_est | μ(x,y), σ_obs²)", 
       "Eq. 3 — Bayesian Posterior")

    # SECTION 4
    story.append(Paragraph("3. Qubit Photonic Neurovascular Array Coils", styles['Heading2']))
    story.append(Paragraph(
        "Hardware limitations in standard volumetric coil transmission can be overcome by mimicking qubit state logic "
        "and algorithmic signal routing. This entails bit-reversal and discrete shuffling schemes applied spatially.", styles['BodyText']))

    story.append(Paragraph("3.1 V1: Strict Bit-Reversal Shuffling", styles['Heading3']))
    story.append(Paragraph(
        "For a 32-element coil (5-bit state representation), contiguous elements often alias adjacent "
        "noise features. By reversing the binary representation, element arrays map to physically disparate sectors:", styles['BodyText']))

    EQ("Shuffle(n) = (n_0, n_1, n_2, n_3, n_4) → (n_4, n_3, n_2, n_1, n_0)_2", "Eq. 4 — Bit Reversal")
    
    EQ("φ_k = exp[ i(π/4) · Shuffle(k) ]", "Eq. 5 — Element Phase")

    story.append(Paragraph("3.2 V2: Depth-Tuned Interleaving Shuffling", styles['Heading3']))
    story.append(Paragraph(
        "For deep cortical penetration in a 64-element matrix, alternating bits dictates radial positioning across double-tiered geometry.", styles['BodyText']))

    EQ("R(n) = R₀ - Δr · (n mod 2)\n"
       "Interleave(n) = U_{m=0}^2 (n_{2m+1}, n_{2m})", "Eq. 6 — Radially Interleaved Geometry")

    story.append(Paragraph(
        "Yielding distinct spatial phases capable of constructively interfering over specific cerebrovascular landmarks.", styles['BodyText']))
    
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                           ParagraphStyle(name='footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(story)
    print(f"Report cleanly generated without parse errors as: {pdf_path}")

if __name__ == "__main__":
    generate_report()
