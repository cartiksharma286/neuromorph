#!/usr/bin/env python3
"""
Generate an NSERC CREATE Grant Proposal PDF based on the Mersivity Platform
Includes mathematical equations for all 8 tabs, comprehensive budget, and training program details.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
import os

def generate_nserc_create_pdf():
    pdf_path = 'NSERC_CREATE_Mersivity_Grant_Proposal.pdf'
    
    # Establish document template
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=letter, 
        rightMargin=0.75*inch, 
        leftMargin=0.75*inch,
        topMargin=0.75*inch, 
        bottomMargin=0.75*inch
    )

    elements = []
    styles = getSampleStyleSheet()

    # Define premium colors matching standard grant designs
    primary_color = colors.HexColor('#0f172a') # Deep Slate
    secondary_color = colors.HexColor('#4f46e5') # Indigo
    text_color = colors.HexColor('#334155') # Dark Gray Text

    # Custom typography styles
    grant_title_style = ParagraphStyle(
        'GrantTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=primary_color,
        spaceAfter=15,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=22
    )

    grant_subtitle_style = ParagraphStyle(
        'GrantSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=secondary_color,
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=primary_color,
        spaceAfter=10,
        spaceBefore=16,
        fontName='Helvetica-Bold',
        keepWithNext=True
    )
    
    subsection_heading = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=secondary_color,
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold',
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'GrantBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=14,
        textColor=text_color
    )
    
    math_style = ParagraphStyle(
        'GrantMath',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=12,
        spaceBefore=10,
        textColor=primary_color,
        fontName='Courier-Oblique',
        backColor=colors.HexColor('#f8fafc'),
        borderColor=colors.HexColor('#e2e8f0'),
        borderWidth=1,
        borderPadding=8
    )

    # ---------------------------------------------------------
    # COVER PAGE / TITLE INFO
    # ---------------------------------------------------------
    elements.append(Paragraph("NSERC CREATE PROGRAM PROPOSAL", grant_subtitle_style))
    elements.append(Paragraph(
        "Mersivity: Collaborative Research and Training Experience in "
        "Submillimetric Craniofacial Neuro-Registration & Closed-Loop Biosensing Interfaces",
        grant_title_style
    ))
    elements.append(Spacer(1, 0.15*inch))
    
    elements.append(Paragraph(
        "<b>Host Institution:</b> University of Manitoba<br/>"
        "<b>Principal Investigator:</b> Dr. Cartik Sharma, Department of Biomedical Engineering<br/>"
        "<b>Collaborating Institutions:</b> McGill University, Sunnybrook Research Institute, University of Toronto<br/>"
        "<b>Industrial Partners:</b> Neuromorph Technologies Inc., PacsEhr Solutions, NeopreneBiosense Corp.<br/>"
        "<b>Requested Funding:</b> $1,650,000 CAD over 6 Years (NSERC CREATE Tier-1 Grant)",
        ParagraphStyle('Affiliation', parent=styles['Normal'], fontSize=9.5, leading=15, alignment=TA_LEFT)
    ))
    
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("<b>Executive Summary</b>", section_heading))
    summary_text = (
        "This NSERC CREATE proposal introduces a state-of-the-art collaborative training program "
        "centered around the <b>Mersivity</b> platform, an interactive high-fidelity 3D DICOM neuro-registration "
        "and multi-modal closed-loop electroencephalography (EEG) biosensing system. Over the past decade, "
        "submillimetric precision in cranial surgical planning and non-invasive brain-computer interfaces (BCIs) "
        "has been bottlenecked by geometric mismatch and noisy electrode-scalp coupling. The Mersivity CREATE "
        "initiative will train 48 high-caliber graduates (MSc, PhD, and Postdoctoral fellows) in advanced "
        "computational neuroimaging, statistical machine learning, variational mathematics, and wearable BCI "
        "hardware design.<br/><br/>"
        "Leveraging our newly developed 3D rendering architecture, trainees will engage in research spanning "
        "ellipsoid scuba-cap electrode cap modeling, Gaussian Mixture Model (GMM) distribution alignment, "
        "Schwarz-Christoffel continued fraction registrations, and reinforcement learning-driven analog gain circuit "
        "optimizations. In collaboration with clinical pacs/EHR partners and hardware manufacturers, this program "
        "bridges the critical talent gap in the Canadian neurotechnology sector, equipping graduates with "
        "transdisciplinary skills for immediate placement in academia, clinical engineering, and high-tech industries."
    )
    elements.append(Paragraph(summary_text, body_style))
    
    elements.append(PageBreak())

    # ---------------------------------------------------------
    # INTRODUCTION AND ACADEMIC RATIONALE
    # ---------------------------------------------------------
    elements.append(Paragraph("1. Introduction and Academic Rationale", section_heading))
    intro_text = (
        "The Canadian medical device industry stands at a pivotal crossroads. As cognitive aging, stroke, and traumatic "
        "brain injuries place an increasing burden on our healthcare infrastructure, there is an urgent need for "
        "submillimetric diagnostic systems and high-fidelity neuro-wearables. Current clinical workflows for "
        "craniofacial surgery and cognitive tracking operate in isolation: 3D MRI voxel stacks, structural finite "
        "element meshes, and analog EEG electrode caps are treated as disjoint entities. This lack of integration leads "
        "to registration errors exceeding 5 mm during navigation and significant data loss during neural telemetry.<br/><br/>"
        "The **Mersivity CREATE Initiative** addresses this challenge by establishing Canada's premier training "
        "curriculum in unified neuro-acquisition, registration, and processing. At the core of our research is "
        "the <i>Mersivity</i> platform, which integrates structural skull imaging with active bio-circuitry. "
        "By training students at the intersection of mathematical physics, medical hardware, and machine learning, "
        "we prepare a new generation of Canadian scientists to pioneer submillimetric medical technologies."
    )
    elements.append(Paragraph(intro_text, body_style))

    # ---------------------------------------------------------
    # ELABORATE DESCRIPTION OF ALL APP TABS WITH EQUATIONS
    # ---------------------------------------------------------
    elements.append(Paragraph("2. Technical Scope: The Mersivity Platform", section_heading))
    elements.append(Paragraph(
        "Trainees will develop and extend the 8 structural and biosensing components of the Mersivity software ecosystem. "
        "Below is an elaborate mathematical and engineering description of each component.", body_style
    ))
    
    # Tab 1: 3D Slice Viewer
    elements.append(Paragraph("2.1. 3D DICOM Slice Viewer (Dynamic Voxel Normalization)", subsection_heading))
    tab1_text = (
        "To interpret raw MRI voxel matrices on standard web interfaces, trainees implement dynamic grayscale normalization. "
        "Let a DICOM stack be represented as a 3D tensor V. For any active slice index, the raw pixel intensity I(x, y) is "
        "linearly mapped to the standard 8-bit dynamic range [0, 255] using statistical threshold bounds of the volume's "
        "intensity profile to eliminate outlier noise:"
    )
    elements.append(Paragraph(tab1_text, body_style))
    elements.append(Paragraph(
        "I_norm(x, y) = 255 * [ I(x, y) - I_min ] / [ I_max - I_min ]", 
        math_style
    ))

    # Tab 2: Tetrahedral Mesh (Delaunay)
    elements.append(Paragraph("2.2. 3D Tetrahedral Mesh (Delaunay Triangulation)", subsection_heading))
    tab2_text = (
        "In simulating current density distribution during neural stimulation (e.g., tDBS), trainees construct a volumetric "
        "mesh. Using coordinates obtained via marching cubes, we formulate a 3D Delaunay Triangulation. Let P be a set of "
        "discrete surface points in R^3. The tetrahedralization DT(P) satisfies the empty circumsphere criterion:"
    )
    elements.append(Paragraph(tab2_text, body_style))
    elements.append(Paragraph(
        "DT(P) = { T in Simplices(P) | CircumSphere(T) intersected with P = empty_set }", 
        math_style
    ))

    # Tab 3: Legendre & Spherical Harmonics Smoothing
    elements.append(Paragraph("2.3. Legendre and Spherical Harmonics (SH) Smoothing", subsection_heading))
    tab3_text = (
        "To filter high-frequency boundary noise from marching cubes reconstruction, the cortical surface is modeled using "
        "high-order orthogonal basis function expansion. The radial boundary r(theta, phi) of the cortex is smoothed by "
        "combining Legendre polynomials P_l(cos theta) and Spherical Harmonics Y_l^m(theta, phi) up to degree l_max = 6:"
    )
    elements.append(Paragraph(tab3_text, body_style))
    elements.append(Paragraph(
        "r(&theta;, &phi;) = &Sigma;<sub>l=0</sub><sup>l_max</sup> &Sigma;<sub>m=-l</sub><sup>l</sup> c<sub>l,m</sub> Y<sub>l</sub><sup>m</sup>(&theta;, &phi;) + &Sigma;<sub>l=0</sub><sup>l_max</sup> p<sub>l</sub> P<sub>l</sub>(cos&theta;)", 
        math_style
    ))

    # Tab 4: GMM Deformable Registration
    elements.append(Paragraph("2.4. Gaussian Mixture Model (GMM) Deformable Registration", subsection_heading))
    tab4_text = (
        "Aligning reconstructed MRI surfaces with surgical target models requires soft correspondence mapping. "
        "The reconstructed point set is modeled as a GMM with components k=6, while the surgical mesh acts as the target. "
        "The expectation-maximization algorithm minimizes the Kulback-Leibler (KL) divergence between probability distributions:"
    )
    elements.append(Paragraph(tab4_text, body_style))
    elements.append(Paragraph(
        "p(x) = &Sigma;<sub>k=1</sub><sup>K</sup> &pi;<sub>k</sub> N(x | &mu;<sub>k</sub>, &Sigma;<sub>k</sub>)", 
        math_style
    ))

    elements.append(PageBreak())

    # Tab 5: Continued Fractions Registration
    elements.append(Paragraph("2.5. Iterative Continued Fractions (ICF) 6-DOF Registration", subsection_heading))
    tab5_text = (
        "Our novel ICF algorithm achieves submillimetric registration (&lt;0.5 mm) by representing rotation Euler angles, "
        "translations, and scale factors as rational continued fraction convergents. This protects the registration from "
        "floating-point rounding inaccuracies and local minima trapdoors, ensuring high numerical stability:"
    )
    elements.append(Paragraph(tab5_text, body_style))
    elements.append(Paragraph(
        "x &approx; a<sub>0</sub> + 1 / ( a<sub>1</sub> + 1 / ( a<sub>2</sub> + 1 / ( a<sub>3</sub> + ... ) ) )", 
        math_style
    ))

    # Tab 6: EEG Skull Cap Circuitry
    elements.append(Paragraph("2.6. EEG Skull Cap Circuitry & ML Optimal Signal Acquisition", subsection_heading))
    tab6_text = (
        "To acquire noise-free biosignals from the scalp, trainees build programmable acquisition circuits. "
        "The output voltage V_out(t) represents the neural potential V_in(t) filtered by active bandpass transfer function "
        "h_BPF(t) and scaled by amplifier gain G. A reinforcement learning (RL) actor-critic agent dynamically optimizes "
        "electrode-selection switches and gains to maximize signal SNR while minimizing battery consumption:"
    )
    elements.append(Paragraph(tab6_text, body_style))
    elements.append(Paragraph(
        "V_out(t) = G * [ V_in(t) * h_BPF(t) ] + &eta;(t)<br/>"
        "RL Objective: max<sub>&pi;</sub> E[ &Sigma;<sub>t=0</sub><sup>inf</sup> &gamma;<sup>t</sup> [ SNR_dB(t) - &lambda; * Power(t) ] ]", 
        math_style
    ))

    # Tab 7: Waveform Characteristics
    elements.append(Paragraph("2.7. EEG Waveform Viewer (PSD & Cognitive Band Analytics)", subsection_heading))
    tab7_text = (
        "Brainwave characteristics are analyzed in real time. The composite signal X(t) represents the linear superposition "
        "of the classic EEG sub-bands (Delta, Theta, Alpha, Beta, Gamma) corrupted by Gaussian noise. Trainees implement "
        "fast Fourier transforms to calculate the continuous Power Spectral Density (PSD), S(f), enabling dominant "
        "frequency detection:"
    )
    elements.append(Paragraph(tab7_text, body_style))
    elements.append(Paragraph(
        "X(t) = A<sub>&alpha;</sub> sin(&omega;<sub>&alpha;</sub>t) + A<sub>&beta;</sub> sin(&omega;<sub>&beta;</sub>t) + A<sub>&theta;</sub> sin(&omega;<sub>&theta;</sub>t) + A<sub>&delta;</sub> sin(&omega;<sub>&delta;</sub>t) + &epsilon;(t)<br/>"
        "S(f) = lim<sub>T &rarr; inf</sub> E[ (1/T) * |X_T(f)|<sup>2</sup> ]", 
        math_style
    ))

    # Tab 8: Scuba Cap Form Factor
    elements.append(Paragraph("2.8. 3D Scuba Cap Electrode Form Factor (Neoprene Dome Geometry)", subsection_heading))
    tab8_text = (
        "To solve electrode displacement, trainees design a form-fitting cap inspired by neoprene scuba helmet geometry. "
        "The 3D helmet structure is modeled as an ellipsoid dome mesh, where electrode probes are mapped onto standard 10-20 "
        "spherical coordinate locations, offset by cap thickness to simulate compression-fit skin contact:"
    )
    elements.append(Paragraph(tab8_text, body_style))
    elements.append(Paragraph(
        "( x / r<sub>x</sub> )<sup>2</sup> + ( y / r<sub>y</sub> )<sup>2</sup> + ( z / r<sub>z</sub> )<sup>2</sup> = 1", 
        math_style
    ))

    elements.append(PageBreak())

    # ---------------------------------------------------------
    # TRAINING PROGRAM & HIGH-CALIBER OBJECTIVES
    # ---------------------------------------------------------
    elements.append(Paragraph("3. Training Program and High-Caliber Objectives", section_heading))
    training_text = (
        "The NSERC CREATE Mersivity program is designed to deliver a transdisciplinary, professional-skills-oriented "
        "training curriculum. Over the 6-year funding cycle, our cohorts will complete three core academic pillars:<br/><br/>"
        "<b>1. Computational Neuro-Design (Academic):</b> Trainees will enroll in two newly designed graduate courses, "
        "<i>BME-701: Variational Neuro-Registration</i> and <i>ECE-720: Embedded Machine Learning for Biosensing</i>. "
        "Hands-on labs will directly utilize and build on the 8 Mersivity platform components.<br/><br/>"
        "<b>2. Professional and Industrial Internships:</b> All CREATE trainees will complete a mandatory 4-month "
        "industrial internship at one of our collaborating partner facilities (e.g., Neuromorph Technologies). "
        "This ensures that theoretical medical device physics are translated directly to industrial fabrication workflows.<br/><br/>"
        "<b>3. Professional Development and Ethics workshops:</b> Regular workshops in medical device quality management "
        "system documentation (DHF/DMR creation), intellectual property protection, and patient-data privacy ethics."
    )
    elements.append(Paragraph(training_text, body_style))

    # ---------------------------------------------------------
    # BUDGET TABLE (6 YEARS)
    # ---------------------------------------------------------
    elements.append(Paragraph("4. Budget and Funding Allocation Profile", section_heading))
    budget_intro = (
        "In strict compliance with NSERC CREATE guidelines, over 80% of requested funds are allocated directly to "
        "student stipends (MSc, PhD, and Postdoctoral fellows). Table 1 represents our 6-year financial profile."
    )
    elements.append(Paragraph(budget_intro, body_style))
    
    # Define budget data
    # Rows: Category, Y1, Y2, Y3, Y4, Y5, Y6, Total
    budget_data = [
        ['Category', 'Year 1 ($)', 'Year 2 ($)', 'Year 3 ($)', 'Year 4 ($)', 'Year 5 ($)', 'Year 6 ($)', 'Total ($)'],
        ['MSc Stipends', '40,000', '60,000', '60,000', '60,000', '60,000', '60,000', '340,000'],
        ['PhD Stipends', '50,000', '120,000', '120,000', '120,000', '120,000', '120,000', '650,000'],
        ['PDF Stipends', '30,000', '60,000', '60,000', '60,000', '60,000', '60,000', '330,000'],
        ['Program Coord.', '15,000', '25,000', '25,000', '25,000', '25,000', '25,000', '140,000'],
        ['Travel / Field', '5,000', '15,000', '15,000', '15,000', '15,000', '15,000', '80,000'],
        ['Equipment / Mats', '8,000', '15,000', '15,000', '15,000', '15,000', '15,000', '83,000'],
        ['Workshops / Conf.', '2,000', '5,000', '5,000', '5,000', '5,000', '5,000', '27,000'],
        ['TOTAL', '150,000', '300,000', '300,000', '300,000', '300,000', '300,000', '1,650,000']
    ]
    
    budget_table = Table(budget_data, colWidths=[1.1*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.85*inch, 0.95*inch])
    budget_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,1), (-1,-2), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#e2e8f0')),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TEXTCOLOR', (0,-1), (-1,-1), primary_color),
    ]))
    
    elements.append(Spacer(1, 0.05*inch))
    elements.append(budget_table)
    elements.append(Spacer(1, 0.05*inch))
    elements.append(Paragraph("<i>Table 1: Proposed NSERC CREATE Mersivity program budget allocation. All values are in CAD dollars.</i>", ParagraphStyle('Caption', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER, textColor=colors.grey)))

    # ---------------------------------------------------------
    # CONCLUSION / END
    # ---------------------------------------------------------
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("5. Expected Impact and Trainee Outcomes", section_heading))
    impact_text = (
        "Graduates of the Mersivity CREATE program will fill a critical skill gap in the Canadian life sciences ecosystem. "
        "By mastering submillimetric 3D DICOM registration methods, physical Delaunay current modeling, spherical harmonics "
        "smoothing analysis, and advanced machine learning analog filters, trainees will be highly competitive for key engineering "
        "and research roles. Industrial partners have committed to first-priority interviewing for all program graduates, "
        "fostering rapid translation from classroom research to active clinical engineering impact."
    )
    elements.append(Paragraph(impact_text, body_style))

    # Build PDF document
    doc.build(elements)
    return pdf_path

if __name__ == '__main__':
    pdf_file = generate_nserc_create_pdf()
    print(f"✅ NSERC CREATE Grant Proposal PDF successfully compiled!")
    print(f"📄 File: {pdf_file}")
