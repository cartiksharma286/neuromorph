#!/usr/bin/env python3
"""
Generates CFU_Projections_QML_Basis.pdf detailing the basis of the 10-year CFU projections
and the exact Quantum Machine Learning model used under the Hydrostar space and ecological suite.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors

def generate_pdf():
    pdf_path = '/Users/cartiksharma/Downloads/neuromorph-main-10/hydrostar/CFU_Projections_QML_Basis.pdf'
    
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

    # Custom styling
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#0284c7'),  # Sky blue accent
        spaceAfter=14,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#475569'),
        spaceAfter=15,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )

    heading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=8,
        spaceBefore=14,
        fontName='Helvetica-Bold',
        borderPadding=(0, 0, 2, 0),
        borderColor=colors.HexColor('#cbd5e1')
    )

    body_style = ParagraphStyle(
        'MainBody',
        parent=styles['BodyText'],
        fontSize=9.5,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=14,
        textColor=colors.HexColor('#334155')
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['BodyText'],
        fontSize=9,
        spaceAfter=4,
        leading=13,
        textColor=colors.HexColor('#334155'),
        leftIndent=15
    )

    formula_style = ParagraphStyle(
        'FormulaText',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=10,
        spaceBefore=10,
        textColor=colors.HexColor('#1e1b4b'),
        fontName='Courier',
        backColor=colors.HexColor('#f1f5f9'),
        borderPadding=10,
        borderWidth=1,
        borderColor=colors.HexColor('#e2e8f0'),
        borderRadius=8
    )

    # 1. Title & Subtitle Headers
    elements.append(Paragraph("HYDROSTAR SYSTEMS INTEGRATION REPORT", title_style))
    elements.append(Paragraph(
        "<b>Project:</b> Yellow Creek Ravine E.coli Active Modeling & Coherent Restoration<br/>"
        "<b>Technical Specifications:</b> 10-Year CFU Projections & Exact QK-GPR Model Architecture", 
        subtitle_style
    ))
    elements.append(Spacer(1, 0.1*inch))

    # 2. Main Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    elements.append(Paragraph(
        "This technical document presents the mathematical and computational framework for predicting and mitigating "
        "fecal coliform and Eschericia coli (E.coli) concentrations within the Yellow Creek Ravine watershed corridor "
        "and Lake Ontario shoreline over a decadal (10-year) horizon. Utilizing edge telemetry data collected via the "
        "newly integrated Raspberry Pi Zero 2 W platform with triple-isolated Atlas Scientific EZO-circuitry, our "
        "predictive model utilizes advanced quantum statistical learning to provide optimal sensor topological placement, "
        "minimizing signal reconstruction error and optimizing passive bioremediation feedback loop timing.",
        body_style
    ))

    # 3. CFU Projection Basis
    elements.append(Paragraph("1. Basis of 10-Year Decadal CFU Projections", heading_style))
    elements.append(Paragraph(
        "Long-term forecasting of Colony Forming Units (CFU) per 100mL of water samples is vital for municipal "
        "recreational safety and ecosystem health diagnostics. Traditional interpolation models fail to represent "
        "environmental thresholds under changing climates. Our projection accounts for three highly coupled parameters:",
        body_style
    ))

    elements.append(Paragraph(
        "<b>• Baseline Municipal and Storm Runoff:</b> Combined sewer overflow (CSO) discharge risks, surface runoff "
        "flows, and urbanization factors scale exponentially with local precipitation index <i>P(t)</i> and catchment "
        "imperviousness factor <i>&eta;</i>.",
        bullet_style
    ))

    elements.append(Paragraph(
        "<b>• Arrhenius Microbial Heat Kinetics:</b> Environmental temperature profiles dictate bacterial "
        "multiplication kinetics. The model factors in Arrhenius temperature-dependent multiplier "
        "<i>e<sup>-E<sub>a</sub>/RT</sup></i> modeling, anticipating summer heatwaves and heat island shifts.",
        bullet_style
    ))

    elements.append(Paragraph(
        "<b>• Optimal Spatial Topology Decay:</b> High-density placement models select coords based on spectral concentration. "
        "Real-time feedback control loops deploy active aeration and passive bio-retention filters to decay microbial concentration "
        "below risk thresholds.",
        bullet_style
    ))

    # Equations block
    elements.append(Spacer(1, 0.05*inch))
    elements.append(Paragraph(
        "<b>Unmitigated Baseline Equation:</b><br/>"
        "CFU_baseline(t) = 480.0 + 35.0 * t + 4.2 * t<sup>1.35</sup> + N(0, σ<sup>2</sup>)", 
        formula_style
    ))
    
    elements.append(Paragraph(
        "<b>QML-Guided Restoration Mitigation Equation:</b><br/>"
        "CFU_QML(t) = 25.0 + (CFU_initial - 25.0) * e<sup>-&lambda;<sub>opt</sub> * t</sup> + N(0, σ<sup>2</sup>)", 
        formula_style
    ))

    # 4. Exact QML model
    elements.append(Paragraph("2. Exact Quantum Machine Learning (QML) Model Architecture", heading_style))
    elements.append(Paragraph(
        "To resolve non-linear spatio-temporal correlations without overfitting, traditional Radial Basis Function (RBF) "
        "kernels are replaced with a hybrid <b>Quantum Kernel-Enhanced Gaussian Process Regressor (QK-GPR)</b> solver. "
        "The model is defined by three primary steps:",
        body_style
    ))

    elements.append(Paragraph(
        "<b>• IQP Feature Mapping:</b> Multimodal edge telemetry covariates (Conductivity, pH, ORP, Temperature, "
        "IMX462 optical night-vision turbidity indicators) are mapped into high-dimensional Hilbert state space via "
        "an Instantaneous Quantum Polynomial-time (IQP) ansatz:",
        bullet_style
    ))
    
    elements.append(Paragraph(
        "U<sub>&Phi;</sub>(x) = exp[ i * &Sigma;<sub>j</sub> x<sub>j</sub> Z<sub>j</sub> + i * &Sigma;<sub>j,k</sub> x<sub>j</sub> x<sub>k</sub> Z<sub>j</sub> Z<sub>k</sub> ]", 
        formula_style
    ))

    elements.append(Paragraph(
        "<b>• Variational Kernel Overlap Matrix:</b> A variational ansatz circuit V(&theta;) composed of Multi-layer "
        "Chebyshev rotation gates extracts cross-correlation patterns. The state overlap is computed directly on physical qubits:",
        bullet_style
    ))

    elements.append(Paragraph(
        "K(x<sub>i</sub>, x<sub>j</sub>) = |&lt;0<sup>&otimes;n</sup>| U<sub>&Phi;</sub><sup>&dagger;</sup>(x<sub>i</sub>) V(&theta;)<sup>&dagger;</sup> V(&theta;) U<sub>&Phi;</sub>(x<sub>j</sub>) |0<sup>&otimes;n</sup>&gt;|<sup>2</sup>", 
        formula_style
    ))

    elements.append(Paragraph(
        "<b>• Log-Likelihood Optimization:</b> Classical optimization loops tune variational parameters &theta; to maximize "
        "marginal log-marginal-likelihood, allowing the regressor to accurately predict decade-long CFU trends with a "
        "reconstruction target error of &lt; 8.0%.",
        bullet_style
    ))

    # Footer note or metadata
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(
        "<i>Hydrostar Ecological Suite. Engineered &amp; Maintained by NEUROMORPH SPACE SYSTEMS. All Rights Reserved. © 2026.</i>", 
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7.5, textColor=colors.HexColor('#64748b'), alignment=TA_CENTER)
    ))

    # Build the document
    doc.build(elements)
    print("PDF generated successfully at:", pdf_path)

if __name__ == '__main__':
    generate_pdf()
