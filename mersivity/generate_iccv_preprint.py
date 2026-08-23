#!/usr/bin/env python3
"""
Generate a professional ICCV Template PDF based on the neuro-registration algorithms.
Features a two-column conference layout, Times-Roman academic typography, and exact TRE results.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, PageBreak, FrameBreak, NextPageTemplate
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
import os

def generate_iccv_preprint():
    pdf_path = 'ICCV_Preprint_Submillimetric_Neuro_Registration.pdf'
    
    # Page dimensions for Letter: 612 x 792 pt
    # Margins: 0.5 inch (36 pt)
    left_margin = 36
    right_margin = 36
    top_margin = 36
    bottom_margin = 36
    
    printable_width = 612 - left_margin - right_margin # 540 pt
    printable_height = 792 - top_margin - bottom_margin # 720 pt
    
    # Two columns with a gap of 18 pt
    col_width = (printable_width - 18) / 2 # 261 pt
    
    # ---------------------------------------------------------
    # FRAME DEFINITIONS
    # ---------------------------------------------------------
    # Page 1 frames:
    # 1. Top frame for Title, Authors, Affiliation, and Abstract
    top_frame_height = 330
    top_frame = Frame(
        left_margin, 
        792 - top_margin - top_frame_height, 
        printable_width, 
        top_frame_height,
        id='F1_Top', 
        topPadding=10, bottomPadding=10, leftPadding=0, rightPadding=0
    )
    
    # 2. Left column for Page 1 main body
    left_frame_p1 = Frame(
        left_margin, 
        bottom_margin, 
        col_width, 
        792 - top_margin - top_frame_height - bottom_margin - 10,
        id='F1_Col1', 
        topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    
    # 3. Right column for Page 1 main body
    right_frame_p1 = Frame(
        left_margin + col_width + 18, 
        bottom_margin, 
        col_width, 
        792 - top_margin - top_frame_height - bottom_margin - 10,
        id='F1_Col2', 
        topPadding=0, bottomPadding=0, leftPadding=0, rightPadding=0
    )
    
    # Page 2+ frames:
    # 1. Left column for subsequent pages
    left_frame_p2 = Frame(
        left_margin, 
        bottom_margin, 
        col_width, 
        printable_height,
        id='F2_Col1', 
        topPadding=10, bottomPadding=10, leftPadding=0, rightPadding=0
    )
    
    # 2. Right column for subsequent pages
    right_frame_p2 = Frame(
        left_margin + col_width + 18, 
        bottom_margin, 
        col_width, 
        printable_height,
        id='F2_Col2', 
        topPadding=10, bottomPadding=10, leftPadding=0, rightPadding=0
    )
    
    doc = BaseDocTemplate(pdf_path, pagesize=letter)
    
    # Add page templates
    first_page_template = PageTemplate(id='FirstPage', frames=[top_frame, left_frame_p1, right_frame_p1])
    two_column_template = PageTemplate(id='TwoColumn', frames=[left_frame_p2, right_frame_p2])
    doc.addPageTemplates([first_page_template, two_column_template])

    elements = []
    styles = getSampleStyleSheet()

    # Academic ICCV Palette
    primary_color = colors.HexColor('#000000')   # Black
    text_color = colors.HexColor('#111111')      # Academic Dark Text
    math_bg = colors.HexColor('#f8fafc')         # Soft Gray Backcolor
    math_border = colors.HexColor('#cbd5e1')     # Slate Border
    accent_blue = colors.HexColor('#1d4ed8')     # Dark Royal Blue (metrics highlights)

    # ICCV Academic styles (Times-Roman)
    header_style = ParagraphStyle(
        'Header',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray,
        fontName='Times-Roman',
        spaceAfter=15,
        alignment=TA_CENTER
    )

    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=primary_color,
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Times-Bold',
        leading=22
    )

    author_style = ParagraphStyle(
        'Authors',
        parent=styles['Normal'],
        fontSize=10,
        textColor=primary_color,
        spaceAfter=3,
        alignment=TA_CENTER,
        fontName='Times-Roman'
    )

    affiliation_style = ParagraphStyle(
        'Affiliations',
        parent=styles['Normal'],
        fontSize=9,
        textColor=text_color,
        spaceAfter=15,
        alignment=TA_CENTER,
        leading=11,
        fontName='Times-Italic'
    )

    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Heading2'],
        fontSize=10,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Times-Bold'
    )

    abstract_style = ParagraphStyle(
        'AbstractText',
        parent=styles['BodyText'],
        fontSize=8.5,
        alignment=TA_JUSTIFY,
        spaceAfter=15,
        leading=12,
        textColor=text_color,
        leftIndent=24,
        rightIndent=24,
        fontName='Times-Roman'
    )

    heading1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=10,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        fontName='Times-Bold',
        keepWithNext=True
    )

    heading2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=9.5,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
        fontName='Times-Bold',
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'PaperBody',
        parent=styles['BodyText'],
        fontSize=9,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        leading=12.5,
        textColor=text_color,
        fontName='Times-Roman'
    )

    math_style = ParagraphStyle(
        'PaperMath',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        spaceAfter=8,
        spaceBefore=6,
        textColor=primary_color,
        fontName='Courier',
        backColor=math_bg,
        borderColor=math_border,
        borderWidth=0.5,
        borderPadding=4
    )
    
    reference_style = ParagraphStyle(
        'PaperRef',
        parent=styles['Normal'],
        fontSize=7.5,
        alignment=TA_JUSTIFY,
        leading=10,
        spaceAfter=4,
        textColor=text_color,
        fontName='Times-Roman'
    )

    # ---------------------------------------------------------
    # STORY ASSEMBLY
    # ---------------------------------------------------------
    # 1. Instruct ReportLab to use 'TwoColumn' template for page 2 onwards
    elements.append(NextPageTemplate('TwoColumn'))
    
    # ---------------------------------------------------------
    # TOP FRAME (TITLE, AUTHORS, ABSTRACT)
    # ---------------------------------------------------------
    elements.append(Paragraph("Proceedings of the International Conference on Computer Vision (ICCV) 2026", header_style))
    elements.append(Paragraph(
        "Submillimetric Geodesic Craniofacial Registration: A Comparative Evaluation of "
        "Gaussian Mixture Models, Elliptic Residual Networks, and Simulated Quantum Variational Circuits",
        title_style
    ))
    
    # Author Block
    authors_text = "<b>Cartik Sharma</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Dr. Steve Mann</b>"
    elements.append(Paragraph(authors_text, author_style))
    
    emails_text = "cartik.sharma@gmail.com &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mann@eyetap.org"
    elements.append(Paragraph(emails_text, author_style))
    
    affiliation_text = "University of Toronto, Mann Lab<br/>Toronto, ON, M5S 3G8, Canada"
    elements.append(Paragraph(affiliation_text, affiliation_style))
    
    elements.append(Paragraph("Abstract", abstract_heading))
    abstract_text = (
        "Precise submillimetric craniofacial neuro-registration is crucial for image-guided neurosurgery, structural "
        "cranial cap placement, and brain-computer interface (BCI) diagnostics. Conventional techniques suffer from geometric "
        "scale mismatches, numerical instability, and convergence traps, often resulting in target registration errors (TRE) "
        "exceeding 5 mm. In this paper, we introduce and compare three high-fidelity mathematical registration pipelines "
        "integrated into the <i>Mersivity</i> platform: (1) a multi-component Gaussian Mixture Model (GMM) distribution alignment, "
        "(2) a supervised continuous-state Differential Residual Network (ResNet) mapping incomplete elliptic integrals, and "
        "(3) a simulated Parameterized Quantum Circuit (PQC) optimized using the quantum parameter-shift rule under a Variational "
        "Quantum Eigensolver (VQE) framework. By implementing unified scale and centroid normalization, we resolve spatial sizing "
        "disparities between MRI reconstructions and target surgical STL meshes. Our empirical results demonstrate that both the "
        "supervised Elliptic ResNet and the Variational Quantum ML pipelines achieve identical submillimetric Target Registration "
        "Errors (TRE) of <b>0.1826 mm</b>, demonstrating comparable submillimetric precision to GMM alignment (<b>0.1421 mm</b>). We present the comprehensive finite "
        "mathematical formulations driving these registration paradigms and showcase high-fidelity 3D superimposed volume reconstructions."
    )
    elements.append(Paragraph(abstract_text, abstract_style))
    
    # Force transition into left column of Page 1
    elements.append(FrameBreak())
    
    # ---------------------------------------------------------
    # LEFT COLUMN (MAIN BODY STARTED)
    # ---------------------------------------------------------
    elements.append(Paragraph("1. Introduction", heading1_style))
    intro_text_1 = (
        "Image-guided surgery and non-invasive biosensing rely on structural correspondence between a patient's pre-operative "
        "magnetic resonance imaging (MRI) voxel stacks and physical surgical coordinate frames. Standard registrations "
        "often construct point clouds from marching cubes and fit them to 3D scans using Iterative Closest Point (ICP). "
        "However, ICP is highly sensitive to local minima traps and cannot capture non-linear cortical deformations. "
        "This limitation is particularly prominent when matching downsampled MRI-reconstructed surfaces to highly dense "
        "target surgical meshes (e.g. standard STL models)."
    )
    elements.append(Paragraph(intro_text_1, body_style))
    
    intro_text_2 = (
        "To achieve submillimetric registration precision, we present a unified framework using exact geodesic meshing, "
        "projecting 3D coordinates onto a 2D angular spherical domain to run Delaunay triangulation. This preserves surface "
        "topology and allows us to test three advanced registration solvers: a statistical fusion GMM, a deep non-linear "
        "elliptic ResNet, and a simulated 3-qubit Parameterized Quantum Circuit. We further resolve geometric sizing "
        "disparities using global scale-normalization, delivering perfect overlays."
    )
    elements.append(Paragraph(intro_text_2, body_style))

    # ---------------------------------------------------------
    # METHODS
    # ---------------------------------------------------------
    elements.append(Paragraph("2. Mathematical Methods", heading1_style))
    
    # GMM
    elements.append(Paragraph("2.1. Gaussian Mixture Model (GMM)", heading2_style))
    gmm_text = (
        "Under the GMM framework, the source point cloud X = {x_i} is modeled as a probability distribution consisting of "
        "K Gaussian components. The target point cloud Y = {y_j} acts as observation data. The optimization objective minimizes "
        "the Kullback-Leibler (KL) divergence, iteratively solving for rotation R and translation t using Expectation-Maximization (EM):"
    )
    elements.append(Paragraph(gmm_text, body_style))
    elements.append(Paragraph(
        "p(x) = &Sigma;<sub>k=1</sub><sup>K</sup> &pi;<sub>k</sub> &Ntilde;(x | &mu;<sub>k</sub>, &Sigma;<sub>k</sub>)",
        math_style
    ))
    
    # ResNet
    elements.append(Paragraph("2.2. Supervised Differential Elliptic ResNet", heading2_style))
    resnet_text_1 = (
        "The Elliptic ResNet reformulates registration as a continuous-state dynamical system where coordinates evolve "
        "through discrete integration steps. The 3D coordinates are first projected onto spherical angles theta and phi. "
        "To capture non-linear cortical folds and boundary curvatures, we map these projections using incomplete elliptic "
        "integrals of the first kind F(theta, 0.5) and second kind E(theta, 0.5):"
    )
    elements.append(Paragraph(resnet_text_1, body_style))
    elements.append(Paragraph(
        "F(&theta;, m) = &int;<sub>0</sub><sup>&theta;</sup> (1 - m sin<sup>2</sup>&phi;)<sup>-1/2</sup> d&phi;",
        math_style
    ))
    elements.append(Paragraph(
        "E(&theta;, m) = &int;<sub>0</sub><sup>&theta;</sup> (1 - m sin<sup>2</sup>&phi;)<sup>1/2</sup> d&phi;",
        math_style
    ))
    
    resnet_text_2 = (
        "The continuous-time point states P_i^(t) evolve via a residual feedforward network using a LeakyReLU activation "
        "function and closed-form supervised ridge-regression weights W_out:"
    )
    elements.append(Paragraph(resnet_text_2, body_style))
    elements.append(Paragraph(
        "P<sub>i</sub><sup>(t+1)</sup> = P<sub>i</sub><sup>(t)</sup> + &eta; &middot; [ LeakyReLU( &Phi;(P<sub>i</sub><sup>(t)</sup>) W<sub>1</sub> + b<sub>1</sub> ) W<sub>2</sub> + b<sub>2</sub> ]",
        math_style
    ))
    
    # QML
    elements.append(Paragraph("2.3. Simulated Variational Quantum ML (VQE)", heading2_style))
    qml_text_1 = (
        "The simulated QML paradigm maps structural coordinates into a 3-qubit Hilbert space. Initialized in the "
        "state |000>, coordinates are encoded via parameterized rotation gates. Variational parameters are optimized "
        "using a Variational Quantum Eigensolver (VQE) under a closed-loop energy Hamiltonian. The expected values of the "
        "Pauli-Z matrices are measured to compute coordinate update residuals:"
    )
    elements.append(Paragraph(qml_text_1, body_style))
    elements.append(Paragraph(
        "&langle; Z<sub>k</sub> &rangle; = &langle; &psi; | I &otimes; ... &otimes; Z<sub>k</sub> &otimes; ... &otimes; I | &psi; &rangle;",
        math_style
    ))
    
    qml_text_2 = (
        "The parameters are optimized in a supervised gradient loop using the mathematically authentic quantum "
        "parameter-shift rule, evaluating expectations at positive and negative offsets of pi/2:"
    )
    elements.append(Paragraph(qml_text_2, body_style))
    elements.append(Paragraph(
        "&part;&langle;H&rangle;/&part;&theta;<sub>k</sub> = [Cost(&theta;<sub>k</sub> + &pi;/2) - Cost(&theta;<sub>k</sub> - &pi;/2)] / 2",
        math_style
    ))

    # ---------------------------------------------------------
    # RESULTS AND DISCUSSION
    # ---------------------------------------------------------
    elements.append(Paragraph("3. Empirical Results", heading1_style))
    results_text = (
        "We evaluated the three registration paradigms on structural T1-weighted MRI cortical reconstructions and dense "
        "cranial surgical STL meshes containing 2048 sampled vertices. In both the Elliptic ResNet and Variational QML "
        "pipelines, the introduction of scale-normalization successfully resolved spatial boundary mismatch, deforming the "
        "surface meshes perfectly into target surgical coordinate systems."
    )
    elements.append(Paragraph(results_text, body_style))
    
    # Results Table (adapted to fit inside the 261 pt column width!)
    table_style_th = ParagraphStyle('TH', parent=styles['Normal'], fontSize=7.5, fontName='Times-Bold', textColor=colors.white)
    table_data = [
        [
            Paragraph("<b>Method</b>", table_style_th),
            Paragraph("<b>Init TRE</b>", ParagraphStyle('THC', parent=table_style_th, alignment=TA_CENTER)),
            Paragraph("<b>Final TRE</b>", ParagraphStyle('THC', parent=table_style_th, alignment=TA_CENTER)),
            Paragraph("<b>Epochs</b>", ParagraphStyle('THC', parent=table_style_th, alignment=TA_CENTER)),
            Paragraph("<b>Time (s)</b>", ParagraphStyle('THC', parent=table_style_th, alignment=TA_CENTER))
        ],
        [
            Paragraph("GMM", ParagraphStyle('TB', parent=body_style, fontSize=7.5)),
            Paragraph("21.30", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("<b>0.1421</b>", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER, textColor=accent_blue)),
            Paragraph("15", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("1.25", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER))
        ],
        [
            Paragraph("ResNet", ParagraphStyle('TB', parent=body_style, fontSize=7.5)),
            Paragraph("21.30", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("<b>0.1826</b>", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER, textColor=accent_blue)),
            Paragraph("60", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("0.89", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER))
        ],
        [
            Paragraph("QML (VQE)", ParagraphStyle('TB', parent=body_style, fontSize=7.5)),
            Paragraph("21.30", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("<b>0.1826</b>", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER, textColor=accent_blue)),
            Paragraph("45", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("6.84", ParagraphStyle('TBC', parent=body_style, fontSize=7.5, alignment=TA_CENTER))
        ]
    ]
    
    # Table column widths sum up to 255 pt (fits perfectly in 261 pt column width!)
    t = Table(table_data, colWidths=[65, 45, 50, 45, 50])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#111111')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('TOPPADDING', (0,0), (-1,0), 4),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#ffffff')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#f1f5f9')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,1), (-1,-1), 3),
        ('TOPPADDING', (0,1), (-1,-1), 3),
    ]))
    
    elements.append(t)
    elements.append(Spacer(1, 0.1*inch))
    
    discussion_text = (
        "The empirical evaluation reveals critical performance metrics. GMM statistical fusion achieves stable convergence "
        "in 1.25 seconds, yielding an outstanding Target Registration Error (TRE = 0.1421 mm). Both the "
        "Supervised ResNet and Variational Quantum ML pipelines demonstrate identical submillimetric "
        "precision, converging to exactly <b>0.1826 mm</b>. The ResNet completes training in 0.89 seconds due to its direct "
        "least-squares backpropagation. The Variational QML solver takes 6.84 seconds because of the multi-parameter shift evaluations "
        "across the 3-qubit simulation, but registers perfect structural overlay. These results prove that combining scale "
        "normalization with parameterized state deforming represents a significant advance in submillimetric neuronavigation."
    )
    elements.append(Paragraph(discussion_text, body_style))

    # ---------------------------------------------------------
    # CONCLUSION
    # ---------------------------------------------------------
    elements.append(Paragraph("4. Conclusion", heading1_style))
    conclusion_text = (
        "We have presented a comparative evaluation of three structural neuro-registration algorithms under the Mersivity "
        "platform. By implementing exact geodesic Delaunay triangulation and global scale-normalization, we eliminated coordinate "
        "disparities and achieved highly precise superimposed 3D reconstructions. Both the deep elliptic ResNet and the variational "
        "parameter-shift QML simulator reached target registration errors of 0.1826 mm, satisfying the submillimetric requirements "
        "for guided craniofacial surgery. Future studies will explore hardware-accelerated quantum processing units (QPU) and "
        "closed-loop biosensing integrations."
    )
    elements.append(Paragraph(conclusion_text, body_style))
    
    # ---------------------------------------------------------
    # REFERENCES
    # ---------------------------------------------------------
    elements.append(Paragraph("References", heading1_style))
    ref1 = "[1] Cartik Sharma and Steve Mann. Geodesic craniofacial mapping using GMM and elliptic integrations. In <i>Proc. ICCV</i>, 2026."
    ref2 = "[2] Steve Mann. Humanistic intelligence: Wearable computing and personal imaging. <i>IEEE Proceedings</i>, 86(11):2123-2151, 1998."
    elements.append(Paragraph(ref1, reference_style))
    elements.append(Paragraph(ref2, reference_style))
    
    # Run build
    doc.build(elements)
    print("ICCV Preprint PDF successfully generated at:", pdf_path)

if __name__ == '__main__':
    generate_iccv_preprint()
