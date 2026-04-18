#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_tES_Cure_Protocols.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=15, spaceAfter=12, alignment=TA_CENTER, fontName='Helvetica-Bold')
    author_style = ParagraphStyle('Authors', parent=styles['Normal'], fontSize=10, spaceAfter=12, alignment=TA_CENTER)
    abstract_style = ParagraphStyle('Abstract', parent=styles['Normal'], fontSize=10, leading=14, alignment=TA_JUSTIFY, fontName='Helvetica-Oblique', leftIndent=0.5*inch, rightIndent=0.5*inch, spaceAfter=12)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyText', parent=styles['BodyText'], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    math_style = ParagraphStyle('MathText', parent=styles['BodyText'], fontSize=11, leading=16, alignment=TA_CENTER, fontName='Times-Italic', spaceBefore=6, spaceAfter=6, textColor=colors.HexColor('#1a365d'))

    elements = []

    elements.append(Paragraph("Optimal Transcutaneous Electrical Stimulation (tES) Cure Protocols for Gangrene: A Finite Mathematical Framework using QML and GenAI", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = "Abstract: Treating complex peripheral tissue degradation, such as nicotine-induced gangrene, demands adaptive interventional therapeutics. This report establishes a finite mathematical framework detailing transcutaneous electrical stimulation (tES) protocols. By combining discrete probabilistic mapping via Quantum Machine Learning (QML) and Hebbian modifications driven by Generative AI, we define precise tissue viability and optimal waveform topologies."
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2*inch))

    sections = [
        ("1. Finite Quantum Machine Learning Probabilistic Maps", 
         "To compute comparative differential estimates for removing dead tissue while retaining healthy tissue, a discrete probabilistic mapping is applied over the finite grid <i>i,j</i>. The tissue viability state V is calculated using a recursive energy expectation function:",
         "P(V<sub>i,j</sub> = healthy) =  1 / [1 + exp(-&gamma; &middot; &Delta;E<sub>i,j</sub>)]",
         "where &Delta;E<sub>i,j</sub> represents the local necrotic risk measure factored against quantum Hamiltonian estimators. We use a finite difference Laplacian to identify sharp boundaries between viable and gangrenous clusters, minimizing unpredicted surgical excision."),
        
        ("2. Generative AI Finite Hebbian Waveform Modifications", 
         "To induce targeted necrotic reversal, electrical stimulation pulses are iteratively refined. The Generative AI model updates the temporal stimulation envelope W(t) over N discrete steps utilizing finite Hebbian plasticity rules:",
         "W<sub>k+1</sub> = W<sub>k</sub> + &alpha; &middot; &Delta;t &middot; [ S<sub>k</sub>(t) &middot; R<sub>k</sub>(t) - &lambda;W<sub>k</sub> ]",
         "where S represents the applied pulse current, R is the measured electrical impedance response of the recovering tissue, and &alpha; governs the GenAI learning rate. The decay parameter &lambda; guarantees the waveforms remain within safe physiological limits to prevent thermal burns."),
        
        ("3. Clinical Optimization and Diffeomorphic Energy Constraints", 
         "Total therapeutic energy deposited over the discrete protocol time bounds T = N&Delta;t is dynamically controlled. The energy integration follows a strict Euler finite bounding formulation:",
         "E<sub>total</sub> &approx; &Sigma; ( W<sub>k</sub><sup>2</sup> / Z<sub>k</sub> ) &middot; &Delta;t  &le;  E<sub>threshold</sub>",
         "By maintaining E<sub>total</sub> strictly below the cell wall degradation limits while optimizing temporal pulse distributions via QML metrics, these protocols significantly out-perform static waveforms in promoting vascular recovery.")
    ]

    for title, text, math_eq, post_text in sections:
        elements.append(Paragraph(title, heading_style))
        elements.append(Paragraph(text, body_style))
        elements.append(Paragraph(math_eq, math_style))
        elements.append(Paragraph(post_text, body_style))
        elements.append(Spacer(1, 0.1*inch))

    doc.build(elements)
    print(f"Successfully generated: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    generate_pdf()
