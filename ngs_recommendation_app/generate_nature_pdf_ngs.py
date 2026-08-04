#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Quantum_NGS_Pipelines_Finite_Math.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=15, spaceAfter=12, alignment=TA_CENTER, fontName='Helvetica-Bold')
    author_style = ParagraphStyle('Authors', parent=styles['Normal'], fontSize=10, spaceAfter=12, alignment=TA_CENTER)
    abstract_style = ParagraphStyle('Abstract', parent=styles['Normal'], fontSize=10, leading=14, alignment=TA_JUSTIFY, fontName='Helvetica-Oblique', leftIndent=0.5*inch, rightIndent=0.5*inch, spaceAfter=12)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyText', parent=styles['BodyText'], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    math_style = ParagraphStyle('MathText', parent=styles['BodyText'], fontSize=11, leading=16, alignment=TA_CENTER, fontName='Times-Italic', spaceBefore=6, spaceAfter=6, textColor=colors.HexColor('#125a56'))

    elements = []

    elements.append(Paragraph("Finite Mathematical Foundations of Quantum-Scored NGS Recommendation Pipelines for Clinical and Wet-Lab Workflows", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = "Abstract: Selecting optimal next-generation sequencing (NGS) pipeline configurations presents a complex multi-criteria optimization problem spanning diverse clinical targets, sample qualities, and computational constraints. In this blueprint, we formalize the finite mathematical and discrete numerical foundations of our hybrid recommendation engine. We define discrete input mappings across wet-lab and bioinformatic variables, construct a finite quantum-inspired scoring state, and specify the scoring boundaries that drive adaptive pipeline mapping. This formal framework stabilizes decision logic and guarantees uniform optimization path selection under variable sample conditions."
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2*inch))

    sections = [
        ("1. Finite Input-Space Mapping", 
         "We represent the clinician and bioinformatic query space by a finite set of features. Let each input attribute correspond to an index in a finite discrete domain. We designate the primary clinical goal, sample preparation status, turnaround urgency, and target complexity as discrete values within a partition space:",
         "S<sub>i</sub> &isin; { s<sub>1</sub>, s<sub>2</sub>, ..., s<sub>K</sub> } &sub; &mathbb{F}<sub>p</sub>",
         "where each input state is mapped onto a finite field representing categories of sample attributes (e.g., blood, tissue, FFPE, saliva). This finite representation ensures a bounded search space for downstream evaluation."),
        
        ("2. Quantum-Inspired scoring state vector", 
         "To compute adaptive pipeline blueprints, we construct a discrete state vector represented in a finite-dimensional Hilbert space. The state is driven by input score parameters and is defined using the discrete components:",
         "|&Psi;&rang; = c<sub>1</sub> |Coherence&rang; + c<sub>2</sub> |Entropy&rang; + c<sub>3</sub> |State&rang;",
         "where coefficients correspond to the calculated values for operational coherence and entropy based on input properties. These weights represent active signals from the wet-lab and bioinformatic constraints."),
        
        ("3. Adaptive scoring update rule", 
         "The recommendation engine modifies the quantum-inspired scoring variables in discrete time-steps based on priority weights. For instance, when urgent conditions are selected, the state variables are updated via the finite mapping:",
         "g<sub>t+1</sub> = g<sub>t</sub> + &Delta; &middot; M<sub>t</sub>",
         "where the change parameter is bounded explicitly to prevent divergence, guaranteeing stable and reproducible recommendations under varying diagnostic priorities."),
         
        ("4. Pipeline alignment confidence logic", 
         "Alignments are selected using a discrete sum-of-squares calculation across the target criteria, comparing platform capabilities against input conditions over our finite asset database consisting of sequencer parameters:",
         "E = &sum;<sub>n=1</sub><sup>M</sup> w<sub>n</sub> ( P<sub>n</sub> - U<sub>n</sub> )<sup>2</sup>",
         "The minimized error distance matches the optimal pipeline configuration (such as NovaSeq or ONT) to satisfy the analytical requirements of the biologists and bioinformaticians.")
    ]

    for title, text, math_eq, post_text in sections:
        elements.append(Paragraph(title, heading_style))
        elements.append(Paragraph(text, body_style))
        elements.append(Paragraph(math_eq, math_style))
        elements.append(Paragraph(post_text, body_style))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("5. Conclusion", heading_style))
    elements.append(Paragraph("Through the use of explicit finite mathematical transforms and quantum-inspired scoring state representations, we provide clinical and bioinformatic operators with a robust recommendation system. This guarantees consistent, high-throughput pipelines optimized for next-generation sequencing.", body_style))

    doc.build(elements)
    print(f"Successfully generated: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    generate_pdf()