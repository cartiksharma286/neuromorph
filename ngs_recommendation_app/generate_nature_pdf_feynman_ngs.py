#!/usr/bin/env python3
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors

def generate_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Nature_Preprint_Quantum_Feynman_NGS_Pipelines_Finite_Math.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=15, spaceAfter=12, alignment=TA_CENTER, fontName='Helvetica-Bold')
    author_style = ParagraphStyle('Authors', parent=styles['Normal'], fontSize=10, spaceAfter=12, alignment=TA_CENTER)
    abstract_style = ParagraphStyle('Abstract', parent=styles['Normal'], fontSize=10, leading=14, alignment=TA_JUSTIFY, fontName='Helvetica-Oblique', leftIndent=0.5*inch, rightIndent=0.5*inch, spaceAfter=12)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, spaceAfter=8, spaceBefore=12, fontName='Helvetica-Bold')
    body_style = ParagraphStyle('BodyText', parent=styles['BodyText'], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    math_style = ParagraphStyle('MathText', parent=styles['BodyText'], fontSize=11, leading=16, alignment=TA_CENTER, fontName='Times-Italic', spaceBefore=6, spaceAfter=6, textColor=colors.HexColor('#0f4c81'))

    elements = []

    elements.append(Paragraph("Feynman Path Integrals and Finite Mathematical Models for Supervised Geodesic Genomics Workflow Engine", title_style))
    elements.append(Paragraph("Cartik Sharma, Neuromorph System", author_style))
    elements.append(Spacer(1, 0.2*inch))

    abstract = "Abstract: Robust genomic pipeline selection presents a multi-criteria network optimization problem across complex sequencing spaces. In this work, we integrate Feynman path integral formulation into a recommendation engine. We define the discrete propagator and action boundaries across transition configurations to produce path-resolved coherence metrics. Combined with supervised geodesic geodesic state minimization, this mathematically grounded approach maps out optimal biological, operational, and sequencing configurations cleanly. Here we formulate the discrete Lagrangian and path sum formulations to define stability conditions."
    elements.append(Paragraph(abstract, abstract_style))
    elements.append(Spacer(1, 0.2*inch))

    sections = [
        ("1. Discrete Path Action Formulation", 
         "To calculate optimal pathway configurations, we describe the transition from an input query state inside a discrete pipeline space using a sequence of intermediate nodes. The total discrete action S along a pathway sequence is evaluated using:",
         "S[x] = &sum;<sub>t=1</sub><sup>N</sup> [ (x<sub>t</sub> - x<sub>t-1</sub>)<sup>2</sup> / (2&Delta;t) - V(x<sub>t</sub>) ] &middot; &Delta;t",
         "where x represents the state checkpoints, &Delta;t is the finite time step, and V describes the operational constraints such as low-quality FFPE samples, bioinformatic throughput overhead, or clinical urgency requirements."),
        
        ("2. Feynman Path Integrals over Pipeline States", 
         "The workflow recommendation and coherence metrics are derived from the propagator K. It summates phase factors across all allowed sequences in our finite network configuration space:",
         "K(x<sub>0</sub>, x<sub>N</sub>) = &sum;<sub>paths</sub> exp( i &middot; S[x] / &#295; )",
         "where &#295; is the scale of the quantum-inspired decision field. The total magnitude of this vector is mapped to the final pipeline recommendation score."),
        
        ("3. Supervised Geodesic Path Prediction", 
         "We model genomic pathway transition trajectories as geodesics on a discrete manifold. Minimizing the path distance metrics yields an optimal path trajectory matching wet-lab constraints:",
         "d(x, y) = inf &sum;<sub>k=1</sub><sup>M</sup> || x<sub>k</sub> - x<sub>k-1</sub> ||<sub>g</sub>",
         "under our coordinate space, mapping the most efficient transition trajectory across DNA sequencing runs directly to clinical outcomes."),
         
        ("4. Pipeline Coherence Bounds", 
         "The final selection confidence is defined as a normalized magnitude. The system guarantees that path coherence is strictly bounded above zero, precluding analytical divergence under irregular sample profiles:",
         "0.0 &lt; |K| / N &le; 1.0",
         "The optimal sequencing pipeline choice is selected from our asset mapping whenever path coherence exceeds the specified threshold.")
    ]

    for title, text, math_eq, post_text in sections:
        elements.append(Paragraph(title, heading_style))
        elements.append(Paragraph(text, body_style))
        elements.append(Paragraph(math_eq, math_style))
        elements.append(Paragraph(post_text, body_style))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("5. Conclusion", heading_style))
    elements.append(Paragraph("Our hybrid recommendation engine utilizes the computational mechanics of Feynman path integrals to map and prioritize high-performance next-generation sequencing pipelines. Modeling decision paths as discrete geodesics preserves analytical accuracy, topological stability, and biological integrity.", body_style))

    doc.build(elements)
    print(f"Successfully generated: {os.path.abspath(pdf_path)}")

if __name__ == '__main__':
    generate_pdf()