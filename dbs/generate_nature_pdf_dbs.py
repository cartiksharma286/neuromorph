from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_nature_pdf(filename):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'NatureTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=14,
        textColor=colors.darkblue
    )
    
    author_style = ParagraphStyle(
        'NatureAuthor',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        spaceAfter=20
    )
    
    body_style = ParagraphStyle(
        'NatureBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        spaceAfter=12,
        leading=14
    )
    
    math_style = ParagraphStyle(
        'NatureMath',
        parent=styles['Normal'],
        fontName='Courier-Oblique',
        fontSize=11,
        leftIndent=20,
        spaceAfter=14,
        textColor=colors.darkred
    )
    
    abstract_style = ParagraphStyle(
        'NatureAbstract',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        spaceAfter=14,
        leftIndent=30,
        rightIndent=30
    )

    story = []
    
    story.append(Paragraph("Generative Artificial Intelligence and Finite Mathematical Paradigms for Deep Brain Stimulation in Dementia", title_style))
    story.append(Paragraph("Neuromorph DBS Research Group | Cartik Sharma", author_style))
    
    abstract_text = ("Abstract: This paper presents a novel integration of generative artificial intelligence "
                     "coupled with finite mathematical paradigms to model longitudinal cognitive trajectories "
                     "in dementia via generalized Deep Brain Stimulation (DBS). We propose an optimal treatment "
                     "protocol alongside finite element analysis (FEA) of cortical surface manifolds for precise "
                     "radiofrequency electromagnetic field shaping.")
    story.append(Paragraph(abstract_text, abstract_style))
    
    story.append(Paragraph("1. Introduction", styles['Heading2']))
    intro_txt = ("Dementia phenotypes exhibit nonlinear temporal progression that confounds static "
                 "Deep Brain Stimulation models. Leveraging finite mathematical decay models paired with "
                 "Hebbian plasticity constraints, we redefine DBS parameters incorporating generative AI retention limits.")
    story.append(Paragraph(intro_txt, body_style))
    
    story.append(Paragraph("2. Finite Math Equations for Cognitive Trajectory", styles['Heading2']))
    body_txt1 = ("We model the baseline cognitive severity decay S(t) utilizing an initial state S_0 "
                 "modulated by a dynamic coefficient beta mapped across a 60-month finite interval. The core mathematical formulation operates as:")
    story.append(Paragraph(body_txt1, body_style))
    
    math1 = ("S(t) = S_0 * exp(-(beta * (1 + 0.1 * sin(t))) * (t/12)) + Delta_DBS")
    story.append(Paragraph(math1, math_style))
    
    body_txt2 = ("Where Delta_DBS is the regenerative Hebbian influence governed by generative parameter amplitude, given by:")
    story.append(Paragraph(body_txt2, body_style))
    
    math2 = ("Delta_DBS = (Amplitude * 0.4) * Phi_AI * (1 - exp(-t/12))")
    story.append(Paragraph(math2, math_style))
    
    story.append(Paragraph("3. Cortical FEA and Toroidal RF Topology", styles['Heading2']))
    body_txt3 = ("To optimize field specificity and limit isotropic conduction bleed, a finite element analysis "
                 "of a cortical icosahedron proxy operates with a toroidally bound RF node mesh. The electromagnetic flux (Phi_RF) constraint functions as:")
    story.append(Paragraph(body_txt3, body_style))
    
    math3 = ("Phi_RF(r, theta) = sigma(r) * B_0 * cos(theta + phi_var) + Sum(epsilon_EM)")
    story.append(Paragraph(math3, math_style))
    
    story.append(Paragraph("Conclusion", styles['Heading2']))
    conclusion = ("Integrating temporal finite regression math via generative constraints with high-fidelity "
                  "radiofrequency electromagnetic volumetric mapping delivers unprecedented theoretical yield "
                  "for non-destructive staging of dementia via deep brain stimulation architectures.")
    story.append(Paragraph(conclusion, body_style))
    
    doc.build(story)

if __name__ == '__main__':
    create_nature_pdf("Nature_DBS_Dementia_FEA.pdf")
    print("Generated Nature_DBS_Dementia_FEA.pdf successfully")
