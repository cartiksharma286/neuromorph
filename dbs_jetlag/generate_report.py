from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import os

def add_equation(paragraphs, eqn):
    # Render LaTeX as plain text for now
    paragraphs.append(Paragraph(f"<b>Equation:</b> {eqn}", styles['Normal']))
    paragraphs.append(Spacer(1, 0.2*inch))

styles = getSampleStyleSheet()
doc = SimpleDocTemplate("DBS_JetLag_CVPR_Report.pdf", pagesize=letter)
story = []

story.append(Paragraph("<b>Finite Element Analysis and Mathematical Modeling of Deep Brain Stimulation for Jet Lag Repair</b>", styles['Title']))
story.append(Paragraph("Author: Cartik Sharma", styles['Normal']))
story.append(Paragraph("Date: April 28, 2026", styles['Normal']))
story.append(Spacer(1, 0.3*inch))

story.append(Paragraph("<b>Introduction</b>", styles['Heading2']))
story.append(Paragraph("Jet lag is a circadian rhythm disorder caused by rapid travel across time zones. Deep Brain Stimulation (DBS) is explored as a novel intervention for accelerating circadian realignment. This report presents a mathematical and finite element analysis (FEA) framework for simulating DBS-based jet lag repair.", styles['Normal']))
story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("<b>Mathematical Model</b>", styles['Heading2']))
story.append(Paragraph("Let a(t) ∈ ℝᴺ be the neuronal activity vector at time t for N neurons. The evolution is governed by:", styles['Normal']))
add_equation(story, "da/dt = W a(t) + u_DBS(t) - d_apnea(t)")
story.append(Paragraph("where W is the synaptic weight matrix, u_DBS(t) is the DBS input, and d_apnea(t) models sleep apnea events.", styles['Normal']))

story.append(Paragraph("<b>DBS Input</b>", styles['Heading3']))
story.append(Paragraph("DBS is modeled as a periodic or continuous input:", styles['Normal']))
add_equation(story, "u_DBS(t) = I_DBS * exp(-||x - x0|| / λ)")
story.append(Paragraph("where I_DBS is the intensity, x0 is the electrode location, λ is the decay constant, and x is the spatial coordinate.", styles['Normal']))

story.append(Paragraph("<b>Sleep Apnea Events</b>", styles['Heading3']))
story.append(Paragraph("Sleep apnea is modeled as a stochastic multiplicative drop:", styles['Normal']))
add_equation(story, "d_apnea(t) = α * a(t) * ξ(t)")
story.append(Paragraph("where α is the apnea factor and ξ(t) is a Bernoulli random variable.", styles['Normal']))

story.append(Paragraph("<b>Finite Element Analysis (FEA) of DBS Field</b>", styles['Heading2']))
story.append(Paragraph("The DBS electric field φ in tissue is governed by the Poisson equation:", styles['Normal']))
add_equation(story, "∇·(σ∇φ) = -I_DBS δ(x - x0)")
story.append(Paragraph("where σ is tissue conductivity and δ is the Dirac delta at the electrode.", styles['Normal']))
add_equation(story, "Σ_{j∈N(i)} (σ_ij/h²) (φ_j - φ_i) = -I_DBS δ_{i,0}")
story.append(Paragraph("where N(i) are neighbors of node i, h is grid spacing, and δ_{i,0} is 1 at the electrode node.", styles['Normal']))

story.append(Paragraph("<b>Simulation Results</b>", styles['Heading2']))
if os.path.exists("activity_heatmap.png"):
    story.append(Paragraph("Neuronal Activity Heatmap:", styles['Heading3']))
    story.append(Image("activity_heatmap.png", width=5*inch, height=2*inch))
    story.append(Spacer(1, 0.2*inch))
if os.path.exists("fea_field.png"):
    story.append(Paragraph("FEA DBS Field:", styles['Heading3']))
    story.append(Image("fea_field.png", width=3*inch, height=3*inch))
    story.append(Spacer(1, 0.2*inch))

story.append(Paragraph("<b>Discussion</b>", styles['Heading2']))
story.append(Paragraph("The model demonstrates that DBS can accelerate circadian realignment by amplifying neuronal activity, while FEA shows the spatial decay of the DBS field. The simulation provides a framework for optimizing DBS parameters for jet lag repair.", styles['Normal']))

doc.build(story)
print("PDF report generated: DBS_JetLag_CVPR_Report.pdf")
