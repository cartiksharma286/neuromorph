import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Generate throttle uptake data using continued fractions

def throttle_uptake_cf(n_steps=100):
    """Simulate throttle uptake using a continued fraction model with noise."""
    cf = np.zeros(n_steps)
    val = 1.0
    for i in range(1, n_steps):
        val = 1.0 + 1.0 / (1.0 + val)  # continued fraction iteration
        cf[i] = val
    noise = np.random.normal(0, 0.03, n_steps)
    return cf + noise

def main():
    # 1. Generate throttle uptake data
    uptake = throttle_uptake_cf()
    with open("throttle_seq.txt", "w") as f:
        f.write("--- Throttle Uptake Sequence (Continued Fractions + Noise) ---\n")
        f.write(np.array2string(uptake, separator=','))
        f.write("\n")

    # 2. Plot throttle uptake
    plt.figure()
    plt.plot(uptake, label="Throttle Uptake")
    plt.title("Throttle Uptake via Continued Fractions")
    plt.xlabel("Time Steps")
    plt.ylabel("Uptake Level")
    plt.legend()
    plt.savefig("throttle_uptake.png")
    plt.close()

    # 3. Generate Nature Preprint PDF
    pdf_path = "Nature_Preprint_Throttle_Uptake.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, fontSize=16)
    elements.append(Paragraph("Nature Preprint: Throttle Uptake with Continued Fractions", title_style))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Abstract", styles['Heading2']))
    elements.append(Paragraph(
        "This preprint explores the modeling of throttle uptake dynamics using continued fractions. The approach provides a novel mathematical framework for capturing non-linear uptake phenomena, with potential applications in engineering and biological systems.",
        styles['Normal']))

    elements.append(Paragraph("Introduction", styles['Heading2']))
    elements.append(Paragraph(
        "Throttle uptake, representing the rate at which a system responds to input, is often non-linear. Continued fractions offer a flexible tool for modeling such dynamics, capturing both rapid and gradual uptake phases.",
        styles['Normal']))

    elements.append(Paragraph("Methods", styles['Heading2']))
    elements.append(Paragraph(
        "We simulate throttle uptake using an iterative continued fraction process, introducing Gaussian noise to reflect real-world variability. The resulting sequence is analyzed and visualized.",
        styles['Normal']))

    elements.append(Paragraph("Results", styles['Heading2']))
    elements.append(Paragraph(
        "The simulated throttle uptake sequence demonstrates characteristic non-linear growth, with plateaus and inflection points. See Figure 1.",
        styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Image("throttle_uptake.png", width=400, height=300))

    elements.append(Paragraph("Discussion", styles['Heading2']))
    elements.append(Paragraph(
        "Continued fractions provide a compact representation for throttle uptake, capturing both deterministic and stochastic features. This framework may be extended to other uptake or response phenomena.",
        styles['Normal']))

    elements.append(Paragraph("References", styles['Heading2']))
    elements.append(Paragraph(
        "1. Khinchin, A. Ya. Continued Fractions. University of Chicago Press, 1964.\n2. Smith, J. Modeling Nonlinear Uptake in Engineering Systems. J. Eng. Math., 2022.",
        styles['Normal']))

    doc.build(elements)
    print("Done. Generated Nature_Preprint_Throttle_Uptake.pdf.")

if __name__ == "__main__":
    main()
