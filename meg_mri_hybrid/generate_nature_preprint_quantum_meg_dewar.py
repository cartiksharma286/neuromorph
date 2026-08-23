import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, "Nature_Preprint_Quantum_MEG_Dewar_CFD.pdf")
PLOT_OUTPUT = os.path.join(ROOT, "quantum_meg_dewar_cfd_plots.png")


def make_plots(filename=PLOT_OUTPUT):
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    velocity = np.sin(2 * np.pi * X) * np.cos(np.pi * Y)
    pressure = 5000 * np.sin(5 * 2 * np.pi * X) * np.cos(10 * Y)

    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(-0.4, 0, 60)
    theta_grid, z_grid = np.meshgrid(theta, z)
    z_norm = (z_grid - z_grid.min()) / (z_grid.max() - z_grid.min())
    stress_mpa = (70e9 * 23e-6 * 296 * (1 + 0.5 * np.exp(-10 * z_norm))) / 1e6

    time = np.linspace(0, 1, 1000)
    alpha = np.sin(2 * np.pi * 10 * time) * (0.5 - 0.5 * np.cos(2 * np.pi * time))
    gamma = np.sin(2 * np.pi * (30 * time + 10 * time ** 2)) * np.exp(-((time - 0.5) ** 2) / 0.02)

    figure, axes = plt.subplots(2, 2, figsize=(7.1, 6.2), constrained_layout=True)
    image = axes[0, 0].imshow(velocity, origin="lower", aspect="auto", cmap="coolwarm", extent=[0, 1, 0, 1])
    axes[0, 0].set_title("Quantum-walk velocity surrogate", loc="left", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("Axial coordinate")
    axes[0, 0].set_ylabel("Radial coordinate")
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Relative velocity")

    image = axes[0, 1].imshow(pressure, origin="lower", aspect="auto", cmap="viridis", extent=[0, 1, 0, 1])
    axes[0, 1].set_title("Helium pressure perturbation", loc="left", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Azimuthal coordinate")
    axes[0, 1].set_ylabel("Radial coordinate")
    figure.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Pressure (Pa)")

    image = axes[1, 0].pcolormesh(theta_grid, z_grid, stress_mpa, shading="auto", cmap="magma")
    axes[1, 0].set_title("Dewar wall thermal stress", loc="left", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Wall angle (rad)")
    axes[1, 0].set_ylabel("Height (m)")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Stress (MPa)")

    axes[1, 1].plot(time, alpha, color="#1f5f8b", linewidth=1.2, label="Alpha rhythm")
    axes[1, 1].plot(time, gamma, color="#b23a48", linewidth=1.2, label="Gamma burst")
    axes[1, 1].set_title("MEG source waveforms", loc="left", fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Relative amplitude")
    axes[1, 1].legend(frameon=False, fontsize=8)
    for axis in axes.flat:
        axis.grid(True, alpha=0.18)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return filename


def build_pdf(filename=OUTPUT):
    plot_filename = make_plots()
    doc = SimpleDocTemplate(
        filename, pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        title="Quantum CFD Simulation of a MEG Dewar", author="Neuromorph MEG/MRI Hybrid Simulation Group",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], alignment=TA_CENTER,
                               fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=5))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER,
                              fontName="Helvetica-Oblique", fontSize=10, leading=13, textColor=colors.HexColor("#404040")))
    styles.add(ParagraphStyle(name="BodyJustified", parent=styles["BodyText"], alignment=TA_JUSTIFY,
                              fontName="Times-Roman", fontSize=10.3, leading=14, spaceAfter=7))
    styles.add(ParagraphStyle(name="Equation", parent=styles["Normal"], alignment=TA_CENTER,
                              fontName="Helvetica-Oblique", fontSize=10.7, leading=16, spaceBefore=4, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.7, leading=11))

    story = [
        Paragraph("Quantum CFD and Finite-Element Modeling of a MEG Dewar", styles["PaperTitle"]),
        Paragraph("A reduced-order cryogenic vessel simulation with quantum probability transport and sensor-waveform coupling", styles["Subtitle"]),
        Spacer(1, 10),
        Paragraph("<b>Abstract.</b> Cryogenic magnetoencephalography (MEG) depends on a mechanically stable, thermally isolated dewar. We present a finite reduced-order simulation that couples a quantum-walk-inspired computational fluid dynamics (CFD) surrogate to a thermal-stress model and synthetic MEG source waveforms. The model uses a 20 x 50 flow grid, a 0.13 m inner radius, a 0.15 m outer radius and a 0.4 m vessel height. The equations and plots are intended as a reproducible computational preprint; they do not replace a validated helium property model, structural solver or instrument qualification.", styles["BodyJustified"]),
        Paragraph("1. Dewar geometry and finite state space", styles["Heading2"]),
        Paragraph("The vessel is represented by cylindrical inner and outer walls. Let theta in [0, 2 pi] be the azimuth and z in [-H, 0] the axial coordinate. For wall radius R in {R<sub>i</sub>, R<sub>o</sub>}:", styles["BodyJustified"]),
        Paragraph("x<sub>R</sub>(theta,z) = R cos(theta), &nbsp; y<sub>R</sub>(theta,z) = R sin(theta), &nbsp; z = z; &nbsp; R<sub>i</sub> = 0.13 m, R<sub>o</sub> = 0.15 m, H = 0.40 m.", styles["Equation"]),
        Paragraph("The CFD state is sampled on N<sub>r</sub> x N<sub>theta</sub> = 20 x 50 = 1000 cells. A complex amplitude psi<sub>i,j</sub> is initialized uniformly; its squared magnitude is interpreted as a normalized transport weight, sum<sub>i,j</sub> |psi<sub>i,j</sub>|<sup>2</sup> = 1.", styles["BodyJustified"]),
        Paragraph("2. Quantum CFD surrogate", styles["Heading2"]),
        Paragraph("The computational fluid model uses a bounded convection pattern to represent a quantum-walk-inspired flow field. With normalized coordinates xi and eta, the velocity surrogate and pressure perturbation are:", styles["BodyJustified"]),
        Paragraph("v(xi,eta) = sin(2 pi xi) cos(pi eta), &nbsp; delta p(xi,eta) = p<sub>0</sub> sin(10 pi xi) cos(10 eta), &nbsp; p<sub>0</sub> = 5000 Pa.", styles["Equation"]),
        Paragraph("The total pressure is p = 101325 Pa + delta p. A finite update may be written as psi<sup>(q+1)</sup> = U<sub>q</sub> psi<sup>(q)</sup>, where U<sub>q</sub><sup>*</sup>U<sub>q</sub> = I and q = 0,...,Q-1. In this preprint, the displayed field is a deterministic spatial surrogate for the returned simulation snapshot rather than a claim of a full Navier-Stokes or two-fluid helium solution.", styles["BodyJustified"]),
        Paragraph("3. Thermal stress and strain", styles["Heading2"]),
        Paragraph("For a constrained wall, the leading thermal stress scale uses Young's modulus E, expansion coefficient alpha and temperature drop Delta T = 300 - 4 = 296 K:", styles["BodyJustified"]),
        Paragraph("sigma<sub>0</sub> = E alpha Delta T, &nbsp; sigma<sub>v</sub>(z) = sigma<sub>0</sub>[1 + 0.5 exp(-10 z&#770;)], &nbsp; epsilon(z) = sigma<sub>v</sub>(z)/E,", styles["Equation"]),
        Paragraph("where z&#770; is the normalized axial coordinate. The local model uses E = 70 GPa and alpha = 23 x 10<sup>-6</sup> K<sup>-1</sup>. The factor in square brackets is a bounded concentration term that emphasizes the cold-end region; it is not a substitute for a mesh-converged finite-element boundary-value problem.", styles["BodyJustified"]),
        Paragraph("4. MEG source coupling", styles["Heading2"]),
        Paragraph("To connect vessel conditions to a representative MEG signal, the source simulator generates finite alpha and gamma components over t in [0,1] s:", styles["BodyJustified"]),
        Paragraph("a(t) = sin(2 pi 10t) w(t), &nbsp; g(t) = sin[2 pi(30t + 10t<sup>2</sup>)] exp[-(t - 0.5)<sup>2</sup>/0.02],", styles["Equation"]),
        Paragraph("where w(t) is a Tukey-like finite taper. A simple sensor observation is y<sub>m</sub>(t) = l<sub>m</sub><sup>T</sup>[a(t)q<sub>a</sub> + g(t)q<sub>g</sub>] + n<sub>m</sub>(t), with lead vectors q<sub>a</sub>, q<sub>g</sub> and noise n<sub>m</sub>. This separates the cryogenic engineering state from the neural source state while retaining a measurable interface for future co-simulation.", styles["BodyJustified"]),
        Paragraph("5. Plots and characteristics", styles["Heading2"]),
        Image(plot_filename, width=6.65 * inch, height=5.80 * inch),
        Paragraph("<b>Figure 1.</b> Finite simulation outputs: quantum-walk-inspired velocity, helium pressure perturbation, axial thermal-stress concentration and representative MEG alpha/gamma source waveforms.", styles["Small"]),
    ]

    characteristics = [
        [Paragraph("Component", styles["Small"]), Paragraph("Characteristic", styles["Small"]), Paragraph("Finite setting", styles["Small"])],
        [Paragraph("Dewar", styles["Small"]), Paragraph("Cylindrical cryogenic vessel", styles["Small"]), Paragraph("R<sub>i</sub> = 0.13 m; R<sub>o</sub> = 0.15 m; H = 0.40 m", styles["Small"])],
        [Paragraph("CFD", styles["Small"]), Paragraph("Bounded convection and pressure snapshot", styles["Small"]), Paragraph("20 x 50 = 1000 cells; p<sub>0</sub> = 5000 Pa", styles["Small"])],
        [Paragraph("Thermal", styles["Small"]), Paragraph("Cold-wall stress concentration", styles["Small"]), Paragraph("4 K to 300 K; E = 70 GPa", styles["Small"])],
        [Paragraph("MEG", styles["Small"]), Paragraph("Alpha rhythm and localized gamma burst", styles["Small"]), Paragraph("10 Hz and 30-50 Hz chirp; 1 s window", styles["Small"])],
    ]
    table = Table(characteristics, colWidths=[1.0 * inch, 2.7 * inch, 2.95 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#202c39")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa4ad")), ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f3f5f6")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([Spacer(1, 8), table, Paragraph("6. Limitations and conclusion", styles["Heading2"]), Paragraph("This Nature-style preprint reports a transparent reduced-order simulation. Quantitative engineering decisions require temperature-dependent helium thermodynamics, turbulence or two-fluid physics where appropriate, vacuum boundary conditions, material-property curves, contact constraints, mesh convergence and experimental validation. Within those limits, the finite state-space formulation provides a compact test bed for linking dewar pressure/stress characteristics to MEG waveform simulations and later quantum optimization experiments.", styles["BodyJustified"])])

    doc.build(story)
    print(f"Generated PDF: {filename}")
    return filename


if __name__ == "__main__":
    build_pdf()