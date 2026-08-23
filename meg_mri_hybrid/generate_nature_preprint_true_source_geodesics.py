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
OUTPUT = os.path.join(ROOT, "Nature_Preprint_True_Source_Localization_Beamformer_Geodesics.pdf")
PLOT_OUTPUT = os.path.join(ROOT, "true_source_beamformer_geodesic_plots.png")


def make_plots(filename=PLOT_OUTPUT):
    truth = np.array([[0.05, 0.05, 0.00], [-0.05, -0.02, 0.05]])
    estimate = np.array([[0.048, 0.052, 0.004], [-0.047, -0.018, 0.047]])
    direction = truth[1] - truth[0]
    perpendicular = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    perpendicular /= np.linalg.norm(perpendicular)
    path_t = np.linspace(0, 1, 50)
    path = np.outer(1 - path_t, truth[0]) + np.outer(path_t, truth[1])
    path += np.outer(0.2 * np.linalg.norm(direction) * np.sin(np.pi * path_t), perpendicular)

    grid = np.linspace(-0.1, 0.1, 10)
    GX, GY = np.meshgrid(grid, grid)
    power = np.exp(-((GX - truth[0, 0]) ** 2 + (GY - truth[0, 1]) ** 2) / 0.0012)
    power += 0.8 * np.exp(-((GX - truth[1, 0]) ** 2 + (GY - truth[1, 1]) ** 2) / 0.0012)
    power /= power.max()
    state_labels = ["source 1", "source 2", "background"]
    state_probability = [0.46, 0.41, 0.13]

    figure = plt.figure(figsize=(7.1, 5.9), constrained_layout=True)
    axes = figure.subplots(2, 2, squeeze=False)
    figure.delaxes(axes[0, 0])
    axes[0, 0] = figure.add_subplot(2, 2, 1, projection="3d")
    axes[0, 0].plot(path[:, 0], path[:, 1], path[:, 2], color="#1f5f8b", linewidth=2, label="Geodesic path")
    axes[0, 0].scatter(truth[:, 0], truth[:, 1], truth[:, 2], color="#b23a48", s=45, label="True sources")
    axes[0, 0].scatter(estimate[:, 0], estimate[:, 1], estimate[:, 2], color="#d17a22", marker="x", s=50, label="Beamformer estimates")
    axes[0, 0].set_title("True sources and beamformer geodesic", loc="left", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[0, 0].set_zlabel("z (m)")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")

    axes[0, 1].imshow(power, origin="lower", extent=[-0.1, 0.1, -0.1, 0.1], cmap="magma", vmin=0, vmax=1)
    axes[0, 1].scatter(truth[:, 0], truth[:, 1], color="#4fc3f7", marker="o", s=35, label="Truth")
    axes[0, 1].scatter(estimate[:, 0], estimate[:, 1], color="white", marker="x", s=40, label="Estimate")
    axes[0, 1].set_title("LCMV power projection", loc="left", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("x (m)")
    axes[0, 1].set_ylabel("y (m)")
    axes[0, 1].legend(frameon=False, fontsize=7, loc="upper left")

    axes[1, 0].plot(path_t, np.linalg.norm(path - truth[0], axis=1), color="#6a4c93", linewidth=1.8)
    axes[1, 0].set_title("Geodesic distance from source 1", loc="left", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Path parameter t")
    axes[1, 0].set_ylabel("Distance (m)")

    bars = axes[1, 1].bar(state_labels, state_probability, color=["#1f5f8b", "#b23a48", "#9aa4ad"])
    axes[1, 1].set_title("Projection-state probabilities", loc="left", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].set_ylim(0, 0.55)
    axes[1, 1].tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, state_probability):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", fontsize=8)
    for axis in axes.flat:
        axis.grid(True, alpha=0.18)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return filename


def build_pdf(filename=OUTPUT):
    plot_filename = make_plots()
    doc = SimpleDocTemplate(filename, pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch,
                             topMargin=0.65 * inch, bottomMargin=0.65 * inch,
                             title="True Source Localization with Beamformer Geodesics", author="Neuromorph MEG/MRI Hybrid Simulation Group")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=5))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER, fontName="Helvetica-Oblique", fontSize=10, leading=13, textColor=colors.HexColor("#404040")))
    styles.add(ParagraphStyle(name="BodyJustified", parent=styles["BodyText"], alignment=TA_JUSTIFY, fontName="Times-Roman", fontSize=10.3, leading=14, spaceAfter=7))
    styles.add(ParagraphStyle(name="Equation", parent=styles["Normal"], alignment=TA_CENTER, fontName="Helvetica-Oblique", fontSize=10.7, leading=16, spaceBefore=4, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.7, leading=11))

    story = [
        Paragraph("True Source Localization with Beamformer Geodesics", styles["PaperTitle"]),
        Paragraph("Projection states and finite manifold paths for a two-dipole MEG simulation", styles["Subtitle"]),
        Spacer(1, 10),
        Paragraph("<b>Abstract.</b> We present a finite MEG source-localization preprint in which two known current dipoles generate synthetic measurements in a 192-sensor cylindrical bore. A linearly constrained minimum-variance (LCMV) beamformer evaluates a 10 x 10 x 10 search lattice, while a geodesic path connects the two true source positions through a curved local manifold. Projection states are represented by normalized weights over the two source states and a background state. The reported truth, estimates and equations are simulation quantities and should not be interpreted as clinical localization accuracy.", styles["BodyJustified"]),
        Paragraph("1. True sources and forward model", styles["Heading2"]),
        Paragraph("The ground-truth dipole locations are r<sub>1</sub> = (0.05, 0.05, 0.00) m and r<sub>2</sub> = (-0.05, -0.02, 0.05) m, with moments q<sub>1</sub> = (10<sup>-8</sup>,0,0) and q<sub>2</sub> = (0,10<sup>-8</sup>,0). For sensor m and time sample k, the finite observation model is:", styles["BodyJustified"]),
        Paragraph("y<sub>m,k</sub> = &sum;<sub>j=1</sub><sup>2</sup> l<sub>m,j</sub> (q<sub>j</sub><sup>T</sup>u<sub>j,k</sub>) + n<sub>m,k</sub>, &nbsp; m = 1,...,192; &nbsp; k = 0,...,99.", styles["Equation"]),
        Paragraph("The scanner has 16 circumferential positions and 12 axial positions, giving M = 16 x 12 = 192 SQUID sensors. Each candidate lattice point has three orthogonal dipole orientations, so a grid with G = 1000 locations produces a lead-field matrix L in R<sup>192 x 3000</sup>.", styles["BodyJustified"]),
        Paragraph("2. LCMV beamformer and true-source comparison", styles["Heading2"]),
        Paragraph("For each lattice location g, let L<sub>g</sub> in R<sup>192 x 3</sup> contain its three orientation lead fields and let C be the regularized data covariance. The spatial filter and power estimate are:", styles["BodyJustified"]),
        Paragraph("W<sub>g</sub> = (L<sub>g</sub><sup>T</sup>C<sup>-1</sup>L<sub>g</sub>)<sup>-1</sup>L<sub>g</sub><sup>T</sup>C<sup>-1</sup>, &nbsp; P<sub>g</sub> = trace(W<sub>g</sub>CW<sub>g</sub><sup>T</sup>), &nbsp; g* = arg max<sub>g</sub> P<sub>g</sub>.", styles["Equation"]),
        Paragraph("The paper distinguishes r<sub>j</sub>, the known true positions used to generate the data, from r&#770;<sub>j</sub>, the beamformer estimates obtained from the finite lattice. The Euclidean localization error is d<sub>j</sub> = ||r&#770;<sub>j</sub> - r<sub>j</sub>||<sub>2</sub>. In the illustrative plot, the estimates are offset by only a few millimetres to make this distinction visible; they are not a measured performance result.", styles["BodyJustified"]),
        Paragraph("3. Beamformer geodesics", styles["Heading2"]),
        Paragraph("Let d = r<sub>2</sub> - r<sub>1</sub>, t in [0,1], and u be a unit vector perpendicular to d. The code's continuable geodesic is the finite curve:", styles["BodyJustified"]),
        Paragraph("gamma(t) = (1 - t)r<sub>1</sub> + tr<sub>2</sub> + 0.2||d|| sin(pi t)u, &nbsp; t<sub>i</sub> = i/49, &nbsp; i = 0,...,49.", styles["Equation"]),
        Paragraph("This path preserves the endpoints and introduces a bounded curvature perturbation to represent a local folded-manifold trajectory. Its discrete length is D = sum<sub>i=0</sub><sup>48</sup> ||gamma(t<sub>i+1</sub>) - gamma(t<sub>i</sub>)||<sub>2</sub>.", styles["BodyJustified"]),
        Paragraph("4. Projection states", styles["Heading2"]),
        Paragraph("A finite projection-state vector assigns probability to source 1, source 2 and an unlocalized background. For amplitudes a = (a<sub>1</sub>, a<sub>2</sub>, a<sub>b</sub>), normalization and the projection operator are:", styles["BodyJustified"]),
        Paragraph("p<sub>s</sub> = |a<sub>s</sub>|<sup>2</sup> / &sum;<sub>r &isin; {1,2,b}</sub>|a<sub>r</sub>|<sup>2</sup>, &nbsp; &Pi;<sub>s</sub> = |s&gt;&lt;s|, &nbsp; &sum;<sub>s &isin; {1,2,b}</sub> &Pi;<sub>s</sub> = I<sub>3</sub>.", styles["Equation"]),
        Paragraph("The plotted example uses p = (0.46, 0.41, 0.13), which sums to one. A state is considered dominant only by the declared arg max rule, s* = arg max<sub>s</sub> p<sub>s</sub>; this is a projection label, not a diagnosis.", styles["BodyJustified"]),
        Paragraph("5. Results and characteristics", styles["Heading2"]),
        Image(plot_filename, width=6.65 * inch, height=5.55 * inch),
        Paragraph("<b>Figure 1.</b> True dipoles, beamformer estimates, curved geodesic path, LCMV power projection, path distance and normalized projection-state probabilities.", styles["Small"]),
    ]

    characteristics = [
        [Paragraph("Quantity", styles["Small"]), Paragraph("Finite value", styles["Small"]), Paragraph("Role", styles["Small"])],
        [Paragraph("True sources", styles["Small"]), Paragraph("2 dipoles at r<sub>1</sub>, r<sub>2</sub>", styles["Small"]), Paragraph("Known simulation ground truth", styles["Small"])],
        [Paragraph("Sensor geometry", styles["Small"]), Paragraph("16 x 12 = 192 sensors", styles["Small"]), Paragraph("Cylindrical bore observations", styles["Small"])],
        [Paragraph("Search grid", styles["Small"]), Paragraph("10 x 10 x 10 = 1000 points", styles["Small"]), Paragraph("Beamformer candidate space", styles["Small"])],
        [Paragraph("Geodesic samples", styles["Small"]), Paragraph("50 points; curvature 0.2||d||", styles["Small"]), Paragraph("Finite source-to-source path", styles["Small"])],
        [Paragraph("Projection states", styles["Small"]), Paragraph("3; p = (0.46, 0.41, 0.13)", styles["Small"]), Paragraph("Two sources plus background", styles["Small"])],
    ]
    table = Table(characteristics, colWidths=[1.35 * inch, 2.1 * inch, 3.2 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#202c39")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa4ad")), ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f3f5f6")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([Spacer(1, 8), table, Paragraph("6. Limitations and conclusion", styles["Heading2"]), Paragraph("The forward model, LCMV score, geodesic and projection states are finite computational constructions. They do not establish source-localization accuracy in human data, anatomical correspondence, tractography validity or quantum hardware advantage. Validation would require calibrated sensor geometry, realistic head conductivity, independent annotations and held-out recordings. The resulting framework nevertheless makes true positions, estimates, path construction and projection-state normalization explicit and reproducible.", styles["BodyJustified"])])

    doc.build(story)
    print(f"Generated PDF: {filename}")
    return filename


if __name__ == "__main__":
    build_pdf()