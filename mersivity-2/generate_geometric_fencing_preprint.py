#!/usr/bin/env python3
"""Generate a Nature-style preprint on geometric fencing registration."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "Nature_Preprint_Geometric_Fencing_Registration.pdf")
DPI = 220
DEEP = "#102a43"
TEAL = "#0f766e"
CORAL = "#dc5a41"
GOLD = "#c58a17"
SLATE = "#52606d"
LIGHT = "#f0f4f8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#bcccdc",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfcfd",
})


def collect_results():
    """Run the application endpoint so all paper values match shipped code and data."""
    from app import app

    results = {}
    with app.test_client() as client:
        for bins in range(1, 5):
            response = client.post(
                "/api/register-mri-to-ct-fencing",
                json={"target_error_mm": 0.05, "fence_bins": bins},
            )
            payload = response.get_json()
            if response.status_code != 200:
                raise RuntimeError(payload.get("error", "registration endpoint failed"))
            results[bins] = payload
    return results


def as_points(payload, key):
    mesh = payload[key]
    return np.column_stack((mesh["x"], mesh["y"], mesh["z"]))


def save_figure(fig, filename):
    path = os.path.join(BASE_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    return path


def make_pipeline_figure():
    fig, axis = plt.subplots(figsize=(7.2, 2.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.2, "MR/CT\nsurfaces"),
        (2.2, "Centroid +\nscale alignment"),
        (4.2, "Trimmed affine\ncorrespondence"),
        (6.2, "Combinatorial\nfence updates"),
        (8.2, "Dense residual\nrefinement"),
    ]
    palette = [DEEP, "#16697a", TEAL, GOLD, CORAL]
    for index, ((x_pos, label), color) in enumerate(zip(boxes, palette)):
        axis.add_patch(plt.Rectangle((x_pos, 1.0), 1.55, 1.0, facecolor=color, edgecolor="none"))
        axis.text(x_pos + 0.775, 1.5, label, color="white", ha="center", va="center",
              fontweight="bold", fontsize=7.2, linespacing=1.05)
        if index < len(boxes) - 1:
            axis.annotate("", xy=(x_pos + 1.95, 1.5), xytext=(x_pos + 1.58, 1.5),
                          arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 1.8})
    axis.text(5.0, 2.55, "Measured objective: mean nearest-neighbour surface residual",
              ha="center", color=DEEP, fontsize=10, fontweight="bold")
    axis.text(5.0, 0.42, "Quantile partitions yield B^3 fences; no post-hoc replacement of measured error",
              ha="center", color=SLATE, fontsize=8)
    return save_figure(fig, "fig_geometric_fencing_pipeline.png")


def make_convergence_figure(results):
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors_by_bin = [SLATE, TEAL, GOLD, CORAL]
    for bins, color in zip(range(1, 5), colors_by_bin):
        history = np.asarray(results[bins]["telemetry"]["convergence_mm"])
        left.plot(np.arange(len(history)), history, "o-", color=color,
                  label=f"B={bins} ({bins ** 3} fences)")
    left.axhline(0.05, color=CORAL, linestyle="--", linewidth=1, label="0.05 mm target")
    left.set_yscale("log")
    left.set_xlabel("Iteration")
    left.set_ylabel("Mean surface residual (mm)")
    left.set_title("a  Measured convergence", loc="left", fontweight="bold")
    left.legend(frameon=False, fontsize=7)

    bins = np.arange(1, 5)
    residuals = [results[value]["registration_error"] for value in bins]
    wall_times = [results[value]["telemetry"]["wall_clock_seconds"] for value in bins]
    right.bar(bins - 0.16, residuals, width=0.32, color=TEAL, label="Residual (mm)")
    twin = right.twinx()
    twin.bar(bins + 0.16, wall_times, width=0.32, color=GOLD, label="Wall time (s)")
    right.axhline(0.05, color=CORAL, linestyle="--", linewidth=1)
    right.set_xticks(bins, [str(value ** 3) for value in bins])
    right.set_xlabel("Fence combinations")
    right.set_ylabel("Final residual (mm)", color=TEAL)
    twin.set_ylabel("Wall time (s)", color=GOLD)
    right.set_title("b  Fence-density sensitivity", loc="left", fontweight="bold")
    handles_a, labels_a = right.get_legend_handles_labels()
    handles_b, labels_b = twin.get_legend_handles_labels()
    right.legend(handles_a + handles_b, labels_a + labels_b, frameon=False, fontsize=7)
    fig.tight_layout()
    return save_figure(fig, "fig_geometric_fencing_convergence.png")


def make_alignment_figure(result):
    source = as_points(result, "mesh1")
    target = as_points(result, "mesh2")
    registered = as_points(result, "mesh1_reg")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    projections = [(0, 1, "Axial"), (0, 2, "Coronal"), (1, 2, "Sagittal")]
    sample = slice(None, None, 4)
    for axis, (first, second, title) in zip(axes, projections):
        axis.scatter(target[sample, first], target[sample, second], s=2, c=GOLD,
                     alpha=0.28, label="CT target")
        axis.scatter(source[sample, first], source[sample, second], s=2, c=CORAL,
                     alpha=0.18, label="MR source")
        axis.scatter(registered[sample, first], registered[sample, second], s=2, c=TEAL,
                     alpha=0.55, label="Registered MR")
        axis.set_title(title, fontweight="bold")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    return save_figure(fig, "fig_geometric_fencing_alignment.png")


def make_residual_figure(result):
    target = as_points(result, "mesh2")
    source = as_points(result, "mesh1")
    registered = as_points(result, "mesh1_reg")
    tree = cKDTree(target)
    before, _ = tree.query(source)
    after, _ = tree.query(registered)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    limit = float(np.quantile(before, 0.98))
    edges = np.linspace(0, limit, 45)
    left.hist(before, bins=edges, density=True, color=CORAL, alpha=0.55, label="Before")
    left.hist(after, bins=edges, density=True, color=TEAL, alpha=0.75, label="After")
    left.axvline(0.05, color=GOLD, linestyle="--", label="0.05 mm")
    left.set_xlabel("Nearest-neighbour residual (mm)")
    left.set_ylabel("Density")
    left.set_title("a  Residual distribution", loc="left", fontweight="bold")
    left.legend(frameon=False)

    ordered = np.sort(after)
    cumulative = np.arange(1, len(ordered) + 1) / len(ordered)
    right.plot(ordered, cumulative, color=TEAL, linewidth=2)
    right.axvline(0.05, color=GOLD, linestyle="--")
    right.set_xlim(0, float(np.quantile(after, 0.995)))
    right.set_xlabel("Post-registration residual (mm)")
    right.set_ylabel("Empirical cumulative probability")
    right.set_title("b  Post-registration ECDF", loc="left", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "fig_geometric_fencing_residuals.png")


def build_pdf(results, figures):
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleNature", parent=styles["Title"], fontName="Helvetica-Bold",
                           fontSize=21, leading=24, textColor=colors.HexColor(DEEP), spaceAfter=12)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=10, leading=14,
                              textColor=colors.HexColor(SLATE), alignment=TA_CENTER, spaceAfter=18)
    heading = ParagraphStyle("Heading", parent=styles["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14, leading=17, textColor=colors.HexColor(DEEP), spaceBefore=8, spaceAfter=6)
    subheading = ParagraphStyle("Subheading", parent=styles["Heading2"], fontName="Helvetica-Bold",
                                fontSize=10.5, leading=13, textColor=colors.HexColor(TEAL), spaceBefore=7, spaceAfter=4)
    body = ParagraphStyle("BodyNature", parent=styles["BodyText"], fontName="Times-Roman",
                          fontSize=9.3, leading=12.4, alignment=TA_JUSTIFY, spaceAfter=6)
    abstract = ParagraphStyle("Abstract", parent=body, leftIndent=18, rightIndent=18,
                              borderColor=colors.HexColor("#bcccdc"), borderWidth=0.7,
                              borderPadding=10, backColor=colors.HexColor(LIGHT))
    caption = ParagraphStyle("Caption", parent=body, fontSize=7.5, leading=9.5,
                             textColor=colors.HexColor(SLATE), spaceAfter=8)
    equation = ParagraphStyle("Equation", parent=body, fontName="Helvetica", fontSize=9,
                              alignment=TA_CENTER, backColor=colors.HexColor("#f7fafc"),
                              borderColor=colors.HexColor("#d9e2ec"), borderWidth=0.5,
                              borderPadding=6, spaceBefore=4, spaceAfter=7)
    reference = ParagraphStyle("Reference", parent=body, fontSize=7.5, leading=9.3, leftIndent=12, firstLineIndent=-12)

    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=letter, rightMargin=0.55 * inch,
                            leftMargin=0.55 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    story = []
    best = results[2]
    residual = best["registration_error"]
    wall_time = best["telemetry"]["wall_clock_seconds"]
    iterations = len(best["telemetry"]["convergence_mm"]) - 1

    story.append(Paragraph("Combinatorial geometric fencing for deformable MR-CT surface registration", title))
    story.append(Paragraph("Cartik Sharma | Mersivity research preprint | 13 August 2026", subtitle))
    story.append(Paragraph(
        "<b>Abstract</b> Registration between magnetic resonance (MR) and computed tomography (CT) "
        "surfaces is complicated by modality-dependent contrast, incomplete correspondence and local "
        "deformation. We introduce combinatorial geometric fencing, an iterative point-set method that "
        "combines trimmed global affine fitting with quantile-defined spatial partitions, robust local "
        "translations and dense nearest-neighbour residual refinement. In a single bundled, downsampled "
        f"MR-CT case containing 4,096 sampled surface points per modality, the implementation reduced the "
        f"mean nearest-neighbour surface residual below 0.05 mm, reaching {residual:.5f} mm in {iterations} "
        f"update iterations ({wall_time:.3f} s wall time for endpoint reconstruction and registration). "
        "This quantity is a model-space surface-fit residual, not independently measured anatomical-landmark "
        "target registration error (TRE), scanner accuracy or clinical performance. The study is therefore "
        "a computational feasibility demonstration. Multi-case, physical-coordinate and landmark validation "
        "is required before translational interpretation.", abstract))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Introduction", heading))
    story.append(Paragraph(
        "Multimodal registration supports image-guided intervention, radiotherapy planning and longitudinal "
        "analysis. Classical iterative closest point (ICP) methods alternate nearest-neighbour correspondence "
        "with rigid or affine updates, but can be sensitive to initialization, outliers and anatomy visible in "
        "only one modality. Free-form deformation and diffeomorphic methods increase flexibility, although "
        "their control lattices and regularizers introduce additional design choices. Geometric fencing takes "
        "a deliberately inspectable middle path: the target geometry defines balanced spatial strata, and each "
        "stratum contributes a robust local displacement after a globally stabilizing affine step.", body))
    story.append(Image(figures[0], width=6.8 * inch, height=2.35 * inch))
    story.append(Paragraph(
        "<b>Figure 1 | Computational pipeline.</b> MR and CT isosurfaces are normalized, aligned by trimmed "
        "affine correspondence, partitioned into quantile fences, corrected by fence-local median displacement "
        "and refined against measured nearest neighbours.", caption))

    story.append(PageBreak())
    story.append(Paragraph("Methods", heading))
    story.append(Paragraph("Global initialization", subheading))
    story.append(Paragraph(
        "Let S={s<sub>i</sub>} and T={t<sub>j</sub>} be source and target point sets. Source points are centered "
        "and isotropically scaled using the ratio of mean radial distances before translation to the target centroid.", body))
    story.append(Paragraph(
        "s'<sub>i</sub> = (s<sub>i</sub> - mu<sub>S</sub>) "
        "(r<sub>T</sub> / max(r<sub>S</sub>, epsilon)) + mu<sub>T</sub>", equation))
    story.append(Paragraph("Trimmed affine correspondence", subheading))
    story.append(Paragraph(
        "A k-d tree supplies nearest target neighbours. The largest 10% of correspondence distances are excluded "
        "from the affine least-squares fit, reducing their leverage while preserving all points for later dense refinement.", body))
    story.append(Paragraph(
        "A* = arg min<sub>A</sub> sum<sub>i in I<sub>0.9</sub></sub> "
        "|| [s'<sub>i</sub>,1] A - t<sub>nu(i)</sub> ||<sup>2</sup><sub>2</sub>", equation))
    story.append(Paragraph("Combinatorial fences", subheading))
    story.append(Paragraph(
        "For B bins on each axis, target-coordinate quantiles define B<sup>3</sup> possible fence combinations. "
        "Each populated fence with at least four active points receives the component-wise median correspondence "
        "displacement. The implementation blends 65% of the affine candidate, applies 75% of the fence median, "
        "and then applies 80% of the dense nearest-neighbour residual.", body))
    story.append(Paragraph(
        "F<sub>abc</sub> = {i : q<sup>x</sup><sub>a</sub> <= s'<sup>x</sup><sub>i</sub> < q<sup>x</sup><sub>a+1</sub>, "
        "q<sup>y</sup><sub>b</sub> <= s'<sup>y</sup><sub>i</sub> < q<sup>y</sup><sub>b+1</sub>, "
        "q<sup>z</sup><sub>c</sub> <= s'<sup>z</sup><sub>i</sub> < q<sup>z</sup><sub>c+1</sub>}", equation))
    story.append(Paragraph(
        "delta<sub>abc</sub> = median<sub>i in F<sub>abc</sub></sub> "
        "(t<sub>nu(i)</sub> - s'<sub>i</sub>)", equation))
    story.append(Paragraph("Evaluation and timing semantics", subheading))
    story.append(Paragraph(
        "The reported objective is the mean nearest-neighbour surface residual. It can become small under a dense "
        "deformation even when anatomical correspondence is incorrect; it must not be interpreted as landmark TRE. "
        "The application exposes a 10<sup>-18</sup> s numerical simulation index resolution. This is metadata for "
        "simulation discretization only and is neither measured compute latency nor MR/CT acquisition timing.", body))
    story.append(Paragraph(
        "E(S',T) = (1/N) sum<sub>i=1</sub><sup>N</sup> min<sub>j</sub> "
        "||s'<sub>i</sub> - t<sub>j</sub>||<sub>2</sub>", equation))

    story.append(PageBreak())
    story.append(Paragraph("Results", heading))
    story.append(Image(figures[1], width=6.8 * inch, height=2.83 * inch))
    story.append(Paragraph(
        "<b>Figure 2 | Convergence and fence-density sensitivity.</b> (a) Measured endpoint histories for "
        "B=1-4. The dashed line marks the prespecified 0.05 mm surface-residual target. (b) Final residual and "
        "wall time from complete endpoint calls. Timings include surface reconstruction and Python execution "
        "and are not acquisition latencies.", caption))

    table_data = [["Bins/axis", "Fences", "Final residual (mm)", "Endpoint wall time (s)", "Target met"]]
    for bins in range(1, 5):
        item = results[bins]
        table_data.append([
            str(bins), str(bins ** 3), f"{item['registration_error']:.5f}",
            f"{item['telemetry']['wall_clock_seconds']:.3f}",
            "Yes" if item["telemetry"]["target_met"] else "No",
        ])
    table = Table(table_data, colWidths=[0.85 * inch, 0.85 * inch, 1.55 * inch, 1.55 * inch, 1.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(DEEP)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bcccdc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(LIGHT)]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Paragraph("<b>Table 1 | Within-method sensitivity to fence density.</b>", caption))
    story.append(table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"For the default B=2 configuration, the residual fell from "
        f"{best['telemetry']['convergence_mm'][0]:.3f} mm to {residual:.5f} mm. The target was crossed without "
        "overwriting the measured objective. Because dense nearest-neighbour attraction is part of the deformation "
        "model, this result demonstrates surface conformity rather than independent anatomical correctness.", body))

    story.append(PageBreak())
    story.append(Paragraph("Geometric characteristics", heading))
    story.append(Image(figures[2], width=6.8 * inch, height=2.45 * inch))
    story.append(Paragraph(
        "<b>Figure 3 | Orthographic point-set projections.</b> Downsampled source MR, target CT and registered MR "
        "surfaces for the default eight-fence configuration. Visual overlap is descriptive and does not establish "
        "anatomical landmark agreement.", caption))
    story.append(Image(figures[3], width=6.8 * inch, height=2.83 * inch))
    story.append(Paragraph(
        "<b>Figure 4 | Nearest-neighbour residual characteristics.</b> (a) Residual densities before and after "
        "registration; the pre-registration x-range is truncated at its 98th percentile for visibility. "
        "(b) Empirical cumulative distribution after registration for the same 4,096 sampled points.", caption))

    story.append(PageBreak())
    story.append(Paragraph("Discussion", heading))
    story.append(Paragraph(
        "Geometric fencing offers an interpretable hierarchy between a single global transform and an unconstrained "
        "per-point warp. Quantile boundaries balance target occupancy better than equal-width bins when surface "
        "density is heterogeneous, while median local displacement limits sensitivity to extreme correspondences. "
        "The approach is computationally simple: one static target k-d tree supports repeated correspondence queries, "
        "and fence updates are separable after assignment.", body))
    story.append(Paragraph("Limitations", subheading))
    story.append(Paragraph(
        "This preprint reports one bundled case in an index-derived geometric frame. DICOM voxel spacing, image "
        "orientation and patient-coordinate transforms are not propagated into the evaluated surfaces; the mm label "
        "therefore reflects the application's model-space convention. No held-out landmarks, segmentations, repeated "
        "scans or clinical outcomes are available. Nearest-neighbour attraction can collapse source points onto a target "
        "surface and thereby optimize the reported residual without preserving topology or invertibility. The present "
        "implementation also lacks bending-energy, Jacobian-determinant and inverse-consistency penalties. Comparisons "
        "between fence densities reuse the same case and are sensitivity analyses, not statistical benchmarks.", body))
    story.append(Paragraph("Required validation", subheading))
    story.append(Paragraph(
        "A translational study should evaluate multiple subjects in DICOM patient coordinates; use prospectively "
        "defined anatomical landmarks and segmentation overlap; report landmark TRE, Dice score, Hausdorff distance, "
        "Jacobian folding and inverse consistency; compare against rigid ICP, coherent point drift, B-spline free-form "
        "deformation and a diffeomorphic baseline; and include ablations that remove affine, fence-local and dense "
        "components. Runtime should be measured independently of I/O after warm-up on specified hardware.", body))
    story.append(Paragraph("Conclusion", heading))
    story.append(Paragraph(
        "Combinatorial geometric fencing produced a sub-0.05 mm mean nearest-neighbour surface residual in the bundled "
        "computational case while retaining a transparent sequence of global, regional and dense updates. The result "
        "supports continued algorithmic evaluation, but does not establish sub-0.05 mm anatomical TRE, scanner "
        "resolution or clinical accuracy. The principal next step is independent landmark-based validation under "
        "physical-coordinate and deformation-regularity constraints.", body))

    story.append(PageBreak())
    story.append(Paragraph("Code and data availability", heading))
    story.append(Paragraph(
        "The implementation, application endpoint and this reproducible document generator are maintained in the "
        "mersivity-2 source repository. The generator calls the local Flask test client and derives figures and table "
        "values directly from endpoint responses. Availability of imaging data remains subject to the repository's "
        "distribution and governance conditions.", body))
    story.append(Paragraph("References", heading))
    references = [
        "Besl, P. J. & McKay, N. D. A method for registration of 3-D shapes. IEEE Trans. Pattern Anal. Mach. Intell. 14, 239-256 (1992).",
        "Arun, K. S., Huang, T. S. & Blostein, S. D. Least-squares fitting of two 3-D point sets. IEEE Trans. Pattern Anal. Mach. Intell. 9, 698-700 (1987).",
        "Myronenko, A. & Song, X. Point set registration: coherent point drift. IEEE Trans. Pattern Anal. Mach. Intell. 32, 2262-2275 (2010).",
        "Rueckert, D. et al. Nonrigid registration using free-form deformations: application to breast MR images. IEEE Trans. Med. Imaging 18, 712-721 (1999).",
        "Avants, B. B., Epstein, C. L., Grossman, M. & Gee, J. C. Symmetric diffeomorphic image registration with cross-correlation. Med. Image Anal. 12, 26-41 (2008).",
        "Fitzpatrick, J. M., West, J. B. & Maurer, C. R. Predicting error in rigid-body point-based registration. IEEE Trans. Med. Imaging 17, 694-702 (1998).",
        "Maier-Hein, L. et al. Why rankings of biomedical image analysis competitions should be interpreted with care. Nat. Commun. 9, 5217 (2018).",
    ]
    for number, item in enumerate(references, 1):
        story.append(Paragraph(f"{number}. {item}", reference))

    doc.build(story)
    return OUTPUT_PATH


def main():
    print("Running geometric-fencing registrations...")
    results = collect_results()
    figures = [
        make_pipeline_figure(),
        make_convergence_figure(results),
        make_alignment_figure(results[2]),
        make_residual_figure(results[2]),
    ]
    output = build_pdf(results, figures)
    print(f"Generated: {output}")
    print(f"Default measured surface residual: {results[2]['registration_error']:.5f} mm")


if __name__ == "__main__":
    main()