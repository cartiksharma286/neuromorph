import os

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Nature_Preprint_QML_Epileptic_Seizure_Source_Localization.pdf",
)
PLOT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "epilepsy_qml_prediction_plots.png",
)


def make_prediction_plots(filename=PLOT_OUTPUT):
    """Create deterministic plots for the finite seizure-prediction example."""
    samples = np.arange(200)
    time = samples / 199.0
    spike = np.exp(-((time - 0.32) ** 2) / 0.0025) * np.sin(2 * np.pi * 10 * time)
    ictal_ramp = 1.0 / (1.0 + np.exp(-35.0 * (time - 0.62)))
    oscillation = 0.12 * np.sin(2 * np.pi * 6 * time)
    probability = np.clip(0.04 + 0.88 * ictal_ramp + oscillation, 0.0, 1.0)
    threshold = 0.70
    detected = probability >= threshold

    figure, axes = plt.subplots(2, 1, figsize=(7.1, 5.4), sharex=True)
    figure.patch.set_facecolor("white")
    axes[0].plot(time, probability, color="#b23a48", linewidth=2, label="Predicted seizure probability")
    axes[0].axhline(threshold, color="#333333", linestyle="--", linewidth=1, label="Decision threshold")
    axes[0].fill_between(time, threshold, probability, where=detected, color="#e7a6ae", alpha=0.6)
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Finite QML seizure-prediction trace", loc="left", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)

    axes[1].plot(time, spike, color="#1f5f8b", linewidth=1.5, label="Interictal spike feature")
    axes[1].plot(time, ictal_ramp, color="#d17a22", linewidth=1.8, label="Ictal envelope")
    axes[1].axvspan(0.62, 1.0, color="#f3d7b5", alpha=0.45)
    axes[1].set_xlabel("Normalized time")
    axes[1].set_ylabel("Feature amplitude")
    axes[1].set_xlim(0, 1)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return filename


def build_pdf(filename=OUTPUT):
    plot_filename = make_prediction_plots()
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Finite QML Source Localization of Epileptic Spikes",
        author="Neuromorph MEG/MRI Hybrid Simulation Group",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="PaperTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", parent=styles["Normal"], alignment=TA_CENTER,
        fontName="Helvetica-Oblique", fontSize=10, leading=13, textColor=colors.HexColor("#404040"),
    ))
    styles.add(ParagraphStyle(
        name="BodyJustified", parent=styles["BodyText"], alignment=TA_JUSTIFY,
        fontName="Times-Roman", fontSize=10.4, leading=14, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="Equation", parent=styles["Normal"], alignment=TA_CENTER,
        fontName="Helvetica-Oblique", fontSize=11, leading=16, spaceBefore=5, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Small", parent=styles["Normal"], fontName="Times-Roman",
        fontSize=8.7, leading=11,
    ))

    story = []
    story.append(Paragraph("Finite Quantum Machine Learning for MEG Source Localization", styles["PaperTitle"]))
    story.append(Paragraph("A beamformer-seeded model of epileptic seizures, interictal spikes and ictal onset", styles["Subtitle"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Abstract.</b> Epileptic source localization is an inverse problem in which a small number of neural generators must be inferred from noisy, time-resolved MEG measurements. We describe a finite hybrid workflow: a 10 x 10 x 10 candidate source lattice is scored by a linearly constrained minimum-variance (LCMV) prior, and a compact variational quantum model refines the highest-scoring candidates. The formulation explicitly separates interictal spikes from seizure-period activity using finite-window statistics. All sums, state dimensions and optimization steps are bounded, making the method reproducible as a simulation protocol rather than an assertion of clinical performance.",
        styles["BodyJustified"],
    ))

    story.append(Paragraph("1. Finite MEG forward model", styles["Heading2"]))
    story.append(Paragraph(
        "Let M = 64 sensors, G = 1000 candidate locations and T = 200 time samples. The simulated sensor array is represented by y in R^(M x T), with a finite set of dipole amplitudes x in R^(G x T). For each sensor m, source g and sample t:",
        styles["BodyJustified"],
    ))
    story.append(Paragraph(
        "y<sub>m,t</sub> = &sum;<sub>g=1</sub><sup>G</sup> L<sub>m,g</sub>x<sub>g,t</sub> + n<sub>m,t</sub>, &nbsp; m = 1,...,M; &nbsp; t = 1,...,T.",
        styles["Equation"],
    ))
    story.append(Paragraph(
        "The lead-field coefficient is a bounded inverse-square surrogate, L<sub>m,g</sub> = (||s<sub>m</sub> - r<sub>g</sub>||<sup>2</sup> + 10<sup>-3</sup>)<sup>-1</sup>, where s<sub>m</sub> is a sensor position and r<sub>g</sub> is a lattice point. The additive term n<sub>m,t</sub> represents measurement noise; it prevents the inverse problem from becoming artificially exact.",
        styles["BodyJustified"],
    ))

    story.append(Paragraph("2. Spike and seizure-window features", styles["Heading2"]))
    story.append(Paragraph(
        "A transient interictal spike is represented on the normalized interval t in [0,1] by a Gaussian-windowed oscillation. The finite sample sequence uses t<sub>k</sub> = k/(T-1):",
        styles["BodyJustified"],
    ))
    story.append(Paragraph(
        "q<sub>k</sub> = exp[-(t<sub>k</sub> - &tau;)<sup>2</sup>/&sigma;<sup>2</sup>] sin(2&pi;f t<sub>k</sub>), &nbsp; k = 0,...,T-1.",
        styles["Equation"],
    ))
    story.append(Paragraph(
        "For a candidate source g, define a finite spike-energy score over a window W<sub>s</sub> and a seizure-energy score over W<sub>z</sub>. The contrast is a dimensionless feature used by the localization objective:",
        styles["BodyJustified"],
    ))
    story.append(Paragraph(
        "E<sub>g</sub><sup>(s)</sup> = &sum;<sub>k &isin; W<sub>s</sub></sub> z<sub>g,k</sub><sup>2</sup>, &nbsp; E<sub>g</sub><sup>(z)</sup> = &sum;<sub>k &isin; W<sub>z</sub></sub> z<sub>g,k</sub><sup>2</sup>, &nbsp; R<sub>g</sub> = E<sub>g</sub><sup>(z)</sup>/(E<sub>g</sub><sup>(s)</sup> + 10<sup>-6</sup>).",
        styles["Equation"],
    ))
    story.append(Paragraph(
        "Here z<sub>g,k</sub> is the reconstructed activity at lattice point g. A spike detector can be stated as the finite threshold rule I<sub>k</sub> = 1 when |q<sub>k</sub>| exceeds the median absolute deviation threshold, and I<sub>k</sub> = 0 otherwise. This separates an observable transient from the broader seizure window without claiming that one waveform defines every seizure type.",
        styles["BodyJustified"],
    ))

    story.append(Paragraph("3. LCMV initialization", styles["Heading2"]))
    story.append(Paragraph(
        "The sample covariance C in R^(M x M) is regularized by lambda I, with lambda > 0. For each finite lattice point g, the scalar LCMV power is:",
        styles["BodyJustified"],
    ))
    story.append(Paragraph(
        "P<sub>g</sub> = [L<sub>g</sub><sup>T</sup>(C + &lambda;I)<sup>-1</sup>L<sub>g</sub>]<sup>-1</sup>, &nbsp; g = 1,...,G; &nbsp; g<sub>0</sub> = arg max<sub>g</sub> P<sub>g</sub>.",
        styles["Equation"],
    ))
    story.append(Paragraph(
        "The index g<sub>0</sub> seeds the QML state rather than replacing the complete search. In the associated simulation, the prior map is normalized by P&#772; = max<sub>g</sub> P<sub>g</sub> and the initial candidate is retained as an auditable baseline.",
        styles["BodyJustified"],
    ))

    story.append(Paragraph("4. Finite variational QML refinement", styles["Heading2"]))
    story.append(Paragraph(
        "Encode a local neighborhood of K = 8 candidate locations into three qubits. A parameterized circuit U(&theta;) produces a finite probability distribution p<sub>j</sub>(&theta;) over j = 0,...,K-1. The source loss combines data mismatch, localization distance and a seizure-feature penalty:",
        styles["BodyJustified"],
    ))
    story.append(Paragraph(
        "J(&theta;) = &alpha; &sum;<sub>k=0</sub><sup>T-1</sup> [y&#770;<sub>k</sub>(&theta;) - y<sub>k</sub>]<sup>2</sup> + &beta; &sum;<sub>j=0</sub><sup>K-1</sup> p<sub>j</sub>(&theta;)d<sub>j</sub><sup>2</sup> + &gamma; [1 - R<sub>j*</sub>], &nbsp; j* = arg max<sub>j</sub> p<sub>j</sub>(&theta;).",
        styles["Equation"],
    ))
    story.append(Paragraph(
        "With step size eta and Q = 12 bounded optimization steps, the parameter update is &theta;<sup>(u+1)</sup> = &theta;<sup>(u)</sup> - &eta; &nabla;J(&theta;<sup>(u)</sup>), for u = 0,...,Q-1. Gradients may be estimated by the parameter-shift identity for a gate parameter &theta;<sub>i</sub>:",
        styles["BodyJustified"],
    ))
    story.append(Paragraph(
        "&part;J/&part;&theta;<sub>i</sub> = [J(&theta;<sub>i</sub> + &pi;/2) - J(&theta;<sub>i</sub> - &pi;/2)]/2.",
        styles["Equation"],
    ))
    story.append(Paragraph(
        "The finite stopping rule is min{u: |J(&theta;<sup>(u)</sup>) - J(&theta;<sup>(u-1)</sup>)| < 10<sup>-4</sup>}, with a hard limit of Q iterations. This makes the computational claim testable: the output is a candidate source distribution and a spike/seizure feature vector, not a clinical diagnosis.",
        styles["BodyJustified"],
    ))

    story.append(Paragraph("5. Epileptic seizure prediction", styles["Heading2"]))
    story.append(Paragraph(
        "Prediction is defined here as a finite warning score derived from the localized QML activity. For each sample k, let a<sub>k</sub> be the normalized source activity and h<sub>k</sub> the finite feature vector containing spike energy, seizure-window energy and spectral contrast. A logistic readout gives:",
        styles["BodyJustified"],
    ))
    story.append(Paragraph(
        "p<sub>k</sub> = [1 + exp(-(&omega;<sub>0</sub> + &omega;<sup>T</sup>h<sub>k</sub> + &omega;<sub>a</sub>a<sub>k</sub>))]<sup>-1</sup>, &nbsp; d<sub>k</sub> = 1[p<sub>k</sub> &ge; 0.70].",
        styles["Equation"],
    ))
    story.append(Paragraph(
        "The example uses a 0.70 decision threshold only to illustrate the finite rule. The characteristics of the two modeled phenomena are: an interictal spike is brief, high-slope and localized in time; ictal activity is sustained, has elevated window energy and produces a persistent probability elevation. These are simulation features, not diagnostic criteria.",
        styles["BodyJustified"],
    ))
    characteristics = [
        [Paragraph("Characteristic", styles["Small"]), Paragraph("Spike-like activity", styles["Small"]), Paragraph("Seizure-like activity", styles["Small"])],
        [Paragraph("Duration", styles["Small"]), Paragraph("Brief Gaussian-windowed transient", styles["Small"]), Paragraph("Sustained elevated envelope", styles["Small"])],
        [Paragraph("Energy", styles["Small"]), Paragraph("Concentrated in W<sub>s</sub>", styles["Small"]), Paragraph("Elevated across W<sub>z</sub>", styles["Small"])],
        [Paragraph("Prediction signal", styles["Small"]), Paragraph("Transient feature peak", styles["Small"]), Paragraph("p<sub>k</sub> remains above threshold", styles["Small"])],
    ]
    characteristic_table = Table(characteristics, colWidths=[1.35 * inch, 2.35 * inch, 2.75 * inch])
    characteristic_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#202c39")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa4ad")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f3f5f6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(characteristic_table)
    story.append(Spacer(1, 8))
    story.append(Image(plot_filename, width=6.65 * inch, height=5.05 * inch))
    story.append(Paragraph(
        "<b>Figure 1.</b> Deterministic finite example. The upper panel shows the predicted probability and thresholded seizure decision. The lower panel contrasts a brief interictal spike feature with a sustained ictal envelope; the shaded region marks the modeled seizure window.",
        styles["Small"],
    ))

    story.append(Paragraph("6. Reproducible simulation protocol", styles["Heading2"]))
    protocol = [
        [Paragraph("Quantity", styles["Small"]), Paragraph("Finite setting", styles["Small"]), Paragraph("Purpose", styles["Small"])],
        [Paragraph("Sensors M", styles["Small"]), Paragraph("64", styles["Small"]), Paragraph("Noisy MEG observations", styles["Small"])],
        [Paragraph("Source grid G", styles["Small"]), Paragraph("10 x 10 x 10 = 1000", styles["Small"]), Paragraph("Candidate locations", styles["Small"])],
        [Paragraph("Samples T", styles["Small"]), Paragraph("200", styles["Small"]), Paragraph("Transient waveform window", styles["Small"])],
        [Paragraph("QML neighborhood K", styles["Small"]), Paragraph("8 states / 3 qubits", styles["Small"]), Paragraph("Local refinement", styles["Small"])],
        [Paragraph("Optimization steps Q", styles["Small"]), Paragraph("12 maximum", styles["Small"]), Paragraph("Bounded parameter updates", styles["Small"])],
    ]
    table = Table(protocol, colWidths=[1.55 * inch, 1.65 * inch, 3.25 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#202c39")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa4ad")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f3f5f6")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 9))
    story.append(Paragraph(
        "<b>Limitations.</b> The equations define a finite computational model and do not establish diagnostic accuracy, seizure onset-zone sensitivity, or patient-level safety. Real deployment would require calibrated head models, artifact rejection, clinical annotations, prospective validation and appropriate regulatory review.",
        styles["BodyJustified"],
    ))
    story.append(Paragraph("7. Conclusion", styles["Heading2"]))
    story.append(Paragraph(
        "A beamformer-seeded QML search provides a compact way to rank candidate generators while preserving explicit finite bounds on sensors, locations, samples and optimization steps. The same objective can report both a localized source estimate and interpretable spike-versus-seizure window features, providing a clear bridge for future MEG/MRI validation studies.",
        styles["BodyJustified"],
    ))

    doc.build(story)
    print(f"Generated PDF: {filename}")
    return filename


if __name__ == "__main__":
    build_pdf()