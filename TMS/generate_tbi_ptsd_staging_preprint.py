#!/usr/bin/env python3
"""Generate a theoretical TBI/PTSD finite-mathematics staging preprint."""

from datetime import date
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from logic.tbi_ptsd_rtms import simulate_tbi_ptsd_rtms


OUTPUT_PDF = Path(__file__).with_name("TBI_PTSD_Finite_Clinical_Staging_Preprint.pdf")
NAVY = colors.HexColor("#123047")
TEAL = colors.HexColor("#0F766E")
PALE = colors.HexColor("#EEF6F5")
SLATE = colors.HexColor("#4B5563")


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18,
            leading=22, alignment=TA_CENTER, textColor=NAVY, spaceAfter=8,
        ),
        "author": ParagraphStyle(
            "author", parent=base["Normal"], fontName="Helvetica", fontSize=9,
            leading=12, alignment=TA_CENTER, textColor=SLATE, spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "heading", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12,
            leading=15, textColor=NAVY, spaceBefore=14, spaceAfter=6,
        ),
        "subheading": ParagraphStyle(
            "subheading", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10,
            leading=13, textColor=TEAL, spaceBefore=9, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=13, alignment=TA_JUSTIFY, spaceAfter=7,
        ),
        "abstract": ParagraphStyle(
            "abstract", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=9.2,
            leading=13, alignment=TA_JUSTIFY, leftIndent=18, rightIndent=18, spaceAfter=8,
        ),
        "equation": ParagraphStyle(
            "equation", parent=base["Code"], fontName="Courier", fontSize=8.7,
            leading=13, alignment=TA_CENTER, textColor=NAVY, backColor=PALE,
            borderColor=colors.HexColor("#C8E3DF"), borderWidth=0.5, borderPadding=6,
            spaceBefore=6, spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=8,
            leading=10, alignment=TA_CENTER, textColor=SLATE, spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontName="Helvetica", fontSize=7.8,
            leading=9.5, alignment=TA_CENTER,
        ),
        "cell_header": ParagraphStyle(
            "cell_header", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=7.8,
            leading=9.5, alignment=TA_CENTER, textColor=colors.white,
        ),
        "reference": ParagraphStyle(
            "reference", parent=base["Normal"], fontName="Helvetica", fontSize=7.7,
            leading=10, leftIndent=14, firstLineIndent=-14, spaceAfter=3,
        ),
    }


def paragraph(text, style):
    return Paragraph(text, style)


def staging_table(styles):
    rows = [
        ["State", "Operational pattern", "Assessment focus", "Decision output"],
        ["S0: acute uncertainty", "Incomplete characterization or unstable trajectory", "Safety, red flags, injury context", "Escalate assessment; no optimization claim"],
        ["S1: early recovery", "Symptoms present; functional course not yet established", "Repeated symptom, function, sleep, and context measures", "Monitor change and refine phenotype"],
        ["S2: persistent symptoms", "Cross-domain burden persists beyond expected recovery window", "Domain profile, comorbidity, exposures, participation", "Multidisciplinary formulation"],
        ["S3: complex persistence", "High uncertainty, discordant measures, or substantial functional impact", "Specialist assessment and longitudinal reassessment", "Shared, individualized care plan"],
    ]
    rendered = []
    for row_index, row in enumerate(rows):
        style = styles["cell_header"] if row_index == 0 else styles["cell"]
        rendered.append([paragraph(value, style) for value in row])
    table = Table(rendered, colWidths=[1.1 * inch, 1.85 * inch, 2.1 * inch, 1.45 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8CFCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def figure_image(figure, width=6.7 * inch, height=2.55 * inch):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(figure)
    buffer.seek(0)
    return Image(buffer, width=width, height=height)


def build_app_figures(simulation):
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    weeks = simulation["weeks"]
    axes[0].plot(weeks, simulation["pcl5_standard"], color="#64748B", linestyle="--", label="App reference arm")
    axes[0].plot(weeks, simulation["pcl5_rtms_only"], color="#0F766E", label="App rTMS arm")
    axes[0].plot(weeks, simulation["pcl5_synergistic"], color="#D97706", label="App rTMS + CBT arm")
    axes[0].set(title="PTSD symptom trajectory (PCL-5)", xlabel="Model week", ylabel="Score")
    axes[0].legend(fontsize=6.5, frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(weeks, simulation["rpq_standard"], color="#64748B", linestyle="--", label="App reference arm")
    axes[1].plot(weeks, simulation["rpq_rtms_only"], color="#0F766E", label="App rTMS arm")
    axes[1].plot(weeks, simulation["rpq_synergistic"], color="#D97706", label="App rTMS + CBT arm")
    axes[1].set(title="Post-concussion trajectory (RPQ)", xlabel="Model week", ylabel="Score")
    axes[1].legend(fontsize=6.5, frameon=False)
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    trajectory = figure_image(figure)

    field = simulation["bem_field"]
    depth = simulation["bem_depth"]
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
    heatmap = axes[0].imshow(field["efield_norm"], cmap="viridis", aspect="auto", interpolation="bilinear")
    axes[0].set(title="App BEM field proxy", xlabel="Azimuth node", ylabel="Polar node")
    colorbar = figure.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)
    colorbar.set_label("Scaled field (V/m)", fontsize=7)
    axes[1].plot(depth["depths_mm"], depth["field_v_m"], color="#0F766E", linewidth=2)
    axes[1].axvline(40, color="#D97706", linestyle="--", linewidth=1)
    axes[1].set(title="App attenuation profile", xlabel="Depth (mm)", ylabel="Scaled field (V/m)")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    bem = figure_image(figure)
    return trajectory, bem


def build_pdf():
    styles = build_styles()
    simulation = simulate_tbi_ptsd_rtms(
        baseline_pcl5=58.0, baseline_rpq=42.0, treatment_weeks=24,
        rtms_freq_hz=10.0, cbt_trauma_gain=0.85,
    )
    trajectory_figure, bem_figure = build_app_figures(simulation)
    document = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        title="Clinical Staging for TBI and PTSD: A Finite-Mathematics Research Framework",
        author="Neuromorph Research Working Paper",
    )
    story = [
        HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=8),
        paragraph("CLINICAL RESEARCH PREPRINT | SIMULATION-BASED METHODS PAPER", styles["author"]),
        paragraph("Clinical Staging for Traumatic Brain Injury and Post-Traumatic Stress Disorder: A Finite-Mathematics Framework with BEM Simulation", styles["title"]),
        paragraph("Neuromorph Research Working Paper | " + date.today().strftime("%B %d, %Y"), styles["author"]),
        HRFlowable(width="100%", thickness=0.7, color=TEAL, spaceAfter=8),
        paragraph("Abstract", styles["heading"]),
        paragraph(
            "TBI and PTSD commonly present with heterogeneous symptoms, changing functional consequences, and uncertain causal attribution. "
            "We describe a finite-state staging framework coupled to a demonstration BEM field proxy and the repository's seeded 24-week symptom simulator. "
            "The simulation depicts reference, rTMS, and rTMS-plus-CBT trajectories; it is not clinical evidence and does not establish efficacy, safety, or cure. "
            "Pharmacological care is represented only as a clinician-managed, protocol-recorded co-intervention that may be compared observationally after indication, dose, adherence, and adverse effects are documented. "
            "The framework assigns uncertainty-aware states and selects the next assessment action subject to safety and equity constraints.",
            styles["abstract"],
        ),
        paragraph("Keywords: traumatic brain injury; post-traumatic stress disorder; clinical staging; finite-state model; decision theory; uncertainty", styles["caption"]),
        paragraph("1. Clinical Scope and Boundary", styles["heading"]),
        paragraph(
            "A clinically useful staging paradigm should distinguish clinical course from diagnosis, avoid treating a single questionnaire as ground truth, "
            "and preserve uncertainty when trauma exposure, injury effects, sleep disturbance, pain, substance use, and social context overlap. "
            "Accordingly, this paper defines a finite decision model whose parameters must be calibrated and externally validated before any clinical use.",
            styles["body"],
        ),
        paragraph("2. App-Derived Simulation Characteristics", styles["heading"]),
        paragraph(
            "The included figures are generated directly from the repository's TBI/PTSD app logic using its default demonstration settings: baseline PCL-5 = 58, baseline RPQ = 42, 24 modeled weeks, rTMS frequency = 10 Hz, and CBT gain = 0.85. "
            "The engine uses seeded noise and fixed analytic trajectories, so these lines are illustrative model outputs rather than patient data, effect estimates, or a treatment recommendation.",
            styles["body"],
        ),
        trajectory_figure,
        paragraph("Figure 1. App-generated symptom trajectories. These seeded curves visualize model behavior only and cannot be interpreted as comparative clinical effectiveness.", styles["caption"]),
        paragraph(
            "The app's BEM component uses a single-layer deformed-sphere cortical surface and scaled field visualization. It reports a peak scaled cortical field of "
            f"{simulation['bem_field']['peak_efield_v_m']:.2f} V/m and an attenuation-profile value of "
            f"{simulation['bem_depth']['target_depth_amygdala_v_m']:.2f} V/m at its nominal 40 mm target. These values are software outputs, not individualized dosimetry or device-safety results.",
            styles["body"],
        ),
        bem_figure,
        paragraph("Figure 2. App-generated BEM field proxy and depth-attenuation visualization. The model is not a validated head model and must not guide coil placement, intensity, or safety decisions.", styles["caption"]),
        paragraph("3. Finite Representation of Longitudinal Assessment", styles["heading"]),
        paragraph(
            "At encounter t, let the bounded observation vector include symptom domains, function, safety, context, and measurement reliability. "
            "Each component is discretized into a finite set using prospectively specified, population-appropriate bins rather than universal cut points:",
            styles["body"],
        ),
        paragraph("x_t = (p_t, c_t, f_t, r_t, u_t) in X = P x C x F x R x U", styles["equation"]),
        paragraph(
            "Here p_t denotes post-concussion symptom burden, c_t denotes trauma-related symptom burden, f_t denotes participation and function, "
            "r_t records safety-relevant observations, and u_t records uncertainty arising from missingness, disagreement, or limited reliability. "
            "The finite set X supports auditability: every assignment can be traced to observed inputs and predefined rules.",
            styles["body"],
        ),
        paragraph("4. Staging States", styles["heading"]),
        paragraph(
            "The state space S is intentionally coarse. It expresses the level of assessment and coordination needed, not disease severity or treatment eligibility. "
            "Transitions are permitted in either direction because recovery and contextual stressors may change over time.",
            styles["body"],
        ),
        staging_table(styles),
        paragraph("Table 1. Conceptual stages. Timing and operational definitions require protocol-specific validation; these labels are not clinical criteria.", styles["caption"]),
        paragraph("5. Uncertainty-Aware State Assignment", styles["heading"]),
        paragraph(
            "For a candidate stage s, define a finite score from domain features with an explicit ambiguity penalty. The chosen stage is the lowest-risk state assignment, "
            "with a distinct review state whenever uncertainty is too high for reliable classification:",
            styles["body"],
        ),
        paragraph("score(s | x_t) = sum_j w_{s,j} phi_j(x_t) - lambda_u u_t", styles["equation"]),
        paragraph("s_t = argmax_{s in S} score(s | x_t),  provided u_t <= tau_u; otherwise s_t = S_review", styles["equation"]),
        paragraph(
            "The basis functions phi_j are finite indicators or ordinal encodings declared before evaluation. Weights w and uncertainty threshold tau_u are estimated only on training data, "
            "then frozen for temporal and external validation. The review state prevents false precision when key observations conflict or are absent.",
            styles["body"],
        ),
        paragraph("6. Pharmacological Co-Intervention and Next Assessment", styles["heading"]),
        paragraph(
            "Medication management may be relevant to a participant's longitudinal course, but medication selection remains outside this model. In a prospective study, the finite record should retain a medication exposure vector m_t with drug class, indication, dose change, adherence, adverse effects, and concurrent psychotherapy. "
            "It supports confounding control and safety review; it cannot prescribe pharmacotherapy or infer causal benefit from non-randomized records.",
            styles["body"],
        ),
        paragraph("x'_t = (x_t, m_t),  m_t in M = Class x Indication x Change x Adherence x AdverseEffects", styles["equation"]),
        paragraph(
            "Let a be an assessment or coordination action from a finite action set A, such as repeat measurement, collateral history, functional evaluation, safety review, or specialist consultation. "
            "A transition matrix captures observed movement across stages over a prespecified interval:",
            styles["body"],
        ),
        paragraph("P_a(i, j) = Pr(s_{t+1} = j | s_t = i, a, z_t),   sum_j P_a(i, j) = 1", styles["equation"]),
        paragraph(
            "where z_t contains fixed protocol covariates and documented context. The next action minimizes expected loss, balancing missed safety concerns, delayed clarification, burden, and inequitable access:",
            styles["body"],
        ),
        paragraph("a*_t = argmin_{a in A} [ sum_j P_a(s_t, j)L(j, a) + kappa B(a) + eta E(a) ]", styles["equation"]),
        paragraph(
            "L is a transparent finite loss table, B is participant or service burden, and E is a disparity penalty estimated across predeclared groups. "
            "The safety rule r_t = 1 supersedes optimization and routes to immediate clinician-led assessment according to local policy.",
            styles["body"],
        ),
        PageBreak(),
        paragraph("7. Evaluation Protocol", styles["heading"]),
        paragraph(
            "The model should be evaluated as a decision-support research object, not by apparent accuracy alone. A prospective protocol should compare it against usual longitudinal assessment using: "
            "(i) calibration of stage probabilities; (ii) agreement with blinded multidisciplinary review; (iii) time to clarified formulation; "
            "(iv) missing-data and subgroup performance; and (v) decision-curve net benefit. Parameters must not be tuned on the held-out cohort.",
            styles["body"],
        ),
        paragraph("Calibration error = sum_{b=1}^B n_b/N | mean(y_b) - mean(q_b) |", styles["equation"]),
        paragraph(
            "This expected calibration error compares observed outcomes y with predicted probabilities q in B fixed bins. The clinical meaning of any benefit must be evaluated separately from statistical performance, including potential harms from over-assessment or delayed escalation.",
            styles["body"],
        ),
        paragraph("8. Governance, Limits, and Clinical Boundary", styles["heading"]),
        paragraph(
            "This framework does not diagnose TBI or PTSD, determine capacity, replace clinical judgment, or recommend rTMS, psychotherapy, medication, or any other intervention. It does not demonstrate a cure. "
            "Clinical deployment would require ethics review, data governance, prospective validation, clinician oversight, local safety workflows, and monitoring for differential error. "
            "Because staging can affect access and attention, a conservative abstention option is preferable to forced classification.",
            styles["body"],
        ),
        paragraph("9. Conclusion", styles["heading"]),
        paragraph(
            "Finite mathematics can make clinical staging explicit: bounded observations map to a small state space, transitions remain empirically estimable, and action selection exposes its safety, burden, and equity assumptions. "
            "The resulting paradigm is a falsifiable research proposal whose value depends on transparent specification and prospective, patient-centered validation.",
            styles["body"],
        ),
        paragraph("References", styles["heading"]),
    ]
    references = [
        "1. National Academies of Sciences, Engineering, and Medicine. Traumatic Brain Injury: A Roadmap for Accelerating Progress. National Academies Press (2022).",
        "2. American Psychiatric Association. Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition, Text Revision. American Psychiatric Association (2022).",
        "3. McGorry, P. D. et al. Clinical staging: a heuristic and practical strategy for new research and better health and social outcomes for psychotic and related mood disorders. Can. J. Psychiatry 55, 486-497 (2010).",
        "4. Steyerberg, E. W. Clinical Prediction Models, 2nd ed. Springer (2019).",
        "5. Vickers, A. J., Van Calster, B. & Steyerberg, E. W. A simple, step-by-step guide to interpreting decision curve analysis. Diagn. Progn. Res. 3, 18 (2019).",
        "6. Obermeyer, Z. et al. Dissecting racial bias in an algorithm used to manage the health of populations. Science 366, 447-453 (2019).",
        "7. Lefaucheur, J.-P. et al. Evidence-based guidelines on the therapeutic use of repetitive transcranial magnetic stimulation (rTMS). Clin. Neurophysiol. 131, 474-528 (2020).",
        "8. Thielscher, A. et al. Field modeling for transcranial magnetic stimulation: a useful tool to understand the physiological effects of TMS? Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. 2011, 222-225 (2011).",
    ]
    story.extend(paragraph(reference, styles["reference"]) for reference in references)
    story.extend([
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1.2, color=NAVY, spaceBefore=4, spaceAfter=4),
        paragraph("Research preprint. Conceptual methodology only; not medical advice and not for clinical decision-making.", styles["caption"]),
    ])
    document.build(story)
    return OUTPUT_PDF


if __name__ == "__main__":
    pdf_path = build_pdf()
    print(f"Generated: {pdf_path}")