#!/usr/bin/env python3
"""Generate a Nature-style computational preprint for the depression rTMS model."""

from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from logic.depression_rtms import simulate_depression_rtms

OUTPUT = Path(__file__).with_name('Nature_Preprint_Depression_rTMS_CBT.pdf')
RED = colors.HexColor('#b42318')
BLUE = colors.HexColor('#175cd3')
GREEN = colors.HexColor('#067647')
AMBER = colors.HexColor('#b54708')
PURPLE = colors.HexColor('#6941c6')
INK = colors.HexColor('#1d2939')
MUTED = colors.HexColor('#667085')
PALE = colors.HexColor('#f2f4f7')


def _line_chart(series, title, y_label, y_min=0.0, y_max=None):
    drawing = Drawing(470, 188)
    chart = LinePlot()
    chart.x = 48; chart.y = 32; chart.width = 395; chart.height = 125
    chart.data = [[(float(x), float(y)) for x, y in zip(xs, ys)] for xs, ys, _ in series]
    max_x = max(max(xs) for xs, _, _ in series)
    max_y = max(max(ys) for _, ys, _ in series)
    chart.xValueAxis.valueMin = 0; chart.xValueAxis.valueMax = float(max_x)
    chart.xValueAxis.valueStep = max(1, int(max_x / 6))
    chart.yValueAxis.valueMin = y_min; chart.yValueAxis.valueMax = float(y_max or max_y * 1.08)
    chart.yValueAxis.valueStep = max(0.1, (chart.yValueAxis.valueMax - y_min) / 5.0)
    for index, (_, _, color) in enumerate(series):
        chart.lines[index].strokeColor = color
        chart.lines[index].strokeWidth = 1.5
    drawing.add(chart)
    drawing.add(String(235, 175, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=9, fillColor=INK))
    drawing.add(String(235, 8, y_label, textAnchor='middle', fontSize=7, fillColor=MUTED))
    return drawing


def _distribution_chart(distribution):
    drawing = Drawing(470, 188)
    chart = VerticalBarChart()
    chart.x = 48; chart.y = 34; chart.width = 395; chart.height = 120
    chart.data = [distribution['baseline_counts'], distribution['post_counts']]
    chart.categoryAxis.categoryNames = [f'{value:.0f}' for value in distribution['bin_centers']]
    chart.categoryAxis.labels.fontSize = 6
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = BLUE; chart.bars[1].fillColor = GREEN
    chart.bars.strokeColor = colors.white
    drawing.add(chart)
    drawing.add(String(235, 175, 'Seeded PHQ-9 cohort distributions (n=600)', textAnchor='middle', fontName='Helvetica-Bold', fontSize=9, fillColor=INK))
    drawing.add(String(235, 8, 'PHQ-9 bin center; count', textAnchor='middle', fontSize=7, fillColor=MUTED))
    return drawing


def build_pdf(output_path=OUTPUT, params=None):
    data = simulate_depression_rtms(**(params or {}))
    sessions = data['sessions']
    metrics = data['metrics']
    trajectories = _line_chart([
        (sessions, data['phq9_usual_care'], MUTED),
        (sessions, data['phq9_cbt_only'], BLUE),
        (sessions, data['phq9_rtms_only'], AMBER),
        (sessions, data['phq9_adaptive_combined'], GREEN),
    ], 'Simulated PHQ-9 trajectories across research arms', 'session; PHQ-9 score', 0, 27)
    control = _line_chart([
        (sessions, data['control_effort'], RED),
        (sessions, data['cognitive_distortion_state'], PURPLE),
    ], 'Bounded control effort and cognitive-distortion state', 'session; normalized state', 0, 1.05)
    objective = _line_chart([
        (sessions, data['objective'], GREEN),
        (sessions, data['number_signature'], AMBER),
    ], 'Finite objective convergence and modular signature', 'session; objective/signature value', 0)
    stage = data['staging']
    stage_trajectory = _line_chart([
        (sessions[:stage['optimal_induction_end'] + 1], data['phq9_adaptive_combined'][:stage['optimal_induction_end'] + 1], RED),
        (sessions[stage['optimal_induction_end']:stage['optimal_consolidation_end'] + 1], data['phq9_adaptive_combined'][stage['optimal_induction_end']:stage['optimal_consolidation_end'] + 1], BLUE),
        (sessions[stage['optimal_consolidation_end']:], data['phq9_adaptive_combined'][stage['optimal_consolidation_end']:], GREEN),
    ], 'Finite optimal staging of the adaptive trajectory', 'session; modeled PHQ-9', 0, 27)
    stage_candidates = _line_chart([
        (stage['candidate_rank'], stage['candidate_cost'], PURPLE),
    ], 'Ranked finite gate candidates (lower cost is preferred)', 'candidate rank; stage objective', 0)
    distribution = _distribution_chart(data['distribution'])

    styles = getSampleStyleSheet()
    title = ParagraphStyle('NatureTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=INK, spaceAfter=7)
    byline = ParagraphStyle('Byline', parent=styles['Normal'], fontSize=9, leading=12, textColor=BLUE, spaceAfter=10)
    heading = ParagraphStyle('NatureHeading', parent=styles['Heading2'], fontSize=11, leading=14, textColor=RED, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle('NatureBody', parent=styles['BodyText'], fontSize=8.7, leading=13, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=6)
    equation = ParagraphStyle('Equation', parent=body, fontName='Courier', fontSize=8.1, leading=12, alignment=TA_CENTER, backColor=PALE, borderPadding=7, spaceBefore=4, spaceAfter=7)
    caption = ParagraphStyle('Caption', parent=body, fontSize=7.5, leading=10, textColor=MUTED, spaceAfter=8)
    notice = ParagraphStyle('Notice', parent=body, fontName='Helvetica-Bold', textColor=RED, borderColor=RED, borderWidth=0.6, borderPadding=7)
    coefficients = data['continued_fraction']['coefficients']
    fractions = ', '.join(item['fraction'] for item in data['continued_fraction']['convergents'])

    story = [
        Paragraph('Finite Optimal-Control Modeling of Combined rTMS and Cognitive Behavioral Therapy for Depression', title),
        Paragraph('Nature Portfolio preprint-style computational study | NeuroMorph rTMS Platform | 18 August 2026', byline),
        Paragraph('Research status', heading),
        Paragraph('This manuscript reports an in silico hypothesis-generating model. It is not a clinical trial, does not demonstrate treatment efficacy, and must not guide individual care. Depression assessment, suicidality screening, rTMS delivery and psychotherapy require qualified clinicians and applicable device, ethics and safety procedures.', notice),
        Paragraph('Abstract', heading),
        Paragraph(f'We developed a finite computational model of symptom-responsive repetitive transcranial magnetic stimulation (rTMS) combined with a cognitive behavioral theory state update for depression research. Seeded PHQ-9 distributions and {data["params"]["sessions"]} finite sessions were used to compare simulated usual-care, CBT-only, rTMS-only and adaptive combined trajectories. A bounded controller minimized symptom error and control variation, while prime-session markers and modular signatures tested number-theoretic scheduling features without assigning biological meaning. In the default simulation, PHQ-9 changed from {metrics["baseline_phq9"]:.1f} to {metrics["final_phq9"]:.2f}, a modeled change of {metrics["modeled_response_pct"]:.1f}%. These are synthetic model outputs, not patient outcomes. The framework exposes assumptions and supplies falsifiable endpoints for prospective validation.', body),
        Paragraph('Introduction', heading),
        Paragraph('Major depressive disorder is heterogeneous, and symptom trajectories vary with baseline severity, comorbidity, medication, psychotherapy engagement and stimulation parameters. rTMS and cognitive behavioral therapy are established clinical domains, but the interaction represented here is a mathematical research abstraction. We ask whether a transparent finite controller can jointly represent symptom error, behavioral-state change, treatment burden and experimental scheduling signatures.', body),
        Paragraph('Finite optimal-control model', heading),
        Paragraph('Let P_k be PHQ-9 score, P* the research target, c_k a normalized cognitive-distortion state and u_k bounded control effort. The finite recurrence uses separate rTMS and CBT attenuation terms:', body),
        Paragraph('e_k = max(0, P_k - P*); &nbsp; u_k = clip(g e_k/P_0, 0, 1)<br/>c_(k+1) = max(c_min, c_k - beta w_CBT c_k)<br/>P_(k+1) = max(P_min, P* + (P_k-P*) exp[-(lambda_rTMS + lambda_CBT)] + epsilon_k)', equation),
        Paragraph('Over N sessions, the controller minimizes symptom error, stimulation burden and abrupt control changes. This objective is a model-selection criterion, not a dosing algorithm.', body),
        Paragraph('J_N = sum_(k=1)^N [q(P_k-P*)^2 + r u_k^2 + s(u_k-u_(k-1))^2]<br/>u* = arg min_(0 <= u_k <= 1) J_N', equation),
        trajectories,
        Paragraph('Figure 1 | Seeded simulated trajectories. Lines are generated by the application model and do not represent trial observations.', caption),
        Paragraph('Statistical distributions', heading),
        Paragraph('A clipped Gaussian baseline cohort and a synthetic post-model distribution illustrate uncertainty and overlap. These distributions support visualization only; no inferential p-values, confidence intervals or effect sizes are claimed.', body),
        distribution,
        Paragraph('Figure 2 | Baseline and synthetic post-model PHQ-9 distributions for 600 generated records.', caption),
        Paragraph('Cognitive behavioral theory state', heading),
        Paragraph('The CBT component represents cognitive reappraisal, behavioral activation and relapse-planning as a decreasing latent state. It does not encode therapist judgment, therapeutic alliance, individualized formulation or real session content.', body),
        Paragraph('c_(k+1) = c_k(1 - beta w_CBT); &nbsp; lambda_CBT = alpha w_CBT(1-c_k)', equation),
        control,
        Paragraph('Figure 3 | Normalized adaptive effort and cognitive-distortion state.', caption),
        Paragraph('Number-theoretic signatures', heading),
        Paragraph('Prime session indices and the modular signature s_k = (k^2 + 3k + 7) mod 17 provide reproducible scheduling labels. A continued fraction approximates a configurable signature ratio. These arithmetic objects test stratification and synchronization hypotheses; they have no established antidepressant mechanism.', body),
        Paragraph('rho = [a_0; a_1, ..., a_m]; &nbsp; p_j = a_j p_(j-1)+p_(j-2); &nbsp; q_j = a_j q_(j-1)+q_(j-2)<br/>s_k = (k^2 + 3k + 7) mod 17', equation),
        Paragraph(f'Expansion: {coefficients}. Convergents: {fractions}. Prime sessions: {data["prime_sessions"]}.', body),
        objective,
        Paragraph('Figure 4 | Finite objective convergence and modular session signature.', caption),
        Paragraph('Optimal finite staging', heading),
        Paragraph('We enumerate every admissible pair of induction and consolidation gates, then rank each pair using target mismatch, consolidation variation, maintenance control burden and terminal symptom error. The minimum is a property of this synthetic objective and is not a clinically optimized treatment schedule.', body),
        Paragraph('G = {(g_1,g_2): ceil(N/4) <= g_1 <= floor(3N/5), g_1+3 <= g_2 <= floor(9N/10)}<br/>J_stage(g_1,g_2) = (P_(g_1)-10)^2 + 0.65(P_(g_2)-7)^2 + 2.5 V_(g_1:g_2) + 1.8 U_(g_2:N) + 0.8(P_N-5)^2<br/>(g_1*,g_2*) = arg min_((g_1,g_2) in G) J_stage(g_1,g_2)', equation),
        Paragraph(f'The default finite search evaluates {stage["candidate_count"]} gate pairs and selects g_1* = {stage["optimal_induction_end"]}, g_2* = {stage["optimal_consolidation_end"]}, with synthetic cost {stage["optimal_cost"]:.3f}.', body),
        stage_trajectory,
        Paragraph('Figure 5 | Selected induction (red), consolidation (blue) and maintenance (green) segments of the modeled trajectory.', caption),
        stage_candidates,
        Paragraph('Figure 6 | The 30 lowest-cost finite gate candidates in ascending objective order.', caption),
    ]

    paradigm = data['paradigm']
    rows = [
        ['Characteristic', 'Research configuration'],
        ['Status', paradigm['status']],
        ['Target abstraction', paradigm['target']],
        ['Frequency input', f'{paradigm["frequency_hz"]:.1f} Hz'],
        ['Finite horizon', f'{paradigm["sessions"]} sessions'],
        ['CBT state', paradigm['cbt_component']],
        ['Controller', paradigm['control_rule']],
        ['Safety boundary', paradigm['safety']],
    ]
    table = Table(rows, colWidths=[4.0*cm, 11.2*cm], repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), RED), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 7.5), ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d0d5dd')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    stage_rows = [['Stage', 'Finite sessions', 'End PHQ-9', 'Mean control', 'Synthetic research goal']]
    for item in stage['stages']:
        stage_rows.append([
            item['name'], f"{item['start']}-{item['end']}", f"{item['end_phq9']:.2f}",
            f"{item['mean_control']:.3f}", item['research_goal'],
        ])
    stage_table = Table(stage_rows, colWidths=[2.5*cm, 2.4*cm, 2.3*cm, 2.3*cm, 5.7*cm], repeatRows=1)
    stage_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 7.2), ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d0d5dd')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    story += [
        Paragraph('Optimal stage characteristics', heading), stage_table, Spacer(1, 8),
        Paragraph('Research paradigm characteristics', heading), table, Spacer(1, 8),
        Paragraph('Prospective validation design', heading),
        Paragraph('A future protocol should preregister primary and secondary outcomes, use sham control and blinded rating where feasible, preserve standard clinical care, and stratify medication and psychotherapy exposure. Outcomes should include validated depression scales, functional measures, durability, cognitive effects, discontinuation and adverse events. Independent monitoring and explicit escalation pathways for worsening depression or suicidality are mandatory.', body),
        Paragraph('A finite stopping rule may be expressed mathematically but must be operationalized by clinicians and ethics oversight:', body),
        Paragraph('stop if P(harm | D_k) >= eta_stop OR symptom worsening >= Delta_max OR safety flag_k = 1', equation),
        Paragraph('Limitations', heading),
        Paragraph('The cohort is synthetic; PHQ-9 dynamics are assumed; the CBT state is reductive; anatomy-specific electric fields, motor threshold, medication effects, bipolar-spectrum risk, psychosis, suicidality and adverse events are omitted. Number-theoretic features are exploratory labels. The model cannot estimate comparative effectiveness or determine an optimal clinical paradigm.', body),
        Paragraph('Methods and reproducibility', heading),
        Paragraph(f'The default model uses NumPy RandomState seed 286, baseline PHQ-9 {metrics["baseline_phq9"]:.1f}, {data["params"]["sessions"]} sessions, {data["params"]["rtms_frequency_hz"]:.1f} Hz frequency input, CBT weight {data["params"]["cbt_weight"]:.2f}, control gain {data["params"]["control_gain"]:.2f}, and signature ratio {data["params"]["signature_ratio"]:.6f}. The Flask endpoint and this PDF call the same pure model function.', body),
        Paragraph('Data availability', heading),
        Paragraph('No participant data were used. Source code and all synthetic values required to reproduce the figures are included with the application.', body),
        Paragraph('Reporting context', heading),
        Paragraph('A submission-ready manuscript requires a verified systematic literature review, complete bibliography, protocol registration and institutional review. This draft intentionally avoids fabricated citations and regulatory claims.', body),
    ]

    document = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2.1*cm, rightMargin=2.1*cm, topMargin=1.8*cm, bottomMargin=1.8*cm, title='Finite Optimal-Control Modeling of Combined rTMS and CBT for Depression', author='NeuroMorph rTMS Platform')
    document.build(story)
    return str(output_path)


if __name__ == '__main__':
    print(build_pdf())
