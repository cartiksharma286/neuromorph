#!/usr/bin/env python3
"""Generate a Nature-style computational preprint from the rTMS sleep-apnea model."""

from pathlib import Path

import numpy as np
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT = Path(__file__).with_name('Nature_Preprint_Sleep_Apnea_rTMS.pdf')
RED = colors.HexColor('#b42318')
BLUE = colors.HexColor('#175cd3')
GREEN = colors.HexColor('#067647')
AMBER = colors.HexColor('#b54708')
INK = colors.HexColor('#1d2939')
MUTED = colors.HexColor('#667085')
PALE = colors.HexColor('#f2f4f7')


def simulate(baseline_ahi=38.0, frequency_hz=10.0, adaptive_gain=1.5, duration_days=30, sync_ratio=1.3416):
    rng = np.random.RandomState(101)
    days = np.arange(1, duration_days + 1, dtype=float)
    adherence = 0.70 + 0.10 * np.sin(days * 0.4)
    open_rate = 0.06 * frequency_hz / 10.0
    adaptive_rate = 0.12 * (frequency_hz / 10.0) * (0.5 + 0.5 * adaptive_gain)
    untreated = []
    cpap = []
    open_loop = []
    closed_loop = []
    for index, day in enumerate(days):
        untreated.append(max(15.0, baseline_ahi + 0.05 * day + rng.normal(0, 1.2)))
        cpap.append(max(2.0, baseline_ahi - (baseline_ahi - 8.0) * 0.85 * adherence[index] + rng.normal(0, 2.5)))
        open_loop.append(max(1.0, 12.0 + (baseline_ahi - 12.0) * np.exp(-open_rate * day) + rng.normal(0, 0.8)))
        closed_loop.append(max(0.5, 3.5 + (baseline_ahi - 3.5) * np.exp(-adaptive_rate * day) + rng.normal(0, 0.3)))
    untreated = np.array(untreated)
    cpap = np.array(cpap)
    open_loop = np.array(open_loop)
    closed_loop = np.array(closed_loop)
    control = np.clip(adaptive_gain * (closed_loop - 5.0) / baseline_ahi, 0.0, 1.0)

    coefficients = []
    value = sync_ratio
    for _ in range(8):
        integer = int(np.floor(value))
        coefficients.append(integer)
        remainder = value - integer
        if abs(remainder) < 1e-10:
            break
        value = 1.0 / remainder

    convergents = []
    errors = []
    p_prev2, p_prev1, q_prev2, q_prev1 = 0, 1, 1, 0
    for coefficient in coefficients:
        numerator = coefficient * p_prev1 + p_prev2
        denominator = coefficient * q_prev1 + q_prev2
        convergents.append((numerator, denominator))
        errors.append(abs(sync_ratio - numerator / denominator))
        p_prev2, p_prev1 = p_prev1, numerator
        q_prev2, q_prev1 = q_prev1, denominator

    return {
        'days': days, 'untreated': untreated, 'cpap': cpap, 'open_loop': open_loop,
        'closed_loop': closed_loop, 'control': control, 'adherence': adherence,
        'coefficients': coefficients, 'convergents': convergents, 'errors': np.array(errors),
        'params': (baseline_ahi, frequency_hz, adaptive_gain, duration_days, sync_ratio),
    }


def line_chart(series, title, y_label, y_min=0.0, y_max=None):
    drawing = Drawing(470, 190)
    chart = LinePlot()
    chart.x = 48; chart.y = 34; chart.width = 395; chart.height = 125
    chart.data = [[(float(x), float(y)) for x, y in zip(xs, ys)] for xs, ys, _ in series]
    max_x = max(max(xs) for xs, _, _ in series)
    max_y = max(max(ys) for _, ys, _ in series)
    chart.xValueAxis.valueMin = 1; chart.xValueAxis.valueMax = float(max_x)
    chart.xValueAxis.valueStep = max(1, int(max_x / 6))
    chart.yValueAxis.valueMin = y_min; chart.yValueAxis.valueMax = float(y_max or max_y * 1.08)
    chart.yValueAxis.valueStep = max(0.1, (chart.yValueAxis.valueMax - y_min) / 5.0)
    for index, (_, _, color) in enumerate(series):
        chart.lines[index].strokeColor = color
        chart.lines[index].strokeWidth = 1.5
    drawing.add(chart)
    drawing.add(String(235, 176, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=9, fillColor=INK))
    drawing.add(String(235, 10, y_label, textAnchor='middle', fontSize=7, fillColor=MUTED))
    return drawing


def build_pdf(output_path=OUTPUT):
    data = simulate()
    days = data['days']
    trajectory = line_chart([
        (days, data['untreated'], MUTED), (days, data['cpap'], BLUE),
        (days, data['open_loop'], AMBER), (days, data['closed_loop'], GREEN),
    ], 'Simulated AHI trajectories: untreated, CPAP, open-loop and adaptive rTMS', 'day; AHI events per hour')
    controller = line_chart([
        (days, data['control'], RED), (days, data['adherence'], BLUE),
    ], 'Adaptive control effort and modeled CPAP adherence', 'day; normalized fraction', 0, 1.1)
    iterations = np.arange(1, len(data['errors']) + 1)
    convergence = line_chart([(iterations, np.maximum(data['errors'], 1e-8), AMBER)],
                             'Finite continued-fraction phase error', 'convergent index; absolute phase-ratio error', 0, max(0.1, float(max(data['errors'])) * 1.1))

    styles = getSampleStyleSheet()
    title = ParagraphStyle('TitleNature', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=INK, spaceAfter=8)
    authors = ParagraphStyle('Authors', parent=styles['Normal'], fontSize=9, leading=12, textColor=BLUE, spaceAfter=12)
    heading = ParagraphStyle('HeadingNature', parent=styles['Heading2'], fontSize=11, leading=14, textColor=RED, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle('BodyNature', parent=styles['BodyText'], fontSize=8.7, leading=13, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=6)
    equation = ParagraphStyle('Equation', parent=body, fontName='Courier', fontSize=8.2, leading=12, alignment=TA_CENTER, backColor=PALE, borderPadding=7, spaceBefore=4, spaceAfter=7)
    caption = ParagraphStyle('Caption', parent=body, fontSize=7.5, leading=10, textColor=MUTED, spaceAfter=8)
    notice = ParagraphStyle('Notice', parent=body, fontName='Helvetica-Bold', textColor=RED, borderColor=RED, borderWidth=0.6, borderPadding=7)

    baseline, frequency, gain, duration, ratio = data['params']
    fractions = ', '.join(f'{p}/{q}' for p, q in data['convergents'])
    story = [
        Paragraph('Finite-Mathematics Modeling of Adaptive Respiratory-Gated rTMS for Sleep Apnea', title),
        Paragraph('Nature Portfolio preprint-style computational study | NeuroMorph rTMS Platform | 18 August 2026', authors),
        Paragraph('Research status', heading),
        Paragraph('This manuscript reports an in silico hypothesis-generating model derived from the application. It is not a clinical trial, does not establish efficacy or safety, and must not be used to prescribe stimulation. rTMS for sleep apnea requires ethics approval, specialist oversight, device-specific safety review and prospective validation.', notice),
        Paragraph('Abstract', heading),
        Paragraph(f'Obstructive sleep apnea is quantified clinically using the apnea-hypopnea index (AHI), while standard treatment commonly relies on positive airway pressure and other established interventions. We evaluated a deterministic simulation of respiratory-gated repetitive transcranial magnetic stimulation (rTMS) as a research hypothesis. Four {duration}-day trajectories were compared: untreated progression, variable-adherence CPAP, open-loop rTMS and adaptive closed-loop rTMS. A finite recurrence model coupled AHI error to bounded control effort, while continued-fraction convergents approximated a respiratory synchronization ratio of {ratio:.4f}. Under the default simulated parameters, closed-loop AHI changed from {baseline:.1f} to {data["closed_loop"][-1]:.1f} events h<super>-1</super>. These values are model outputs, not observed patient outcomes. The framework provides reproducible equations and falsifiable targets for future feasibility studies.', body),
        Paragraph('Introduction', heading),
        Paragraph('Sleep apnea produces recurrent airflow reduction, oxygen desaturation and sleep fragmentation. A computational neuromodulation model may help formulate questions about whether stimulation timing, feedback and adherence-sensitive control deserve preclinical evaluation. The present work formalizes the application model without asserting therapeutic benefit. Its purpose is to expose assumptions, finite update rules and measurable failure criteria.', body),
        Paragraph('Finite mathematical formulation', heading),
        Paragraph('Let A_k denote AHI on day k, A_inf the model floor, f the stimulation frequency and g the adaptive gain. Open-loop and adaptive rates are finite parameter maps:', body),
        Paragraph('lambda_open = 0.06 (f / 10)<br/>lambda_adapt = 0.12 (f / 10) (0.5 + 0.5 g)<br/>A_k = max(A_min, A_inf + (A_0 - A_inf) exp(-lambda k) + epsilon_k)', equation),
        Paragraph('The controller uses a bounded proportional recurrence around the research target A* = 5 events h^-1. A safety-governed implementation would additionally require hard device limits, clinician enablement and independent respiratory monitoring.', body),
        Paragraph('e_k = A_k - A*; &nbsp; u_k = clip(g e_k / A_0, 0, 1)<br/>x_(k+1) = F x_k + G u_k + w_k; &nbsp; y_k = H x_k + v_k<br/>J_N = sum_(k=1)^N [q e_k^2 + r u_k^2 + s (u_k-u_(k-1))^2]', equation),
        trajectory,
        Paragraph('Figure 1 | Deterministic simulated AHI trajectories. Curves are generated by the application equations with seeded noise and are not clinical observations.', caption),
        Paragraph('Respiratory phase synchronization', heading),
        Paragraph('A finite continued fraction approximates the target respiratory-to-stimulation phase ratio. For coefficients a_j, numerator p_j and denominator q_j satisfy the recurrence below. Smaller approximation error does not imply biological synchronization; it only quantifies arithmetic agreement.', body),
        Paragraph('rho = [a_0; a_1, ..., a_m]<br/>p_j = a_j p_(j-1) + p_(j-2); &nbsp; q_j = a_j q_(j-1) + q_(j-2)<br/>delta_j = |rho - p_j/q_j|', equation),
        Paragraph(f'For rho = {ratio:.4f}, the finite convergents are {fractions}.', body),
        convergence,
        Paragraph('Figure 2 | Absolute arithmetic error across finite continued-fraction convergents.', caption),
        Paragraph('Adaptive characteristics', heading),
        controller,
        Paragraph('Figure 3 | Bounded adaptive effort and modeled CPAP adherence. Control effort is dimensionless and does not map directly to stimulator output.', caption),
    ]

    results = [
        ['Arm', 'Day 30 AHI', 'Change from baseline', 'Interpretation'],
        ['Untreated simulation', f'{data["untreated"][-1]:.1f}', f'{data["untreated"][-1]-baseline:+.1f}', 'Comparator only'],
        ['Variable-adherence CPAP', f'{data["cpap"][-1]:.1f}', f'{data["cpap"][-1]-baseline:+.1f}', 'Synthetic adherence model'],
        ['Open-loop rTMS', f'{data["open_loop"][-1]:.1f}', f'{data["open_loop"][-1]-baseline:+.1f}', 'Unvalidated model'],
        ['Adaptive rTMS', f'{data["closed_loop"][-1]:.1f}', f'{data["closed_loop"][-1]-baseline:+.1f}', 'Unvalidated model'],
    ]
    table = Table(results, colWidths=[4.1*cm, 2.8*cm, 3.5*cm, 5.0*cm], repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), RED), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 7.5), ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d0d5dd')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    story += [
        Paragraph('Simulated results', heading), table, Spacer(1, 8),
        Paragraph('Study design for prospective validation', heading),
        Paragraph('A future study should begin with device and respiratory-physiology feasibility rather than efficacy claims. Prespecified outcomes should include adverse events, arousal index, oxygen-desaturation index, overnight AHI measured by validated polysomnography, sleep-stage effects and durability. Sham control, blinded scoring, preregistered stopping rules and independent safety monitoring are essential. CPAP should not be withdrawn solely for participation.', body),
        Paragraph('Candidate finite decision rule: stop stimulation if any device safety boundary is exceeded, if respiratory status worsens beyond a preregistered margin, or if posterior harm probability crosses eta_stop. Continue escalation only when safety constraints remain satisfied.', body),
        Paragraph('stop if P(harm | D_k) >= eta_stop OR u_k > u_max OR oxygenation_k < O_min', equation),
        Paragraph('Limitations', heading),
        Paragraph('The application uses synthetic noise, assumed exponential response, no anatomy-specific electric-field calculation, no oxygen saturation dynamics, no sleep-stage state, no comorbidity model and no patient data. The CPAP comparator is simplified and should not be interpreted as a realistic effectiveness estimate. AHI below five in the adaptive arm is a consequence of the chosen asymptote and rate equation, not evidence that rTMS normalizes breathing. Continued fractions solve a timing approximation problem but do not demonstrate neural entrainment.', body),
        Paragraph('Methods and reproducibility', heading),
        Paragraph(f'The simulation uses seed 101, baseline AHI {baseline:.1f}, frequency {frequency:.1f} Hz, adaptive gain {gain:.2f}, duration {duration} days and target ratio {ratio:.4f}. NumPy generates finite arrays and ReportLab renders vector figures and the PDF. The generator is stored with the rTMS application so all reported default values can be regenerated.', body),
        Paragraph('Data availability', heading),
        Paragraph('No participant data were used. All values in this manuscript are generated by the included deterministic simulation. The PDF and source generator are local application artifacts.', body),
        Paragraph('References and reporting context', heading),
        Paragraph('Clinical interpretation should follow current sleep-medicine diagnostic guidance, established treatment guidelines, applicable rTMS device labeling and consensus stimulation-safety recommendations. This computational draft intentionally avoids inventing study citations or claiming regulatory authorization; a submission-ready manuscript requires a verified systematic literature review and complete bibliography.', body),
    ]

    document = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2.1*cm, rightMargin=2.1*cm, topMargin=1.8*cm, bottomMargin=1.8*cm, title='Finite-Mathematics Modeling of Adaptive Respiratory-Gated rTMS for Sleep Apnea', author='NeuroMorph rTMS Platform')
    document.build(story)
    return str(output_path)


if __name__ == '__main__':
    print(build_pdf())
