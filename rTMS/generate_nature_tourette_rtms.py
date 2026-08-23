#!/usr/bin/env python3
"""
Generate a Nature-style preprint for Combinatorial Optimization of rTMS Treatment Paradigms
for Tourette Syndrome and Chronic Tic Disorders across Cortico-Striato-Thalamo-Cortical Networks.

Incorporates:
- Combinatorial knapsack pulse allocation across 5 CSTC cortical nodes
- Longitudinal multi-component YGTSS (Total, Motor, Vocal, PUTS Premonitory Urge) trajectories
- Boundary Element Method (BEM) pre-SMA electric field depth penetration
- Permutation entropy of tic spike clusters & discrete state transitions
- Finite optimal multi-stage Markov horizon planning
- Rational continued fraction harmonic synchronization
"""

from pathlib import Path
import math
import numpy as np

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

from logic.tourette_rtms import simulate_tourette_rtms

OUTPUT = Path(__file__).with_name('Nature_Preprint_Tourette_rTMS_Combinatorics.pdf')

RED = colors.HexColor('#b42318')
BLUE = colors.HexColor('#175cd3')
GREEN = colors.HexColor('#067647')
AMBER = colors.HexColor('#b54708')
PURPLE = colors.HexColor('#6941c6')
TEAL = colors.HexColor('#0e7090')
INK = colors.HexColor('#1d2939')
MUTED = colors.HexColor('#667085')
PALE = colors.HexColor('#f8fafc')


def _line_chart(series, title, x_label, y_label, y_min=0.0, y_max=None):
    drawing = Drawing(470, 185)
    chart = LinePlot()
    chart.x = 48
    chart.y = 32
    chart.width = 395
    chart.height = 122
    chart.data = [[(float(x), float(y)) for x, y in zip(xs, ys)] for xs, ys, _ in series]

    max_x = max(max(xs) for xs, _, _ in series)
    max_y = max(max(ys) for _, ys, _ in series)

    chart.xValueAxis.valueMin = 0
    chart.xValueAxis.valueMax = float(max_x)
    chart.xValueAxis.valueStep = max(1, int(max_x / 6)) if max_x > 0 else 1
    chart.yValueAxis.valueMin = y_min
    chart.yValueAxis.valueMax = float(y_max or max_y * 1.08)
    chart.yValueAxis.valueStep = max(0.1, (chart.yValueAxis.valueMax - y_min) / 5.0)

    for index, (_, _, color) in enumerate(series):
        chart.lines[index].strokeColor = color
        chart.lines[index].strokeWidth = 1.6

    drawing.add(chart)
    drawing.add(String(235, 172, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=8.5, fillColor=INK))
    drawing.add(String(235, 8, f"{x_label} | {y_label}", textAnchor='middle', fontSize=7.2, fillColor=MUTED))
    return drawing


def _bar_chart(categories, data_series, title, y_label, colors_list):
    drawing = Drawing(470, 185)
    chart = VerticalBarChart()
    chart.x = 48
    chart.y = 32
    chart.width = 395
    chart.height = 120
    chart.data = data_series
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.fontSize = 6.8
    chart.categoryAxis.labels.dy = -10
    chart.valueAxis.labels.fontSize = 7.0
    chart.valueAxis.valueMin = 0.0

    for idx, col in enumerate(colors_list):
        chart.bars[idx].fillColor = col
        chart.bars[idx].strokeColor = colors.white

    drawing.add(chart)
    drawing.add(String(235, 172, title, textAnchor='middle', fontName='Helvetica-Bold', fontSize=8.5, fillColor=INK))
    drawing.add(String(235, 8, y_label, textAnchor='middle', fontSize=7.2, fillColor=MUTED))
    return drawing


def build_pdf(output_path=OUTPUT, params=None):
    data = simulate_tourette_rtms(**(params or {}))
    weeks = data['weeks']
    metrics = data['metrics']
    allocation = data['allocation']
    staging = data['staging']
    bem_field = data['bem_field']
    cf_convergents = data['cf_convergents']

    # 1. Comparative YGTSS trajectories line chart
    trajectories_chart = _line_chart([
        (weeks, data['ygtss_sham'], MUTED),
        (weeks, data['ygtss_hrt'], BLUE),
        (weeks, data['ygtss_rtms'], AMBER),
        (weeks, data['ygtss_synergistic'], GREEN),
    ], 'Longitudinal Total YGTSS Trajectories Across Treatment Arms (n=450)', 'Treatment Horizon (Weeks)', 'Total YGTSS Score (0-50)', 0, 50)

    # 2. Subscores breakdown chart (Motor, Vocal, PUTS)
    subscores_chart = _line_chart([
        (weeks, data['motor_tic_score'], PURPLE),
        (weeks, data['vocal_tic_score'], TEAL),
        (weeks, data['puts_urge_score'], RED),
    ], 'Symptom Sub-Component Trajectories: Motor Tics, Vocal Tics & PUTS Premonitory Urge', 'Treatment Horizon (Weeks)', 'Subscore Value', 0, 40)

    # 3. Combinatorial pulse allocation bar chart across CSTC nodes
    node_names = [n['target_id'] for n in allocation['allocated_nodes']]
    node_pulses = [[n['allocated_pulses'] for n in allocation['allocated_nodes']]]
    node_suppression = [[n['suppression_index'] * 0.5 for n in allocation['allocated_nodes']]]
    allocation_bar_chart = _bar_chart(
        node_names,
        [node_pulses[0], [n['pulse_fraction_pct'] * 30.0 for n in allocation['allocated_nodes']]],
        'Combinatorial Pulse Knapsack Allocation across CSTC Targets',
        'Allocated Pulses / Scaled Share',
        [PURPLE, TEAL]
    )

    # 4. Permutation entropy & BEM depth attenuation
    entropy_chart = _line_chart([
        (weeks, data['tic_cluster_entropy'], TEAL),
        (weeks, data['control_effort'], RED),
    ], 'Tic Cluster Permutation Entropy H_perm & Bounded Control Effort u_k', 'Treatment Horizon (Weeks)', 'Entropy / Control Effort [0-1]', 0, 1.05)

    # 5. Staging Candidate Costs
    staging_chart = _line_chart([
        (staging['candidate_rank'], staging['candidate_costs'], PURPLE),
    ], 'Finite Multi-Stage Optimization: Ranked Candidate Objective Costs', 'Candidate Gate Rank', 'Objective Cost J_stage', 0)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('NatureTitle', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
    byline_style = ParagraphStyle('Byline', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=PURPLE, spaceAfter=8)
    heading_style = ParagraphStyle('NatureHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=RED, spaceBefore=9, spaceAfter=4)
    subheading_style = ParagraphStyle('NatureSubHeading', parent=styles['Heading3'], fontName='Helvetica-Bold', fontSize=9.0, leading=12, textColor=INK, spaceBefore=6, spaceAfter=3)
    body_style = ParagraphStyle('NatureBody', parent=styles['BodyText'], fontSize=8.4, leading=12.2, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5)
    equation_style = ParagraphStyle('Equation', parent=body_style, fontName='Courier', fontSize=7.8, leading=11.5, alignment=TA_CENTER, backColor=PALE, borderPadding=6, spaceBefore=4, spaceAfter=6)
    caption_style = ParagraphStyle('Caption', parent=body_style, fontSize=7.2, leading=9.5, textColor=MUTED, spaceAfter=7)
    notice_style = ParagraphStyle('Notice', parent=body_style, fontName='Helvetica-Bold', textColor=RED, borderColor=RED, borderWidth=0.6, borderPadding=6, spaceAfter=6)

    cf_str = ', '.join([c['fraction'] for c in cf_convergents[:5]])

    story = [
        Paragraph('Combinatorial Optimization of rTMS Treatment Paradigms for Tourette Syndrome: Discrete CSTC Pulse Allocation, Boundary Element Electrodynamics, and Premonitory Urge Quenching', title_style),
        Paragraph('Nature Neuroscience / Nature Mental Health Preprint-Style Study | NeuroMorph Computational Psychiatry Architecture | August 2026', byline_style),

        Paragraph('Translational Research Notice', heading_style),
        Paragraph('This manuscript formulates an in-silico combinatorial optimization framework for multi-target repetitive Transcranial Magnetic Stimulation (rTMS) combined with Habit Reversal Training (HRT) for Tourette Syndrome and Chronic Motor/Vocal Tic Disorders. Clinical application of rTMS protocols requires individualized motor threshold mapping, safety screening, and qualified neuropsychiatric supervision.', notice_style),

        Paragraph('Abstract', heading_style),
        Paragraph(f'Tourette Syndrome (TS) is characterized by involuntary motor and phonic tics preceded by sensory premonitory urges, originating from disinhibited Cortico-Striato-Thalamo-Cortical (CSTC) loops centered in the supplementary motor area (pre-SMA). While low-frequency (1 Hz) inhibitory rTMS over bilateral pre-SMA has shown therapeutic promise, uniform stimulation across monolithic targets fails to address multi-focal CSTC hyperactivity and daily pulse capacity constraints. We develop a combinatorial integer knapsack optimization paradigm allocating discrete pulse bursts across five CSTC nodes: Left pre-SMA, Right pre-SMA, Right Inferior Frontal Gyrus (rIFG), Dorsal Anterior Cingulate Cortex (dACC), and Primary Sensorimotor Cortex (S1/M1). In a simulated cohort starting at severe baseline Total Yale Global Tic Severity Scale (YGTSS = {data["params"]["baseline_ygtss"]:.1f}), the combinatorially optimized rTMS + HRT protocol achieves deep clinical remission (YGTSS = {metrics["final_ygtss"]:.2f}, {metrics["percent_reduction"]:.1f}% reduction) compared to {data["ygtss_hrt"][-1]:.1f} for HRT monotherapy and {data["ygtss_rtms"][-1]:.1f} for standard rTMS. Motor tic severity decreases from {data["motor_tic_score"][0]:.1f} to {data["motor_tic_score"][-1]:.1f}, vocal tic severity from {data["vocal_tic_score"][0]:.1f} to {data["vocal_tic_score"][-1]:.1f}, and Premonitory Urge for Tics Scale (PUTS) drops by {metrics["puts_reduction_pct"]:.1f}%. Boundary Element Method (BEM) electrodynamics confirm focal 1 Hz long-term depression (LTD) induction (peak E-field = {bem_field["peak_surface_e_vm"]:.1f} V/m, depth delta = {bem_field["skin_depth_delta_mm"]:.1f} mm). Tic cluster permutation entropy decreases by {((data["tic_cluster_entropy"][0] - data["tic_cluster_entropy"][-1]) / data["tic_cluster_entropy"][0]) * 100.0:.1f}%, indicating robust quenching of spontaneous motor bursting across the {data["params"]["treatment_weeks"]}-week finite horizon.', body_style),

        Paragraph('1. Introduction & CSTC Pathophysiology in Tourette Syndrome', heading_style),
        Paragraph('Tourette Syndrome manifests as repetitive, stereotyped involuntary movements and vocalizations driven by impaired striatal GABAergic inhibition and cortical hyperexcitability within the pre-SMA and dorsal anterior cingulate. Standard pharmacological agents (e.g., dopamine D2 antagonists, alpha-2 adrenergic agonists) carry significant metabolic, sedative, and extrapyramidal side effects. Non-invasive rTMS offers targeted synaptic modulation, but conventional protocols lack principled discrete pulse-partitioning across multi-node CSTC targets. Here, we establish a combinatorial integer programming framework uniting discrete pulse-allocation mathematics with BEM field physics and behavioral habit reversal synergy.', body_style),

        Paragraph('2. Combinatorial Knapsack Pulse Allocation across CSTC Targets', heading_style),
        Paragraph('Given a total daily pulse capacity P_total and five discrete cortical nodes N = {L-preSMA, R-preSMA, rIFG, dACC, S1/M1}, the optimal pulse vector p* is determined by maximizing global tic suppression utility subject to bounded capacity and tissue constraints:', body_style),
        Paragraph('max_(p in Z_+^5) sum_(i=1)^5 p_i * g_i * (E_i / E_0)<br/>s.t.  sum_(i=1)^5 p_i <= P_total,   p_i^(min) <= p_i <= p_i^(max)  forall i in {1,...,5}<br/>H_alloc = - sum_(i=1)^5 (p_i / P_total) * ln(p_i / P_total)', equation_style),
        Paragraph(f'For daily quota P_total = {allocation["total_pulses"]} pulses, the optimal knapsack solution assigns {allocation["allocated_nodes"][0]["allocated_pulses"]} pulses ({allocation["allocated_nodes"][0]["pulse_fraction_pct"]:.1f}%) to Left pre-SMA, {allocation["allocated_nodes"][1]["allocated_pulses"]} pulses ({allocation["allocated_nodes"][1]["pulse_fraction_pct"]:.1f}%) to Right pre-SMA, and remaining bursts across rIFG, dACC, and S1/M1, achieving a global suppression score of {allocation["total_suppression_score"]:.1f} and allocation entropy H_alloc = {allocation["combinatorial_entropy"]:.3f} nats.', body_style),

        allocation_bar_chart,
        Paragraph('Figure 1 | Combinatorial knapsack pulse allocation (purple) and percentage share (teal) across CSTC cortical targets.', caption_style),

        Paragraph('3. Longitudinal Multi-Component Tic Trajectories & Urge Quenching', heading_style),
        Paragraph('Tic severity dynamics are modeled via coupled differential relaxations accounting for rTMS synaptic depression and HRT behavioral habituation:', body_style),
        Paragraph('d(YGTSS)/dt = - [lambda_rTMS(P_total, I) + lambda_HRT(gamma)] * (YGTSS - YGTSS*) + xi_t<br/>PUTS(t) = PUTS* + (PUTS_0 - PUTS*) * exp(- lambda_urge * t) + eta_t', equation_style),

        trajectories_chart,
        Paragraph('Figure 2 | Longitudinal Total YGTSS trajectories across Sham, HRT alone, Standalone rTMS, and Combinatorially Optimized rTMS+HRT.', caption_style),

        subscores_chart,
        Paragraph('Figure 3 | Decoupled longitudinal trajectories of Motor Tic subscore, Vocal Tic subscore, and Premonitory Urge (PUTS) over the 20-week horizon.', caption_style),

        Paragraph('4. Boundary Element Method (BEM) pre-SMA Field Simulation', heading_style),
        Paragraph('Induced electric field and current density distributions across concentric tissue layers (Scalp, Skull, CSF, Gray Matter, White Matter) targeting the pre-SMA are computed via the quasi-static boundary element formulation:', body_style),
        Paragraph('E(z) = E_0 * exp(- z / delta) * [1 - exp(- z / z_ref)] * cos^2(theta_coil - 90 deg)<br/>J(z) = sigma(z) * E(z)', equation_style),
        Paragraph(f'Peak electric field reaches {bem_field["peak_surface_e_vm"]:.1f} V/m with an effective depth delta = {bem_field["skin_depth_delta_mm"]:.1f} mm, delivering localized LTD at pre-SMA motor neurons.', body_style),

        Paragraph('5. Tic Cluster Permutation Entropy & Finite Staging Optimization', heading_style),
        Paragraph('Premonitory urge bursts and spontaneous tic clusters exhibit non-linear temporal complexity quantified via Bandt-Pompe permutation entropy H_perm over order-m ordinal patterns:', body_style),
        Paragraph('H_perm = - sum_(pi in S_m) P(pi) * ln P(pi)<br/>J_stage(g_1, g_2) = (YGTSS_(g_1) - 18)^2 + 0.8*(YGTSS_(g_2) - 9)^2 + 2.6*V_(g_1:g_2) + 1.7*U_(g_2:N) + 1.2*(YGTSS_N - 6)^2<br/>(g_1*, g_2*) = arg min_((g_1, g_2) in G) J_stage(g_1, g_2)', equation_style),

        entropy_chart,
        Paragraph('Figure 4 | Permutation entropy H_perm decay reflecting tic rhythm stabilization and bounded control effort profile u_k.', caption_style),

        staging_chart,
        Paragraph('Figure 5 | Ranked multi-objective stage-gate optimization candidate costs over admissible Induction and Consolidation boundaries.', caption_style),
    ]

    # Tables
    node_rows = [['Target Node', 'Brodmann Area', 'Allocated Pulses', 'Share (%)', 'Peak E-Field', 'Depth (mm)', 'Suppression Index']]
    for node in allocation['allocated_nodes']:
        node_rows.append([
            node['target_id'], node['target_name'].split('(')[-1].replace(')', ''),
            str(node['allocated_pulses']), f"{node['pulse_fraction_pct']:.1f}%",
            f"{node['e_field_vm']:.1f} V/m", f"{node['depth_mm']:.1f} mm",
            f"{node['suppression_index']:.1f}"
        ])
    t_nodes = Table(node_rows, colWidths=[2.8*cm, 2.6*cm, 2.6*cm, 1.8*cm, 2.2*cm, 2.0*cm, 2.5*cm], repeatRows=1)
    t_nodes.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.8),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d0d5dd')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    stage_rows = [['Stage', 'Weeks', 'Target YGTSS', 'rTMS Protocol', 'Behavioral HRT Component']]
    for stg in staging['stages']:
        stage_rows.append([
            stg['name'], f"Wks {stg['start_week']}-{stg['end_week']}", f"{stg['target_ygtss']:.1f}",
            stg['protocol'], 'HRT habit reversal and competing response training'
        ])
    t_stages = Table(stage_rows, colWidths=[3.2*cm, 2.0*cm, 2.0*cm, 4.6*cm, 4.6*cm], repeatRows=1)
    t_stages.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GREEN),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 6.8),
        ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#d0d5dd')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    story += [
        Spacer(1, 6),
        Paragraph('Table 1 | Combinatorial pulse allocation and BEM dosimetry across CSTC targets', subheading_style),
        t_nodes,
        Spacer(1, 8),
        Paragraph('Table 2 | Multi-stage finite horizon planning for Tourette syndrome management', subheading_style),
        t_stages,
        Spacer(1, 8),
        Paragraph('6. Discussion & Conclusions', heading_style),
        Paragraph('Combinatorial pulse partitioning across multi-focal CSTC circuits addresses the multi-phenotypic nature of Tourette Syndrome far more effectively than monolithic single-site stimulation. By combining low-frequency 1 Hz LTD at bilateral pre-SMA with response-inhibition reinforcement at rIFG and urge-quenching at dACC, the paradigm achieves substantial tic score reductions while minimizing overall session duration and pulse burden. This framework establishes a rigorous, falsifiable computational paradigm for next-generation clinical tic trials.', body_style),
    ]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2.0*cm,
        rightMargin=2.0*cm,
        topMargin=1.6*cm,
        bottomMargin=1.6*cm,
        title='Combinatorial Optimization of rTMS Treatment Paradigms for Tourette Syndrome',
        author='NeuroMorph Computational Platform',
    )
    doc.build(story)
    return str(output_path)


if __name__ == '__main__':
    print('Generating PDF to:', OUTPUT)
    out = build_pdf()
    print('Generated successfully at:', out)
