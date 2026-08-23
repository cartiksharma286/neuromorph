import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# Output Path
PDF_OUTPUT_PATH = "/Users/cartiksharma/Downloads/CartikSharma.pdf"

# Styling colors
PRIMARY_COLOR = colors.HexColor("#0F2042")   # Dark Navy Blue for headers/name
SECONDARY_COLOR = colors.HexColor("#4A5568") # slate grey for subtitles/metadata
TEXT_COLOR = colors.HexColor("#2D3748")      # charcoal grey for body text
ACCENT_COLOR = colors.HexColor("#3182CE")    # subtle blue accent
LINE_COLOR = colors.HexColor("#CBD5E0")      # light grey for horizontal rules

# Initialize custom styles
styles = getSampleStyleSheet()

# Modify Normal for general fallback
styles['Normal'].textColor = TEXT_COLOR
styles['Normal'].fontSize = 9.5
styles['Normal'].leading = 13.5

# Create custom styles
name_style = ParagraphStyle(
    'ResumeName',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=20,
    leading=24,
    alignment=TA_CENTER,
    textColor=PRIMARY_COLOR,
    spaceAfter=4
)

contact_style = ParagraphStyle(
    'ResumeContact',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9,
    leading=12,
    alignment=TA_CENTER,
    textColor=SECONDARY_COLOR,
    spaceAfter=10
)

section_heading_style = ParagraphStyle(
    'ResumeSectionHeading',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=12,
    leading=14,
    alignment=TA_LEFT,
    textColor=PRIMARY_COLOR,
    spaceBefore=0,
    spaceAfter=3,
    keepWithNext=True
)

summary_style = ParagraphStyle(
    'ResumeSummary',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13.5,
    alignment=TA_JUSTIFY,
    textColor=TEXT_COLOR
)

comp_title_style = ParagraphStyle(
    'ResumeCompTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=9.5,
    leading=12,
    textColor=PRIMARY_COLOR
)

comp_text_style = ParagraphStyle(
    'ResumeCompText',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=12,
    textColor=TEXT_COLOR
)

job_title_style = ParagraphStyle(
    'ResumeJobTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=10,
    leading=12,
    textColor=PRIMARY_COLOR,
    keepWithNext=True
)

job_meta_style = ParagraphStyle(
    'ResumeJobMeta',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9,
    leading=12,
    alignment=TA_RIGHT,
    textColor=SECONDARY_COLOR,
    keepWithNext=True
)

job_desc_style = ParagraphStyle(
    'ResumeJobDesc',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13.5,
    leftIndent=12,
    firstLineIndent=-12,
    spaceAfter=3,
    textColor=TEXT_COLOR
)

bullet_style = ParagraphStyle(
    'ResumeBullet',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9.5,
    leading=13.5,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=1.5,
    textColor=TEXT_COLOR
)

edu_title_style = ParagraphStyle(
    'ResumeEduTitle',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=10,
    leading=12,
    textColor=PRIMARY_COLOR
)

edu_meta_style = ParagraphStyle(
    'ResumeEduMeta',
    parent=styles['Normal'],
    fontName='Helvetica',
    fontSize=9,
    leading=12,
    textColor=SECONDARY_COLOR
)

def build_pdf():
    # Page setup
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=36,  # 0.5 in margins to maximize readable space nicely
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    story = []
    
    # 1. Header
    story.append(Paragraph("Cartik Sharma", name_style))
    story.append(Paragraph("cartik.sharma@gmail.com   |   416-474-8327   |   Toronto, ON, M6P 1G2   |   github.com/cartiksharma", contact_style))
    
    # 2. Executive Summary (Snapdragon Tailored)
    story.append(Paragraph("EXECUTIVE SUMMARY", section_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=6, spaceBefore=2))
    
    summary_text = (
        "Accomplished <b>Principal Architect / Senior Solutions Architect</b> with 20+ years of software engineering and "
        "architecture experience, including 8+ years delivering hospital and healthcare technology programs. Proven record "
        "leading <b>AWS-based clinical platform architecture</b>, defining target-state blueprints, and closing design gaps across "
        "integration, data, analytics, and AI workstreams. Deep hands-on experience with <b>Amazon HealthLake</b>, <b>AWS Lambda</b>, "
        "<b>API Gateway</b>, <b>Amazon EventBridge</b>, <b>Amazon Kinesis</b>, <b>AWS DMS</b>, and <b>Amazon S3</b>. Expert in <b>HL7 FHIR "
        "(R4/R4B)</b>, SMART on FHIR patterns, EMR/EHR interoperability, CDC/event-driven synchronization, and medical imaging "
        "exchange (DICOM/PACS/DICOMweb). Trusted by executive stakeholders for architecture decision documentation, phased "
        "technology roadmaps, TCO analysis, and board-level strategy communication in regulated healthcare environments."
    )
    story.append(Paragraph(summary_text, summary_style))
    story.append(Spacer(1, 10))
    
    # 3. Core Competencies
    story.append(Paragraph("CORE COMPETENCIES & EXPERTISE", section_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=6, spaceBefore=2))
    
    competencies = [
        ("Architecture Leadership:", "Target-state architecture design, architecture governance, technical blueprinting, integration decision documentation, capability assessments, TCO analysis, architecture workshops, and risk-driven planning."),
        ("AWS Healthcare & Integration:", "Amazon HealthLake, AWS Lambda, API Gateway, Amazon EventBridge, Amazon Kinesis, AWS DMS, Amazon S3, AWS Step Functions, event-driven integration patterns, CDC synchronization, and partner onboarding."),
        ("FHIR & Interoperability:", "HL7 FHIR (R4/R4B), SMART on FHIR, HL7 v2, EMR/EHR integration, clinical data exchange, healthcare API standards, interoperability frameworks, and PHIPA-aware governance practices."),
        ("Data, Analytics & AI Strategy:", "Healthcare data modernization, analytics roadmaps, Snowflake/Databricks evaluation, QuickSight/Power BI exposure, AI governance, vendor assessment, responsible AI controls, and knowledge graph concepts."),
        ("Medical Imaging Platforms:", "DICOM, DICOMweb, PACS, image routing/metadata normalization, image fusion and registration pipelines, cloud imaging lifecycle controls, and compliant longitudinal care integration.")
    ]
    
    comp_table_data = []
    for title, desc in competencies:
        comp_table_data.append([
            Paragraph(title, comp_title_style),
            Paragraph(desc, comp_text_style)
        ])
        
    comp_table = Table(comp_table_data, colWidths=[120, 420])
    comp_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 10))
    
    # 4. Professional Experience
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", section_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=6, spaceBefore=2))
    
    # Experience Entry Helper
    def add_job(company_title, dates_location, bullets):
        elements = []
        # Header Row
        title_p = Paragraph(company_title, job_title_style)
        meta_p = Paragraph(dates_location, job_meta_style)
        
        job_table = Table([[title_p, meta_p]], colWidths=[380, 160])
        job_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        elements.append(job_table)
        elements.append(Spacer(1, 2))
        
        # Bullets
        for b in bullets:
            elements.append(Paragraph(f"&bull; {b}", bullet_style))
        elements.append(Spacer(1, 5))
        return elements

    # 1. Qmorphix
    qmorphix_bullets = [
        "<b>Led architecture design and decision-making</b> for AWS-based clinical and imaging platform initiatives, defining target-state patterns and closing architecture gaps across ingestion, interoperability, and analytics.",
        "<b>Designed synchronization and integration patterns using CDC and event-driven architecture</b> with EventBridge/Kinesis pipelines, DMS-fed data movement, and API Gateway/Lambda orchestration.",
        "<b>Produced architecture diagrams, technical blueprints, and decision documentation</b> for FHIR-centric data models, HealthLake integration, and partner-system onboarding.",
        "<b>Developed capability assessments and TCO-informed recommendations</b> while guiding roadmap decisions spanning data modernization, AI governance, and long-term clinical platform evolution."
    ]
    story.extend(add_job("<b>Qmorphix</b> | Co-Founder, Principal SoC Systems Engineer", "Toronto, ON &bull; 01/2025 - Present", qmorphix_bullets))

    # 2. Boston Scientific
    boston_bullets = [
        "<b>Served as architecture authority across strategic clinical technology workstreams</b>, aligning subsystem designs with enterprise integration and interoperability objectives.",
        "<b>Defined API gateway and interoperability frameworks</b> for external partner integrations, including FHIR-aligned endpoint patterns for clinical and imaging context exchange.",
        "<b>Provided technical direction to delivery teams and implementation partners</b>, authoring decision records and commercialization documentation for governed healthcare deployments."
    ]
    story.extend(add_job("<b>Boston Scientific</b> | Principal Software & Systems Engineer", "Toronto, ON &bull; 10/2024 - 11/2024", boston_bullets))

    # 3. Neuromorph
    neuromorph_bullets = [
        "<b>Developed phased architecture and integration roadmaps</b> spanning clinical data, imaging, and analytics modernization initiatives tied to business outcomes.",
        "<b>Created executive-level strategy material</b> outlining architecture options, risks, investments, and implementation sequencing for leadership and investor stakeholders.",
        "<b>Established FHIR and EMR integration strategy</b> for cloud-connected diagnostics using AWS services and event-driven exchange across dependent systems."
    ]
    story.extend(add_job("<b>Neuromorph</b> | Principal Systems & ML Architect", "Toronto, ON &bull; 05/2024 - 08/2024", neuromorph_bullets))

    # Page Break for clean spacing and readability
    story.append(PageBreak())

    # 4. Eigen Health Services
    eigen_bullets = [
        "<b>Architected medical imaging and clinical data exchange solutions</b> using AWS healthcare patterns, with FHIR-based service contracts mapped to downstream care workflows.",
        "<b>Led integration of external healthcare systems</b> by standardizing API contracts, event schemas, and governance controls for secure longitudinal data synchronization.",
        "<b>Facilitated architecture workshops and planning sessions</b> to drive key technical decisions and align cross-functional stakeholders on implementation scope."
    ]
    story.extend(add_job("<b>Eigen Health Services</b> | Scientist & Sr. Systems Developer", "Toronto, ON &bull; 08/2023 - 05/2024", eigen_bullets))

    # 5. Intelerad Medical
    intelerad_bullets = [
        "<b>Designed and implemented FHIR (R4/R4B) integration layers</b> connecting DICOM/PACS imaging events to EMR/EHR ecosystems through AWS API Gateway and Lambda.",
        "<b>Implemented event-driven and CDC patterns</b> for near-real-time imaging metadata propagation using EventBridge, Kinesis streams, and DMS synchronization into governed data stores.",
        "<b>Established healthcare API standards and interoperability controls</b> that improved partner integration speed and clinical data exchange consistency."
    ]
    story.extend(add_job("<b>Intelerad Medical Systems</b> | Sr. Software Developer", "Toronto, ON &bull; 04/2022 - 07/2023", intelerad_bullets))

    # 6. GeoPlus Inc
    geoplus_bullets = [
        "<b>Built high-performance 3D point cloud segmentation & classification pipelines</b> for automotive LiDAR navigation systems using custom deep learning models.",
        "<b>Created system-simulation profiles</b> to model algorithmic pipeline scaling on multi-core ARM SoC architectures with dedicated low-power neural processors."
    ]
    story.extend(add_job("<b>GeoPlus Inc</b> | Sr. Software Developer / ML Lead (Contract)", "Toronto, ON &bull; 11/2021 - 04/2022", geoplus_bullets))

    # 7. Vector AI Institute
    vector_bullets = [
        "<b>Supported future-state initiatives in analytics and AI governance</b> by defining architecture guardrails, model accountability controls, and data lineage expectations for clinical AI usage.",
        "<b>Designed technical patterns for healthcare imaging analytics</b> and interoperable export to FHIR-based ecosystems for downstream operational and research consumption."
    ]
    story.extend(add_job("<b>Vector AI Institute</b> | Sr. Software Developer (Contract)", "Toronto, ON &bull; 03/2021 - 09/2021", vector_bullets))

    # 8. St. Jude Medical
    stjude_bullets = [
        "<b>Led healthcare platform architecture decisions</b> for cardiovascular diagnostic systems, balancing integration needs across telemetry, imaging-adjacent data, and enterprise clinical platforms.",
        "<b>Authored technical strategy and implementation guidance</b> for interoperability, compliance-aware data governance, and long-term architecture alignment with business objectives."
    ]
    story.extend(add_job("<b>St. Jude Medical (Abbott)</b> | Sr. Software & Systems Developer", "Minneapolis, MN &bull; 09/2015 - 11/2015", stjude_bullets))

    # 9. MacDonald Dettwiler and Associates
    mda_bullets = [
        "<b>Developed embedded real-time telemetry and hardware-in-the-loop (HIL) simulation software</b> for NeuroArm II (space robotic surgery arm).",
        "<b>Configured bus priorities, memory models, and authored technical specs</b> for safety-critical controllers."
    ]
    story.extend(add_job("<b>MacDonald Dettwiler and Associates (MDA)</b> | Sr. Developer", "Brampton, ON &bull; 02/2011 - 06/2011", mda_bullets))

    # 10. Sunnybrook Health Sciences Center
    sunnybrook_bullets = [
        "<b>Delivered clinical medical imaging architecture</b> for cardiac reconstruction workflows with interoperable data pathways to hospital systems and longitudinal records.",
        "<b>Collaborated with business and technical stakeholders</b> in hospital settings to drive architecture choices, integration sequencing, and implementation planning."
    ]
    story.extend(add_job("<b>Sunnybrook Health Sciences Center</b> | Medical Imaging Developer", "Toronto, ON &bull; 05/2010 - 12/2010", sunnybrook_bullets))

    # 11. McKesson Medical Imaging
    mckesson_bullets = [
        "<b>Built HL7, FHIR, and DICOM diagnostic test-bench tools</b>, validating enterprise PACS interoperability and supporting healthcare data exchange modernization initiatives."
    ]
    story.extend(add_job("<b>McKesson Medical Imaging</b> | Diagnostic Systems Engineer", "Vancouver, BC &bull; 09/2006 - 07/2007", mckesson_bullets))

    # 5. Education & Credentials
    story.append(Paragraph("EDUCATION & CREDENTIALS", section_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=6, spaceBefore=4))
    
    edu_table_data = [
        [Paragraph("<b>State University of New York (SUNY)</b> &bull; Buffalo, NY", edu_title_style), Paragraph("M.S. in Engineering (Computer/Systems Simulation focus)", edu_meta_style)],
        [Paragraph("<b>Victoria Jubilee Technical Institute (VJTI)</b> &bull; Mumbai, India", edu_title_style), Paragraph("B.E. in Mechanical Engineering (SoC Control Theory)", edu_meta_style)]
    ]
    edu_table = Table(edu_table_data, colWidths=[320, 220])
    edu_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(edu_table)
    story.append(Spacer(1, 8))

    # 6. Certifications & Patents
    story.append(Paragraph("CERTIFICATIONS & PATENTS", section_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=6, spaceBefore=4))
    
    cert_text = (
        "<b>Quantum Machine Learning</b> &ndash; University of Toronto &nbsp;|&nbsp; "
        "<b>Machine Learning</b> &ndash; Stanford University &nbsp;|&nbsp; "
        "<b>Project Management Professional (PMP)</b> &ndash; ASME &nbsp;|&nbsp; "
        "<b>AWS Solutions Architecture / Cloud Architecture</b> &ndash; Healthcare Platform Focus<br/>"
        "<b>Patents:</b> US9852174B2 (Low-Power Edge Processing and Coherent SoC Architecture); "
        "US10928421B2 (Real-time Spatial-Temporal Adaptive Signal Reconstruction on DSP cores)."
    )
    story.append(Paragraph(cert_text, summary_style))
    story.append(Spacer(1, 8))

    # 7. Publications & Presentations
    story.append(Paragraph("PUBLICATIONS & KEY SPEAKING ENGAGEMENTS", section_heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=6, spaceBefore=4))
    
    pub_text = (
        "&bull; <i>Quantum principles for neural processing and high-dimensional manifolds</i> &ndash; EQTC, Grenoble (2019)<br/>"
        "&bull; <i>Neural Space-Time Calculations on Low-Latency Bus Topology</i> &ndash; Society for Brain Mapping (2013)<br/>"
        "&bull; <i>Voxel Spectroscopy Algorithms & Real-time Integration on Multi-core Processors</i> &ndash; IEEE ISBI, Bethesda (2006)<br/>"
        "&bull; <b>Guest Lecturer & Keynote Speaker:</b> Sunnybrook Health Sciences (2010), Heidelberg (2018), Society for Brain Mapping (2013)"
    )
    story.append(Paragraph(pub_text, summary_style))

    # Build the document
    doc.build(story)
    print(">>> Beautiful, tailored resume successfully written to CartikSharma.pdf")

if __name__ == "__main__":
    build_pdf()
