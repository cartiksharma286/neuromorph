#!/usr/bin/env python3
"""
Generate PDF report with embedded simulation images.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import re
import os

import matplotlib.pyplot as plt

IMAGE_DIR = os.path.join(os.getcwd(), 'report_images')

# ... (rest of file) ...

if __name__ == '__main__':
    md_to_pdf(
        os.path.join(os.getcwd(), 'coil_circuits_report.md'),
        os.path.join(os.getcwd(), 'NeuroPulse_Circuit_Design_Report.pdf')
    )
