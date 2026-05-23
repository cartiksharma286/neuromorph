from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Read the manuscript text
with open('Quantum_Elliptic_Dementia_Nature_Manuscript.md', 'r') as f:
    lines = f.readlines()

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import re

# Read the manuscript
with open('Quantum_Elliptic_Dementia_Nature_Manuscript.md', 'r') as f:
    md_text = f.read()

# Split into lines for block equation detection
lines = md_text.split('\n')
