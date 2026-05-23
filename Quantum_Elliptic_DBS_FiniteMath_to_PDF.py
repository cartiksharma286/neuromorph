from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import re

# Read the manuscript
with open('Quantum_Elliptic_DBS_FiniteMath_Nature_Preprint.md', 'r') as f:
    md_text = f.read()

# Split into lines for block equation detection
lines = md_text.split('\n')

pdf_file = 'Quantum_Elliptic_DBS_FiniteMath_Nature_Preprint.pdf'
doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=72)
styles = getSampleStyleSheet()
mono_style = ParagraphStyle('Mono', parent=styles['Normal'], fontName='Courier', fontSize=10, leading=14)
story = []

in_block_eq = False
block_eq_lines = []
buffer = []

def flush_buffer():
    if buffer:
        para = '\n'.join(buffer)
        story.append(Paragraph(para, styles["Normal"]))
        story.append(Spacer(1, 0.1*inch))
        buffer.clear()

for line in lines:
    # Detect block equations $$ ... $$
    if line.strip().startswith('$$') and line.strip().endswith('$$') and len(line.strip()) > 4:
        # Single-line block equation
        flush_buffer()
        eq = line.strip()[2:-2].strip()
        story.append(Preformatted(eq, mono_style))
        story.append(Spacer(1, 0.1*inch))
        continue
    if line.strip().startswith('$$'):
        flush_buffer()
        in_block_eq = True
        block_eq_lines = []
        continue
    if line.strip().endswith('$$') and in_block_eq:
        in_block_eq = False
        eq = '\n'.join(block_eq_lines).strip()
        story.append(Preformatted(eq, mono_style))
        story.append(Spacer(1, 0.1*inch))
        block_eq_lines = []
        continue
    if in_block_eq:
        block_eq_lines.append(line)
        continue
    # Inline equations: replace $...$ with [ ... ] for visual separation (no HTML tags)
    def inline_eq(m):
        return '[' + m.group(1) + ']'
    line = re.sub(r'\$(.+?)\$', inline_eq, line)
    buffer.append(line)

flush_buffer()

doc.build(story)
print(f"PDF generated: {pdf_file}")