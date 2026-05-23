import markdown2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import os

md_file = "space_combustion_physics/Canada_Arm3_Nature_Preprint.md"
pdf_file = "Canada_Arm3_Nature_Preprint.pdf"

# Read markdown content
with open(md_file, "r") as f:
    md_content = f.read()

# Convert markdown to plain text (basic)
html_content = markdown2.markdown(md_content)
# Remove HTML tags for simple PDF export
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_content, "html.parser")
plain_text = soup.get_text()

# Write to PDF
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter
margin = 50
max_width = width - 2 * margin
max_height = height - 2 * margin

lines = simpleSplit(plain_text, 'Helvetica', 12, max_width)
y = height - margin
for line in lines:
    if y < margin:
        c.showPage()
        y = height - margin
    c.drawString(margin, y, line)
    y -= 14
c.save()

print(f"PDF generated: {pdf_file}")
