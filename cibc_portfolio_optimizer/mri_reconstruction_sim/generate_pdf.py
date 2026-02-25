#!/usr/bin/env python3
"""
Generate PDF report with embedded simulation images.
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import re
import os

def md_to_pdf(md_path, pdf_path):
    """
    Converts a Markdown file to a PDF report.
    Handles headers, paragraphs, and images.
    """
    try:
        # Check if file exists
        if not os.path.exists(md_path):
            print(f"Error: Markdown file not found at {md_path}")
            return False

        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        with open(md_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.1*inch))
                continue

            if line.startswith('# '):
                # H1
                p = Paragraph(line[2:], styles['Title'])
                story.append(p)
                story.append(Spacer(1, 0.2*inch))
            elif line.startswith('## '):
                # H2
                p = Paragraph(line[3:], styles['Heading2'])
                story.append(p)
                story.append(Spacer(1, 0.1*inch))
            elif line.startswith('### '):
                # H3
                p = Paragraph(line[4:], styles['Heading3'])
                story.append(p)
            elif line.startswith('![') and '](' in line:
                # Image syntax: ![Alt](path)
                try:
                    match = re.search(r'\((.*?)\)', line)
                    if match:
                        img_rel_path = match.group(1)
                        # Resolve path relative to MD file base dir
                        base_dir = os.path.dirname(md_path)
                        img_path = os.path.join(base_dir, img_rel_path)
                        
                        if os.path.exists(img_path):
                            # Add image to story
                            im = Image(img_path)
                            # Resize if too large (fits within 6 inches width)
                            target_width = 6 * inch
                            aspect = im.imageHeight / im.imageWidth
                            im.drawWidth = target_width
                            im.drawHeight = target_width * aspect
                            
                            story.append(im)
                            story.append(Spacer(1, 0.2*inch))
                        else:
                            story.append(Paragraph(f"[Image missing: {img_rel_path}]", styles['BodyText']))
                except Exception as e:
                    print(f"Error processing image line: {line}. {e}")
            elif line.startswith('- ') or line.startswith('* '):
                 story.append(Paragraph(f"• {line[2:]}", styles['BodyText']))
            else:
                story.append(Paragraph(line, styles['BodyText']))

        doc.build(story)
        print(f"PDF generated successfully at {pdf_path}")
        return True

    except Exception as e:
        print(f"Failed to generate PDF: {e}")
        return False

if __name__ == '__main__':
    # Test execution
    test_md = os.path.join(os.getcwd(), 'test_report.md')
    if not os.path.exists(test_md):
        with open(test_md, 'w') as f:
            f.write("# Test Report\n\nThis is a test.\n\n## Section 1\nContent.")
        
    md_to_pdf(
        test_md,
        os.path.join(os.getcwd(), 'test_report.pdf')
    )
