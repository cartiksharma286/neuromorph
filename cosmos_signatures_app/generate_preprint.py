from __future__ import annotations

import html
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "preprint.md"
FIGURE = ROOT / "static" / "black_hole_schematic.png"
PDF = ROOT / "Nature_Preprint_Cosmological_Modular_Signatures.pdf"


def create_black_hole_figure() -> None:
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "#070c0c")
    pixels = image.load()
    for y in range(height):
        amount = y / height
        color = (7 + int(4 * amount), 12 + int(7 * amount), 13 + int(8 * amount))
        for x in range(width):
            pixels[x, y] = color

    randomizer = random.Random(240925)
    stars = ImageDraw.Draw(image)
    for _ in range(460):
        x = randomizer.randrange(width)
        y = randomizer.randrange(height)
        brightness = randomizer.randrange(55, 155)
        stars.point((x, y), fill=(brightness, min(210, brightness + 18), min(205, brightness + 8)))

    cx, cy = width * 0.52, height * 0.53
    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    for offset in range(90, 0, -1):
        alpha = max(1, int(1.2 * (90 - offset)))
        glow_draw.ellipse((cx - 430 - offset, cy - 118 - offset * 0.28,
                           cx + 430 + offset, cy + 118 + offset * 0.28),
                          outline=(234, 104 + offset, 62, alpha), width=2)
    image = Image.alpha_composite(image.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(24)))

    disk = ImageDraw.Draw(image)
    colorset = [(255, 185, 112, 135), (240, 117, 73, 175), (143, 218, 204, 125), (255, 224, 174, 115)]
    for index in range(55):
        rx = 240 + index * 4.5
        ry = 66 + index * 1.15
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        start = (index * 31) % 360
        extent = 118 + (index * 17) % 140
        disk.arc(bbox, start, start + extent, fill=colorset[index % len(colorset)], width=2 if index % 4 else 4)
        disk.arc(bbox, start + 190, start + 190 + extent // 2, fill=(235, 111, 73, 90), width=1)

    photon_ring = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(photon_ring)
    ring_draw.ellipse((cx - 132, cy - 89, cx + 132, cy + 89), outline=(255, 201, 132, 235), width=8)
    image = Image.alpha_composite(image, photon_ring.filter(ImageFilter.GaussianBlur(5)))
    disk = ImageDraw.Draw(image)
    disk.ellipse((cx - 116, cy - 75, cx + 116, cy + 75), fill=(1, 3, 3, 255))
    disk.ellipse((cx - 132, cy - 89, cx + 132, cy + 89), outline=(131, 213, 203, 185), width=2)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(FIGURE, optimize=True)


def inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    return re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)


def build_story() -> list[object]:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("PreprintTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=25, alignment=TA_LEFT, textColor=colors.HexColor("#173e37"), spaceAfter=13)
    meta_style = ParagraphStyle("PreprintMeta", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=13, alignment=TA_LEFT, textColor=colors.HexColor("#607169"), spaceAfter=12)
    abstract_style = ParagraphStyle("PreprintAbstract", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY, leftIndent=0.22 * inch, rightIndent=0.22 * inch, textColor=colors.HexColor("#2c3831"), spaceAfter=13)
    heading_style = ParagraphStyle("PreprintHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#a7522e"), spaceBefore=13, spaceAfter=6, keepWithNext=True)
    body_style = ParagraphStyle("PreprintBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3, leading=13.2, alignment=TA_JUSTIFY, textColor=colors.HexColor("#202b25"), spaceAfter=7)
    bullet_style = ParagraphStyle("PreprintBullet", parent=body_style, leftIndent=15, firstLineIndent=-9)
    code_style = ParagraphStyle("PreprintCode", parent=styles["Code"], fontName="Courier", fontSize=8.4, leading=11.5, leftIndent=12, rightIndent=8, borderColor=colors.HexColor("#bcc9c0"), borderWidth=0.5, borderPadding=7, backColor=colors.HexColor("#f1f5f0"), textColor=colors.HexColor("#183a32"), spaceBefore=4, spaceAfter=10)

    story: list[object] = []
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    in_code = False
    abstract = False
    title_seen = False

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(paragraph_lines)
        style = abstract_style if abstract else meta_style if not title_seen else body_style
        story.append(Paragraph(inline_markup(text), style))
        paragraph_lines.clear()

    for line in MANUSCRIPT.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), code_style, maxLineLength=98))
                code_lines.clear()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            flush_paragraph()
            continue
        image_match = re.match(r"!\[[^]]*\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            figure = PDFImage(str(ROOT / image_match.group(1)))
            scale = min(6.8 * inch / figure.imageWidth, 4.2 * inch / figure.imageHeight)
            figure.drawWidth = figure.imageWidth * scale
            figure.drawHeight = figure.imageHeight * scale
            story.extend([figure, Spacer(1, 5)])
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), title_style))
            title_seen = True
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            heading = stripped[3:]
            abstract = heading.lower() == "abstract"
            story.append(Paragraph(inline_markup(heading), heading_style))
            continue
        if stripped.startswith("---"):
            flush_paragraph()
            story.append(Spacer(1, 4))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            story.append(Paragraph("&#8226; " + inline_markup(stripped[2:]), bullet_style))
            continue
        paragraph_lines.append(stripped)
    flush_paragraph()

    def page_chrome(canvas, document) -> None:
        canvas.saveState()
        page_width, _ = letter
        canvas.setStrokeColor(colors.HexColor("#d1d9d2"))
        canvas.setLineWidth(0.5)
        canvas.line(0.72 * inch, 0.58 * inch, page_width - 0.72 * inch, 0.58 * inch)
        canvas.setFillColor(colors.HexColor("#748178"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(0.72 * inch, 0.4 * inch, "HORIZON ATLAS · UNREVIEWED CONCEPT NOTE")
        canvas.drawRightString(page_width - 0.72 * inch, 0.4 * inch, str(document.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF), pagesize=letter, leftMargin=0.72 * inch, rightMargin=0.72 * inch, topMargin=0.68 * inch, bottomMargin=0.76 * inch, title="Cosmological Signature Forks and Modular Harmonics", author="Horizon Atlas Working Note")
    doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
    return story


def main() -> None:
    create_black_hole_figure()
    build_story()
    print(f"Generated figure: {FIGURE}")
    print(f"Generated preprint: {PDF}")


if __name__ == "__main__":
    main()