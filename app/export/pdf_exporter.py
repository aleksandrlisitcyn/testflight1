import io
from typing import Optional, Dict, Any

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from PIL import Image


def export_pdf(pattern: Dict[str, Any], preview: Optional[bytes] = None) -> bytes:
    """
    Simple PDF exporter.

    pattern: pattern dict (as stored in JSON / passed from jobs.py)
    preview: optional PNG bytes with rendered grid preview.
    """

    buffer = io.BytesIO()

    # horizontal A4
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(buffer, pagesize=(page_w, page_h))

    # -----------------------------------------------------------------
    # 1) Title
    # -----------------------------------------------------------------
    title = pattern.get("meta", {}).get("title") or "Cross-Stitch Pattern"
    c.setFont("Helvetica-Bold", 20)
    c.drawString(20 * mm, page_h - 20 * mm, title)

    # -----------------------------------------------------------------
    # 2) Preview image (if provided)
    # -----------------------------------------------------------------
    preview_block_w = page_w * 0.7
    preview_block_h = page_h - 50 * mm
    preview_top_y = page_h - 30 * mm
    preview_left_x = 20 * mm

    if preview:
        try:
            img = Image.open(io.BytesIO(preview))
            img_w, img_h = img.size

            scale = min(preview_block_w / img_w, preview_block_h / img_h, 1.0)
            draw_w = img_w * scale
            draw_h = img_h * scale

            c.drawImage(
                ImageReader(img),
                preview_left_x,
                preview_top_y - draw_h,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                anchor="sw",
            )
        except Exception as e:
            # if something goes wrong, just skip preview
            print("[WARN] PDF preview insert failed:", e)

    # -----------------------------------------------------------------
    # 3) Legend block
    # -----------------------------------------------------------------
    palette = pattern.get("palette", [])
    c.setFont("Helvetica", 11)

    legend_x = page_w * 0.75
    legend_y = page_h - 30 * mm

    c.drawString(legend_x, legend_y + 10, "Legend:")

    # each entry: symbol – BRAND CODE Name
    for p in palette:
        legend_y -= 12
        if legend_y < 15 * mm:
            # new page if we run out of space
            c.showPage()
            legend_y = page_h - 20 * mm
            c.setFont("Helvetica", 11)
        symbol = p.get("symbol", "?")
        brand = p.get("brand", "")
        code = p.get("code", "")
        name = p.get("name", "")
        c.drawString(
            legend_x,
            legend_y,
            f"{symbol}: {brand} {code} {name}",
        )

    # -----------------------------------------------------------------
    # 4) Grid information / meta
    # -----------------------------------------------------------------
    grid = pattern.get("canvasGrid", {})
    w = grid.get("width", 0)
    h = grid.get("height", 0)

    total_stitches = pattern.get("meta", {}).get("total_stitches", 0)
    palette_size = pattern.get("meta", {}).get("palette_size", len(palette))

    c.setFont("Helvetica", 10)
    c.drawString(
        20 * mm,
        20 * mm,
        f"Grid size: {w} × {h} cells; Colors: {palette_size}; Stitches: {total_stitches}",
    )

    c.showPage()
    c.save()

    return buffer.getvalue()