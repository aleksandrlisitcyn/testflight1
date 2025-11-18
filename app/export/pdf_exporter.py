import io
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_CANDIDATES = [
    Path("assets/fonts/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/local/share/fonts/DejaVuSans.ttf"),
]
FONT_NAME = "DejaVuSans"


def _ensure_font() -> str:
    try:
        pdfmetrics.getFont(FONT_NAME)
        return FONT_NAME
    except KeyError:
        pass

    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            try:
                pdfmetrics.registerFont(TTFont(FONT_NAME, str(candidate)))
                return FONT_NAME
            except Exception as exc:  # fallback to built-in fonts if file is broken
                print(f"[WARN] Failed to register font {candidate}: {exc}")
                continue
    return "Helvetica"


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
    font_name = _ensure_font()
    title_font_name = f"{font_name}-Bold"
    if title_font_name not in pdfmetrics.getRegisteredFontNames():
        title_font_name = font_name
    c.setFont(title_font_name, 20)
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
    legend_entries = pattern.get("meta", {}).get("legend") or palette
    c.setFont(font_name, 11)

    legend_x = page_w * 0.7
    legend_y = page_h - 30 * mm
    c.drawString(legend_x, legend_y + 10, "Legend:")

    for entry in legend_entries:
        legend_y -= 12
        if legend_y < 15 * mm:
            c.showPage()
            legend_y = page_h - 20 * mm
            c.setFont(font_name, 11)
        symbol = entry.get("symbol") or entry.get("symbol", "?")
        brand = entry.get("brand", "")
        code = entry.get("code", "")
        name = entry.get("name", "") or ""
        count = entry.get("count")
        line = f"{symbol:<3} {brand} {code} {name}"
        if count:
            line += f" ({count})"
        c.drawString(legend_x, legend_y, line)

    # -----------------------------------------------------------------
    # 4) Grid information / meta
    # -----------------------------------------------------------------
    grid = pattern.get("canvasGrid", {})
    w = grid.get("width", 0)
    h = grid.get("height", 0)

    total_stitches = pattern.get("meta", {}).get("total_stitches", 0)
    palette_size = pattern.get("meta", {}).get("palette_size", len(palette))

    meta = pattern.get("meta", {})
    detail_level = meta.get("detail_level")
    c.setFont(font_name, 10)
    c.drawString(
        20 * mm,
        20 * mm,
        f"Grid: {w} × {h} cells · Colors: {palette_size} · Stitches: {total_stitches}"
        + (f" · Detail: {detail_level}" if detail_level else ""),
    )

    c.showPage()
    c.save()

    return buffer.getvalue()
