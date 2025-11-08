from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image
from weasyprint import HTML

from ..core.legend import build_legend
from ..core.pipeline import render_preview

STITCHES_PER_CM = 5.5  # Approximate 14ct Aida cloth


@lru_cache(maxsize=1)
def _get_template():
    env = Environment(
        loader=FileSystemLoader(Path(__file__).resolve().parent / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template("pattern_pdf.html")


def _rgb_to_hex(rgb) -> str:
    if not rgb:
        return "#FFFFFF"
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _encode_preview(pattern: dict, mode: str) -> str:
    png_bytes = render_preview(pattern, mode=mode)
    with Image.open(io.BytesIO(png_bytes)) as img:
        img.thumbnail((1200, 1200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_pdf_context(pattern: dict) -> Dict[str, object]:
    if hasattr(pattern, "model_dump"):
        pattern = pattern.model_dump()
    elif hasattr(pattern, "dict"):
        pattern = pattern.dict()

    grid = pattern.get("canvasGrid", {"width": 0, "height": 0})
    legend = build_legend(pattern)

    for row in legend:
        row["hex"] = _rgb_to_hex(row.get("rgb"))

    total_stitches = pattern.get("meta", {}).get("total_stitches") or sum(
        row.get("count", 0) for row in legend
    )
    palette_size = pattern.get("meta", {}).get("palette_size") or len(legend)

    width_cm = round(grid.get("width", 0) / STITCHES_PER_CM, 1)
    height_cm = round(grid.get("height", 0) / STITCHES_PER_CM, 1)

    return {
        "title": pattern.get("meta", {}).get("title", "Generated Pattern"),
        "brand": pattern.get("meta", {}).get("brand", "Unknown"),
        "grid": grid,
        "legend": legend,
        "total_stitches": total_stitches,
        "palette_size": palette_size,
        "stitch_per_cm": STITCHES_PER_CM,
        "estimated_size": {"cm_width": width_cm, "cm_height": height_cm},
        "previews": {
            "color": _encode_preview(pattern, mode="color"),
            "symbols": _encode_preview(pattern, mode="symbols"),
        },
    }


def export_pdf(pattern: dict) -> bytes:
    context = _build_pdf_context(pattern)
    html = _get_template().render(**context)
    return HTML(string=html).write_pdf()


__all__ = ["export_pdf", "_build_pdf_context"]
