from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image
from weasyprint import HTML

from ..core.pipeline import render_preview
from .context import build_export_context, rgb_to_hex

STITCHES_PER_CM = 5.5  # Approximate 14ct Aida cloth


@lru_cache(maxsize=1)
def _get_template():
    env = Environment(
        loader=FileSystemLoader(Path(__file__).resolve().parent / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template("pattern_pdf.html")


def _encode_preview(pattern: dict, mode: str) -> str:
    png_bytes = render_preview(pattern, mode=mode)
    with Image.open(io.BytesIO(png_bytes)) as img:
        img.thumbnail((1200, 1200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_pdf_context(pattern: dict) -> Dict[str, object]:
    base = build_export_context(pattern)
    legend_with_hex = []
    for entry in base["legend"]:
        enriched = dict(entry)
        enriched["hex"] = rgb_to_hex(enriched.get("rgb"))
        legend_with_hex.append(enriched)

    width_cm = round(base["grid"]["width"] / STITCHES_PER_CM, 1)
    height_cm = round(base["grid"]["height"] / STITCHES_PER_CM, 1)

    return {
        "title": base["meta"].get("title", "Generated Pattern"),
        "brand": base["meta"].get("brand", "Unknown"),
        "grid": base["grid"],
        "legend": legend_with_hex,
        "total_stitches": base["meta"].get("total_stitches", 0),
        "palette_size": base["meta"].get("palette_size", len(legend_with_hex)),
        "stitch_per_cm": STITCHES_PER_CM,
        "estimated_size": {"cm_width": width_cm, "cm_height": height_cm},
        "previews": {
            "color": _encode_preview(base["pattern"], mode="color"),
            "symbols": _encode_preview(base["pattern"], mode="symbols"),
        },
    }


def export_pdf(pattern: dict) -> bytes:
    context = _build_pdf_context(pattern)
    html = _get_template().render(**context)
    return HTML(string=html).write_pdf()


__all__ = ["export_pdf", "_build_pdf_context"]
