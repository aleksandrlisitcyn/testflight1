from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image
from weasyprint import HTML

from ..core.pipeline import render_preview
from .context import build_export_context, rgb_to_hex

STITCHES_PER_CM = 5.5  # ≈14ct Aida


# =====================================================================
#  TEMPLATE LOADER
# =====================================================================

@lru_cache(maxsize=1)
def _get_template():
    env = Environment(
        loader=FileSystemLoader(Path(__file__).resolve().parent / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template("pattern_pdf.html")


# =====================================================================
#  PREVIEW HELPERS
# =====================================================================

def _encode_png_bytes(png_bytes: bytes) -> str:
    """Safely shrink PNG and convert to base64."""
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            img.thumbnail((1400, 1400))
            buf = io.BytesIO()
            img.save(buf, "PNG")
        data = buf.getvalue()
    except Exception:
        data = png_bytes
    return base64.b64encode(data).decode("ascii")


def _encode_preview_from_pattern(pattern: dict, mode: str) -> str:
    """Render preview from pattern and encode into base64."""
    png_bytes = render_preview(pattern, mode=mode)
    return _encode_png_bytes(png_bytes)


# =====================================================================
#  CONTEXT BUILDER
# =====================================================================

def _build_pdf_context(pattern: dict, preview_override: Optional[bytes]) -> Dict[str, object]:
    """
    Build context passed into Jinja2 template.
    preview_override: PNG bytes from jobs pipeline (optional).
    """
    base = build_export_context(pattern)

    # Legend with HEX
    legend_with_hex = []
    for item in base["legend"]:
        row = dict(item)
        row["hex"] = rgb_to_hex(row.get("rgb"))
        legend_with_hex.append(row)

    # Estimated size in cm
    width_cm = round(base["grid"]["width"] / STITCHES_PER_CM, 1)
    height_cm = round(base["grid"]["height"] / STITCHES_PER_CM, 1)

    # Previews
    if preview_override:
        preview_color = _encode_png_bytes(preview_override)
        preview_symbols = _encode_preview_from_pattern(base["pattern"], mode="symbols")
    else:
        preview_color = _encode_preview_from_pattern(base["pattern"], mode="color")
        preview_symbols = _encode_preview_from_pattern(base["pattern"], mode="symbols")

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
            "color": preview_color,
            "symbols": preview_symbols,
        },
    }


# =====================================================================
#  PUBLIC API
# =====================================================================

def export_pdf(pattern: Dict[str, Any], preview: Optional[bytes] = None) -> bytes:
    """
    Create PDF using Jinja2 → HTML → WeasyPrint.
    pattern: dict or Pydantic model.
    preview: optional PNG preview provided externally (e.g. jobs pipeline).
    """
    context = _build_pdf_context(pattern, preview_override=preview)
    html = _get_template().render(**context)
    return HTML(string=html).write_pdf()


__all__ = ["export_pdf", "_build_pdf_context"]