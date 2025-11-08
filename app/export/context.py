from __future__ import annotations

import copy
from typing import Any, Dict

from ..core.legend import build_legend


def _as_dict(pattern: Any) -> Dict[str, Any]:
    if hasattr(pattern, "model_dump"):
        return copy.deepcopy(pattern.model_dump())
    if hasattr(pattern, "dict"):
        return copy.deepcopy(pattern.dict())
    return copy.deepcopy(pattern)


def _normalize_palette(palette: list[dict]) -> list[dict]:
    normalised: list[dict] = []
    for entry in palette:
        item = dict(entry)
        rgb = item.get("rgb")
        if isinstance(rgb, tuple):
            item["rgb"] = [int(v) for v in rgb]
        elif isinstance(rgb, list):
            item["rgb"] = [int(v) for v in rgb]
        normalised.append(item)
    return normalised


def rgb_to_hex(rgb) -> str:
    if not rgb:
        return "#FFFFFF"
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def build_export_context(pattern: Any) -> Dict[str, Any]:
    pattern_dict = _as_dict(pattern)

    grid = pattern_dict.get("canvasGrid") or {}
    grid_dict = {
        "width": int(grid.get("width", 0) or 0),
        "height": int(grid.get("height", 0) or 0),
    }

    legend = build_legend(pattern_dict)

    meta = dict(pattern_dict.get("meta") or {})
    title = meta.get("title") or "Generated Pattern"
    brand = meta.get("brand") or (legend[0]["brand"] if legend else "Unknown")

    total_stitches = meta.get("total_stitches")
    if not total_stitches:
        total_stitches = sum(row.get("count", 0) for row in legend)
    palette_size = meta.get("palette_size")
    if not palette_size:
        palette_size = len({(row.get("brand"), row.get("code")) for row in legend})

    palette = pattern_dict.get("palette") or []
    stitches = pattern_dict.get("stitches") or []

    return {
        "pattern": pattern_dict,
        "grid": grid_dict,
        "legend": legend,
        "meta": {
            **meta,
            "title": title,
            "brand": brand,
            "total_stitches": int(total_stitches),
            "palette_size": int(palette_size),
        },
        "palette": _normalize_palette(palette),
        "stitches": [dict(stitch) if isinstance(stitch, dict) else stitch for stitch in stitches],
    }


__all__ = ["build_export_context", "rgb_to_hex"]
