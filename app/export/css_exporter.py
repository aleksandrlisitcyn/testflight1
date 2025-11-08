from __future__ import annotations

from typing import Iterable

from .context import build_export_context, rgb_to_hex


def _format_legend_rows(legend: Iterable[dict]) -> list[str]:
    rows: list[str] = []
    for entry in legend:
        brand = entry.get("brand", "")
        code = entry.get("code", "")
        name = entry.get("name") or "—"
        symbol = entry.get("symbol") or "?"
        hex_colour = rgb_to_hex(entry.get("rgb"))
        count = entry.get("count", 0)
        percent = entry.get("percent", 0)
        rows.append(
            f"{symbol}\t{brand} {code}\t{name}\t{hex_colour}\t{count} stitches ({percent:.2f}%)"
        )
    return rows


def export_css(pattern: dict) -> str:
    context = build_export_context(pattern)
    header = [
        f"# {context['meta'].get('title', 'Generated Pattern')}",
        f"Brand: {context['meta'].get('brand', 'Unknown')}",
        f"Grid: {context['grid']['width']} x {context['grid']['height']}",
        "",
        "[Legend]",
    ]
    legend_rows = _format_legend_rows(context["legend"])
    stitches_section = ["", "[Stitches]"]
    for stitch in context["stitches"]:
        thread = stitch.get("thread") if isinstance(stitch, dict) else {}
        brand = thread.get("brand", "") if isinstance(thread, dict) else ""
        code = thread.get("code", "") if isinstance(thread, dict) else ""
        stitches_section.append(f"{stitch.get('x', 0)},{stitch.get('y', 0)},{brand},{code}")

    return "\n".join(header + legend_rows + stitches_section)


__all__ = ["export_css"]
