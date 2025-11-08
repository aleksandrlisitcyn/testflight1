from __future__ import annotations

import xml.etree.ElementTree as ET

from .context import build_export_context


def export_xsd(pattern: dict) -> str:
    context = build_export_context(pattern)

    root = ET.Element("CrossStitchDesign", version="1.0")
    meta = ET.SubElement(root, "Meta")
    for key in ("title", "brand", "total_stitches", "palette_size"):
        value = context["meta"].get(key)
        if value is not None:
            meta.set(key, str(value))

    grid = ET.SubElement(root, "Canvas")
    grid.set("width", str(context["grid"].get("width", 0)))
    grid.set("height", str(context["grid"].get("height", 0)))

    legend_el = ET.SubElement(root, "Legend")
    for index, entry in enumerate(context["legend"], start=1):
        color_el = ET.SubElement(legend_el, "Color")
        color_el.set("index", str(index))
        color_el.set("brand", str(entry.get("brand", "")))
        color_el.set("code", str(entry.get("code", "")))
        if entry.get("name"):
            color_el.set("name", entry.get("name"))
        color_el.set("symbol", entry.get("symbol", ""))
        color_el.set("stitches", str(entry.get("count", 0)))
        color_el.set("percent", f"{entry.get('percent', 0):.2f}")

        rgb_el = ET.SubElement(color_el, "RGB")
        rgb = entry.get("rgb") or (0, 0, 0)
        r, g, b = [int(v) for v in rgb]
        rgb_el.set("r", str(r))
        rgb_el.set("g", str(g))
        rgb_el.set("b", str(b))

    stitches_el = ET.SubElement(root, "Stitches")
    for stitch in context["stitches"]:
        thread = stitch.get("thread") or {}
        stitch_el = ET.SubElement(stitches_el, "Stitch")
        stitch_el.set("x", str(stitch.get("x", 0)))
        stitch_el.set("y", str(stitch.get("y", 0)))
        stitch_el.set("brand", str(thread.get("brand", "")))
        stitch_el.set("code", str(thread.get("code", "")))

    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


__all__ = ["export_xsd"]
