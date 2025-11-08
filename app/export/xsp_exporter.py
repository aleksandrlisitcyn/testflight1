from __future__ import annotations

import json

from .context import build_export_context, rgb_to_hex


def export_xsp(pattern: dict) -> str:
    context = build_export_context(pattern)
    payload = {
        "meta": {
            key: value
            for key, value in context["meta"].items()
            if value is not None
        },
        "canvas": context["grid"],
        "legend": [
            {
                "brand": entry.get("brand"),
                "code": entry.get("code"),
                "name": entry.get("name"),
                "symbol": entry.get("symbol"),
                "rgb": entry.get("rgb"),
                "hex": rgb_to_hex(entry.get("rgb")),
                "count": entry.get("count"),
                "percent": entry.get("percent"),
            }
            for entry in context["legend"]
        ],
        "stitches": [
            {
                "x": stitch.get("x", 0),
                "y": stitch.get("y", 0),
                "brand": (stitch.get("thread") or {}).get("brand"),
                "code": (stitch.get("thread") or {}).get("code"),
            }
            for stitch in context["stitches"]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = ["export_xsp"]
