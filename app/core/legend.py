from __future__ import annotations

from collections import Counter
from typing import List


def build_legend(pattern) -> List[dict]:
    """Return a legend enriched with counts, colour, and symbol info."""

    if hasattr(pattern, "model_dump"):
        pattern = pattern.model_dump()
    elif hasattr(pattern, "dict"):
        pattern = pattern.dict()

    if pattern.get("meta", {}).get("legend"):
        return list(pattern["meta"]["legend"])  # already prepared

    stitches = pattern.get("stitches", [])
    palette_lookup = {
        (p["thread"]["brand"], p["thread"]["code"]): p["thread"]
        if "thread" in p
        else p
        for p in pattern.get("palette", [])
    }

    counts = Counter((s["thread"]["brand"], s["thread"]["code"]) for s in stitches)
    total = sum(counts.values()) or 1
    legend: List[dict] = []
    for key, count in counts.most_common():
        palette_item = palette_lookup.get(key, {"brand": key[0], "code": key[1]})
        legend.append(
            {
                "brand": palette_item.get("brand", key[0]),
                "code": palette_item.get("code", key[1]),
                "name": palette_item.get("name"),
                "symbol": palette_item.get("symbol"),
                "rgb": palette_item.get("rgb"),
                "count": count,
                "percent": round(count / total * 100, 2),
            }
        )
    return legend
