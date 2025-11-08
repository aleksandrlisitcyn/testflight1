from __future__ import annotations

from collections import Counter
from typing import List


def _as_dict(pattern):
    if hasattr(pattern, "model_dump"):
        return pattern.model_dump()
    if hasattr(pattern, "dict"):
        return pattern.dict()
    return pattern


def build_legend(pattern, *, force: bool = False) -> List[dict]:
    """Return a legend enriched with counts, colour, and symbol info."""

    pattern_dict = _as_dict(pattern)

    meta = pattern_dict.get("meta") or {}
    if not force and meta.get("legend"):
        # Return a copy so callers can safely mutate the result.
        return [dict(entry) for entry in meta["legend"]]

    stitches = pattern_dict.get("stitches", [])
    palette_lookup: dict[tuple[str, str], dict] = {}
    for p in pattern_dict.get("palette", []):
        brand = p.get("brand")
        code = p.get("code")
        if brand and code:
            palette_lookup[(brand, code)] = p
    for stitch in pattern_dict.get("stitches", []):
        thread = stitch.get("thread") if isinstance(stitch, dict) else None
        if not thread:
            continue
        brand = thread.get("brand")
        code = thread.get("code")
        if brand and code and (brand, code) not in palette_lookup:
            palette_lookup[(brand, code)] = thread

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
