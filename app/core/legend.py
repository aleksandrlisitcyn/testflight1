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

    def _thread_info(value):
        data = _as_dict(value)
        if not isinstance(data, dict):
            return None
        if "thread" in data and isinstance(data["thread"], dict):
            return _thread_info(data["thread"])
        brand = data.get("brand")
        code = data.get("code")
        if not (brand and code):
            return None
        info = {"brand": brand, "code": code}
        for key in ("name", "symbol", "rgb"):
            if data.get(key) is not None:
                info[key] = data[key]
        return info

    palette_lookup: dict[tuple[str, str], dict] = {}
    for entry in pattern_dict.get("palette", []):
        info = _thread_info(entry)
        if not info:
            continue
        palette_lookup[(info["brand"], info["code"])] = info

    counts: Counter[tuple[str, str]] = Counter()
    for stitch in pattern_dict.get("stitches", []):
        info = _thread_info(stitch)
        if not info:
            continue
        key = (info["brand"], info["code"])
        counts[key] += 1
        if key not in palette_lookup:
            palette_lookup[key] = info

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
