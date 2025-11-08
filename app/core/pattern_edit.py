from __future__ import annotations

from typing import Iterable

from .legend import build_legend


def apply_meta_updates(pattern: dict, updates: dict) -> dict:
    meta = pattern.setdefault("meta", {})
    meta.update(updates)
    return pattern


def apply_legend_updates(pattern: dict, entries: Iterable[dict]) -> dict:
    updates_map = {(entry["brand"], entry["code"]): entry for entry in entries}

    palette = pattern.get("palette", [])
    palette_lookup = {(p["brand"], p["code"]): p for p in palette}

    for key, update in updates_map.items():
        target = palette_lookup.get(key)
        if not target:
            continue
        if update.get("name") is not None:
            target["name"] = update["name"]
        if update.get("symbol") is not None:
            target["symbol"] = update["symbol"]

    for stitch in pattern.get("stitches", []):
        thread = stitch.get("thread", {})
        key = (thread.get("brand"), thread.get("code"))
        update_entry = updates_map.get(key)
        if not update_entry:
            continue
        if update_entry.get("name") is not None:
            thread["name"] = update_entry["name"]
        if update_entry.get("symbol") is not None:
            thread["symbol"] = update_entry["symbol"]

    legend = build_legend(pattern, force=True)
    meta = pattern.setdefault("meta", {})
    meta["legend"] = legend
    meta["palette_size"] = len({(p["brand"], p["code"]) for p in pattern.get("palette", [])})
    meta["total_stitches"] = sum(row["count"] for row in legend)
    return pattern
