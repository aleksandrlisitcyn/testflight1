from __future__ import annotations

import csv
import io
import json
import zipfile

from .context import build_export_context, rgb_to_hex


def export_dize(pattern: dict) -> bytes:
    context = build_export_context(pattern)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        meta_payload = {
            "meta": context["meta"],
            "canvas": context["grid"],
            "legend": context["legend"],
        }
        zf.writestr("pattern.json", json.dumps(meta_payload, ensure_ascii=False, indent=2))

        legend_csv = io.StringIO()
        legend_writer = csv.writer(legend_csv)
        legend_writer.writerow(["index", "brand", "code", "name", "symbol", "hex", "count", "percent"])
        for idx, entry in enumerate(context["legend"], start=1):
            legend_writer.writerow(
                [
                    idx,
                    entry.get("brand", ""),
                    entry.get("code", ""),
                    entry.get("name", ""),
                    entry.get("symbol", ""),
                    rgb_to_hex(entry.get("rgb")),
                    entry.get("count", 0),
                    entry.get("percent", 0),
                ]
            )
        zf.writestr("legend.csv", legend_csv.getvalue())

        stitches_csv = io.StringIO()
        stitches_writer = csv.writer(stitches_csv)
        stitches_writer.writerow(["x", "y", "brand", "code"])
        for stitch in context["stitches"]:
            thread = stitch.get("thread") or {}
            stitches_writer.writerow(
                [
                    stitch.get("x", 0),
                    stitch.get("y", 0),
                    thread.get("brand", ""),
                    thread.get("code", ""),
                ]
            )
        zf.writestr("stitches.csv", stitches_csv.getvalue())

    return buf.getvalue()


__all__ = ["export_dize"]
