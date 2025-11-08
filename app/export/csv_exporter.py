import io, csv

def export_csv(pattern: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["x", "y", "brand", "code", "r", "g", "b"])
    palette_lookup = {
        (p["brand"], p["code"]): p.get("rgb")
        for p in pattern.get("palette", [])
    }
    for s in pattern["stitches"]:
        key = (s["thread"]["brand"], s["thread"]["code"])
        rgb = palette_lookup.get(key) or ("", "", "")
        writer.writerow([
            s["x"],
            s["y"],
            key[0],
            key[1],
            *(int(v) if isinstance(v, (int, float)) else "" for v in rgb),
        ])
    return buf.getvalue()
