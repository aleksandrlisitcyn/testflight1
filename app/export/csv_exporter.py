import io, csv

def export_csv(pattern: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["x","y","brand","code","r","g","b"])
    for s in pattern["stitches"]:
        writer.writerow([s["x"], s["y"], s["thread"]["brand"], s["thread"]["code"], "", "", ""])
    return buf.getvalue()
