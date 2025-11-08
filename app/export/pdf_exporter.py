from weasyprint import HTML
from jinja2 import Template

HTML_TEMPLATE = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Pattern PDF</title></head>
<body>
<h1>{{ title }}</h1>
<h2>Legend</h2>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>Brand</th><th>Code</th><th>Name</th><th>Count</th></tr>
{% for row in legend %}
<tr>
  <td>{{ row.brand }}</td>
  <td>{{ row.code }}</td>
  <td>{{ row.name or "" }}</td>
  <td>{{ row.count }}</td>
</tr>
{% endfor %}
</table>
</body>
</html>
"""

def export_pdf(pattern: dict) -> bytes:
    # Build simple legend
    counts = {}
    for s in pattern["stitches"]:
        key = (s["thread"]["brand"], s["thread"]["code"])
        counts[key] = counts.get(key, 0) + 1
    palette_lookup = {(p["brand"], p["code"]): p for p in pattern["palette"]}
    legend = []
    for key, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        p = palette_lookup.get(key, {"brand": key[0], "code": key[1], "name": None})
        legend.append({"brand": p["brand"], "code": p["code"], "name": p.get("name"), "count": count})

    html = Template(HTML_TEMPLATE).render(
        title=pattern.get("meta", {}).get("title", "Pattern"),
        legend=legend
    )
    return HTML(string=html).write_pdf()
