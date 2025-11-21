import base64
import json

from ..core.pipeline import render_pattern_image


def export_json(pattern):
    pat = pattern.dict() if hasattr(pattern, "dict") else pattern
    preview = render_pattern_image(pat, with_symbols=True)
    pat["preview_png"] = base64.b64encode(preview).decode()
    return json.dumps(pat, ensure_ascii=False, indent=2)
