from ..core.pipeline import render_pattern_image


def export_png(pattern):
    pat = pattern.dict() if hasattr(pattern, "dict") else pattern
    return render_pattern_image(pat, with_symbols=True)
