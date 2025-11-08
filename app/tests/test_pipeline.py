from app.core.pipeline import process_image_to_pattern
from app.export.pdf_exporter import _build_pdf_context
import numpy as np

def test_process_minimal():
    img = np.zeros((100,100,3), dtype=np.uint8)
    pat = process_image_to_pattern(img)
    assert pat.canvasGrid.width > 0 and pat.canvasGrid.height > 0


def test_pattern_meta_contains_legend():
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    pattern = process_image_to_pattern(img)
    legend = pattern.meta.get("legend")
    assert isinstance(legend, list)
    assert legend, "legend should contain at least one colour"
    first = legend[0]
    assert {"brand", "code", "count", "percent"}.issubset(first.keys())
    assert first["percent"] >= 0


def test_pdf_context_has_previews():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    pattern = process_image_to_pattern(img)
    context = _build_pdf_context(pattern)
    assert context["legend"], "legend should not be empty"
    previews = context["previews"]
    assert previews["color"].startswith("iVBOR"[:4]) or len(previews["color"]) > 10
    assert previews["symbols"].startswith("iVBOR"[:4]) or len(previews["symbols"]) > 10
