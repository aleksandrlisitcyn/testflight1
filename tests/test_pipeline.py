from __future__ import annotations

from app.core.pipeline import process_image_to_pattern, render_preview
from tests.utils import make_grid_image


def test_process_image_to_pattern_produces_palette_and_meta():
    img = make_grid_image(10, 8, cell=18, noise=True)
    pattern = process_image_to_pattern(
        img,
        brand="DMC",
        min_cells=5,
        detail_level="medium",
        pattern_mode="mixed",
    )
    assert pattern.canvasGrid.width >= 1
    assert pattern.canvasGrid.height >= 1
    assert pattern.meta["palette_size"] >= 1
    assert pattern.meta.get("grid_confidence") is not None
    assert pattern.meta.get("legend")
    assert pattern.meta.get("pattern_mode") == "mixed"
    assert "sanity" in pattern.meta

    preview = render_preview(pattern, mode="symbols")
    assert isinstance(preview, (bytes, bytearray))
    assert len(preview) > 0


def test_multi_colour_input_preserves_palette():
    img = make_grid_image(6, 4, cell=14, noise=False)
    pattern = process_image_to_pattern(
        img,
        brand="DMC",
        min_cells=1,
        detail_level="medium",
        pattern_mode="color",
    )
    assert pattern.meta["palette_size"] >= 2
    sanity = pattern.meta.get("sanity", {})
    assert sanity.get("metrics", {}).get("palette_size") == len(pattern.palette)


def test_symbol_mode_surfaces_in_meta():
    img = make_grid_image(4, 4, cell=20, noise=True)
    pattern = process_image_to_pattern(
        img,
        brand="DMC",
        min_cells=1,
        detail_level="medium",
        pattern_mode="symbol",
    )
    assert pattern.meta["pattern_mode"] == "symbol"
    metrics = pattern.meta.get("sanity", {}).get("metrics", {})
    assert metrics.get("grid_width") == pattern.canvasGrid.width
