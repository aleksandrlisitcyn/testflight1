from __future__ import annotations

from app.cv.grid_detector import detect_and_rectify_grid
from tests.utils import make_grid_image


def test_detect_and_rectify_grid_matches_expected_dimensions():
    img = make_grid_image(16, 12, cell=14)
    result = detect_and_rectify_grid(img)
    assert abs(result["width"] - 16) <= 4
    assert abs(result["height"] - 12) <= 4
    assert result["cell_w"] >= 6
    assert result["cell_h"] >= 6
    assert result["confidence"] >= 0.2
