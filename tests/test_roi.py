from __future__ import annotations

import pytest

from app.cv.grid_detector import detect_pattern_roi
from tests.utils import make_grid_image


def test_detect_pattern_roi_rectifies_perspective():
    pytest.importorskip("cv2")
    img = make_grid_image(18, 12, cell=18, perspective=True)
    result = detect_pattern_roi(img)
    assert result.roi.size > 0
    assert result.confidence >= 0.3
    # ROI should be tighter than original canvas when perspective is applied
    assert result.roi.shape[0] <= img.shape[0]
    assert result.roi.shape[1] <= img.shape[1]
