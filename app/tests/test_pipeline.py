from app.core.pipeline import process_image_to_pattern
import numpy as np

def test_process_minimal():
    img = np.zeros((100,100,3), dtype=np.uint8)
    pat = process_image_to_pattern(img)
    assert pat.canvasGrid.width > 0 and pat.canvasGrid.height > 0
