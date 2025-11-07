import cv2
import numpy as np

def detect_pattern_roi(img: np.ndarray) -> np.ndarray:
    # Placeholder: return the image as-is. Real impl should crop decorative borders.
    return img

def detect_and_rectify_grid(img: np.ndarray) -> dict:
    # Placeholder grid: estimate roughly 100x100
    h, w = img.shape[:2]
    cells_x = max(10, min(120, w // 20))
    cells_y = max(10, min(120, h // 20))
    return {'width': int(cells_x), 'height': int(cells_y)}
