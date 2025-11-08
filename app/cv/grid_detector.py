from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def detect_pattern_roi(img: np.ndarray) -> np.ndarray:
    """Extract and rectify the region that contains the stitch grid."""
    if cv2 is None:
        return img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img.copy()

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) >= 4:
        pts = approx.reshape(-1, 2).astype(np.float32)
        ordered = _order_points(pts[:4])
        width_top = np.linalg.norm(ordered[0] - ordered[1])
        width_bottom = np.linalg.norm(ordered[2] - ordered[3])
        width = max(int((width_top + width_bottom) / 2), 1)
        height_left = np.linalg.norm(ordered[0] - ordered[3])
        height_right = np.linalg.norm(ordered[1] - ordered[2])
        height = max(int((height_left + height_right) / 2), 1)
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(img, matrix, (width, height))
        return warped

    x, y, w, h = cv2.boundingRect(contour)
    pad = int(0.02 * max(w, h))
    return img[max(y - pad, 0) : min(y + h + pad, img.shape[0]), max(x - pad, 0) : min(x + w + pad, img.shape[1])].copy()


def _estimate_cells(length: int, projection: np.ndarray) -> int:
    if length <= 0:
        return 1
    threshold = projection.max() * 0.4 if projection.max() > 0 else projection.mean()
    peaks = np.where(projection >= threshold)[0]
    if len(peaks) < 2:
        return max(1, min(200, length // 20 or 1))
    diffs = np.diff(peaks)
    diffs = diffs[diffs > 1]
    if diffs.size == 0:
        return max(1, min(200, length // 20 or 1))
    spacing = int(np.median(diffs))
    if spacing <= 0:
        spacing = max(1, length // 20)
    return max(1, min(400, int(round(length / spacing))))


def detect_and_rectify_grid(img: np.ndarray) -> dict:
    """Estimate grid dimensions from the region of interest."""
    h, w = img.shape[:2]
    if cv2 is None:
        cells_x = max(10, min(120, max(1, w) // 20 or 1))
        cells_y = max(10, min(120, max(1, h) // 20 or 1))
        return {"width": int(cells_x), "height": int(cells_y)}

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    projection_x = np.mean(np.abs(edges_x), axis=0)
    projection_y = np.mean(np.abs(edges_y), axis=1)

    cells_x = _estimate_cells(w, projection_x)
    cells_y = _estimate_cells(h, projection_y)

    return {"width": int(max(1, cells_x)), "height": int(max(1, cells_y))}
