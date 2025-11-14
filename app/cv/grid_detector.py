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
    """
    Detect real stitch grid using probabilistic Hough lines.

    Returns dict with:
      - width: number of cells in X
      - height: number of cells in Y
      - cell_w, cell_h: estimated cell size in pixels
      - roi: cropped rect (numpy array) that contains the grid
    """
    h, w = img.shape[:2]
    if cv2 is None:
        return {
            "width": max(1, w // 10),
            "height": max(1, h // 10),
            "cell_w": 10,
            "cell_h": 10,
            "roi": img.copy(),
        }

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(40, min(200, (w + h) // 50)),
        minLineLength=max(20, min(w, h) // 20),
        maxLineGap=max(5, min(w, h) // 80),
    )

    if lines is None or len(lines) < 8:
        # fallback: coarse estimate
        return {
            "width": max(1, w // 12),
            "height": max(1, h // 12),
            "cell_w": max(1, w // 12),
            "cell_h": max(1, h // 12),
            "roi": img.copy(),
        }

    vertical = []
    horizontal = []

    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = l
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < 3 and dy > 10:
            vertical.append(int(round((x1 + x2) / 2)))
        elif dy < 3 and dx > 10:
            horizontal.append(int(round((y1 + y2) / 2)))

    vertical = sorted(set(vertical))
    horizontal = sorted(set(horizontal))

    # If insufficient lines found, try a more permissive pass
    if len(vertical) < 2 or len(horizontal) < 2:
        # lower thresholds and try again on a resized image
        small = cv2.resize(blur, (max(200, w // 2), max(200, h // 2)))
        edges2 = cv2.Canny(small, 30, 120)
        lines2 = cv2.HoughLinesP(edges2, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10)
        if lines2 is not None:
            for l in lines2.reshape(-1, 4):
                x1, y1, x2, y2 = l
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                # scale back coordinates
                sx = int(round(x1 * (w / small.shape[1])))
                sy = int(round(y1 * (h / small.shape[0])))
                sx2 = int(round(x2 * (w / small.shape[1])))
                sy2 = int(round(y2 * (h / small.shape[0])))
                if abs(sx2 - sx) < 6 and abs(sy2 - sy) > 6:
                    vertical.append(int(round((sx + sx2) / 2)))
                elif abs(sy2 - sy) < 6 and abs(sx2 - sx) > 6:
                    horizontal.append(int(round((sy + sy2) / 2)))
        vertical = sorted(set(vertical))
        horizontal = sorted(set(horizontal))

    if len(vertical) < 2 or len(horizontal) < 2:
        # final fallback
        return {
            "width": max(1, w // 12),
            "height": max(1, h // 12),
            "cell_w": max(1, w // 12),
            "cell_h": max(1, h // 12),
            "roi": img.copy(),
        }

    # compute median spacing -> cell size
    v_diffs = np.diff(vertical)
    h_diffs = np.diff(horizontal)
    # filter out outliers
    if v_diffs.size == 0:
        cell_w = max(1, w // 12)
    else:
        cell_w = int(max(1, np.median(v_diffs)))

    if h_diffs.size == 0:
        cell_h = max(1, h // 12)
    else:
        cell_h = int(max(1, np.median(h_diffs)))

    width_cells = max(1, len(vertical) - 1)
    height_cells = max(1, len(horizontal) - 1)

    # ROI: crop tightly to grid lines with small padding
    x0, x1 = vertical[0], vertical[-1]
    y0, y1 = horizontal[0], horizontal[-1]
    pad_x = max(2, int(cell_w * 0.1))
    pad_y = max(2, int(cell_h * 0.1))
    x0c = max(0, x0 - pad_x)
    x1c = min(w, x1 + pad_x)
    y0c = max(0, y0 - pad_y)
    y1c = min(h, y1 + pad_y)

    roi = img[y0c:y1c, x0c:x1c].copy()

    return {
        "width": int(width_cells),
        "height": int(height_cells),
        "cell_w": int(cell_w),
        "cell_h": int(cell_h),
        "roi": roi,
    }
