from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)

from ..core.types import PatternMode


@dataclass
class ROIResult:
    """Represents a detected region of interest for the stitch grid."""

    roi: np.ndarray
    confidence: float
    corners: Optional[np.ndarray] = None
    transform: Optional[np.ndarray] = None
    bounding_box: Optional[Tuple[int, int, int, int]] = None


def _order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _fallback_roi(img: np.ndarray) -> ROIResult:
    gray = img.mean(axis=2)
    mask = gray < np.percentile(gray, 98)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size and cols.size:
        y0, y1 = int(rows[0]), int(rows[-1] + 1)
        x0, x1 = int(cols[0]), int(cols[-1] + 1)
        pad_y = max(2, int(0.02 * img.shape[0]))
        pad_x = max(2, int(0.02 * img.shape[1]))
        y0 = max(0, y0 - pad_y)
        y1 = min(img.shape[0], y1 + pad_y)
        x0 = max(0, x0 - pad_x)
        x1 = min(img.shape[1], x1 + pad_x)
        roi = img[y0:y1, x0:x1].copy()
        return ROIResult(roi=roi, confidence=0.35, bounding_box=(x0, y0, x1, y1))
    return ROIResult(
        roi=img.copy(),
        confidence=0.2,
        bounding_box=(0, 0, img.shape[1], img.shape[0]),
    )


def detect_pattern_roi(img: np.ndarray, pattern_mode: PatternMode = "color") -> ROIResult:
    """
    Extract and rectify the region that contains the stitch grid.
    pattern_mode lets us tweak how aggressively we smooth / close details. For antique charts we
    keep it conservative by default and preserve ink strokes in symbol-heavy scans.
    """
    if img.size == 0:
        return ROIResult(roi=img.copy(), confidence=0.0)

    if cv2 is None:  # pragma: no cover - fallback when cv2 is unavailable
        return _fallback_roi(img)

    orig_h, orig_w = img.shape[:2]
    scale = 900.0 / float(max(orig_h, orig_w))
    scale = min(1.0, max(scale, 0.2))
    working = (
        cv2.resize(img, (int(orig_w * scale), int(orig_h * scale))) if scale < 1 else img.copy()
    )

    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur_kernel = 5
    close_iterations = 2
    thresh_block = 35
    thresh_c = 4
    if pattern_mode == "symbol":
        blur_kernel = 3
        close_iterations = 1
        thresh_block = 25
        thresh_c = 2
    elif pattern_mode == "mixed":
        close_iterations = 2
        blur_kernel = 5
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), 0)

    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(3, thresh_block),
        thresh_c,
    )
    edges = cv2.Canny(blurred, 40, 120)
    mask = cv2.bitwise_or(thresh, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("ROI detection: no contours found, falling back to heuristic crop")
        return _fallback_roi(img)

    best_roi = None
    best_conf = 0.0

    working_area = working.shape[0] * working.shape[1]
    scale_back = 1.0 / scale

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        contour_conf = float(area / float(working_area))
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        bbox = cv2.boundingRect(approx)
        pad = int(0.01 * max(bbox[2], bbox[3]))
        x, y, w_box, h_box = bbox
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(working.shape[1], x + w_box + pad)
        y1 = min(working.shape[0], y + h_box + pad)

        ordered = None
        matrix = None
        roi_section: np.ndarray
        confidence = contour_conf

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
            abs_pts = ordered * scale_back
            abs_matrix = cv2.getPerspectiveTransform(abs_pts, dst)
            warped = cv2.warpPerspective(img, abs_matrix, (width, height))
            matrix = abs_matrix
            roi_section = warped
            confidence *= 1.2
        else:
            abs_box = (
                int(round(x0 * scale_back)),
                int(round(y0 * scale_back)),
                int(round(x1 * scale_back)),
                int(round(y1 * scale_back)),
            )
            x0_abs, y0_abs, x1_abs, y1_abs = abs_box
            roi_section = img[y0_abs:y1_abs, x0_abs:x1_abs]

        if confidence > best_conf:
            abs_box = (
                int(round(x0 * scale_back)),
                int(round(y0 * scale_back)),
                int(round(x1 * scale_back)),
                int(round(y1 * scale_back)),
            )
            best_roi = ROIResult(
                roi=roi_section.copy(),
                confidence=float(min(1.0, confidence)),
                corners=ordered * scale_back if ordered is not None else None,
                transform=matrix,
                bounding_box=abs_box,
            )
            best_conf = confidence

    if best_roi is None:
        return _fallback_roi(img)

    if best_roi.roi.size == 0:
        logger.warning("ROI detection produced empty crop, falling back to original")
        return ROIResult(roi=img.copy(), confidence=0.1, bounding_box=(0, 0, orig_w, orig_h))

    return best_roi


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _peak_centers(mask: np.ndarray) -> List[int]:
    centers: List[int] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            centers.append((start + idx - 1) // 2)
            start = None
    if start is not None:
        centers.append((start + len(mask) - 1) // 2)
    return centers


def _estimate_axis(profile: np.ndarray, length: int) -> Tuple[int, int, float]:
    if length <= 0:
        return 1, 1, 0.0
    norm = profile - profile.min()
    if norm.max() <= 1e-6:
        fallback_cells = max(1, length // 12 or 1)
        return fallback_cells, max(1, length // fallback_cells), 0.0

    window = max(5, int(max(5, length * 0.01)))
    if window % 2 == 0:
        window += 1
    smooth = _smooth_signal(norm, window)
    threshold = np.percentile(smooth, 65)
    mask = smooth >= threshold
    centers = _peak_centers(mask)

    if len(centers) < 3:
        fallback_cells = max(1, length // 10 or 1)
        return fallback_cells, max(1, int(round(length / fallback_cells))), 0.2

    diffs = np.diff(centers)
    diffs = diffs[diffs > 1]
    if diffs.size == 0:
        fallback_cells = max(1, length // 12 or 1)
        return fallback_cells, max(1, int(round(length / fallback_cells))), 0.25

    median_spacing = float(np.median(diffs))
    approx_from_spacing = max(1, int(round(length / max(median_spacing, 1))))
    approx_from_lines = max(1, len(centers) - 1)
    approx_cells = max(1, int(round((approx_from_spacing + approx_from_lines) / 2)))
    approx_cells = max(approx_from_lines, approx_cells)
    approx_cell_px = max(1, int(round(length / approx_cells)))
    spacing_std = float(np.std(diffs) / (np.mean(diffs) + 1e-6))
    normalized_density = min(len(centers) / (approx_cells + 1e-6), 1.0)
    confidence = float(
        min(
            1.0,
            0.4 + 0.3 * normalized_density + (0.3 if spacing_std < 0.35 else 0),
        )
    )
    return approx_cells, approx_cell_px, confidence


def detect_and_rectify_grid(
    img: np.ndarray,
    detail_level: Literal["low", "medium", "high"] = "medium",
    pattern_mode: PatternMode = "color",
) -> dict:
    """
    Detect the stitch grid using gradient projections. Returns dict with:
      - width / height: number of cells
      - cell_w / cell_h: estimated cell size in pixels
      - roi: the (already rectified) ROI image
      - confidence: 0..1 measure of detection confidence

    NOTE: detail_level is intentionally ignored for now to keep recognition deterministic. We keep
    the argument for compatibility and only use pattern_mode for mild kernel tweaks.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return {
            "width": 1,
            "height": 1,
            "cell_w": 1,
            "cell_h": 1,
            "roi": img.copy(),
            "confidence": 0.0,
        }

    # detail_level is neutral – kernel depends only on chart type.
    kernel_size = 5
    if pattern_mode == "symbol":
        kernel_size = 3
    elif pattern_mode == "mixed":
        kernel_size = 5
    if kernel_size % 2 == 0:
        kernel_size += 1

    if cv2 is not None:  # pragma: no branch - prefer cv2 path when available
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        grad_x = np.abs(cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3))
    else:  # pragma: no cover - fallback branch without cv2
        gray = img.mean(axis=2)
        kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
        blur = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=gray)
        blur = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=blur)
        grad_x = np.abs(np.gradient(blur, axis=1))
        grad_y = np.abs(np.gradient(blur, axis=0))

    vertical_profile = grad_x.mean(axis=0)
    horizontal_profile = grad_y.mean(axis=1)

    width_cells, cell_w, conf_w = _estimate_axis(vertical_profile, w)
    height_cells, cell_h, conf_h = _estimate_axis(horizontal_profile, h)
    confidence = float(min(1.0, 0.3 + 0.35 * conf_w + 0.35 * conf_h))

    return {
        "width": int(max(1, width_cells)),
        "height": int(max(1, height_cells)),
        "cell_w": int(max(1, cell_w)),
        "cell_h": int(max(1, cell_h)),
        "roi": img.copy(),
        "confidence": confidence,
    }
