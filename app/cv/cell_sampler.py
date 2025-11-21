from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np

from ..core.types import PatternMode


def _aggregate_patch(sample: np.ndarray, pattern_mode: PatternMode) -> Tuple[int, int, int]:
    """
    Consolidate colours inside a single cell. This is intentionally local so we do not collapse
    distinct hues globally – we only smooth tiny paper artefacts within the cell itself.
    """
    flat = sample.reshape(-1, sample.shape[-1]).astype(np.float32)
    if flat.size == 0:
        return (0, 0, 0)

    trimmed = flat
    if flat.shape[0] >= 6:
        low = np.quantile(flat, 0.1, axis=0)
        high = np.quantile(flat, 0.9, axis=0)
        mask = np.all((flat >= low) & (flat <= high), axis=1)
        if np.any(mask):
            trimmed = flat[mask]

    mean = trimmed.mean(axis=0)
    median = np.median(flat, axis=0)

    if pattern_mode == "symbol":
        # Symbols are often dark ink on lighter paper – median keeps edges without drifting
        # towards paper tones.
        value = median
    elif pattern_mode == "mixed":
        value = 0.5 * mean + 0.5 * median
    else:
        value = mean

    return (
        int(round(float(value[0]))),
        int(round(float(value[1]))),
        int(round(float(value[2]))),
    )


def split_into_cells_and_average(
    img: np.ndarray,
    grid: dict,
    *,
    method: Literal["mean", "centroid"] = "mean",
    pattern_mode: PatternMode = "color",
) -> Dict[Tuple[int, int], Tuple[int, int, int]]:
    """
    Split a rectified ROI into logical cells and compute an average colour per cell.

    method:
      - "mean": average all pixels
      - "centroid": use a smaller patch near the centre to reduce border bleeding
    pattern_mode:
      - influences how local aggregation handles paper noise vs ink strokes
    """

    h, w = img.shape[:2]
    width = max(1, int(grid.get("width") or 1))
    height = max(1, int(grid.get("height") or 1))

    cell_w = w / float(width)
    cell_h = h / float(height)

    colors: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    for y in range(height):
        top = int(round(y * cell_h))
        bottom = int(round((y + 1) * cell_h))
        bottom = min(h, max(top + 1, bottom))
        for x in range(width):
            left = int(round(x * cell_w))
            right = int(round((x + 1) * cell_w))
            right = min(w, max(left + 1, right))

            patch = img[top:bottom, left:right]
            if patch.size == 0:
                continue

            if method == "centroid":
                cy = patch.shape[0] // 2
                cx = patch.shape[1] // 2
                radius_y = max(1, min(patch.shape[0] // 8, 2))
                radius_x = max(1, min(patch.shape[1] // 8, 2))
                sub = patch[
                    max(0, cy - radius_y) : min(patch.shape[0], cy + radius_y),
                    max(0, cx - radius_x) : min(patch.shape[1], cx + radius_x),
                ]
                sample = sub if sub.size > 0 else patch
            else:
                sample = patch

            colors[(x, y)] = _aggregate_patch(sample, pattern_mode=pattern_mode)

    return colors
