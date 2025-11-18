from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency for synthetic warps
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def make_grid_image(
    cols: int,
    rows: int,
    cell: int = 20,
    border: int = 20,
    perspective: bool = False,
    noise: bool = False,
) -> np.ndarray:
    width = cols * cell + border * 2
    height = rows * cell + border * 2
    canvas = np.full((height, width, 3), 240, dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            top = border + y * cell
            bottom = top + cell
            left = border + x * cell
            right = left + cell
            color = (180, 80, 60) if (x + y) % 2 == 0 else (60, 90, 180)
            canvas[top:bottom, left:right] = color

    if noise:
        noise_mask = np.random.randint(0, 15, size=canvas.shape[:2], dtype=np.uint8)
        canvas[:, :, 0] = np.clip(canvas[:, :, 0] + noise_mask, 0, 255)

    if perspective and cv2 is not None:
        h, w = canvas.shape[:2]
        pts_src = np.float32(
            [
                [border // 2, border],
                [w - border, border // 3],
                [w - border // 2, h - border],
                [border, h - border // 3],
            ]
        )
        pts_dst = np.float32(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ]
        )
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(canvas, matrix, (w, h))
        return warped

    return canvas
