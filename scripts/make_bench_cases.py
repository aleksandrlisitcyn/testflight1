from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def make_case(
    cols: int,
    rows: int,
    cell: int,
    border: int = 24,
    perspective: bool = False,
) -> np.ndarray:
    width = cols * cell + border * 2
    height = rows * cell + border * 2
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)

    colors = [
        (180, 70, 80),
        (70, 140, 200),
        (200, 200, 80),
    ]

    for y in range(rows):
        for x in range(cols):
            color = colors[(x + y) % len(colors)]
            top = border + y * cell
            bottom = top + cell
            left = border + x * cell
            right = left + cell
            canvas[top:bottom, left:right] = color

    if perspective and cv2 is not None:
        h, w = canvas.shape[:2]
        src = np.float32(
            [
                [border // 2, border],
                [w - border, border + 10],
                [w - border // 2, h - border],
                [border, h - border // 3],
            ]
        )
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        canvas = cv2.warpPerspective(canvas, matrix, (w, h))

    return canvas


def main() -> None:
    cases_dir = Path("data/bench/cases")
    cases_dir.mkdir(parents=True, exist_ok=True)

    case_specs = [
        ("case01_basic.png", dict(cols=12, rows=12, cell=20, perspective=False)),
        ("case02_perspective.png", dict(cols=16, rows=10, cell=16, perspective=True)),
        ("case03_noise.png", dict(cols=18, rows=14, cell=14, perspective=False)),
    ]

    for name, kwargs in case_specs:
        img = make_case(**kwargs)
        if name.startswith("case03"):
            noise = np.random.randint(-12, 12, size=img.shape, dtype=np.int16)
            noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = noisy
        Image.fromarray(img).save(cases_dir / name)


if __name__ == "__main__":
    main()
