import io
from collections import Counter
from typing import Literal, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..color.palette_loader import load_palette
from ..color.palette_matcher import build_kd, nearest_color
from ..cv.cell_sampler import split_into_cells_and_average
from ..cv.grid_detector import detect_and_rectify_grid, detect_pattern_roi
from ..models.pattern import CanvasGrid, Pattern, Stitch, ThreadRef
from .legend import build_legend
from .symbols import assign_symbols_to_palette


def resize_to_max_cells(image: np.ndarray, target_cells: int) -> np.ndarray:
    if target_cells <= 0:
        return image

    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side <= 0:
        return image

    if max_side <= target_cells:
        return image

    scale = float(target_cells) / float(max_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    import cv2  # local import to avoid mandatory dependency at module import time

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def _color_distance(rgb_a: Tuple[int, int, int], rgb_b: Tuple[int, int, int]) -> float:
    arr_a = np.array(rgb_a, dtype=float)
    arr_b = np.array(rgb_b, dtype=float)
    return float(np.linalg.norm(arr_a - arr_b))


def process_image_to_pattern(
    image: np.ndarray,
    brand: Literal["DMC", "Gamma", "Anchor", "auto"] = "DMC",
    min_cells: int = 30,
) -> Pattern:
    # === Улучшено: адаптивное масштабирование ===
    # определяем длину большей стороны
    h, w = image.shape[:2]
    max_side = max(h, w)

    # определяем количество клеток в зависимости от размера изображения
    # для крупных изображений — до 400-600 клеток, для мелких — не уменьшаем
    target_cells = min(max_side // 4, 600)
    if target_cells <= 0:
        target_cells = max_side

    # изменяем размер, сохраняя пропорции
    image = resize_to_max_cells(image, target_cells)

    # небольшое сглаживание для уменьшения шумов
    import cv2

    image = cv2.GaussianBlur(image, (3, 3), 0)

    roi = detect_pattern_roi(image)
    grid_dict = detect_and_rectify_grid(roi)
    colors = split_into_cells_and_average(roi, grid_dict)

    resolved_brand = brand if brand != "auto" else "DMC"
    palette = load_palette(resolved_brand)
    kd = build_kd(palette)

    used_palette: dict[Tuple[str, str], dict] = {}
    stitches_raw: list[dict] = []
    counts: Counter[Tuple[str, str]] = Counter()

    for (x, y), rgb in colors.items():
        matched = nearest_color(kd, rgb, palette)
        key = (matched["brand"], matched["code"])
        used_palette[key] = matched
        counts[key] += 1
        stitches_raw.append({"x": int(x), "y": int(y), "key": key})

    if counts and min_cells > 1:
        major_keys = {key for key, count in counts.items() if count >= min_cells}
        if not major_keys:
            major_keys = set(counts.keys())
        palette_rgb = {key: tuple(used_palette[key].get("rgb", (0, 0, 0))) for key in used_palette}
        for stitch in stitches_raw:
            if stitch["key"] in major_keys:
                continue
            if not major_keys:
                continue
            origin_rgb = palette_rgb[stitch["key"]]
            target = min(major_keys, key=lambda k: _color_distance(palette_rgb[k], origin_rgb))
            stitch["key"] = target
        used_palette = {key: used_palette[key] for key in major_keys}
    final_counts = Counter()
    for stitch in stitches_raw:
        final_counts[stitch["key"]] += 1

    palette_items = sorted(
        used_palette.items(),
        key=lambda item: final_counts.get(item[0], 0),
        reverse=True,
    )
    palette_list = [dict(value) for _, value in palette_items]
    palette_list = assign_symbols_to_palette(palette_list)
    thread_map: dict[Tuple[str, str], ThreadRef] = {}
    for entry in palette_list:
        rgb_tuple = tuple(int(v) for v in entry.get("rgb", (0, 0, 0)))
        thread = ThreadRef(
            brand=entry["brand"],
            code=entry["code"],
            name=entry.get("name"),
            rgb=rgb_tuple,
            symbol=entry.get("symbol"),
        )
        thread_map[(thread.brand, thread.code)] = thread

    stitches: list[Stitch] = []
    for stitch in stitches_raw:
        thread = thread_map.get(stitch["key"])
        if not thread:
            continue
        stitches.append(Stitch(x=stitch["x"], y=stitch["y"], thread=thread))

    total_stitches = sum(final_counts.values())

    pattern = Pattern(
        canvasGrid=CanvasGrid(
            width=int(grid_dict["width"]),
            height=int(grid_dict["height"]),
        ),
        palette=list(thread_map.values()),
        stitches=stitches,
        meta={
            "title": "Generated Pattern",
            "dpi": 300,
            "brand": resolved_brand,
            "palette_size": len(thread_map),
            "total_stitches": total_stitches,
        },
    )
    legend = build_legend(pattern, force=True)
    pattern.meta["legend"] = legend
    if len({s.thread.code for s in pattern.stitches}) < 5 and min_cells > 1:
        return process_image_to_pattern(
            image,
            brand=brand,
            min_cells=max(1, int(min_cells * 0.5)),
        )

    return pattern


def render_preview(pattern: dict, mode: str = "color") -> bytes:
    if hasattr(pattern, "model_dump"):
        pattern = pattern.model_dump()
    elif hasattr(pattern, "dict"):
        pattern = pattern.dict()
    w = pattern["canvasGrid"]["width"]
    h = pattern["canvasGrid"]["height"]
    cell = 20  # px per cell for a clearer preview
    img = np.full((h * cell, w * cell, 3), 255, dtype=np.uint8)

    color_map: dict[Tuple[str, str], Tuple[int, int, int]] = {}
    symbol_map: dict[Tuple[str, str], str] = {}
    for p in pattern["palette"]:
        rgb = p.get("rgb") or (120, 120, 120)
        key = (p["brand"], p["code"])
        color_map[key] = tuple(int(v) for v in rgb)
        if p.get("symbol"):
            symbol_map[key] = p["symbol"]

    for s in pattern["stitches"]:
        x, y = s["x"], s["y"]
        key = (s["thread"]["brand"], s["thread"]["code"])
        rgb = color_map.get(key, (200, 200, 200))
        img[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = rgb

    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil)

    # Draw grid lines for clarity
    base_color = (210, 210, 210)
    accent_color = (120, 120, 120)
    for x in range(w + 1):
        color = accent_color if x % 10 == 0 else base_color
        draw.line((x * cell, 0, x * cell, h * cell), fill=color, width=1)
    for y in range(h + 1):
        color = accent_color if y % 10 == 0 else base_color
        draw.line((0, y * cell, w * cell, y * cell), fill=color, width=1)

    if mode == "symbols":
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, cell - 6))
        except Exception:  # pragma: no cover - optional font availability
            font = ImageFont.load_default()
        for s in pattern["stitches"]:
            x, y = s["x"], s["y"]
            key = (s["thread"]["brand"], s["thread"]["code"])
            symbol = symbol_map.get(key, "?")
            try:
                bbox = font.getbbox(symbol)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except Exception:  # pragma: no cover
                text_width = font.getlength(symbol) if hasattr(font, "getlength") else len(symbol) * (cell / 2)
                text_height = getattr(font, "size", cell / 2)
            draw_x = x * cell + (cell - text_width) / 2
            draw_y = y * cell + (cell - text_height) / 2
            draw.text((draw_x, draw_y), symbol, fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
