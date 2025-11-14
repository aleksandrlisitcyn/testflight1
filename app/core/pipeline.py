from __future__ import annotations

import io
from collections import Counter
from typing import Literal, Tuple, Dict, List, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ..color.palette_loader import load_palette
from ..color.palette_matcher import build_kd, nearest_color
from ..cv.cell_sampler import split_into_cells_and_average
from ..cv.grid_detector import detect_and_rectify_grid, detect_pattern_roi
from ..models.pattern import CanvasGrid, Pattern, Stitch, ThreadRef
from .legend import build_legend
from .symbols import assign_symbols_to_palette


# =====================================================================
#  OPTIONAL CV2 + PIL FALLBACK
# =====================================================================

def _maybe_import_cv2():
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover
        return None
    return cv2


# Pillow 9/10 compatibility
_Resampling = getattr(Image, "Resampling", Image)


def _pil_resize(image: np.ndarray, size: Tuple[int, int], resample) -> np.ndarray:
    pil = Image.fromarray(image)
    resized = pil.resize(size, resample=resample)
    return np.array(resized)


# =====================================================================
#  RESIZE / DETAIL
# =====================================================================

def resize_for_detail(image: np.ndarray, detail_level: Literal["low", "medium", "high"]) -> np.ndarray:
    """
    Adaptive resize + лёгкая предобработка для детализации.
    Работает с cv2, но имеет fallback на PIL.
    """
    cv2 = _maybe_import_cv2()

    h, w = image.shape[:2]
    max_side = max(h, w) or 1

    if detail_level == "low":
        target_cells = min(max_side // 8, 250)
        alpha, beta = 1.05, 5
    elif detail_level == "high":
        target_cells = min(max_side // 2, 800)
        alpha, beta = 1.2, 12
    else:  # medium
        target_cells = min(max_side // 4, 450)
        alpha, beta = 1.12, 8

    if target_cells <= 0:
        target_cells = max_side

    scale = max(target_cells / float(max_side), 1e-6)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if cv2 is not None:
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    # Fallback: PIL
    resample = _Resampling.LANCZOS if scale < 1.0 else _Resampling.BICUBIC
    img = _pil_resize(image, (new_w, new_h), resample)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.asarray(pil_img, dtype=np.float32)
    arr = arr * alpha + beta
    img = np.clip(arr, 0, 255).astype(np.uint8)
    return img


def _color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))


# =====================================================================
#  MAIN PIPELINE: image → Pattern
# =====================================================================

def process_image_to_pattern(
    image: np.ndarray,
    brand: Literal["DMC", "Gamma", "Anchor", "auto"] = "DMC",
    min_cells: int = 30,
    detail_level: Literal["low", "medium", "high"] = "medium",
) -> Pattern:
    """
    Основной конвейер:
      1) предобработка/масштабирование под уровень детализации
      2) поиск ROI и сетки
      3) усреднение цвета по ячейкам
      4) привязка к палитре, фильтрация редких цветов
      5) построение Pattern + legend/meta
    """

    # 1) предобработка
    img_proc = resize_for_detail(image, detail_level)

    # 2) ROI + grid
    roi = detect_pattern_roi(img_proc)
    grid_dict = detect_and_rectify_grid(roi)

    # 3) усреднение цвета в ячейках
    colors: Dict[Tuple[int, int], Tuple[int, int, int]] = split_into_cells_and_average(roi, grid_dict)

    # запасной вариант — хотя бы один цвет
    if not colors:
        mean_rgb = (
            tuple(int(v) for v in roi.reshape(-1, 3).mean(axis=0))
            if roi.size
            else (0, 0, 0)
        )
        colors = {(0, 0): mean_rgb}
        grid_dict["width"] = int(grid_dict.get("width") or 1)
        grid_dict["height"] = int(grid_dict.get("height") or 1)

    # 4) детекция фона по квантизованному цвету
    quant_counter: Counter[Tuple[int, int, int]] = Counter()
    for _, rgb in colors.items():
        q = tuple((int(c) // 16) * 16 for c in rgb)
        quant_counter[q] += 1

    background_quant = quant_counter.most_common(1)[0][0] if quant_counter else None

    # 5) палитра и KD-дерево
    resolved_brand = "DMC" if brand == "auto" else brand
    palette = load_palette(resolved_brand)
    kd = build_kd(palette)

    used_palette: Dict[Tuple[str, str], Dict] = {}
    counts: Counter[Tuple[str, str]] = Counter()
    stitches_raw: List[Dict] = []

    for (x, y), rgb in colors.items():
        # пропускаем фон
        if background_quant:
            q = tuple((int(c) // 16) * 16 for c in rgb)
            if q == background_quant:
                continue

        matched = nearest_color(kd, rgb, palette)
        key = (matched["brand"], matched["code"])
        used_palette[key] = matched
        counts[key] += 1
        stitches_raw.append({"x": int(x), "y": int(y), "key": key})

    # 6) адаптивный порог по min_cells в зависимости от detail_level
    if detail_level == "low":
        min_cells_bias = 1.2
    elif detail_level == "high":
        min_cells_bias = 0.6
    else:
        min_cells_bias = 1.0

    effective_min_cells = max(1, int(round(min_cells * min_cells_bias)))

    # слияние редких цветов в ближайшие "мажорные"
    if counts and effective_min_cells > 1:
        major = {k for k, c in counts.items() if c >= effective_min_cells}
        if not major:
            major = set(counts.keys())
    else:
        major = set(counts.keys())

    if counts and effective_min_cells > 1:
        palette_rgb = {
            key: tuple(used_palette[key].get("rgb", (0, 0, 0)))
            for key in used_palette
        }
        for s in stitches_raw:
            if s["key"] in major:
                continue
            if not major:
                continue
            origin_rgb = palette_rgb[s["key"]]
            target = min(major, key=lambda k: _color_distance(palette_rgb[k], origin_rgb))
            s["key"] = target
        used_palette = {key: used_palette[key] for key in major}

    # пересчёт итоговых количеств
    final_counts: Counter[Tuple[str, str]] = Counter()
    for s in stitches_raw:
        final_counts[s["key"]] += 1

    # 7) автоудаление доминирующего фона, если он явно один
    background_key: Tuple[str, str] | None = None
    if final_counts:
        total_cells = sum(final_counts.values())
        key, cnt = final_counts.most_common(1)[0]
        if len(final_counts) > 1 and cnt >= max(50, int(total_cells * 0.4)):
            background_key = key

    if background_key is not None:
        stitches_raw = [s for s in stitches_raw if s["key"] != background_key]
        final_counts.pop(background_key, None)
        used_palette.pop(background_key, None)

    # 8) собираем палитру и присваиваем символы
    palette_items = sorted(
        used_palette.items(),
        key=lambda item: final_counts.get(item[0], 0),
        reverse=True,
    )
    palette_list = [dict(v) for _, v in palette_items]
    palette_list = assign_symbols_to_palette(palette_list)

    thread_map: Dict[Tuple[str, str], ThreadRef] = {}
    for entry in palette_list:
        rgb = tuple(int(v) for v in entry.get("rgb", (0, 0, 0)))
        thread = ThreadRef(
            brand=entry["brand"],
            code=entry["code"],
            name=entry.get("name"),
            rgb=rgb,
            symbol=entry.get("symbol"),
        )
        thread_map[(thread.brand, thread.code)] = thread

    # 9) строим стежки
    stitches: List[Stitch] = []
    for s in stitches_raw:
        t = thread_map.get(s["key"])
        if not t:
            continue
        stitches.append(Stitch(x=int(s["x"]), y=int(s["y"]), thread=t))

    # 10) размер сетки в клетках
    width_cells = int(grid_dict.get("width") or 0) or (
        max((st.x for st in stitches), default=-1) + 1
    )
    height_cells = int(grid_dict.get("height") or 0) or (
        max((st.y for st in stitches), default=-1) + 1
    )

    total_stitches = int(sum(final_counts.values()))

    pattern = Pattern(
        canvasGrid=CanvasGrid(width=width_cells, height=height_cells),
        palette=list(thread_map.values()),
        stitches=stitches,
        meta={
            "title": "Generated Pattern",
            "dpi": 300,
            "brand": resolved_brand,
            "palette_size": len(thread_map),
            "total_stitches": total_stitches,
            "detail_level": detail_level,
        },
    )

    # 11) легенда в meta
    legend = build_legend(pattern, force=True)
    pattern.meta["legend"] = legend

    # если цветов слишком мало — автоматически ослабляем min_cells и пробуем ещё раз
    if len({s.thread.code for s in pattern.stitches}) < 5 and effective_min_cells > 1:
        return process_image_to_pattern(
            image,
            brand=brand,
            min_cells=max(1, int(effective_min_cells * 0.5)),
            detail_level=detail_level,
        )

    return pattern


# =====================================================================
#  PREVIEW (PNG)
# =====================================================================

def render_preview(pattern: Union[dict, Pattern], mode: str = "color") -> bytes:
    """
    Рендер превью в PNG:
      - цветные блоки
      - опционально символы поверх
    """
    if hasattr(pattern, "model_dump"):
        p = pattern.model_dump()
    elif hasattr(pattern, "dict"):
        p = pattern.dict()
    else:
        p = pattern

    w = int(p["canvasGrid"]["width"])
    h = int(p["canvasGrid"]["height"])
    cell = 20  # px per cell

    # палитра → цвет + символ
    color_map: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    symbol_map: Dict[Tuple[str, str], str] = {}

    for entry in p["palette"]:
        brand = entry.get("brand")
        code = entry.get("code")
        rgb = entry.get("rgb", (200, 200, 200))
        color_map[(brand, code)] = tuple(int(c) for c in rgb)
        if entry.get("symbol"):
            symbol_map[(brand, code)] = entry["symbol"]

    img = np.full((h * cell, w * cell, 3), 255, dtype=np.uint8)

    # закрашиваем клетки
    for s in p["stitches"]:
        x, y = int(s["x"]), int(s["y"])
        th = s["thread"]
        key = (th["brand"], th["code"])
        rgb = color_map.get(key, (200, 200, 200))

        x0, y0 = x * cell, y * cell
        x1, y1 = x0 + cell, y0 + cell
        img[y0:y1, x0:x1] = rgb

    # сетка
    base_grid = (210, 210, 210)
    accent_grid = (120, 120, 120)

    for x in range(w + 1):
        color = accent_grid if x % 10 == 0 else base_grid
        img[:, x * cell:x * cell + 1] = color
    for y in range(h + 1):
        color = accent_grid if y % 10 == 0 else base_grid
        img[y * cell:y * cell + 1, :] = color

    # символы
    pil = Image.fromarray(img, "RGB")
    draw = ImageDraw.Draw(pil)

    if mode == "symbols":
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, cell - 6))
        except Exception:  # pragma: no cover
            font = ImageFont.load_default()

        for s in p["stitches"]:
            x, y = int(s["x"]), int(s["y"])
            th = s["thread"]
            sym = symbol_map.get((th["brand"], th["code"]))
            if not sym:
                continue

            # центр клетки
            cx = x * cell + cell / 2
            cy = y * cell + cell / 2

            # аккуратное центрирование
            try:
                bbox = font.getbbox(sym)
                tw = bbox[2] - bbox[0]
                th_h = bbox[3] - bbox[1]
            except Exception:
                tw = font.getlength(sym) if hasattr(font, "getlength") else len(sym) * (cell / 2)
                th_h = getattr(font, "size", cell / 2)

            draw_x = cx - tw / 2
            draw_y = cy - th_h / 2
            draw.text((draw_x, draw_y), sym, fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def render_pattern_image(pattern: Union[dict, Pattern], with_symbols: bool = True) -> bytes:
    """Backwards-compatible helper."""
    return render_preview(pattern, "symbols" if with_symbols else "color")