import io
from collections import Counter
from typing import Literal, Tuple, Dict, List, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..color.palette_loader import load_palette
from ..color.palette_matcher import build_kd, nearest_color
from ..cv.cell_sampler import split_into_cells_and_average
from ..cv.grid_detector import detect_and_rectify_grid, detect_pattern_roi
from ..models.pattern import CanvasGrid, Pattern, Stitch, ThreadRef
from .legend import build_legend
from .symbols import assign_symbols_to_palette


# =====================================================================
#  Helpers
# =====================================================================

def resize_for_detail(image: np.ndarray, detail_level: Literal["low", "medium", "high"]) -> np.ndarray:
    """
    Adaptive resize based on detail level.
    Keeps aspect ratio, reduces very large photos while not upscaling small ones too aggressively.
    """
    import cv2

    h, w = image.shape[:2]
    max_side = max(h, w) or 1

    if detail_level == "low":
        target_cells = min(max_side // 8, 250)
        contrast_alpha = 1.05
        contrast_beta = 5
    elif detail_level == "high":
        target_cells = min(max_side // 2, 800)
        contrast_alpha = 1.2
        contrast_beta = 12
    else:
        # medium
        target_cells = min(max_side // 4, 450)
        contrast_alpha = 1.12
        contrast_beta = 8

    if target_cells <= 0:
        target_cells = max_side

    scale = max(target_cells / float(max_side), 1e-6)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Light blur + contrast to stabilise colors / suppress noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.convertScaleAbs(img, alpha=contrast_alpha, beta=contrast_beta)
    return img


def _color_distance(rgb_a: Tuple[int, int, int], rgb_b: Tuple[int, int, int]) -> float:
    a = np.array(rgb_a, dtype=float)
    b = np.array(rgb_b, dtype=float)
    return float(np.linalg.norm(a - b))


# =====================================================================
#  IMAGE → PATTERN
# =====================================================================

def process_image_to_pattern(
    image: np.ndarray,
    brand: Literal["DMC", "Gamma", "Anchor", "auto"] = "DMC",
    min_cells: int = 30,
    detail_level: Literal["low", "medium", "high"] = "medium",
) -> Pattern:
    """
    Main pipeline:
      1) adaptive resize & pre-processing
      2) detect ROI (stitch grid or whole image)
      3) detect grid & sample average color per cell
      4) detect background color and drop it
      5) palette matching (DMC / Gamma / Anchor / auto)
      6) min_cells filtering
      7) assign symbols
      8) build Pattern model
    """
    # 1) adaptive resize / denoise / contrast
    image_proc = resize_for_detail(image, detail_level=detail_level)

    # 2) ROI and grid detection
    roi = detect_pattern_roi(image_proc)
    grid_dict = detect_and_rectify_grid(roi)

    # 3) sample average colors per logical cell
    colors: Dict[Tuple[int, int], Tuple[int, int, int]] = split_into_cells_and_average(roi, grid_dict)

    # 4) background detection (most frequent quantised color)
    bg_counter: Counter[Tuple[int, int, int]] = Counter()
    for (_x, _y), rgb in colors.items():
        q = tuple(int(int(c) / 16) * 16 for c in rgb)
        bg_counter[q] += 1

    background_quant = None
    if bg_counter:
        background_quant, _ = bg_counter.most_common(1)[0]

    # 5) palette matching
    resolved_brand = "DMC" if brand == "auto" else brand
    palette = load_palette(resolved_brand)
    kd = build_kd(palette)

    used_palette: Dict[Tuple[str, str], Dict] = {}
    counts: Counter[Tuple[str, str]] = Counter()
    stitches_raw: List[Dict[str, object]] = []

    for (x, y), rgb in colors.items():
        # Skip background / canvas color
        if background_quant is not None:
            q = tuple(int(int(c) / 16) * 16 for c in rgb)
            if q == background_quant:
                continue

        matched = nearest_color(kd, rgb, palette)
        key = (matched["brand"], matched["code"])
        used_palette[key] = matched
        counts[key] += 1
        stitches_raw.append({"x": int(x), "y": int(y), "key": key})

    # 6) min_cells filtering
    if counts and min_cells > 1:
        major_keys = {key for key, cnt in counts.items() if cnt >= min_cells}
        if not major_keys:
            major_keys = set(counts.keys())
    else:
        major_keys = set(counts.keys())

    # Build thread_map limited to major_keys
    thread_map: Dict[Tuple[str, str], ThreadRef] = {}
    final_counts: Counter[Tuple[str, str]] = Counter()

    for key, entry in used_palette.items():
        if key not in major_keys:
            continue
        rgb_tuple = tuple(int(c) for c in (entry.get("rgb") or entry.get("color") or (0, 0, 0)))
        thread = ThreadRef(
            brand=entry["brand"],
            code=entry["code"],
            name=entry.get("name"),
            rgb=rgb_tuple,
            symbol=None,
        )
        thread_map[key] = thread
        final_counts[key] = counts[key]

    # 7) build stitches list using filtered thread_map
    stitches: List[Stitch] = []
    for s in stitches_raw:
        key = s["key"]  # type: ignore[assignment]
        thread = thread_map.get(key)  # type: ignore[arg-type]
        if not thread:
            continue
        stitches.append(
            Stitch(
                x=int(s["x"]),  # type: ignore[arg-type]
                y=int(s["y"]),  # type: ignore[arg-type]
                thread=thread,
            )
        )

    # 8) assign symbols to palette
    palette_list = []
    for (_brand_code, thread) in thread_map.items():
        palette_list.append(
            {
                "brand": thread.brand,
                "code": thread.code,
                "name": thread.name,
                "rgb": thread.rgb,
            }
        )

    palette_list = assign_symbols_to_palette(palette_list)

    # rebuild thread_map with symbols
    new_thread_map: Dict[Tuple[str, str], ThreadRef] = {}
    for p in palette_list:
        key = (p["brand"], p["code"])
        new_thread_map[key] = ThreadRef(
            brand=p["brand"],
            code=p["code"],
            name=p.get("name"),
            rgb=tuple(p["rgb"]),
            symbol=p.get("symbol"),
        )

    # reassign threads on stitches
    final_stitches: List[Stitch] = []
    for s in stitches:
        key = (s.thread.brand, s.thread.code)
        thread = new_thread_map.get(key)
        if not thread:
            continue
        final_stitches.append(
            Stitch(
                x=s.x,
                y=s.y,
                thread=thread,
            )
        )

    # 9) build Pattern
    width_cells = int(grid_dict.get("width") or 0) or max((s.x for s in final_stitches), default=-1) + 1
    height_cells = int(grid_dict.get("height") or 0) or max((s.y for s in final_stitches), default=-1) + 1

    total_stitches = int(sum(final_counts.values()))

    pattern = Pattern(
        canvasGrid=CanvasGrid(
            width=int(width_cells),
            height=int(height_cells),
        ),
        palette=list(new_thread_map.values()),
        stitches=final_stitches,
        meta={
            "title": "Generated Pattern",
            "dpi": 300,
            "brand": resolved_brand,
            "palette_size": len(new_thread_map),
            "total_stitches": total_stitches,
            "detail_level": detail_level,
        },
    )

    # precompute legend
    pattern.meta["legend"] = build_legend(pattern, force=True)
    return pattern


# =====================================================================
#  PREVIEW RENDERING
# =====================================================================

def render_preview(pattern: Union[dict, Pattern], mode: str = "color") -> bytes:
    """
    Render a PNG preview of the pattern.

    mode:
      - "color"   – filled cells with thread colors
      - "symbols" – same but with symbol overlay
    """
    if hasattr(pattern, "model_dump"):
        pattern_dict = pattern.model_dump()
    elif hasattr(pattern, "dict"):
        pattern_dict = pattern.dict()
    else:
        pattern_dict = pattern  # type: ignore[assignment]

    w = int(pattern_dict["canvasGrid"]["width"])
    h = int(pattern_dict["canvasGrid"]["height"])
    cell = 20  # px per cell for a clearer preview

    # palette lookup
    color_map: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    for p in pattern_dict.get("palette", []):
        if isinstance(p, dict):
            brand = p.get("brand")
            code = p.get("code")
            rgb = p.get("rgb") or p.get("color") or (200, 200, 200)
        else:
            brand = getattr(p, "brand", None)
            code = getattr(p, "code", None)
            rgb = getattr(p, "rgb", (200, 200, 200))
        if brand is None or code is None:
            continue
        color_map[(brand, code)] = tuple(int(c) for c in rgb)

    # base white canvas
    img = np.full((h * cell, w * cell, 3), 255, dtype=np.uint8)

    # draw stitches
    for s in pattern_dict["stitches"]:
        if isinstance(s, dict):
            x = int(s.get("x"))
            y = int(s.get("y"))
            thread = s.get("thread")
            if isinstance(thread, dict):
                brand = thread.get("brand")
                code = thread.get("code")
            else:
                brand = getattr(thread, "brand", None)
                code = getattr(thread, "code", None)
        else:
            x = int(getattr(s, "x"))
            y = int(getattr(s, "y"))
            thread = getattr(s, "thread")
            brand = getattr(thread, "brand", None)
            code = getattr(thread, "code", None)

        if brand is None or code is None:
            continue
        rgb = color_map.get((brand, code), (200, 200, 200))

        x0, x1 = x * cell, (x + 1) * cell
        y0, y1 = y * cell, (y + 1) * cell

        # clamp to image bounds
        x0 = max(0, min(img.shape[1], x0))
        x1 = max(0, min(img.shape[1], x1))
        y0 = max(0, min(img.shape[0], y0))
        y1 = max(0, min(img.shape[0], y1))
        if x0 >= x1 or y0 >= y1:
            continue
        img[y0:y1, x0:x1] = rgb

    # draw grid (light + every 10th darker)
    base_grid = (210, 210, 210)
    accent_grid = (120, 120, 120)

    # vertical
    for x in range(w + 1):
        px = x * cell
        if 0 <= px < img.shape[1]:
            color = accent_grid if x % 10 == 0 else base_grid
            img[:, px:px + 1] = color

    # horizontal
    for y in range(h + 1):
        py = y * cell
        if 0 <= py < img.shape[0]:
            color = accent_grid if y % 10 == 0 else base_grid
            img[py:py + 1, :] = color

    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil)

    # optional symbols overlay
    if mode == "symbols":
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, cell - 6))
        except Exception:
            font = ImageFont.load_default()

        for s in pattern_dict["stitches"]:
            if isinstance(s, dict):
                x = int(s.get("x"))
                y = int(s.get("y"))
                thread = s.get("thread")
                if isinstance(thread, dict):
                    symbol = thread.get("symbol")
                else:
                    symbol = getattr(thread, "symbol", None)
            else:
                x = int(getattr(s, "x"))
                y = int(getattr(s, "y"))
                thread = getattr(s, "thread")
                symbol = getattr(thread, "symbol", None)

            if not symbol:
                continue

            cx = x * cell + cell // 2
            cy = y * cell + cell // 2
            draw.text((cx, cy), str(symbol), fill=(0, 0, 0), anchor="mm", font=font)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def render_pattern_image(pattern: Union[dict, Pattern], with_symbols: bool = True) -> bytes:
    """
    Convenience wrapper used by exporters: returns PNG bytes.
    """
    mode = "symbols" if with_symbols else "color"
    return render_preview(pattern, mode=mode)