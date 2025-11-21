import io
import logging
from collections import Counter
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..color.palette_loader import load_palette
from ..color.palette_matcher import build_kd, nearest_color
from ..cv.cell_sampler import split_into_cells_and_average
from ..cv.grid_detector import detect_and_rectify_grid, detect_pattern_roi
from ..models.pattern import CanvasGrid, Pattern, Stitch, ThreadRef
from .legend import build_legend
from .symbols import assign_symbols_to_palette
from .types import PatternMode

logger = logging.getLogger(__name__)


# =====================================================================
#  Helpers
# =====================================================================


def resize_for_detail(
    image: np.ndarray,
    detail_level: Literal["low", "medium", "high"],
) -> np.ndarray:
    """
    Adaptive resize based on detail level.
    Keeps aspect ratio, reduces very large photos while not upscaling small ones too aggressively.

    NOTE: detail_level is currently ignored intentionally to keep recognition neutral. The argument
    remains for API compatibility and will be repurposed for output-only tweaks later on.
    """
    import cv2

    h, w = image.shape[:2]
    max_side = max(h, w) or 1

    target_cells = min(max_side // 4, 450)
    contrast_alpha = 1.12
    contrast_beta = 8

    if target_cells <= 0:
        target_cells = max_side

    if max_side <= 700:
        scale = 1.0
    else:
        scale = max(target_cells / float(max_side), 0.2)
        scale = min(1.0, scale)
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


def _estimate_color_spread(colors: Dict[Tuple[int, int], Tuple[int, int, int]]) -> float:
    if not colors:
        return 0.0
    arr = np.array(list(colors.values()), dtype=float)
    center = arr.mean(axis=0)
    distances = np.linalg.norm(arr - center, axis=1)
    return float(distances.mean())


def _build_sanity_report(
    palette_size: int,
    grid_size: Tuple[int, int],
    roi_size: Tuple[int, int],
    filled_cells: int,
    color_spread: float,
) -> Dict[str, object]:
    grid_w, grid_h = grid_size
    roi_w, roi_h = roi_size
    total_cells = max(1, grid_w * grid_h)
    fill_ratio = filled_cells / total_cells

    issues: List[Dict[str, object]] = []
    if palette_size < 2 and color_spread > 12:
        issues.append(
            {
                "code": "palette_too_small",
                "details": {
                    "color_spread": round(color_spread, 2),
                    "filled_cells": filled_cells,
                },
            }
        )
    roi_area = roi_w * roi_h
    if roi_area > 0 and total_cells <= 9:
        issues.append(
            {
                "code": "grid_too_coarse",
                "details": {
                    "roi_area": roi_area,
                    "grid_cells": total_cells,
                },
            }
        )
    if fill_ratio < 0.3 and filled_cells > 4:
        issues.append(
            {
                "code": "sparse_fill",
                "details": {
                    "fill_ratio": round(fill_ratio, 3),
                    "filled_cells": filled_cells,
                },
            }
        )

    return {
        "status": "ok" if not issues else "flagged",
        "issues": issues,
        "metrics": {
            "palette_size": palette_size,
            "grid_width": grid_w,
            "grid_height": grid_h,
            "roi_width": roi_w,
            "roi_height": roi_h,
            "filled_cells": filled_cells,
            "color_spread": round(color_spread, 2),
        },
        "fallback_applied": False,
    }


# =====================================================================
#  IMAGE → PATTERN
# =====================================================================


def _detect_background_color(
    samples: Dict[Tuple[int, int], Tuple[int, int, int]],
) -> Tuple[int, int, int] | None:
    if not samples:
        return None
    quantised = [
        (
            int(round(rgb[0] / 12.0) * 12),
            int(round(rgb[1] / 12.0) * 12),
            int(round(rgb[2] / 12.0) * 12),
        )
        for rgb in samples.values()
    ]
    counter: Counter[Tuple[int, int, int]] = Counter(quantised)
    (color, count), *_ = counter.most_common(1) or [((0, 0, 0), 0)]
    ratio = count / max(len(quantised), 1)
    brightness = sum(color) / 3.0
    saturation = max(color) - min(color)

    if ratio >= 0.48 and (brightness >= 185 or brightness <= 40 or saturation <= 18):
        return color
    return None


def process_image_to_pattern(
    image: np.ndarray,
    brand: Literal["DMC", "Gamma", "Anchor", "auto"] = "DMC",
    min_cells: int = 30,
    detail_level: Literal["low", "medium", "high"] = "medium",
    pattern_mode: PatternMode = "color",
) -> Pattern:
    """
    Main pipeline:
      1) adaptive resize & pre-processing
      2) detect ROI (stitch grid or whole image)
      3) detect grid & sample average color per cell
      4) detect background colour candidate (metadata only)
      5) palette matching (DMC / Gamma / Anchor / auto)
      6) local per-cell consolidation (no global collapse)
      7) assign symbols
      8) build Pattern model
    """
    # 1) ROI detection on original image to keep geometry intact
    roi_result = detect_pattern_roi(image, pattern_mode=pattern_mode)
    roi = roi_result.roi

    # 2) adaptive resize & grid detection inside ROI
    roi_for_grid = resize_for_detail(roi, detail_level=detail_level)
    grid_dict = detect_and_rectify_grid(
        roi_for_grid,
        detail_level=detail_level,
        pattern_mode=pattern_mode,
    )
    grid_roi = grid_dict.get("roi")
    if grid_roi is None or getattr(grid_roi, "size", 0) == 0:
        grid_roi = roi
    if grid_roi.size == 0:
        logger.warning("Grid ROI is empty, falling back to original ROI crop")
        grid_roi = roi

    # 3) sample average colors per logical cell
    sampling_method: Literal["mean", "centroid"]
    sampling_method = "centroid" if pattern_mode == "symbol" else "mean"
    colors: Dict[Tuple[int, int], Tuple[int, int, int]] = split_into_cells_and_average(
        grid_roi,
        grid_dict,
        method=sampling_method,
        pattern_mode=pattern_mode,
    )
    if not colors:
        logger.warning("No cells detected after sampling; using fallback ROI averaging")
        grid_dict["width"] = grid_dict.get("width") or 1
        grid_dict["height"] = grid_dict.get("height") or 1
        h, w = grid_roi.shape[:2]
        colors[(0, 0)] = tuple(int(c) for c in grid_roi.reshape(-1, 3).mean(axis=0))

    # 4) background detection (metadata only – nothing is dropped automatically)
    background_quant = _detect_background_color(colors)

    # 5) palette matching
    resolved_brand = "DMC" if brand == "auto" else brand
    palette = load_palette(resolved_brand)
    kd = build_kd(palette)

    used_palette: Dict[Tuple[str, str], Dict] = {}
    counts: Counter[Tuple[str, str]] = Counter()
    stitches_raw: List[Dict[str, object]] = []

    for (x, y), rgb in colors.items():
        matched = nearest_color(kd, rgb, palette)
        key = (matched["brand"], matched["code"])
        used_palette[key] = matched
        counts[key] += 1
        stitches_raw.append({"x": int(x), "y": int(y), "key": key})

    # Build thread map without collapsing “minor” colours globally.
    thread_map: Dict[Tuple[str, str], ThreadRef] = {}
    final_counts: Counter[Tuple[str, str]] = Counter()

    for key, entry in used_palette.items():
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
    for _brand_code, thread in thread_map.items():
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
    width_cells = int(grid_dict.get("width") or 0) or (
        max((s.x for s in final_stitches), default=-1) + 1
    )
    height_cells = int(grid_dict.get("height") or 0) or (
        max((s.y for s in final_stitches), default=-1) + 1
    )

    if not final_stitches:
        logger.warning("Pipeline produced no stitches; falling back to single grayscale cell")
        fallback_thread = ThreadRef(
            brand=resolved_brand,  # type: ignore[arg-type]
            code="000",
            name="Fallback",
            rgb=(30, 30, 30),
            symbol="■",
        )
        new_thread_map = {(fallback_thread.brand, fallback_thread.code): fallback_thread}
        final_stitches = [Stitch(x=0, y=0, thread=fallback_thread)]
        width_cells = height_cells = 1
        final_counts = Counter({(fallback_thread.brand, fallback_thread.code): 1})

    total_stitches = int(sum(final_counts.values()))

    meta: Dict[str, object] = {
        "title": "Generated Pattern",
        "dpi": 300,
        "brand": resolved_brand,
        "palette_size": len(new_thread_map),
        "total_stitches": total_stitches,
        "detail_level": detail_level,
        "min_cells_per_color": min_cells,
        "pattern_mode": pattern_mode,
        "roi_confidence": getattr(roi_result, "confidence", 0.0),
        "grid_confidence": grid_dict.get("confidence"),
    }
    if background_quant:
        meta["background_candidate"] = {
            "rgb": tuple(int(c) for c in background_quant),
            "note": (
                "Detected via a frequency heuristic only. We keep this colour available because "
                "antique charts can rely on the background hue."
            ),
        }

    pattern = Pattern(
        canvasGrid=CanvasGrid(
            width=int(width_cells),
            height=int(height_cells),
        ),
        palette=list(new_thread_map.values()),
        stitches=final_stitches,
        meta=meta,
    )

    # precompute legend and build a sanity snapshot
    pattern.meta["legend"] = build_legend(pattern, force=True)
    roi_h, roi_w = grid_roi.shape[:2] if grid_roi.size else roi.shape[:2]
    pattern.meta["sanity"] = _build_sanity_report(
        palette_size=len(pattern.palette),
        grid_size=(pattern.canvasGrid.width, pattern.canvasGrid.height),
        roi_size=(roi_w, roi_h),
        filled_cells=len(colors),
        color_spread=_estimate_color_spread(colors),
    )
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
            x = int(s.x)
            y = int(s.y)
            thread = s.thread
            brand = thread.brand
            code = thread.code

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
            thickness = 2 if x % 10 == 0 else 1
            img[:, px : min(img.shape[1], px + thickness)] = color

    # horizontal
    for y in range(h + 1):
        py = y * cell
        if 0 <= py < img.shape[0]:
            color = accent_grid if y % 10 == 0 else base_grid
            thickness = 2 if y % 10 == 0 else 1
            img[py : min(img.shape[0], py + thickness), :] = color

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
                x = int(s.x)
                y = int(s.y)
                thread = s.thread
                symbol = thread.symbol

            if not symbol:
                continue

            text = str(symbol)
            cx = x * cell + cell // 2
            cy = y * cell + cell // 2
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                draw.text((cx - tw / 2, cy - th / 2), text, fill=(0, 0, 0), font=font)
            except Exception:
                draw.text((cx, cy), text, fill=(0, 0, 0), anchor="mm", font=font)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def render_pattern_image(pattern: Union[dict, Pattern], with_symbols: bool = True) -> bytes:
    """
    Convenience wrapper used by exporters: returns PNG bytes.
    """
    mode = "symbols" if with_symbols else "color"
    return render_preview(pattern, mode=mode)
