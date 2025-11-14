import io
from typing import Literal, Tuple, Dict, List, Union
from collections import Counter
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
#  RESIZE / DETAIL
# =====================================================================

def resize_for_detail(image: np.ndarray, detail_level: Literal["low", "medium", "high"]) -> np.ndarray:
    """Adaptive resize: reduce large images, stabilize contrast & noise."""
    import cv2

    h, w = image.shape[:2]
    max_side = max(h, w) or 1

    if detail_level == "low":
        target_cells = min(max_side // 8, 250)
        alpha, beta = 1.05, 5
    elif detail_level == "high":
        target_cells = min(max_side // 2, 800)
        alpha, beta = 1.2, 12
    else:
        target_cells = min(max_side // 4, 450)
        alpha, beta = 1.12, 8

    scale = target_cells / float(max_side)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def _color_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(a, float) - np.array(b, float)))


# =====================================================================
#  PIPELINE: image → Pattern
# =====================================================================

def process_image_to_pattern(
    image: np.ndarray,
    brand: Literal["DMC", "Gamma", "Anchor", "auto"] = "DMC",
    min_cells: int = 30,
    detail_level: Literal["low", "medium", "high"] = "medium",
) -> Pattern:

    # 1) pre-process
    img_proc = resize_for_detail(image, detail_level)

    # 2) detect region of interest and grid
    roi = detect_pattern_roi(img_proc)
    grid = detect_and_rectify_grid(roi)

    # 3) sample colors for each cell
    colors: Dict[Tuple[int, int], Tuple[int, int, int]] = split_into_cells_and_average(roi, grid)

    # fallback if grid not detected or no colors extracted
    if not colors:
        mean_rgb = tuple(int(v) for v in roi.reshape(-1, 3).mean(axis=0)) if roi.size else (0, 0, 0)
        colors = {(0, 0): mean_rgb}
        grid["width"] = grid.get("width") or 1
        grid["height"] = grid.get("height") or 1

    # 4) background detection
    quant_counter = Counter()
    for _, rgb in colors.items():
        q = tuple((int(c) // 16) * 16 for c in rgb)
        quant_counter[q] += 1

    background_quant = quant_counter.most_common(1)[0][0] if quant_counter else None

    # 5) palette and KD-tree
    resolved_brand = "DMC" if brand == "auto" else brand
    palette = load_palette(resolved_brand)
    kd = build_kd(palette)

    used_palette: Dict[Tuple[str, str], Dict] = {}
    counts: Counter[Tuple[str, str]] = Counter()
    stitches_raw: List[Dict] = []

    for (x, y), rgb in colors.items():

        # Skip canvas color
        if background_quant:
            q = tuple((int(c) // 16) * 16 for c in rgb)
            if q == background_quant:
                continue

        matched = nearest_color(kd, rgb, palette)
        key = (matched["brand"], matched["code"])
        used_palette[key] = matched
        counts[key] += 1
        stitches_raw.append({"x": x, "y": y, "key": key})

    # 6) min_cells filtering
    if counts and min_cells > 1:
        major = {k for k, c in counts.items() if c >= min_cells}
        if not major:
            major = set(counts.keys())
    else:
        major = set(counts.keys())

    # 7) thread_map
    thread_map: Dict[Tuple[str, str], ThreadRef] = {}
    final_counts: Counter = Counter()

    for key, entry in used_palette.items():
        if key not in major:
            continue
        rgb = tuple(entry.get("rgb") or entry.get("color") or (0, 0, 0))
        thread_map[key] = ThreadRef(
            brand=entry["brand"],
            code=entry["code"],
            name=entry.get("name"),
            rgb=tuple(int(c) for c in rgb),
            symbol=None,
        )
        final_counts[key] = counts[key]

    # build stitches
    stitches: List[Stitch] = []
    for s in stitches_raw:
        t = thread_map.get(s["key"])
        if t:
            stitches.append(Stitch(x=s["x"], y=s["y"], thread=t))

    # 8) assign symbols
    palette_list = []
    for key, t in thread_map.items():
        palette_list.append(
            {
                "brand": t.brand,
                "code": t.code,
                "name": t.name,
                "rgb": t.rgb,
            }
        )

    palette_list = assign_symbols_to_palette(palette_list)

    # rebuild thread_map with assigned symbols
    new_thread_map: Dict[Tuple[str, str], ThreadRef] = {}
    for p in palette_list:
        new_thread_map[(p["brand"], p["code"])] = ThreadRef(
            brand=p["brand"],
            code=p["code"],
            name=p.get("name"),
            rgb=tuple(int(c) for c in p["rgb"]),
            symbol=p.get("symbol"),
        )

    # reassign threads in stitches
    final_stitches: List[Stitch] = []
    for s in stitches:
        key = (s.thread.brand, s.thread.code)
        thread = new_thread_map.get(key)
        if thread:
            final_stitches.append(Stitch(x=s.x, y=s.y, thread=thread))

    # 9) build final Pattern object
    width_cells = int(grid.get("width") or 0) or max((s.x for s in final_stitches), default=-1) + 1
    height_cells = int(grid.get("height") or 0) or max((s.y for s in final_stitches), default=-1) + 1

    pattern = Pattern(
        canvasGrid=CanvasGrid(width=width_cells, height=height_cells),
        palette=list(new_thread_map.values()),
        stitches=final_stitches,
        meta={
            "title": "Generated Pattern",
            "dpi": 300,
            "brand": resolved_brand,
            "palette_size": len(new_thread_map),
            "total_stitches": int(sum(final_counts.values())),
            "detail_level": detail_level,
        },
    )

    # generate legend
    pattern.meta["legend"] = build_legend(pattern, force=True)

    return pattern


# =====================================================================
#  PREVIEW (PNG)
# =====================================================================

def render_preview(pattern: Union[dict, Pattern], mode: str = "color") -> bytes:
    """Render PNG preview: color blocks + optional symbol overlay."""
    if hasattr(pattern, "model_dump"):
        p = pattern.model_dump()
    elif hasattr(pattern, "dict"):
        p = pattern.dict()
    else:
        p = pattern

    w = int(p["canvasGrid"]["width"])
    h = int(p["canvasGrid"]["height"])
    cell = 20

    # build palette map
    color_map = {}
    symbol_map = {}

    for entry in p["palette"]:
        brand = entry.get("brand")
        code = entry.get("code")
        rgb = entry.get("rgb", (200, 200, 200))
        color_map[(brand, code)] = tuple(int(c) for c in rgb)
        if entry.get("symbol"):
            symbol_map[(brand, code)] = entry["symbol"]

    img = np.full((h * cell, w * cell, 3), 255, dtype=np.uint8)

    # draw stitches
    for s in p["stitches"]:
        x, y = int(s["x"]), int(s["y"])
        th = s["thread"]
        key = (th["brand"], th["code"])
        rgb = color_map.get(key, (200, 200, 200))

        x0, y0 = x * cell, y * cell
        x1, y1 = x0 + cell, y0 + cell
        img[y0:y1, x0:x1] = rgb

    # grid
    base_grid = (210, 210, 210)
    accent_grid = (120, 120, 120)

    for x in range(w + 1):
        color = accent_grid if x % 10 == 0 else base_grid
        img[:, x * cell:x * cell + 1] = color
    for y in range(h + 1):
        color = accent_grid if y % 10 == 0 else base_grid
        img[y * cell:y * cell + 1, :] = color

    # symbols
    pil = Image.fromarray(img, "RGB")
    draw = ImageDraw.Draw(pil)

    if mode == "symbols":
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, cell - 6))
        except Exception:
            font = ImageFont.load_default()

        for s in p["stitches"]:
            x, y = int(s["x"]), int(s["y"])
            th = s["thread"]
            sym = symbol_map.get((th["brand"], th["code"]))
            if not sym:
                continue
            cx = x * cell + cell // 2
            cy = y * cell + cell // 2
            draw.text((cx, cy), sym, fill=(0, 0, 0), anchor="mm", font=font)

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def render_pattern_image(pattern: Union[dict, Pattern], with_symbols: bool = True) -> bytes:
    return render_preview(pattern, "symbols" if with_symbols else "color")