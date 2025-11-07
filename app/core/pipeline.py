import numpy as np
from typing import Literal
from ..models.pattern import Pattern, CanvasGrid, ThreadRef, Stitch
from ..cv.grid_detector import detect_pattern_roi, detect_and_rectify_grid
from ..cv.cell_sampler import split_into_cells_and_average
from ..color.palette_loader import load_palette
from ..color.palette_matcher import build_kd, nearest_color
from .symbols import assign_symbols_to_palette
from PIL import Image
import io

def process_image_to_pattern(img: np.ndarray, brand: Literal["DMC","Gamma","Anchor","auto"]="DMC", min_cells: int = 30) -> Pattern:
    roi = detect_pattern_roi(img)
    grid_dict = detect_and_rectify_grid(roi)
    colors = split_into_cells_and_average(roi, grid_dict)

    palette = load_palette(brand if brand != "auto" else "DMC")
    kd = build_kd(palette)

    # Build palette map of used colors
    used_palette = {}
    stitches = []
    for (x, y), rgb in colors.items():
        p = nearest_color(kd, rgb, palette)
        key = (p["brand"], p["code"])
        used_palette[key] = p
        stitches.append(Stitch(x=int(x), y=int(y),
                               thread=ThreadRef(brand=p["brand"], code=p["code"], name=p.get("name"))))

    # Assign symbols to palette
    palette_list = [dict(v) for v in used_palette.values()]
    palette_list = assign_symbols_to_palette(palette_list)

    pattern = Pattern(
        canvasGrid=CanvasGrid(width=int(grid_dict["width"]), height=int(grid_dict["height"])),
        palette=[ThreadRef(**{**p, "rgb": tuple(p.get("rgb", (0,0,0)))}) for p in palette_list],
        stitches=stitches,
        meta={"title": "Generated Pattern", "dpi": 300}
    )
    return pattern

def render_preview(pattern: dict, mode: str = "color") -> bytes:
    w = pattern["canvasGrid"]["width"]
    h = pattern["canvasGrid"]["height"]
    cell = 8  # px per cell
    img = np.full((h*cell, w*cell, 3), 255, dtype=np.uint8)
    # brand/code -> rgb lookup (fallback gray)
    color_map = {}
    for p in pattern["palette"]:
        rgb = p.get("rgb") or (120,120,120)
        color_map[(p["brand"], p["code"])] = tuple(int(v) for v in rgb)
    for s in pattern["stitches"]:
        x, y = s["x"], s["y"]
        key = (s["thread"]["brand"], s["thread"]["code"])
        rgb = color_map.get(key, (120,120,120))
        img[y*cell:(y+1)*cell, x*cell:(x+1)*cell] = rgb
    # TODO: symbols overlay for mode=="symbols"
    pil = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
