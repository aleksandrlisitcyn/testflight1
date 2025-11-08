import numpy as np

def split_into_cells_and_average(img: np.ndarray, grid: dict) -> dict:
    h, w = img.shape[:2]
    gw, gh = grid['width'], grid['height']
    cell_w = max(1, int(round(w / gw)))
    cell_h = max(1, int(round(h / gh)))
    colors = {}
    for y in range(gh):
        for x in range(gw):
            xs, ys = x*cell_w, y*cell_h
            xe, ye = min(w, xs+cell_w), min(h, ys+cell_h)
            patch = img[ys:ye, xs:xe]
            if patch.size == 0:
                continue
            rgb = patch.reshape(-1, 3).mean(axis=0).astype(int).tolist()
            # simple "empty" detection: skip near-white patches
            if sum(rgb) > 720:  # ~ near white
                continue
            colors[(x, y)] = rgb
    return colors
