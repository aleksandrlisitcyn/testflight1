import numpy as np
from sklearn.neighbors import KDTree


def build_kd(palette: list[dict]) -> KDTree:
    arr = np.array([p["rgb"] for p in palette], dtype=float)
    return KDTree(arr)


def nearest_color(kd: KDTree, rgb: list[int], palette: list[dict]) -> dict:
    dist, ind = kd.query([rgb], k=1)
    return palette[int(ind[0][0])]
