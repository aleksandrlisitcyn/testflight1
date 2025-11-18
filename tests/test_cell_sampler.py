from __future__ import annotations

import numpy as np

from app.cv.cell_sampler import split_into_cells_and_average


def test_centroid_sampling_focuses_center():
    img = np.full((20, 20, 3), 240, dtype=np.uint8)
    img[8:12, 8:12] = (10, 10, 200)
    grid = {"width": 1, "height": 1}

    mean_color = split_into_cells_and_average(img, grid, method="mean")[(0, 0)]
    centroid_color = split_into_cells_and_average(img, grid, method="centroid")[(0, 0)]

    assert centroid_color[2] < mean_color[2]
    assert centroid_color[0] <= mean_color[0]
