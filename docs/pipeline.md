# Pipeline

This document outlines the recognition pipeline that powers the Stitch Converter backend. The guiding
principle is faithful “as-is” digitisation of antique charts. We prefer conservative heuristics and
record potential issues instead of silently altering the result.

## 1. ROI detection

`detect_pattern_roi` isolates the region that contains the stitched grid. We combine CLAHE contrast
enhancement, adaptive thresholding, Canny edges and conservative morphological closing. The code
accepts `pattern_mode` so symbol-heavy charts (`symbol`) keep thinner strokes, while colour blocks
(`color`) apply a bit more smoothing. When OpenCV is missing we fall back to a light intensity-based
crop.

## 2. Grid detection and alignment

The cropped ROI passes to `detect_and_rectify_grid`, which projects Sobel gradients onto the axes,
estimates the number of cells and derives cell size. `detail_level` stays in the function signature
for compatibility but is intentionally ignored to keep recognition deterministic. Only
`pattern_mode` affects the blur kernel (e.g. narrower kernels for symbol scans). Once we know the
grid, we rectify the ROI and propagate the grid confidence score downstream.

## 3. Per-cell sampling & local consolidation

`split_into_cells_and_average` walks each logical cell, extracts either the full patch (`mean`) or a
centre patch (`centroid`) and consolidates colours locally. We use trimmed means/medians to mop up
tiny paper artefacts without touching neighbouring cells. `pattern_mode` selects the consolidation
strategy:

- `color` – trimmed mean with outlier rejection.
- `symbol` – median-biased sampling to keep ink strokes on top of paper.
- `mixed` – blend mean and median so we keep both colour blocks and drawn details.

No global “minor to major” merging happens: every cell keeps the closest thread colour independently.

## 4. Palette building

Sampled colours go through the KD-tree palette matcher for the requested thread brand (or `auto`
which currently falls back to DMC). We store every matched colour, assign symbols and keep the palette
size untouched. `min_cells_per_color` remains part of the API but does not remove colours in this
iteration – it is meta-information describing what the user requested.

## 5. Background handling

We still detect a background candidate via coarse quantised histograms, but this value is metadata
only. Background hues are kept in the palette because the dominant colour is often part of the design.
Future work may add a structure-aware remover, but not in the current “as-is” pipeline.

## 6. Pattern modes

`PatternMode = Literal["color", "symbol", "mixed"]` flows through ROI detection, grid detection,
sampling and metadata. Behaviour today:

- `color`: prioritise colour stability, treat dust/dark speckles as noise.
- `symbol`: preserve thin lines and darker strokes, bias sampling to median values.
- `mixed`: balance both, keeping small dark details without over-amplifying noise.

All modes keep the same exported structure; the difference is just how we sample each cell.

## 7. Sanity pass

Before returning the pattern we compute a sanity report with palette size, ROI/grid metrics and fill
ratios. Suspicious outcomes (e.g. palette collapsed to 1 colour while spread suggests otherwise, or a
huge ROI with only a handful of grid cells) are flagged via `pattern.meta["sanity"]`. The pipeline
never “fixes” such results silently – it simply records the issue so the caller can inspect and decide
on a re-run or manual intervention.

## Detail level

`detail_level` stays in every public signature for forwards compatibility, but as of this refactor it
does not change recognition semantics. A note in the code explains that it is reserved for future
output-only tweaks (e.g. preview scaling) so we keep the API stable while guaranteeing faithful
recognition of antique charts.
