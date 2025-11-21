# Overview

Stitch Converter is a lightweight service that digitises antique cross-stitch charts. Users upload
scans (JPEG / PNG), the backend reads the historical grid “as-is”, constructs a digital pattern and
exports it as PDF/SAGA along with previews.

High level pipeline:

1. **ROI detection** – crop just the embroidery chart while keeping the original perspective.
2. **Grid alignment** – estimate grid spacing, cell size and rectify the crop.
3. **Per-cell sampling** – average colours inside every stitch cell with robust local filtering.
4. **Palette building** – match sampled colours to the requested thread set without global merging.
5. **Export** – assign symbols, produce previews, PDF and SAGA packages.

The current focus is “as-is” digitisation: no aggressive palette merges or background stripping, so
that aged paper texture and subtle ornament colours remain intact. Future iterations may add
optional enhancement passes, but the default experience stays faithful to the source charts.
