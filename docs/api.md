# API

The service exposes both HTTP endpoints (FastAPI) and a CLI/bench helper.

## HTTP endpoints

### `POST /api/v1/jobs`

Create a new conversion job.

Multipart fields:

- `file` – required JPEG/PNG.
- Query/Form params:
  - `brand`: one of `DMC`, `Gamma`, `Anchor`, `auto` (default `DMC`).
  - `min_cells_per_color`: integer ≥ 1 (stored as metadata; colours are not filtered).
  - `detail_level`: `low|medium|high` (accepted for compatibility, currently neutral).
  - `pattern_mode`: `color|symbol|mixed` (controls ROI/grid/sample heuristics).

Response: `{"job_id": "...", "status": "done"}` once processing completes.

### `GET /api/v1/jobs/{job_id}`

Returns `JobStatus` with progress, stored metadata and grid dimensions when ready.

### `GET /api/v1/jobs/{job_id}/preview?mode=color|symbols`

Streams a PNG preview (with optional symbol overlay) generated from the stored pattern.

### `GET /api/v1/jobs/{job_id}/legend`

Returns the palette legend (brand/code/name/symbol). Useful for custom exports.

### `GET /data/{job_id}/pattern.(pdf|saga|json)`

Static files served directly after generation. The FastAPI app mounts `/data` so files are reachable
via HTTP. This is what the UI “Download PDF/SAGA” buttons call.

### `POST /upload`

Helper endpoint used by the built-in UI. It accepts the same form fields as `/api/v1/jobs`, forwards
them to the API and re-renders `templates/index.html` with preview/download links.

## CLI / scripts

### `scripts/run_bench.py`

```
python scripts/run_bench.py \
    --cases data/bench/cases \
    --output data/bench/bench.json \
    --results data/bench/results \
    --pattern-mode color
```

Runs the pipeline on a folder of PNG cases, renders previews and records simple stats. The script is
useful for regression testing and for validating that palette sizes stay >1 on multi-colour inputs.

## Pattern metadata

Every pattern exposes informative metadata in `pattern.meta`:

- `pattern_mode`, `detail_level`, `min_cells_per_color`
- `background_candidate` (colour recorded, never removed automatically)
- `legend` (pre-computed palette legend)
- `sanity` (status, issues, metrics)

Downstream consumers should rely on this metadata to detect suspicious outputs instead of re-running
heuristics themselves.
