import numpy as np

from app.core.jobs import store as job_store
from app.core.pipeline import process_image_to_pattern
from app.core.pattern_edit import apply_legend_updates, apply_meta_updates
from app.core.legend import build_legend


def _prepare_job() -> str:
    img = np.full((16, 16, 3), 180, dtype=np.uint8)
    pattern = process_image_to_pattern(img)
    job_id = "job-test-meta"
    job_store.create(job_id, status="done", progress=1.0)
    grid = {"width": pattern.canvasGrid.width, "height": pattern.canvasGrid.height}
    job_store.set_pattern(job_id, pattern.model_dump(), grid=grid)
    return job_id


def test_meta_and_legend_updates():
    job_id = _prepare_job()

    record = job_store.get(job_id)
    legend_initial = build_legend(record.pattern)
    assert legend_initial, "Legend should not be empty"
    first_entry = legend_initial[0]

    updates = {"title": "Custom Title", "notes": "Hand tuned"}
    job_store.update_pattern(job_id, lambda pattern: apply_meta_updates(pattern, updates))
    updated = job_store.get(job_id)
    assert updated.pattern["meta"]["title"] == "Custom Title"
    assert updated.pattern["meta"]["notes"] == "Hand tuned"

    payload = [{
        "brand": first_entry["brand"],
        "code": first_entry["code"],
        "symbol": "★",
        "name": "Updated Name",
    }]
    job_store.update_pattern(job_id, lambda pattern: apply_legend_updates(pattern, payload))
    updated_after_legend = job_store.get(job_id)
    updated_legend = build_legend(updated_after_legend.pattern)
    match = next(item for item in updated_legend if item["code"] == first_entry["code"])
    assert match["symbol"] == "★"
    assert match["name"] == "Updated Name"
