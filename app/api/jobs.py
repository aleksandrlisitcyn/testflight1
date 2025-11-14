from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from uuid import uuid4
from typing import Literal
import io
import numpy as np
import math

from PIL import Image as PILImage, Image

from ..core.pipeline import process_image_to_pattern, render_preview
from ..core.legend import build_legend
from ..core.pattern_edit import apply_legend_updates, apply_meta_updates
from ..core.jobs import store as job_store

from ..export.saga_exporter import export_saga
from ..export.pdf_exporter import export_pdf
from ..export.json_exporter import export_json
from ..export.csv_exporter import export_csv
from ..export.xsd_exporter import export_xsd
from ..export.xsp_exporter import export_xsp
from ..export.css_exporter import export_css
from ..export.dize_exporter import export_dize
from ..export.png_exporter import export_png

from ..models.api_schemas import (
    ExportRequest,
    JobStatus,
    LegendUpdateRequest,
    MetaUpdateRequest,
)

from ..storage import get_storage

router = APIRouter()

# =====================================================================
#   JOB CREATE
# =====================================================================

@router.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    brand: Literal["DMC", "Gamma", "Anchor", "auto"] = "DMC",
    min_cells_per_color: int = 30,
    detail_level: Literal["low", "medium", "high"] = "medium",
):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    job_id = str(uuid4())

    job_store.create(
        job_id,
        status="processing",
        progress=0.2,
        meta={
            "filename": file.filename,
            "brand": brand,
            "min_cells": min_cells_per_color,
            "detail_level": detail_level,
        },
    )

    # ============================================
    # PROCESS PATTERN
    # ============================================
    pattern = process_image_to_pattern(
        np.array(img),
        brand=brand,
        min_cells=min_cells_per_color,
        detail_level=detail_level,
    )
    pattern_dict = pattern.dict()

    # ============================================
    # PREVIEW (safe, high-res)
    # ============================================
    preview_bytes = None
    try:
        preview_obj = render_preview(pattern, mode="color")

        if isinstance(preview_obj, (bytes, bytearray)):
            preview_img = PILImage.open(io.BytesIO(preview_obj)).convert("RGBA")
        elif isinstance(preview_obj, np.ndarray):
            preview_img = PILImage.fromarray(preview_obj).convert("RGBA")
        elif isinstance(preview_obj, PILImage.Image):
            preview_img = preview_obj.convert("RGBA")
        else:
            preview_img = None

        if preview_img is not None:
            canvas = pattern_dict["canvasGrid"]
            grid_w = int(canvas["width"])
            grid_h = int(canvas["height"])

            CELL_PX = 16
            max_total_w = 1400
            target_w = min(max_total_w, grid_w * CELL_PX)
            target_h = min(2000, grid_h * CELL_PX)

            if preview_img.width < target_w:
                scale = math.ceil(target_w / preview_img.width)
                preview_img = preview_img.resize(
                    (preview_img.width * scale, preview_img.height * scale),
                    resample=PILImage.NEAREST,
                )

            preview_img.thumbnail((max_total_w, target_h), resample=PILImage.NEAREST)

            buf = io.BytesIO()
            preview_img.save(buf, format="PNG")
            preview_bytes = buf.getvalue()

    except Exception as e:
        print(f"[WARN] preview generation failed: {e}")

    # ============================================
    # SAVE PREVIEW & PATTERN
    # ============================================
    storage = get_storage()

    storage.save_json(f"{job_id}/pattern.json", pattern_dict)

    if preview_bytes:
        storage.save_bytes(f"{job_id}/preview.png", preview_bytes)

    # Update job store
    job_store.update(job_id, status="done", progress=1.0)
    grid = {
        "width": pattern.canvasGrid.width,
        "height": pattern.canvasGrid.height,
    }
    job_store.set_pattern(job_id, pattern_dict, grid=grid)

    # ============================================
    # AUTO-EXPORT PDF + SAGA
    # ============================================
    try:
        pdf_bytes = export_pdf(pattern_dict, preview=preview_bytes)
        storage.save_bytes(f"{job_id}/pattern.pdf", pdf_bytes)

        saga_data = export_saga(pattern_dict)
        saga_bytes = saga_data.encode("utf-8") if isinstance(saga_data, str) else saga_data
        storage.save_bytes(f"{job_id}/pattern.saga", saga_bytes)

    except Exception as e:
        print(f"[WARN] Export failed: {e}")

    return JSONResponse({"job_id": job_id, "status": "done"})


# =====================================================================
#   JOB GET
# =====================================================================

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    record = job_store.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    job = record.to_dict(include_pattern=True)
    return JobStatus(**{
        "status": job["status"],
        "progress": job["progress"],
        "meta": job.get("meta", {}),
        "grid": job.get("grid"),
    })


# =====================================================================
#   LEGEND
# =====================================================================

@router.get("/jobs/{job_id}/legend")
async def get_legend(job_id: str):
    record = job_store.get(job_id)
    if not record or record.status != "done" or not record.pattern:
        raise HTTPException(status_code=404, detail="Job not ready")

    legend = build_legend(record.pattern)
    return legend


# =====================================================================
#   PREVIEW
# =====================================================================

@router.get("/jobs/{job_id}/preview")
async def preview(job_id: str, mode: Literal["color", "symbols"] = "color"):
    record = job_store.get(job_id)
    if not record or record.status != "done":
        raise HTTPException(status_code=404, detail="Job not ready")

    try:
        img_bytes = render_preview(record.pattern, mode=mode)
        if isinstance(img_bytes, np.ndarray):
            buf = io.BytesIO()
            PILImage.fromarray(img_bytes).save(buf, format="PNG")
            img_bytes = buf.getvalue()
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        print(f"[WARN] preview failed: {e}")
        raise HTTPException(status_code=500, detail="Preview render failed")