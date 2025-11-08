from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from uuid import uuid4
from typing import Literal, Optional
import io
import numpy as np
from PIL import Image

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
from ..models.api_schemas import (
    ExportRequest,
    JobStatus,
    LegendUpdateRequest,
    MetaUpdateRequest,
)
from ..storage import get_storage

router = APIRouter()

@router.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    brand: Literal["DMC","Gamma","Anchor","auto"] = "DMC",
    min_cells_per_color: int = 30,
    detail_level: Literal["low", "medium", "high"] = "medium",
):
    if file.content_type not in {"image/jpeg","image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    if max(img.size) > 3000:
        raise HTTPException(status_code=400, detail="Image too large (max side 3000px)")

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

    # Synchronous pipeline for scaffold (replace with background tasks if needed)
    pattern = process_image_to_pattern(
        np.array(img),
        brand=brand,
        min_cells=min_cells_per_color,
        detail_level=detail_level,
    )
    job_store.update(job_id, status="done", progress=1.0)
    grid = {"width": pattern.canvasGrid.width, "height": pattern.canvasGrid.height}
    job_store.set_pattern(job_id, pattern.dict(), grid=grid)

    # Store internal JSON
    storage = get_storage()
    storage.save_json(f"{job_id}/pattern.json", pattern.dict())

    record = job_store.get(job_id)
    status_payload = {
        "job_id": job_id,
        "status": record.status if record else "processing",
        "progress": record.progress if record else 0.2,
    }
    if record and record.meta:
        status_payload["meta"] = record.meta
    return JSONResponse(status_payload)

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
        "grid": job.get("grid")
    })

@router.get("/jobs/{job_id}/legend")
async def get_legend(job_id: str):
    record = job_store.get(job_id)
    if not record or record.status != "done" or not record.pattern:
        raise HTTPException(status_code=404, detail="Job not ready")
    job = record.to_dict()
    legend = build_legend(job["pattern"])
    return legend

@router.get("/jobs/{job_id}/preview")
async def preview(job_id: str, mode: Literal["color","symbols"] = "color"):
    record = job_store.get(job_id)
    if not record or record.status != "done" or not record.pattern:
        raise HTTPException(status_code=404, detail="Job not ready")
    img_bytes = render_preview(record.pattern, mode=mode)
    return Response(content=img_bytes, media_type="image/png")

@router.post("/jobs/{job_id}/export")
async def export(job_id: str, req: ExportRequest):
    record = job_store.get(job_id)
    if not record or record.status != "done" or not record.pattern:
        raise HTTPException(status_code=404, detail="Job not ready")
    pattern = record.pattern
    storage = get_storage()

    links = []
    for fmt in req.formats:
        if fmt == "saga":
            data = export_saga(pattern)
            path = f"{job_id}/pattern.saga"
            storage.save_bytes(path, data.encode("utf-8"))
            links.append({"format": "saga", "path": path})
        elif fmt == "pdf":
            pdf = export_pdf(pattern)
            path = f"{job_id}/pattern.pdf"
            storage.save_bytes(path, pdf)
            links.append({"format": "pdf", "path": path})
        elif fmt == "json":
            data = export_json(pattern)
            path = f"{job_id}/pattern.json"
            storage.save_bytes(path, data.encode("utf-8"))
            links.append({"format": "json", "path": path})
        elif fmt == "csv":
            data = export_csv(pattern)
            path = f"{job_id}/pattern.csv"
            storage.save_bytes(path, data.encode("utf-8"))
            links.append({"format": "csv", "path": path})
        elif fmt == "xsd":
            data = export_xsd(pattern)
            path = f"{job_id}/pattern.xsd"
            storage.save_bytes(path, data.encode("utf-8"))
            links.append({"format": "xsd", "path": path})
        elif fmt == "xsp":
            data = export_xsp(pattern)
            path = f"{job_id}/pattern.xsp"
            storage.save_bytes(path, data.encode("utf-8"))
            links.append({"format": "xsp", "path": path})
        elif fmt == "css":
            data = export_css(pattern)
            path = f"{job_id}/pattern.css"
            storage.save_bytes(path, data.encode("utf-8"))
            links.append({"format": "css", "path": path})
        elif fmt == "dize":
            payload = export_dize(pattern)
            path = f"{job_id}/pattern.dize"
            storage.save_bytes(path, payload)
            links.append({"format": "dize", "path": path})
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {fmt}")
    return {"files": links}


@router.patch("/jobs/{job_id}/meta")
async def update_meta(job_id: str, payload: MetaUpdateRequest):
    record = job_store.get(job_id)
    if not record or record.pattern is None:
        raise HTTPException(status_code=404, detail="Job not ready")

    updates = {k: v for k, v in payload.dict(exclude_unset=True).items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No meta fields provided")

    def apply(pattern: dict) -> dict:
        return apply_meta_updates(pattern, updates)

    updated_pattern = job_store.update_pattern(job_id, apply)
    if updated_pattern is None:
        raise HTTPException(status_code=404, detail="Job not ready")

    combined_meta = {**(record.meta or {}), **updates}
    job_store.update(job_id, meta=combined_meta)
    return {"meta": updated_pattern.get("meta", {})}


@router.patch("/jobs/{job_id}/legend")
async def update_legend(job_id: str, payload: LegendUpdateRequest):
    record = job_store.get(job_id)
    if not record or record.status != "done" or record.pattern is None:
        raise HTTPException(status_code=404, detail="Job not ready")

    updates = [entry.model_dump(exclude_none=True) for entry in payload.entries]

    def apply(pattern: dict) -> dict:
        return apply_legend_updates(pattern, updates)

    updated_pattern = job_store.update_pattern(job_id, apply)
    if updated_pattern is None:
        raise HTTPException(status_code=404, detail="Job not ready")

    legend = build_legend(updated_pattern)
    return {"legend": legend}
