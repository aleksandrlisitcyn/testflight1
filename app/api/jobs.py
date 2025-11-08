from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from uuid import uuid4
from typing import Literal, Optional
import io
import numpy as np
from PIL import Image

from ..core.pipeline import process_image_to_pattern, render_preview
from ..core.jobs import store as job_store
from ..export.saga_exporter import export_saga
from ..export.pdf_exporter import export_pdf
from ..export.json_exporter import export_json
from ..export.csv_exporter import export_csv
from ..models.api_schemas import JobStatus, ExportRequest
from ..storage import get_storage

router = APIRouter()

@router.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    brand: Literal["DMC","Gamma","Anchor","auto"] = "DMC",
    min_cells_per_color: int = 30
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
        meta={"filename": file.filename, "brand": brand, "min_cells": min_cells_per_color},
    )

    # Synchronous pipeline for scaffold (replace with background tasks if needed)
    pattern = process_image_to_pattern(np.array(img), brand=brand, min_cells=min_cells_per_color)
    job_store.update(job_id, status="done", progress=1.0)
    grid = {"width": pattern.canvasGrid.width, "height": pattern.canvasGrid.height}
    job_store.set_pattern(job_id, pattern.dict(), grid=grid)

    # Store internal JSON
    storage = get_storage()
    storage.save_json(f"{job_id}/pattern.json", pattern.dict())

    return JSONResponse({"job_id": job_id, "status": "processing"})

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
    legend = []
    palette = job["pattern"]["palette"]
    # Count stitches per code
    counts = {}
    for s in job["pattern"]["stitches"]:
        key = (s["thread"]["brand"], s["thread"]["code"])
        counts[key] = counts.get(key, 0) + 1
    for p in palette:
        key = (p["brand"], p["code"])
        legend.append({
            "brand": p["brand"], "code": p["code"],
            "name": p.get("name"), "count": counts.get(key, 0)
        })
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
            links.append({"format":"saga","path":path})
        elif fmt == "pdf":
            pdf = export_pdf(pattern)
            path = f"{job_id}/pattern.pdf"
            storage.save_bytes(path, pdf)
            links.append({"format":"pdf","path":path})
        elif fmt == "json":
            data = export_json(pattern)
            path = f"{job_id}/pattern.json"
            storage.save_bytes(path, data.encode("utf-8"))
            links.append({"format":"json","path":path})
        elif fmt == "csv":
            data = export_csv(pattern)
            path = f"{job_id}/pattern.csv"
            storage.save_bytes(path, data.encode("utf-8"))
            links.append({"format":"csv","path":path})
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {fmt}")
    return {"files": links}
