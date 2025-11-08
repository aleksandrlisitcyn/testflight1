from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..core.jobs import store as job_store

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "admin" / "templates"))


def _datetimeformat(value: float | int | None) -> str:
    if not value:
        return "—"
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


templates.env.filters["datetimeformat"] = _datetimeformat


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, status: Optional[str] = None, query: Optional[str] = None):
    records = job_store.list(status=status, query=query)
    jobs = [r.to_dict() for r in records]
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "jobs": jobs,
            "status": status or "",
            "query": query or "",
        },
    )


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None, query: Optional[str] = None):
    records = job_store.list(status=status, query=query)
    return {"items": [r.to_dict() for r in records], "total": len(records)}
