from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.jobs import router as jobs_router
from .api.admin import router as admin_router
from fastapi.staticfiles import StaticFiles
from .settings import DATA_DIR

app = FastAPI(title="Stitch Converter Service")

# Монтируем локальную папку с результатами
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# ✅ Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # при желании можно ограничить: ["http://127.0.0.1:8000", "http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Подключаем роутеры
app.include_router(jobs_router, prefix="/api/v1", tags=["jobs"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])

# ✅ Простой эндпоинт для проверки
@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request, UploadFile, Form
from fastapi import File
import httpx

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    brand: str = Form("DMC"),
    min_cells_per_color: int = Form(30),
):
    files = {"file": (file.filename, await file.read(), file.content_type)}
    params = {"brand": brand, "min_cells_per_color": min_cells_per_color}

    # 🔧 формируем URL динамически (работает и в Codespaces, и локально)
    base_url = str(request.base_url).rstrip("/")
    api_url = f"{base_url}/api/v1/jobs"

    # ✅ используем асинхронный httpx вместо requests
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, files=files, params=params)

    job = response.json()
    job_id = job["job_id"]

    preview_url = f"/api/v1/jobs/{job_id}/preview?mode=color"
    # FSStorage сохраняет файлы в DATA_DIR/<job_id>/pattern.*, поэтому используем такой URL
    pdf_url = f"/data/{job_id}/pattern.pdf"
    saga_url = f"/data/{job_id}/pattern.saga"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "job": {"job_id": job_id, "status": "done"},
            "preview_url": preview_url,
            "pdf_url": pdf_url,
            "saga_url": saga_url,
        },
    )