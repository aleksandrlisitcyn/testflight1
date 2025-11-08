from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.jobs import router as jobs_router
from .api.admin import router as admin_router

app = FastAPI(title="Stitch Converter Service")

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
