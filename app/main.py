from fastapi import FastAPI
from .api.jobs import router as jobs_router
from .api.admin import router as admin_router

app = FastAPI(title="Stitch Converter Service")

app.include_router(jobs_router, prefix="/api/v1", tags=["jobs"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])

@app.get("/health")
def health():
    return {"status": "ok"}
