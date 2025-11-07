from pydantic import BaseModel
from typing import Optional, List, Literal, Any

class JobStatus(BaseModel):
    status: Literal["done","processing","failed"]
    progress: float
    meta: dict[str, Any] = {}
    grid: Optional[dict] = None

class ExportRequest(BaseModel):
    formats: List[Literal["saga","pdf","json","csv","xsd","xsp","css","dize"]] = ["saga","pdf"]
