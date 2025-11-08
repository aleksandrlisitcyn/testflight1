from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal, Any

class JobStatus(BaseModel):
    status: Literal["done", "processing", "failed"]
    progress: float
    meta: dict[str, Any] = Field(default_factory=dict)
    grid: Optional[dict] = None

class ExportRequest(BaseModel):
    formats: List[Literal["saga","pdf","json","csv","xsd","xsp","css","dize"]] = ["saga","pdf"]


class MetaUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = None
    description: Optional[str] = None
    brand: Optional[str] = None
    author: Optional[str] = None
    notes: Optional[str] = None


class LegendEntryPatch(BaseModel):
    brand: str
    code: str
    name: Optional[str] = None
    symbol: Optional[str] = None


class LegendUpdateRequest(BaseModel):
    entries: List[LegendEntryPatch] = Field(..., min_length=1)
