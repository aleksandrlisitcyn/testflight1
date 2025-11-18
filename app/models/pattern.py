from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, Field


class CanvasGrid(BaseModel):
    width: int
    height: int


class ThreadRef(BaseModel):
    brand: Literal["DMC", "Gamma", "Anchor"]
    code: str
    name: str | None = None
    rgb: Tuple[int, int, int] | None = None
    symbol: str | None = None


class Stitch(BaseModel):
    x: int
    y: int
    thread: ThreadRef


class Pattern(BaseModel):
    canvasGrid: CanvasGrid
    palette: List[ThreadRef]
    stitches: List[Stitch]
    meta: dict[str, Any] = Field(default_factory=dict)
