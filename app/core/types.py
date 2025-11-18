"""Common lightweight type aliases used across the pipeline."""

from typing import Literal

PatternMode = Literal["color", "symbol", "mixed"]
DetailLevel = Literal["low", "medium", "high"]

__all__ = ["PatternMode", "DetailLevel"]
