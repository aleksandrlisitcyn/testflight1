from __future__ import annotations

from typing import Optional

from ..settings import STORAGE_BACKEND
from .fs_storage import FSStorage

try:
    from .s3_storage import S3Storage  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    S3Storage = None  # type: ignore


_storage_instance: Optional[object] = None


def get_storage():
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    backend = STORAGE_BACKEND.lower()
    if backend == "s3" and S3Storage is not None:
        _storage_instance = S3Storage()
    else:
        _storage_instance = FSStorage()
    return _storage_instance
