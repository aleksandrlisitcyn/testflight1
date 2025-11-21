from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable, Dict, Iterable, Optional


@dataclass
class JobRecord:
    job_id: str
    status: str = "pending"
    progress: float = 0.0
    meta: dict = field(default_factory=dict)
    grid: Optional[dict] = None
    pattern: Optional[dict] = None
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self, include_pattern: bool = False) -> dict:
        data = {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "meta": copy.deepcopy(self.meta),
            "grid": copy.deepcopy(self.grid),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_pattern and self.pattern is not None:
            data["pattern"] = copy.deepcopy(self.pattern)
        return data


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def create(
        self,
        job_id: str,
        *,
        status: str = "processing",
        progress: float = 0.0,
        meta: Optional[dict] = None,
    ) -> JobRecord:
        record = JobRecord(job_id=job_id, status=status, progress=progress, meta=meta or {})
        with self._lock:
            self._jobs[job_id] = record
        return record

    def update(self, job_id: str, **fields) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            for key, value in fields.items():
                if hasattr(record, key):
                    setattr(record, key, value)
                else:
                    record.meta[key] = value
            record.updated_at = time.time()
            return record

    def set_pattern(
        self,
        job_id: str,
        pattern: dict,
        grid: Optional[dict] = None,
    ) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            record.pattern = copy.deepcopy(pattern)
            if grid is not None:
                record.grid = copy.deepcopy(grid)
            record.updated_at = time.time()
            return record

    def update_pattern(self, job_id: str, updater: Callable[[dict], dict]) -> Optional[dict]:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record or record.pattern is None:
                return None
            pattern = copy.deepcopy(record.pattern)
            pattern = updater(pattern)
            record.pattern = pattern
            record.updated_at = time.time()
            return copy.deepcopy(pattern)

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return None
            # Return a shallow copy to avoid accidental external mutation
            return copy.deepcopy(record)

    def list(
        self,
        *,
        status: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Iterable[JobRecord]:
        with self._lock:
            records = list(self._jobs.values())
        if status:
            records = [r for r in records if r.status == status]
        if query:
            query_lower = query.lower()
            records = [
                r
                for r in records
                if query_lower in r.job_id.lower()
                or any(query_lower in str(value).lower() for value in r.meta.values())
            ]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return [copy.deepcopy(r) for r in records]


store = JobStore()
