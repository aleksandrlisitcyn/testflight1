import json
from pathlib import Path

from ..settings import DATA_DIR


class FSStorage:
    def __init__(self):
        self.root = Path(DATA_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, path: str, data: bytes):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def save_json(self, path: str, obj):
        self.save_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))
