import os
from pathlib import Path

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "fs")  # 'fs' or 's3'
DATA_DIR = os.getenv("DATA_DIR", str(Path(__file__).resolve().parent / "data"))
S3_BUCKET = os.getenv("S3_BUCKET", "stitch-results")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")