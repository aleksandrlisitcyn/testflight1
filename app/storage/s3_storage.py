from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.exceptions import ClientError

from ..settings import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_SECRET_KEY,
)


class S3Storage:
    """Minimal S3-compatible storage backend."""

    def __init__(self) -> None:
        self.bucket = S3_BUCKET
        self.client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except ClientError:
                # Give up silently; uploads will raise later
                pass

    def save_bytes(self, path: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=path, Body=data)

    def save_json(self, path: str, obj: Any) -> None:
        payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.save_bytes(path, payload)
