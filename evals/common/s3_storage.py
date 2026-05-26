"""S3/MinIO storage helpers for persisting evaluation artifacts.

Handles upload and download of predictions and results between
evaluation phases. Credentials and endpoint are read from
environment variables:

    AWS_ACCESS_KEY_ID     (from K8s secret minio-credentials / MINIO_ROOT_USER)
    AWS_SECRET_ACCESS_KEY (from K8s secret minio-credentials / MINIO_ROOT_PASSWORD)
    S3_ENDPOINT_URL       (from K8s secret minio-credentials / MINIO_ENDPOINT_URL)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import boto3

logger = logging.getLogger(__name__)


def _get_s3_client():
    """Create an S3 client using environment variables for config."""
    kwargs = {}
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an s3://bucket/key URI into (bucket, key)."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def upload_file(local_path: str | Path, s3_uri: str) -> None:
    """Upload a local file to S3/MinIO.

    Args:
        local_path: Path to the local file.
        s3_uri: S3 URI (e.g. s3://bucket/path/predictions.jsonl).
    """
    bucket, key = parse_s3_uri(s3_uri)
    s3 = _get_s3_client()
    logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key)
    logger.info(f"Upload complete")


def upload_dir(local_dir: str | Path, s3_prefix: str) -> None:
    """Upload all files in a directory tree to S3/MinIO.

    Skips files that are not inside a per-instance subdirectory (e.g.
    prompted_dataset.jsonl, predictions.jsonl) to avoid re-uploading
    bulk files. Only files one level deep (instance_id/filename) are
    uploaded.

    Args:
        local_dir: Local directory containing per-instance subdirectories.
        s3_prefix: S3 URI prefix (e.g. s3://bucket/runs/my-run).
    """
    bucket, prefix = parse_s3_uri(s3_prefix)
    prefix = prefix.rstrip("/")
    s3 = _get_s3_client()
    local_dir = Path(local_dir)
    uploaded = 0
    for instance_dir in local_dir.iterdir():
        if not instance_dir.is_dir():
            continue
        for file in instance_dir.rglob("*"):
            if not file.is_file():
                continue
            rel = file.relative_to(local_dir).as_posix()
            key = f"{prefix}/{rel}" if prefix else rel
            uploaded += 1
    logger.info(f"Uploaded {uploaded} per-instance files to s3://{bucket}/{prefix}/")


def download_file(s3_uri: str, local_path: str | Path) -> None:
    """Download a file from S3/MinIO to a local path.

    Args:
        s3_uri: S3 URI (e.g. s3://bucket/path/predictions.jsonl).
        local_path: Path to write the downloaded file.
    """
    bucket, key = parse_s3_uri(s3_uri)
    s3 = _get_s3_client()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
    s3.download_file(bucket, key, str(local_path))
    logger.info(f"Download complete")
