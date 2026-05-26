"""Low-level S3 utilities for the image-crop-retrieval project.

This module provides a thin, typed wrapper around ``boto3`` that handles:

* Discovering datasets by listing bucket prefixes.
* Finding the **latest** ``split_<date>.csv`` annotation file under a dataset
  prefix (lexicographic sort on the date portion is equivalent to chronological
  sort for ISO-8601 dates: ``YYYY-MM-DD``).
* Downloading / uploading index artefacts (``index.faiss``,
  ``metadata.parquet``) between S3 and a local cache directory.
* Loading images from ``s3://bucket/key`` URIs on-demand for result display.

Credentials are resolved via the standard boto3 credential chain:
environment variables (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``),
``~/.aws/credentials``, or an IAM instance / task role.

For S3-compatible stores (MinIO, Yandex Cloud Object Storage, …) set
``S3Config.endpoint_url``.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from PIL import Image as PILImage

from .config import S3Block as S3Config  # S3Config alias kept for clarity

if TYPE_CHECKING:
    # boto3-stubs[s3] is a dev dep; import only for type checking
    from mypy_boto3_s3 import S3Client as BotoS3Client

logger = logging.getLogger(__name__)

# Matches filenames like split_2024-01-15.csv or split_20240115.csv
_SPLIT_RE = re.compile(r"split_(\d{4}-?\d{2}-?\d{2})\.csv$")

# Matches s3://bucket/key URIs
_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")


class S3Client:
    """Typed wrapper around ``boto3`` for image-crop-retrieval S3 operations.

    Args:
        config: S3 connection and bucket parameters.
    """

    def __init__(self, config: S3Config) -> None:
        self._config = config
        # Pass kwargs explicitly so mypy can match the typed "s3" overload in
        # boto3-stubs instead of falling back to the unresolvable **kwargs path.
        self._s3: BotoS3Client = boto3.client(
            "s3",
            region_name=config.region,
            endpoint_url=config.endpoint_url,
        )

    def list_dataset_names(self) -> list[str]:
        """Return names of dataset sub-prefixes under ``config.prefix``.

        Datasets are identified as *common prefixes* (virtual directories)
        directly under the configured prefix.  A dataset named ``"my_dataset"``
        must correspond to a prefix ``{config.prefix}my_dataset/`` in the
        bucket.

        Returns:
            Sorted list of dataset name strings.
        """
        response = self._s3.list_objects_v2(
            Bucket=self._config.bucket,
            Prefix=self._config.prefix,
            Delimiter="/",
        )
        common_prefixes = response.get("CommonPrefixes") or []
        names: list[str] = []
        for entry in common_prefixes:
            full_prefix: str = entry.get("Prefix", "")
            # Strip the parent prefix and trailing slash to get the name
            name = full_prefix.removeprefix(self._config.prefix).rstrip("/")
            if name:
                names.append(name)
        return sorted(names)

    def find_latest_split_key(self, dataset_name: str) -> str | None:
        """Find the S3 key of the most recent ``split_<date>.csv`` file.

        Files are compared by the date string in their filename.  ISO-8601
        dates (``YYYY-MM-DD``) sort lexicographically in chronological order,
        so no date parsing is needed.

        Args:
            dataset_name: Name of the dataset sub-prefix.

        Returns:
            The full S3 key of the latest split file, or ``None`` if no split
            files exist under the dataset prefix.
        """
        prefix = self._config.dataset_prefix(dataset_name)
        response = self._s3.list_objects_v2(
            Bucket=self._config.bucket,
            Prefix=prefix,
        )
        contents = response.get("Contents") or []

        candidates: list[tuple[str, str]] = []  # (date_str, key)
        for obj in contents:
            key: str = obj.get("Key", "")
            filename = key.split("/")[-1]
            match = _SPLIT_RE.match(filename)
            if match:
                # Normalise by stripping dashes so both YYYYMMDD and YYYY-MM-DD
                # compare correctly together
                date_str = match.group(1).replace("-", "")
                candidates.append((date_str, key))

        if not candidates:
            logger.warning(
                "No split_<date>.csv files found under s3://%s/%s",
                self._config.bucket,
                prefix,
            )
            return None

        candidates.sort(key=lambda t: t[0], reverse=True)
        latest_key = candidates[0][1]
        logger.info("Latest split for '%s': %s", dataset_name, latest_key)
        return latest_key

    def download_file(self, s3_key: str, local_path: Path) -> None:
        """Download *s3_key* from the configured bucket to *local_path*.

        The parent directory is created automatically.

        Args:
            s3_key: Object key within ``config.bucket``.
            local_path: Destination path on the local filesystem.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "Downloading s3://%s/%s → %s", self._config.bucket, s3_key, local_path
        )
        self._s3.download_file(self._config.bucket, s3_key, str(local_path))

    def download_uri(self, uri: str, local_path: Path) -> None:
        """Download an ``s3://bucket/key`` URI (or bare key) to *local_path*.

        Unlike :meth:`download_file` this method parses the bucket and key
        from the URI, so it works with URIs that reference a different bucket
        than ``config.bucket`` (e.g. model checkpoints stored in a separate
        bucket).

        The parent directory is created automatically.

        Args:
            uri: ``s3://bucket/key`` URI or bare key relative to
                ``config.bucket``.
            local_path: Destination path on the local filesystem.
        """
        bucket, key = self._parse_uri(uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Downloading %s → %s", uri, local_path)
        self._s3.download_file(bucket, key, str(local_path))

    def upload_file(self, local_path: Path, s3_key: str) -> None:
        """Upload *local_path* to *s3_key* in the configured bucket.

        Args:
            local_path: Source file on the local filesystem.
            s3_key: Destination object key within ``config.bucket``.
        """
        logger.debug(
            "Uploading %s → s3://%s/%s", local_path, self._config.bucket, s3_key
        )
        self._s3.upload_file(str(local_path), self._config.bucket, s3_key)

    def upload_file_atomic(self, local_path: Path, s3_key: str) -> None:
        """Upload *local_path* to a ``*.tmp`` key then rename to *s3_key*.

        S3 does not support atomic rename natively, but the *copy + delete*
        pattern ensures the target key is never in a partially-written state
        when readers poll the bucket.  The window between the upload completing
        and the copy/delete completing is short.

        Args:
            local_path: Source file.
            s3_key: Final destination key in the configured bucket.
        """
        tmp_key = s3_key + ".tmp"
        self.upload_file(local_path, tmp_key)
        self._s3.copy_object(
            Bucket=self._config.bucket,
            CopySource={"Bucket": self._config.bucket, "Key": tmp_key},
            Key=s3_key,
        )
        self._s3.delete_object(Bucket=self._config.bucket, Key=tmp_key)
        logger.debug("Atomic upload complete: s3://%s/%s", self._config.bucket, s3_key)

    def load_image(self, uri: str) -> PILImage.Image:
        """Load a PIL Image from an S3 URI or bare key.

        Supports two URI formats:
        * ``s3://bucket/path/to/image.jpg`` — explicit bucket in URI.
        * ``path/to/image.jpg`` — key relative to ``config.bucket``.

        Args:
            uri: S3 URI or bare key string.

        Returns:
            RGB PIL image.

        Raises:
            ValueError: If the URI cannot be parsed.
            botocore.exceptions.ClientError: If the object does not exist.
        """
        bucket, key = self._parse_uri(uri)
        response = self._s3.get_object(Bucket=bucket, Key=key)
        data: bytes = response["Body"].read()
        return PILImage.open(io.BytesIO(data)).convert("RGB")

    def load_image_to_tmp(self, uri: str) -> Path:
        """Download an image to a temp file and return its path.

        Useful during batch indexing where the same image is accessed multiple
        times — callers can cache the returned path keyed by URI.

        Args:
            uri: S3 URI or bare key.

        Returns:
            Path to a temporary file (caller is responsible for deletion).
        """
        bucket, key = self._parse_uri(uri)
        suffix = Path(key).suffix or ".jpg"
        fd, tmp_str = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        tmp_path = Path(tmp_str)
        self._s3.download_file(bucket, key, str(tmp_path))
        return tmp_path

    def get_last_modified(self, s3_key: str) -> float | None:
        """Return the ``LastModified`` timestamp of *s3_key* as a POSIX float.

        Returns ``None`` if the object does not exist.

        Args:
            s3_key: Object key within ``config.bucket``.
        """
        try:
            head = self._s3.head_object(Bucket=self._config.bucket, Key=s3_key)
            return head["LastModified"].timestamp()
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            # botocore.exceptions.ClientError for 404
            if _is_not_found(exc):
                return None
            raise

    def is_remote_newer(self, s3_key: str, local_path: Path) -> bool:
        """Return ``True`` if the S3 object is newer than *local_path*.

        Also returns ``True`` when *local_path* does not exist (treat as
        infinitely old) or when *s3_key* does not exist on S3 (no update
        available, return ``False``).

        Args:
            s3_key: Object key within ``config.bucket``.
            local_path: Local file to compare against.
        """
        if not local_path.exists():
            return True
        remote_ts = self.get_last_modified(s3_key)
        if remote_ts is None:
            return False
        local_ts = local_path.stat().st_mtime
        return remote_ts > local_ts

    def index_key(self, dataset_name: str) -> str:
        """Return the S3 key for ``index.faiss`` of *dataset_name*."""
        return self._config.dataset_prefix(dataset_name) + "index.faiss"

    def metadata_key(self, dataset_name: str) -> str:
        """Return the S3 key for ``metadata.parquet`` of *dataset_name*."""
        return self._config.dataset_prefix(dataset_name) + "metadata.parquet"

    def _parse_uri(self, uri: str) -> tuple[str, str]:
        """Parse ``s3://bucket/key`` or bare ``key`` into ``(bucket, key)``.

        Args:
            uri: URI or bare key string.

        Returns:
            ``(bucket, key)`` tuple.
        """
        match = _S3_URI_RE.match(uri)
        if match:
            return match.group(1), match.group(2)
        # Bare key — use the configured bucket
        return self._config.bucket, uri


def _is_not_found(exc: Exception) -> bool:
    """Return ``True`` if *exc* is a boto3 ClientError for a 404/NoSuchKey."""
    try:
        # botocore.exceptions.ClientError carries a ``response`` dict.
        code: str = exc.response["Error"]["Code"]  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return False
    return code in {"404", "NoSuchKey"}
