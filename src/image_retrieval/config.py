"""Application configuration — Pydantic v2 block-based design.

One :class:`Configuration` object holds all subsystem blocks.  Each block is
a frozen Pydantic model so fields are validated on construction and values are
immutable after creation.

Loading priority (highest → lowest):

1. **Consul KV** — if ``consul_url`` is provided (or ``CONSUL_URL`` env var)
   and the server is reachable, the YAML stored at the configured key is
   fetched and parsed.
2. **YAML file** — ``config_path`` (or ``CONFIG_PATH`` env var, defaulting to
   ``config.yaml`` in the working directory) if the file exists.
3. **Built-in defaults** — all fields have sensible defaults; a
   :class:`Configuration` with no arguments is fully usable.

Example ``config.yaml``::

    app:
      top_k: 20
      device: cpu

    s3:
      bucket: my-datasets-bucket
      prefix: datasets/
      region: eu-west-1

    encoder:
      checkpoint: s3://my-datasets-bucket/models/encoder.pth

    cvat:
      url: https://cvat.example.com
      token: my-api-token
      project_id: 42

Backward-compatible aliases :data:`AppConfig` and :data:`S3Config` are
provided so existing code continues to compile without changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


def _default_datasets_dir() -> Path:
    """Return ``<project_root>/datasets`` as the default datasets directory."""
    return (Path(__file__).resolve().parents[3] / "datasets").resolve()


class AppBlock(BaseModel):
    """UI and runtime settings for the Streamlit application.

    Attributes:
        datasets_dir: Directory containing one sub-folder per indexed dataset
            (local mode only; ignored when :attr:`Configuration.s3` is set).
        top_k: Default number of nearest-neighbour results to retrieve.
        device: PyTorch device string (``"cpu"`` for CPU-only inference).
        min_crop_px: Minimum side length (pixels) for a drawn crop to be valid.
        canvas_width: Maximum display width of the drawable canvas (px).
        canvas_height: Maximum display height of the drawable canvas (px).
        results_columns: Number of columns in the results grid.
    """

    model_config = ConfigDict(frozen=True)

    datasets_dir: Path = Field(default_factory=_default_datasets_dir)
    top_k: int = 10
    device: str = "cpu"
    min_crop_px: int = 8
    canvas_width: int = 800
    canvas_height: int = 600
    results_columns: int = 3

    @field_validator("datasets_dir", mode="before")
    @classmethod
    def _resolve_datasets_dir(cls, v: Any) -> Path:  # noqa: ANN401
        return Path(v).resolve()


class S3Block(BaseModel):
    """Connection parameters for an S3-compatible object store.

    Attributes:
        bucket: Name of the S3 bucket.
        prefix: Key prefix under which all datasets live
            (e.g. ``"datasets/"``).  May be empty for bucket-root access.
        region: AWS region string (e.g. ``"us-east-1"``).
        endpoint_url: Custom endpoint for S3-compatible stores (MinIO, Yandex
            Cloud Object Storage, …).  ``None`` → use standard AWS endpoints.
        check_interval_seconds: How often (seconds) the app polls S3 to detect
            a new index.
    """

    model_config = ConfigDict(frozen=True)

    bucket: str
    prefix: str = ""
    region: str = "us-east-1"
    endpoint_url: str | None = None
    check_interval_seconds: int = 300

    def dataset_prefix(self, dataset_name: str) -> str:
        """Return the S3 key prefix for *dataset_name* (with trailing slash)."""
        base = self.prefix.rstrip("/")
        return f"{base}/{dataset_name}/" if base else f"{dataset_name}/"


class EncoderBlock(BaseModel):
    """Configuration for the SSL model used to embed image crops.

    Attributes:
        checkpoint: Path to the ``.pt`` / ``.pth`` checkpoint.  Accepts both
            local filesystem paths and ``s3://bucket/key`` URIs.  When an S3
            URI is given the file is downloaded to *cache_dir* on first use and
            re-used on subsequent starts (unless the remote object is newer).
        model_module: Optional ``"module.path:ClassName"`` for state-dict
            checkpoints that need an explicit model class.  ``None`` → full
            model pickle.
        cache_dir: Directory for caching S3-downloaded model files.  Defaults
            to ``<tmp>/image-retrieval-models/``.
    """

    model_config = ConfigDict(frozen=True)

    checkpoint: str
    model_module: str | None = None
    cache_dir: Path | None = None

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_cache_dir(cls, v: Any) -> Path | None:  # noqa: ANN401
        return Path(v).resolve() if v is not None else None


class CVATBlock(BaseModel):
    """Connection parameters for the CVAT annotation platform (v2.x API).

    Authentication priority: *token* → ``username`` + ``password``.

    Attributes:
        url: Base URL of the CVAT instance (e.g.
            ``"https://cvat.example.com"``).
        token: CVAT REST API token.  Takes precedence over username/password.
        username: CVAT username (only used when *token* is absent).
        password: CVAT password.
        project_id: Default project to attach new tasks to.  ``None`` →
            tasks are created without a project.
        task_label: Name of the bounding-box label in exported tasks.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    token: str | None = None
    username: str | None = None
    password: str | None = None
    project_id: int | None = None
    task_label: str = "crop"

    @field_validator("url", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, v: Any) -> str:  # noqa: ANN401
        return str(v).rstrip("/")


class Configuration(BaseModel):
    """Root configuration object containing all subsystem blocks.

    All blocks are optional except *app* which always carries defaults.
    Use :meth:`load` as the canonical factory — it handles the
    Consul → YAML → defaults fallback chain automatically.
    """

    model_config = ConfigDict(frozen=True)

    app: AppBlock = Field(default_factory=AppBlock)
    s3: S3Block | None = None
    encoder: EncoderBlock | None = None
    cvat: CVATBlock | None = None

    @classmethod
    def load(
        cls,
        *,
        config_path: Path | None = None,
        consul_url: str | None = None,
        consul_key: str | None = None,
    ) -> Configuration:
        """Load configuration using Consul → YAML → defaults priority chain.

        All parameters default to environment variables, so callers can simply
        invoke ``Configuration.load()`` and rely on ``CONSUL_URL``,
        ``CONSUL_KEY``, and ``CONFIG_PATH`` env vars.

        Args:
            config_path: YAML config file path.  Falls back to ``CONFIG_PATH``
                env var, then ``config.yaml`` in the current working directory.
            consul_url: Consul agent URL (e.g. ``"http://consul:8500"``).
                Falls back to ``CONSUL_URL`` env var.
            consul_key: Consul KV key whose value is the YAML config.  Falls
                back to ``CONSUL_KEY`` env var, then the default key.
        """
        effective_consul_url = consul_url or os.environ.get("CONSUL_URL", "")
        effective_consul_key = (
            consul_key
            or os.environ.get("CONSUL_KEY", "config/image-crop-retrieval")
        )

        if effective_consul_url:
            try:
                cfg = cls.from_consul(effective_consul_url, effective_consul_key)
                logger.info(
                    "Configuration loaded from Consul key '%s'.",
                    effective_consul_key,
                )
                return cfg
            except Exception as exc:
                logger.warning(
                    "Consul unavailable (%s) — falling back to YAML / defaults.",
                    exc,
                )

        effective_path = config_path or Path(
            os.environ.get("CONFIG_PATH", "config.yaml")
        )
        if effective_path.exists():
            cfg = cls.from_yaml(effective_path)
            logger.info("Configuration loaded from '%s'.", effective_path)
            return cfg

        logger.info("No Consul / config file found — using built-in defaults.")
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> Configuration:
        """Parse *path* as YAML and return a validated :class:`Configuration`.

        Args:
            path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If *path* does not exist.
            pydantic.ValidationError: If the YAML content fails validation.
        """
        import yaml  # local import keeps pyyaml optional until this method is called

        with path.open(encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    @classmethod
    def from_consul(cls, url: str, key: str) -> Configuration:
        """Fetch a YAML document from a Consul KV store and parse it.

        Args:
            url: Consul agent base URL (scheme + host + port).
            key: KV key whose value is a YAML document.

        Raises:
            KeyError: If *key* does not exist in the KV store.
        """
        from urllib.parse import urlparse

        import consul
        import yaml

        parsed = urlparse(url)
        c = consul.Consul(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8500,
            scheme=parsed.scheme or "http",
        )
        _, kv_data = c.kv.get(key)
        if kv_data is None:
            raise KeyError(f"Consul key '{key}' not found at {url}")
        raw: str = kv_data["Value"].decode("utf-8")
        data: dict[str, Any] = yaml.safe_load(raw) or {}
        return cls.model_validate(data)


@dataclass(frozen=True)
class DatasetMeta:
    """Resolved paths for one indexed dataset.

    Created by :class:`~image_retrieval.registry.DatasetRegistry` at startup;
    passed to :class:`~image_retrieval.indexer.FAISSIndex` and the results
    viewer so they know where to find source images.

    Attributes:
        name: Human-readable dataset name (sub-folder name under
            ``datasets_dir``).
        index_path: Absolute path to the ``index.faiss`` file.
        metadata_path: Absolute path to the ``metadata.parquet`` file.
        images_root: Absolute path to the folder containing source images.
    """

    name: str
    index_path: Path
    metadata_path: Path
    images_root: Path


#: Alias for :class:`AppBlock` — keeps existing ``AppConfig`` imports working.
AppConfig = AppBlock

#: Alias for :class:`S3Block` — keeps existing ``S3Config`` imports working.
S3Config = S3Block
