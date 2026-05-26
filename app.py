"""Streamlit application entry point.

Run with::

    uv run streamlit run app.py

Configuration is loaded via :class:`~image_retrieval.config.Configuration`
using the Consul → YAML → defaults priority chain.  The following environment
variables are the main knobs:

Consul (primary config source)::

    CONSUL_URL    Consul agent URL, e.g. ``http://consul:8500``
    CONSUL_KEY    KV key that stores the YAML config
                  (default: ``config/image-crop-retrieval``)

YAML file (fallback)::

    CONFIG_PATH   Path to a YAML config file (default: ``config.yaml``)

Quick env-var overrides (bypass YAML/Consul for common settings)::

    MODEL_PATH    Pre-fill the model checkpoint path field
    S3_BUCKET     Enable S3 mode (sets ``s3.bucket`` in config)
    S3_PREFIX     S3 key prefix for datasets
    S3_REGION     AWS region (default: ``us-east-1``)
    S3_ENDPOINT_URL  Custom S3 endpoint (MinIO, Yandex Cloud, …)
    S3_CHECK_INTERVAL  S3 poll interval in seconds (default: 300)
    DATASETS_DIR  Local datasets directory (local mode only)

See ``config.yaml.example`` for the full YAML schema.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from image_retrieval.config import (
    AppBlock,
    Configuration,
    EncoderBlock,
    S3Block,
)
from image_retrieval.embedder import TorchEmbedder
from image_retrieval.registry import DatasetRegistry, S3DatasetRegistry
from image_retrieval.s3_client import S3Client
from image_retrieval.ui.crop_selector import render_crop_selector
from image_retrieval.ui.cvat_exporter import render_cvat_exporter
from image_retrieval.ui.results_viewer import render_results

logger = logging.getLogger(__name__)

_AnyRegistry = DatasetRegistry | S3DatasetRegistry


@st.cache_resource
def _load_config() -> Configuration:
    """Load and cache the root :class:`Configuration` for the process lifetime.

    Reads ``CONSUL_URL`` / ``CONFIG_PATH`` env vars automatically.  Quick env
    overrides (``S3_BUCKET``, ``DATASETS_DIR``, …) are applied on top of the
    loaded config afterwards in :func:`_apply_env_overrides`.
    """
    return Configuration.load()


def _apply_env_overrides(cfg: Configuration) -> Configuration:
    """Patch *cfg* with any environment-variable quick overrides.

    Env vars take precedence over the YAML / Consul config so operators can
    inject secrets (tokens, passwords) without storing them in config files.
    """
    overrides: dict[str, object] = {}

    # S3 block — any S3_* env var enables S3 mode
    s3_bucket = os.environ.get("S3_BUCKET", "").strip()
    if s3_bucket:
        existing_s3 = cfg.s3 or S3Block(bucket=s3_bucket)
        overrides["s3"] = S3Block(
            bucket=s3_bucket,
            prefix=os.environ.get("S3_PREFIX", existing_s3.prefix),
            region=os.environ.get("S3_REGION", existing_s3.region),
            endpoint_url=(
                os.environ.get("S3_ENDPOINT_URL") or existing_s3.endpoint_url
            ),
            check_interval_seconds=int(
                os.environ.get(
                    "S3_CHECK_INTERVAL",
                    str(existing_s3.check_interval_seconds),
                )
            ),
        )

    # App block
    datasets_dir_env = os.environ.get("DATASETS_DIR", "").strip()
    top_k_env = os.environ.get("TOP_K", "").strip()
    device_env = os.environ.get("DEVICE", "").strip()
    if datasets_dir_env or top_k_env or device_env:
        overrides["app"] = AppBlock(
            datasets_dir=(
                Path(datasets_dir_env) if datasets_dir_env else cfg.app.datasets_dir
            ),
            top_k=int(top_k_env) if top_k_env else cfg.app.top_k,
            device=device_env or cfg.app.device,
            min_crop_px=cfg.app.min_crop_px,
            canvas_width=cfg.app.canvas_width,
            canvas_height=cfg.app.canvas_height,
            results_columns=cfg.app.results_columns,
        )

    # Encoder block — MODEL_PATH env var
    model_path_env = os.environ.get("MODEL_PATH", "").strip()
    if model_path_env and cfg.encoder is None:
        overrides["encoder"] = EncoderBlock(checkpoint=model_path_env)

    if not overrides:
        return cfg
    return cfg.model_copy(update=overrides)


@st.cache_resource
def _load_local_registry(datasets_dir: str) -> DatasetRegistry:
    """Instantiate a DatasetRegistry once and cache for the process lifetime."""
    return DatasetRegistry(AppBlock(datasets_dir=Path(datasets_dir)))


@st.cache_resource
def _load_s3_registry(
    bucket: str,
    prefix: str,
    region: str,
    endpoint_url: str,
    check_interval: int,
) -> S3DatasetRegistry:
    """Instantiate an S3DatasetRegistry once; cache by connection params."""
    s3_block = S3Block(
        bucket=bucket,
        prefix=prefix,
        region=region,
        endpoint_url=endpoint_url or None,
        check_interval_seconds=check_interval,
    )
    return S3DatasetRegistry(s3_block)


@st.cache_resource
def _load_s3_client(
    bucket: str,
    prefix: str,
    region: str,
    endpoint_url: str,
    check_interval: int,
) -> S3Client:
    """Create and cache an S3Client for on-demand image loading."""
    s3_block = S3Block(
        bucket=bucket,
        prefix=prefix,
        region=region,
        endpoint_url=endpoint_url or None,
        check_interval_seconds=check_interval,
    )
    return S3Client(s3_block)


@st.cache_resource
def _load_embedder(model_path: str, device: str) -> TorchEmbedder:
    """Load the PyTorch checkpoint once; cache by (model_path, device)."""
    return TorchEmbedder(checkpoint_path=Path(model_path), device=device)


def _resolve_model_path(
    encoder_cfg: EncoderBlock,
    s3_client: S3Client | None,
) -> Path:
    """Return a local Path to the model checkpoint.

    If ``encoder_cfg.checkpoint`` is a local path, returns it directly.
    If it is an ``s3://`` URI:
    * Derives a cache path under ``encoder_cfg.cache_dir`` (or a system temp
      dir) keyed by a hash of the URI.
    * Downloads the file on the first call.
    * On subsequent calls, checks whether the S3 object is newer than the
      cached file and re-downloads if necessary.

    Args:
        encoder_cfg: Encoder configuration block.
        s3_client: S3 client (required for ``s3://`` checkpoints).

    Raises:
        ValueError: If an S3 URI is given but no *s3_client* is available.
        FileNotFoundError: If a local path does not exist.
    """
    checkpoint = encoder_cfg.checkpoint
    if not checkpoint.startswith("s3://"):
        local = Path(checkpoint)
        if not local.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint}"
            )
        return local

    if s3_client is None:
        raise ValueError(
            "Encoder checkpoint is an S3 URI but no S3 config was found.  "
            "Set 's3' block in config or S3_BUCKET env var."
        )

    # Derive a stable local cache path from the URI
    cache_dir = (
        encoder_cfg.cache_dir
        or Path(tempfile.gettempdir()) / "image-retrieval-models"
    )
    uri_hash = hashlib.md5(checkpoint.encode()).hexdigest()[:16]
    suffix = Path(checkpoint.rsplit("/", 1)[-1]).suffix or ".pth"
    local_path = cache_dir / f"model_{uri_hash}{suffix}"

    if s3_client.is_remote_newer(_s3_key_from_uri(checkpoint), local_path):
        logger.info("Downloading model checkpoint from %s …", checkpoint)
        s3_client.download_uri(checkpoint, local_path)
        logger.info("Model cached at %s", local_path)

    return local_path


def _s3_key_from_uri(uri: str) -> str:
    """Extract the key portion from an ``s3://bucket/key`` URI."""
    # Strip "s3://bucket/"
    parts = uri.split("/", 3)
    return parts[3] if len(parts) >= 4 else uri


def _render_dataset_status(
    registry: _AnyRegistry,
    dataset_name: str,
) -> None:
    """Show index build timestamp and a Rescan button for *dataset_name*."""
    reload_info = registry.last_reload_info(dataset_name)
    if reload_info is not None:
        index_mtime_ns, _ = reload_info
        ts = datetime.datetime.fromtimestamp(
            index_mtime_ns / 1_000_000_000, tz=datetime.UTC
        ).astimezone()
        st.caption(f"🕒 Index built: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if st.button(
        "🔄 Rescan datasets",
        help=(
            "In local mode: re-scan the datasets directory.  "
            "In S3 mode: re-list bucket prefixes."
        ),
        use_container_width=True,
    ):
        new_names = registry.rescan()
        if new_names:
            st.success(f"Found new datasets: {', '.join(new_names)}")
            st.rerun()
        else:
            st.info("No new datasets found.")


def main() -> None:
    """Application entry point."""
    base_cfg = _load_config()
    cfg = _apply_env_overrides(base_cfg)

    st.set_page_config(
        page_title="Image Crop Retrieval",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 Image Crop Retrieval")
    st.caption(
        "Upload an image, draw a bounding box, and find the most visually "
        "similar crops in your dataset."
    )

    with st.sidebar:
        st.header("⚙️ Settings")

        default_checkpoint = cfg.encoder.checkpoint if cfg.encoder else ""
        model_input = st.text_input(
            label="Model checkpoint (.pt / .pth or s3://)",
            value=default_checkpoint,
            placeholder="s3://bucket/models/encoder.pth  or  /local/encoder.pth",
            help=(
                "Local path or s3:// URI to the SSL model checkpoint.  "
                "S3 models are downloaded to a local cache on first use."
            ),
        )

        st.divider()

        registry: _AnyRegistry
        s3_client: S3Client | None = None

        if cfg.s3 is not None:
            s3 = cfg.s3
            prefix_display = s3.prefix or "/"
            st.caption(f"☁️ S3: `s3://{s3.bucket}/{prefix_display}`")

            registry = _load_s3_registry(
                bucket=s3.bucket,
                prefix=s3.prefix,
                region=s3.region,
                endpoint_url=s3.endpoint_url or "",
                check_interval=s3.check_interval_seconds,
            )
            s3_client = _load_s3_client(
                bucket=s3.bucket,
                prefix=s3.prefix,
                region=s3.region,
                endpoint_url=s3.endpoint_url or "",
                check_interval=s3.check_interval_seconds,
            )
        else:
            registry = _load_local_registry(str(cfg.app.datasets_dir))

        available_datasets = registry.available()
        if not available_datasets:
            if cfg.s3 is not None:
                msg = (
                    f"No datasets found in S3 bucket `{cfg.s3.bucket}` "
                    f"under prefix `{cfg.s3.prefix or '/'}`.  \n"
                    "Run `scripts/build_index.py s3 …` to build an index."
                )
            else:
                msg = (
                    f"No indexed datasets found in `{cfg.app.datasets_dir}`.  \n"
                    "Run `scripts/build_index.py local …` to create one first."
                )
            st.error(msg, icon="🚫")
            st.stop()

        dataset_name = st.selectbox(
            label="Dataset",
            options=available_datasets,
            help="Select the dataset to search in.",
        )
        assert dataset_name is not None

        _render_dataset_status(registry, dataset_name)

        st.divider()

        top_k: int = st.slider(
            label="Top K results",
            min_value=1,
            max_value=50,
            value=cfg.app.top_k,
            help="Number of nearest neighbours to retrieve.",
        )

        st.divider()
        if cfg.s3 is None:
            st.caption(f"📁 `{cfg.app.datasets_dir}`")

    if not model_input:
        st.info(
            "👈 Enter the path to your SSL model checkpoint in the sidebar.",
            icon="ℹ️",
        )
        st.stop()

    crop_result = render_crop_selector(cfg.app)
    if crop_result is None:
        st.stop()
    _full_image, crop = crop_result

    st.divider()
    if not st.button("🔍 Search", type="primary", use_container_width=False):
        st.stop()

    encoder_block = cfg.encoder or EncoderBlock(checkpoint=model_input)
    if model_input != default_checkpoint:
        encoder_block = EncoderBlock(
            checkpoint=model_input,
            model_module=encoder_block.model_module,
            cache_dir=encoder_block.cache_dir,
        )

    try:
        local_model_path = _resolve_model_path(encoder_block, s3_client)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc), icon="🚫")
        st.stop()

    try:
        embedder = _load_embedder(
            model_path=str(local_model_path),
            device=cfg.app.device,
        )
    except FileNotFoundError as exc:
        st.error(f"Model checkpoint not found: {exc}", icon="🚫")
        st.stop()
    except RuntimeError as exc:
        st.error(f"Failed to load model: {exc}", icon="🚫")
        st.stop()

    with st.spinner("Computing embedding…"):
        query_vec: np.ndarray = embedder.embed([crop])

    dataset_meta, faiss_index = registry.get(dataset_name)
    with st.spinner(f"Searching top {top_k} in '{dataset_name}'…"):
        search_results = faiss_index.search(query_vec, top_k)

    render_results(search_results, dataset_meta.images_root, cfg.app, s3_client)

    if cfg.cvat is not None:
        render_cvat_exporter(
            all_results=search_results,
            images_root=dataset_meta.images_root,
            cvat_config=cfg.cvat,
            s3_client=s3_client,
        )


if __name__ == "__main__":
    main()
