#!/usr/bin/env python3
"""CLI script: build a FAISS index from an annotated image dataset.

Supports two modes: **local** and **S3**.

Local mode
----------
Reads annotations from a local CSV/Parquet file and images from a local
directory.  Writes the index to ``datasets/{dataset_name}/``.

::

    uv run python scripts/build_index.py local \\
        --annotations /data/annotations.csv \\
        --images-root /data/images/ \\
        --checkpoint   /models/encoder.pth \\
        --dataset-name my_dataset

S3 mode
-------
Finds the latest ``split_<date>.csv`` under the given S3 dataset prefix,
downloads it, fetches images from S3 (via the ``s3_image_path`` column),
builds the index, and uploads ``index.faiss`` + ``metadata.parquet`` back
to S3.

::

    uv run python scripts/build_index.py s3 \\
        --bucket    my-bucket \\
        --dataset   my_dataset \\
        --checkpoint /models/encoder.pth

Column requirements
-------------------
*Local mode* — annotations must contain: ``image_path, x1, y1, x2, y2``.

*S3 mode* — split CSV must contain: ``s3_image_path, x1, y1, x2, y2``.
The ``s3_image_path`` values must be valid ``s3://bucket/key`` URIs.

A ``box_id`` column is auto-generated if absent in either mode.

State-dict checkpoints
----------------------
Pass ``--model-module module.path:ClassName`` to use a state-dict checkpoint
(in either mode)::

    uv run python scripts/build_index.py s3 \\
        --bucket    my-bucket \\
        --dataset   my_dataset \\
        --checkpoint /models/weights.pt \\
        --model-module mypackage.models:MyEncoder

.. warning::
    ``torch.load(weights_only=False)`` executes arbitrary pickle code.
    Only load checkpoints from **trusted sources**.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image as PILImage
from tqdm import tqdm

from image_retrieval.embedder import TorchEmbedder

logger = logging.getLogger(__name__)


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    """Add embedder / tuning arguments shared by both modes."""
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        metavar="FILE",
        help="Path to the SSL model checkpoint (.pt / .pth)",
    )
    p.add_argument(
        "--model-module",
        default=None,
        metavar="MODULE:CLASS",
        help=(
            "For state-dict checkpoints: 'module.path:ClassName'.  "
            "Example: mypackage.models:ResNetEncoder"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Number of crops per forward pass",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device string",
    )
    p.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("H", "W"),
        help="Crop size fed to the model (height width)",
    )


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="build_index",
        description="Build a FAISS index for image-crop-retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(dest="mode", required=True)

    local = sub.add_parser(
        "local",
        help="Build from a local CSV/Parquet + local images directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    local.add_argument(
        "--annotations",
        required=True,
        type=Path,
        metavar="FILE",
        help="CSV or Parquet with columns: image_path, x1, y1, x2, y2",
    )
    local.add_argument(
        "--images-root",
        required=True,
        type=Path,
        metavar="DIR",
        help="Root directory for images referenced in --annotations",
    )
    local.add_argument(
        "--dataset-name",
        required=True,
        metavar="NAME",
        help="Name of the output dataset sub-directory",
    )
    local.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        metavar="DIR",
        help="Root datasets directory (output: DATASETS_DIR/DATASET_NAME/)",
    )
    _add_shared_args(local)

    s3p = sub.add_parser(
        "s3",
        help="Build from S3: reads latest split_<date>.csv, writes index to S3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    s3p.add_argument(
        "--bucket",
        required=True,
        metavar="BUCKET",
        help="S3 bucket name",
    )
    s3p.add_argument(
        "--dataset",
        required=True,
        metavar="NAME",
        help="Dataset name (prefix within the bucket)",
    )
    s3p.add_argument(
        "--prefix",
        default="",
        metavar="PREFIX",
        help="Key prefix under which datasets are stored (e.g. 'datasets/')",
    )
    s3p.add_argument(
        "--region",
        default="us-east-1",
        metavar="REGION",
        help="AWS region",
    )
    s3p.add_argument(
        "--endpoint-url",
        default=None,
        metavar="URL",
        help="Custom S3-compatible endpoint (e.g. for MinIO)",
    )
    s3p.add_argument(
        "--image-cache-dir",
        default=None,
        type=Path,
        metavar="DIR",
        help=(
            "Local directory for caching downloaded images during indexing.  "
            "Defaults to a temp directory that is cleaned up after the run."
        ),
    )
    _add_shared_args(s3p)

    return root


def _resolve_model_class(module_spec: str) -> type[nn.Module]:
    """Import and return an ``nn.Module`` subclass from ``'module:Class'``.

    Raises:
        ValueError: If *module_spec* does not contain ``:``.
        TypeError: If the resolved object is not an ``nn.Module`` subclass.
    """
    if ":" not in module_spec:
        raise ValueError(
            f"--model-module must be 'module.path:ClassName', got: '{module_spec}'"
        )
    module_path, class_name = module_spec.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise TypeError(
            f"'{module_spec}' resolved to {cls!r}, not an nn.Module subclass."
        )
    return cls


def _validate_and_fill_box_id(
    df: pd.DataFrame,
    required: set[str],
    context: str = "Annotations",
) -> pd.DataFrame:
    """Raise ValueError if required columns are missing; auto-fill box_id."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{context} missing columns: {sorted(missing)}")
    if "box_id" not in df.columns:
        df["box_id"] = [f"box_{i}" for i in range(len(df))]
    return df.reset_index(drop=True)


def _load_annotations_local(path: Path) -> pd.DataFrame:
    """Load a local CSV or Parquet annotations file.

    Required columns: ``image_path, x1, y1, x2, y2``.
    ``box_id`` is auto-generated if absent.

    Returns:
        Validated DataFrame with ``image_path`` pointing to local files.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = _validate_and_fill_box_id(df, {"image_path", "x1", "y1", "x2", "y2"})
    logger.info("Loaded %d local annotations from '%s'.", len(df), path)
    return df


def _load_annotations_s3(
    s3_client: object,  # S3Client, typed loosely to avoid circular
    dataset_name: str,
    tmp_dir: Path,
) -> pd.DataFrame:
    """Download the latest ``split_<date>.csv`` from S3 and load it.

    Required columns in the CSV: ``s3_image_path, x1, y1, x2, y2``.

    The ``s3_image_path`` values are copied to ``image_path`` so the metadata
    stored in the index uses ``s3://`` URIs — the app detects these at query
    time to route image loading through S3.

    Returns:
        Validated DataFrame ready for :func:`_embed_all_s3`.
    """
    from image_retrieval.s3_client import S3Client

    assert isinstance(s3_client, S3Client)

    split_key = s3_client.find_latest_split_key(dataset_name)
    if split_key is None:
        raise FileNotFoundError(
            f"No split_<date>.csv found for dataset '{dataset_name}' in bucket "
            f"'{s3_client._config.bucket}'."
        )

    local_csv = tmp_dir / "annotations.csv"
    logger.info(
        "Downloading annotations: s3://%s/%s", s3_client._config.bucket, split_key
    )
    s3_client.download_file(split_key, local_csv)

    df = pd.read_csv(local_csv)
    df = df.copy()
    df["image_path"] = df["s3_image_path"]
    df = _validate_and_fill_box_id(
        df,
        {"s3_image_path", "x1", "y1", "x2", "y2"},
        f"Split CSV '{split_key}'",
    )
    logger.info("Loaded %d annotations from S3 split '%s'.", len(df), split_key)
    return df


def _open_local(path: Path) -> PILImage.Image:
    """Open a local image file as an RGB PIL Image."""
    return PILImage.open(path).convert("RGB")


def _embed_all_local(
    df: pd.DataFrame,
    images_root: Path,
    embedder: TorchEmbedder,
    batch_size: int,
) -> np.ndarray:
    """Crop + embed using local images.

    Rows where the image cannot be loaded are zero-filled and warned.
    """
    return _run_embedding_loop(
        df=df,
        load_image_fn=lambda img_path_str: _open_local(images_root / img_path_str),
        image_col="image_path",
        embedder=embedder,
        batch_size=batch_size,
    )


def _embed_all_s3(
    df: pd.DataFrame,
    s3_client: object,
    embedder: TorchEmbedder,
    batch_size: int,
    cache_dir: Path,
) -> np.ndarray:
    """Crop + embed S3 images, caching each unique image to *cache_dir*.

    Each unique ``s3_image_path`` is downloaded once and kept in *cache_dir*
    for the duration of the run.  For 100K boxes from ~10K images this avoids
    re-downloading the same image for every box that shares it.
    """
    from image_retrieval.s3_client import S3Client

    assert isinstance(s3_client, S3Client)

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_cache: dict[str, Path] = {}

    def load_s3_image(uri: str) -> PILImage.Image:
        if uri not in local_cache:
            local_cache[uri] = s3_client.load_image_to_tmp(uri)
        cached = local_cache[uri]
        return PILImage.open(cached).convert("RGB")

    try:
        return _run_embedding_loop(
            df=df,
            load_image_fn=load_s3_image,
            image_col="s3_image_path",
            embedder=embedder,
            batch_size=batch_size,
        )
    finally:
        # Clean up temp image files
        for tmp_path in local_cache.values():
            with contextlib.suppress(Exception):
                tmp_path.unlink(missing_ok=True)


def _run_embedding_loop(
    df: pd.DataFrame,
    load_image_fn: object,  # Callable[[str], PILImage.Image]
    image_col: str,
    embedder: TorchEmbedder,
    batch_size: int,
) -> np.ndarray:
    """Core embedding loop shared by local and S3 modes.

    Args:
        df: Annotations DataFrame.
        load_image_fn: Callable that takes an image path/URI string and returns
            a PIL Image (RGB).  May raise on failure.
        image_col: Name of the column in *df* that holds image paths/URIs.
        embedder: Embedder instance.
        batch_size: Crops per forward pass.

    Returns:
        Float32 ndarray of shape ``(len(df), embedding_dim)``.
    """
    from collections.abc import Callable

    assert callable(load_image_fn)
    loader: Callable[[str], PILImage.Image] = load_image_fn

    all_vecs: list[np.ndarray] = []
    placeholder: PILImage.Image | None = None

    progress = tqdm(total=len(df), desc="Embedding boxes", unit="box")
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        crops: list[PILImage.Image] = []
        valid: list[bool] = []

        for _, row in batch.iterrows():
            try:
                img = loader(str(row[image_col]))
                crops.append(
                    img.crop(
                        (int(row["x1"]), int(row["y1"]),
                         int(row["x2"]), int(row["y2"]))
                    )
                )
                valid.append(True)
            except Exception:
                logger.warning(
                    "Skipping box (image='%s'): could not load/crop.",
                    row[image_col],
                    exc_info=True,
                )
                if placeholder is None:
                    placeholder = PILImage.new("RGB", (16, 16), color=0)
                crops.append(placeholder)
                valid.append(False)

        vecs = embedder.embed(crops)
        for i, ok in enumerate(valid):
            if not ok:
                vecs[i] = 0.0

        all_vecs.append(vecs)
        progress.update(len(batch))

    progress.close()
    return np.vstack(all_vecs).astype(np.float32)


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create and populate an IndexFlatIP (in-place L2-normalise first)."""
    faiss.normalize_L2(embeddings)
    _, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    logger.info("Built FAISS index: %d vectors, dim=%d.", index.ntotal, d)
    return index


def _write_outputs_local(
    index: faiss.IndexFlatIP,
    df: pd.DataFrame,
    out_dir: Path,
    images_root: Path,
) -> None:
    """Write index, metadata, and images_root atomically to *out_dir*."""
    index_path = out_dir / "index.faiss"
    metadata_path = out_dir / "metadata.parquet"

    tmp_index: Path | None = None
    tmp_meta: Path | None = None
    try:
        fd, tmp_str = tempfile.mkstemp(suffix=".faiss.tmp", dir=out_dir)
        tmp_index = Path(tmp_str)
        os.close(fd)
        faiss.write_index(index, str(tmp_index))

        fd, tmp_str = tempfile.mkstemp(suffix=".parquet.tmp", dir=out_dir)
        tmp_meta = Path(tmp_str)
        os.close(fd)
        df.to_parquet(tmp_meta, index=False)

        tmp_index.replace(index_path)
        tmp_index = None
        tmp_meta.replace(metadata_path)
        tmp_meta = None

        (out_dir / "images_root.txt").write_text(str(images_root), encoding="utf-8")
    except Exception:
        if tmp_index is not None:
            tmp_index.unlink(missing_ok=True)
        if tmp_meta is not None:
            tmp_meta.unlink(missing_ok=True)
        raise


def _write_outputs_s3(
    index: faiss.IndexFlatIP,
    df: pd.DataFrame,
    s3_client: object,
    dataset_name: str,
) -> None:
    """Serialize index and metadata to temp files then upload to S3 atomically."""
    from image_retrieval.s3_client import S3Client

    assert isinstance(s3_client, S3Client)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        index_tmp = tmp_dir / "index.faiss"
        meta_tmp = tmp_dir / "metadata.parquet"

        faiss.write_index(index, str(index_tmp))
        df.to_parquet(meta_tmp, index=False)

        logger.info("Uploading index to S3…")
        s3_client.upload_file_atomic(index_tmp, s3_client.index_key(dataset_name))

        logger.info("Uploading metadata to S3…")
        s3_client.upload_file_atomic(meta_tmp, s3_client.metadata_key(dataset_name))


def main() -> int:
    """Entry point; returns an exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()

    model_class: type[nn.Module] | None = None
    if args.model_module:
        model_class = _resolve_model_class(args.model_module)

    input_size: tuple[int, int] = (args.input_size[0], args.input_size[1])
    embedder = TorchEmbedder(
        checkpoint_path=args.checkpoint,
        model_class=model_class,
        input_size=input_size,
        device=args.device,
    )

    if args.mode == "local":
        df = _load_annotations_local(args.annotations)

        logger.info(
            "Starting embedding pass (batch_size=%d, device=%s)…",
            args.batch_size, args.device,
        )
        embeddings = _embed_all_local(df, args.images_root, embedder, args.batch_size)
        index = _build_faiss_index(embeddings)

        out_dir: Path = args.datasets_dir / args.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs_local(index, df, out_dir, args.images_root.resolve())

        logger.info("✓ index    → %s", out_dir / "index.faiss")
        logger.info("✓ metadata → %s", out_dir / "metadata.parquet")
        return 0

    from image_retrieval.config import S3Config
    from image_retrieval.s3_client import S3Client

    s3_config = S3Config(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        endpoint_url=args.endpoint_url,
    )
    s3_client = S3Client(s3_config)

    # Use a managed temp dir unless the user specified a persistent cache dir
    if args.image_cache_dir is not None:
        cache_dir = args.image_cache_dir
        _tmp_mgr = None
    else:
        _tmp_mgr = tempfile.TemporaryDirectory()
        cache_dir = Path(_tmp_mgr.name)

    try:
        with tempfile.TemporaryDirectory() as anno_tmp:
            df = _load_annotations_s3(s3_client, args.dataset, Path(anno_tmp))

        logger.info(
            "Starting S3 embedding pass (batch_size=%d, device=%s)…",
            args.batch_size, args.device,
        )
        embeddings = _embed_all_s3(df, s3_client, embedder, args.batch_size, cache_dir)
        index = _build_faiss_index(embeddings)

        _write_outputs_s3(index, df, s3_client, args.dataset)

        logger.info(
            "✓ index    → s3://%s/%s", args.bucket, s3_client.index_key(args.dataset)
        )
        logger.info(
            "✓ metadata → s3://%s/%s", args.bucket, s3_client.metadata_key(args.dataset)
        )
    finally:
        if _tmp_mgr is not None:
            _tmp_mgr.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
