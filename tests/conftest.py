"""Shared pytest fixtures for image-crop-retrieval tests."""

from __future__ import annotations

import io
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest
from PIL import Image as PILImage

DIM = 64  # small embedding dimension for speed


@pytest.fixture()
def embedding_dim() -> int:
    """Return the embedding dimensionality used in test fixtures."""
    return DIM


@pytest.fixture()
def sample_embeddings() -> np.ndarray:
    """Return 10 random L2-normalised float32 embeddings of shape (10, DIM)."""
    rng = np.random.default_rng(42)
    vecs = rng.random((10, DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms  # type: ignore[no-any-return]


@pytest.fixture()
def sample_meta_df() -> pd.DataFrame:
    """Return a metadata DataFrame with 10 rows (matching sample_embeddings)."""
    return pd.DataFrame(
        {
            "box_id": [f"box_{i:03d}" for i in range(10)],
            "image_path": [f"img_{i:03d}.jpg" for i in range(10)],
            "x1": [i * 10 for i in range(10)],
            "y1": [i * 5 for i in range(10)],
            "x2": [i * 10 + 50 for i in range(10)],
            "y2": [i * 5 + 30 for i in range(10)],
        }
    )


@pytest.fixture()
def dataset_dir(
    tmp_path: Path,
    sample_embeddings: np.ndarray,
    sample_meta_df: pd.DataFrame,
) -> Path:
    """Write a complete dataset directory (index.faiss + metadata.parquet)."""
    ds = tmp_path / "test_dataset"
    ds.mkdir()

    index = faiss.IndexFlatIP(DIM)
    index.add(sample_embeddings)
    faiss.write_index(index, str(ds / "index.faiss"))

    sample_meta_df.to_parquet(ds / "metadata.parquet", index=False)
    return ds


@pytest.fixture()
def small_image() -> PILImage.Image:
    """Return a tiny 32×32 RGB PIL image (solid colour)."""
    return PILImage.new("RGB", (32, 32), color=(128, 64, 32))


@pytest.fixture()
def small_image_bytes(small_image: PILImage.Image) -> bytes:
    """Return JPEG bytes for small_image."""
    buf = io.BytesIO()
    small_image.save(buf, format="JPEG", quality=80)
    return buf.getvalue()
