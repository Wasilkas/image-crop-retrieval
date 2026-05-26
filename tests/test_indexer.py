"""Tests for image_retrieval.indexer — FAISSIndex and SearchResult."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

from image_retrieval.indexer import FAISSIndex, SearchResult

DIM = 64


def _write_dataset(
    tmp_path: Path,
    n: int = 10,
    dim: int = DIM,
    seed: int = 0,
) -> tuple[Path, Path, np.ndarray]:
    """Write index.faiss + metadata.parquet into tmp_path.

    Returns paths to both files and the L2-normalised embedding matrix.
    """
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    index_path = tmp_path / "index.faiss"
    faiss.write_index(index, str(index_path))

    meta = pd.DataFrame(
        {
            "box_id": [f"box_{i}" for i in range(n)],
            "image_path": [f"img_{i}.jpg" for i in range(n)],
            "x1": list(range(n)),
            "y1": list(range(n)),
            "x2": [i + 50 for i in range(n)],
            "y2": [i + 30 for i in range(n)],
        }
    )
    meta_path = tmp_path / "metadata.parquet"
    meta.to_parquet(meta_path, index=False)

    return index_path, meta_path, vecs


class TestFAISSIndexConstruction:
    def test_loads_valid_dataset(self, tmp_path: Path) -> None:
        idx_path, meta_path, _ = _write_dataset(tmp_path)
        fi = FAISSIndex(idx_path, meta_path)
        assert fi.ntotal == 10
        assert fi.embedding_dim == DIM

    def test_missing_index_file(self, tmp_path: Path) -> None:
        _, meta_path, _ = _write_dataset(tmp_path)
        with pytest.raises(FileNotFoundError, match="index"):
            FAISSIndex(tmp_path / "nonexistent.faiss", meta_path)

    def test_missing_metadata_file(self, tmp_path: Path) -> None:
        idx_path, _, _ = _write_dataset(tmp_path)
        with pytest.raises(FileNotFoundError, match="Metadata"):
            FAISSIndex(idx_path, tmp_path / "nonexistent.parquet")

    def test_missing_required_column(self, tmp_path: Path) -> None:
        idx_path, meta_path, _ = _write_dataset(tmp_path)
        # Overwrite metadata without 'box_id' column
        bad_meta = pd.DataFrame(
            {
                "image_path": [f"img_{i}.jpg" for i in range(10)],
                "x1": list(range(10)),
                "y1": list(range(10)),
                "x2": [i + 50 for i in range(10)],
                "y2": [i + 30 for i in range(10)],
            }
        )
        bad_meta.to_parquet(meta_path, index=False)
        with pytest.raises(ValueError, match="box_id"):
            FAISSIndex(idx_path, meta_path)

    def test_row_count_mismatch(self, tmp_path: Path) -> None:
        idx_path, meta_path, _ = _write_dataset(tmp_path, n=10)
        # Write metadata with 5 rows (mismatch with 10-vector index)
        bad_meta = pd.DataFrame(
            {
                "box_id": [f"b{i}" for i in range(5)],
                "image_path": [f"img_{i}.jpg" for i in range(5)],
                "x1": list(range(5)),
                "y1": list(range(5)),
                "x2": [i + 50 for i in range(5)],
                "y2": [i + 30 for i in range(5)],
            }
        )
        bad_meta.to_parquet(meta_path, index=False)
        with pytest.raises(ValueError, match="match"):
            FAISSIndex(idx_path, meta_path)


class TestFAISSIndexSearch:
    def _build(self, tmp_path: Path, n: int = 10, seed: int = 0) -> FAISSIndex:
        idx_path, meta_path, _ = _write_dataset(tmp_path, n=n, seed=seed)
        return FAISSIndex(idx_path, meta_path)

    def _unit_query(self, seed: int = 99) -> np.ndarray:
        rng = np.random.default_rng(seed)
        q = rng.random((1, DIM)).astype(np.float32)
        q /= np.linalg.norm(q)
        return q

    def test_top_k_results_ordered_by_score(self, tmp_path: Path) -> None:
        fi = self._build(tmp_path)
        results = fi.search(self._unit_query(), top_k=5)
        assert len(results) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_returns_search_result_dataclass(self, tmp_path: Path) -> None:
        fi = self._build(tmp_path)
        results = fi.search(self._unit_query(seed=0), top_k=1)
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, SearchResult)
        assert r.box_id.startswith("box_")
        assert r.image_path.endswith(".jpg")
        assert r.x2 > r.x1
        assert r.y2 > r.y1
        assert -1.0 <= r.score <= 1.0

    def test_exact_self_match(self, tmp_path: Path) -> None:
        """The best match for a stored vector should be itself (score ~1.0)."""
        idx_path, meta_path, vecs = _write_dataset(tmp_path)
        fi = FAISSIndex(idx_path, meta_path)
        results = fi.search(vecs[0:1].copy(), top_k=1)
        assert abs(results[0].score - 1.0) < 1e-5
        assert results[0].box_id == "box_0"

    def test_top_k_capped_at_ntotal(self, tmp_path: Path) -> None:
        fi = self._build(tmp_path, n=5)
        results = fi.search(self._unit_query(), top_k=100)
        assert len(results) == 5

    def test_bad_query_shape_raises(self, tmp_path: Path) -> None:
        fi = self._build(tmp_path)
        bad = np.ones((DIM,), dtype=np.float32)  # 1-D, not (1, D)
        with pytest.raises(ValueError, match="shape"):
            fi.search(bad, top_k=1)

    def test_top_k_zero_raises(self, tmp_path: Path) -> None:
        fi = self._build(tmp_path)
        with pytest.raises(ValueError, match="top_k"):
            fi.search(np.ones((1, DIM), dtype=np.float32), top_k=0)

    def test_empty_index_raises(self, tmp_path: Path) -> None:
        index = faiss.IndexFlatIP(DIM)
        idx_path = tmp_path / "empty.faiss"
        faiss.write_index(index, str(idx_path))

        meta = pd.DataFrame(
            {col: [] for col in ["box_id", "image_path", "x1", "y1", "x2", "y2"]}
        )
        meta_path = tmp_path / "meta.parquet"
        meta.to_parquet(meta_path, index=False)

        fi = FAISSIndex(idx_path, meta_path)
        with pytest.raises(ValueError, match="empty"):
            fi.search(np.ones((1, DIM), dtype=np.float32), top_k=1)


def test_search_result_frozen() -> None:
    r = SearchResult(
        box_id="b1", image_path="img.jpg", x1=0, y1=0, x2=100, y2=100, score=0.9
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.score = 0.5  # type: ignore[misc]
