"""Tests for scripts/build_index.py — pure helper functions."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import pytest
import torch.nn as nn

# build_index.py lives in scripts/ which is not an installed package.
# We add the project root to sys.path so it can be imported directly.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_index import (  # type: ignore[import-not-found]  # noqa: E402
    _build_faiss_index,
    _load_annotations_local,
    _resolve_model_class,
    _write_outputs_local,
)


class TestLoadAnnotationsLocal:
    def _make_csv(
        self,
        tmp_path: Path,
        extra_cols: dict[str, Any] | None = None,
        box_id: bool = False,
    ) -> Path:
        """Write a minimal annotations CSV and return its path."""
        data: dict[str, Any] = {
            "image_path": ["img1.jpg", "img2.jpg"],
            "x1": [0, 10],
            "y1": [0, 10],
            "x2": [50, 60],
            "y2": [50, 60],
        }
        if box_id:
            data["box_id"] = ["b0", "b1"]
        if extra_cols:
            data.update(extra_cols)
        csv_path = tmp_path / "annot.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)
        return csv_path

    def test_loads_csv(self, tmp_path: Path) -> None:
        df = _load_annotations_local(self._make_csv(tmp_path))
        assert len(df) == 2
        assert "image_path" in df.columns

    def test_loads_parquet(self, tmp_path: Path) -> None:
        pq = tmp_path / "annot.parquet"
        pd.DataFrame({
            "image_path": ["img.jpg"],
            "x1": [0], "y1": [0], "x2": [50], "y2": [50],
            "box_id": ["b0"],
        }).to_parquet(pq, index=False)
        assert len(_load_annotations_local(pq)) == 1

    def test_auto_generates_box_id(self, tmp_path: Path) -> None:
        df = _load_annotations_local(self._make_csv(tmp_path, box_id=False))
        assert "box_id" in df.columns
        assert list(df["box_id"]) == ["box_0", "box_1"]

    def test_existing_box_id_preserved(self, tmp_path: Path) -> None:
        df = _load_annotations_local(self._make_csv(tmp_path, box_id=True))
        assert list(df["box_id"]) == ["b0", "b1"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_annotations_local(tmp_path / "nope.csv")

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        csv = tmp_path / "bad.csv"
        pd.DataFrame(
            {"image_path": ["a.jpg"], "x1": [0], "y1": [0], "x2": [50]}
        ).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="y2"):
            _load_annotations_local(csv)

    def test_index_reset(self, tmp_path: Path) -> None:
        df = _load_annotations_local(self._make_csv(tmp_path))
        assert list(df.index) == [0, 1]


class TestBuildFaissIndex:
    def test_returns_index_flat_ip(self) -> None:
        vecs = np.random.default_rng(0).random((5, 8)).astype(np.float32)
        assert isinstance(_build_faiss_index(vecs), faiss.IndexFlatIP)

    def test_ntotal_matches_input(self) -> None:
        vecs = np.random.default_rng(1).random((12, 16)).astype(np.float32)
        assert _build_faiss_index(vecs).ntotal == 12

    def test_vectors_are_l2_normalised(self) -> None:
        """After _build_faiss_index the in-place normalised vectors have unit norm."""
        n, d = 6, 8
        vecs = np.ones((n, d), dtype=np.float32)
        _build_faiss_index(vecs)
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), np.ones(n), atol=1e-5)

    def test_exact_self_search_after_build(self) -> None:
        vecs = np.random.default_rng(99).random((4, 8)).astype(np.float32)
        faiss.normalize_L2(vecs)
        original = vecs.copy()
        index = _build_faiss_index(vecs)

        q = original[0:1].copy()
        faiss.normalize_L2(q)
        distances, indices = index.search(q, 1)
        assert indices[0][0] == 0
        assert abs(distances[0][0] - 1.0) < 1e-5


class TestWriteOutputsLocal:
    @pytest.fixture()
    def out_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "out"
        d.mkdir()
        return d

    def _make_index_and_df(
        self, n: int = 5, d: int = 8
    ) -> tuple[faiss.IndexFlatIP, pd.DataFrame]:
        vecs = np.random.default_rng(7).random((n, d)).astype(np.float32)
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(d)
        index.add(vecs)
        df = pd.DataFrame({
            "box_id": [f"b{i}" for i in range(n)],
            "image_path": [f"img_{i}.jpg" for i in range(n)],
            "x1": list(range(n)),
            "y1": list(range(n)),
            "x2": [i + 10 for i in range(n)],
            "y2": [i + 10 for i in range(n)],
        })
        return index, df

    def test_creates_index_faiss(self, tmp_path: Path, out_dir: Path) -> None:
        index, df = self._make_index_and_df()
        _write_outputs_local(index, df, out_dir, tmp_path)
        assert (out_dir / "index.faiss").exists()

    def test_creates_metadata_parquet(self, tmp_path: Path, out_dir: Path) -> None:
        index, df = self._make_index_and_df()
        _write_outputs_local(index, df, out_dir, tmp_path)
        assert (out_dir / "metadata.parquet").exists()

    def test_creates_images_root_txt(self, tmp_path: Path, out_dir: Path) -> None:
        index, df = self._make_index_and_df()
        images_root = tmp_path / "images"
        _write_outputs_local(index, df, out_dir, images_root)
        txt = out_dir / "images_root.txt"
        assert txt.exists()
        assert str(images_root) in txt.read_text()

    def test_roundtrip_loadable(self, tmp_path: Path, out_dir: Path) -> None:
        """Written files can be read back by FAISSIndex."""
        from image_retrieval.indexer import FAISSIndex

        index, df = self._make_index_and_df(n=5, d=8)
        _write_outputs_local(index, df, out_dir, tmp_path)
        fi = FAISSIndex(out_dir / "index.faiss", out_dir / "metadata.parquet")
        assert fi.ntotal == 5
        assert fi.embedding_dim == 8

    def test_no_stale_tmp_files(self, tmp_path: Path, out_dir: Path) -> None:
        """After a successful write no .tmp files remain in out_dir."""
        index, df = self._make_index_and_df()
        _write_outputs_local(index, df, out_dir, tmp_path)
        assert list(out_dir.glob("*.tmp")) == []


class TestResolveModelClass:
    def test_resolves_builtin_module(self) -> None:
        cls = _resolve_model_class("torch.nn:Linear")
        assert issubclass(cls, nn.Module)
        assert cls is nn.Linear

    def test_missing_colon_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="module.path:ClassName"):
            _resolve_model_class("torch.nn.Linear")

    def test_non_module_class_raises_type_error(self) -> None:
        """pathlib.Path is not an nn.Module subclass → TypeError."""
        with pytest.raises(TypeError, match="nn.Module"):
            _resolve_model_class("pathlib:Path")

    def test_missing_module_raises_module_not_found(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            _resolve_model_class("nonexistent_package:SomeClass")
