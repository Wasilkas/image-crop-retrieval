"""Tests for image_retrieval.registry — DatasetRegistry (local mode)."""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

from image_retrieval.config import AppBlock
from image_retrieval.registry import DatasetRegistry, _resolve_images_root

DIM = 16


def _write_dataset(ds_dir: Path, n: int = 5, dim: int = DIM) -> None:
    """Write a minimal valid dataset into ds_dir."""
    ds_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(ds_dir / "index.faiss"))

    pd.DataFrame(
        {
            "box_id": [f"b{i}" for i in range(n)],
            "image_path": [f"img_{i}.jpg" for i in range(n)],
            "x1": list(range(n)),
            "y1": list(range(n)),
            "x2": [i + 10 for i in range(n)],
            "y2": [i + 10 for i in range(n)],
        }
    ).to_parquet(ds_dir / "metadata.parquet", index=False)


def _registry(datasets_dir: Path) -> DatasetRegistry:
    return DatasetRegistry(AppBlock(datasets_dir=datasets_dir))


class TestDatasetRegistryDiscovery:
    @pytest.fixture()
    def datasets_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "datasets"
        d.mkdir()
        return d

    def test_empty_datasets_dir(self, datasets_dir: Path) -> None:
        reg = _registry(datasets_dir)
        assert reg.available() == []
        assert len(reg) == 0

    def test_nonexistent_datasets_dir(self, tmp_path: Path) -> None:
        reg = _registry(tmp_path / "nonexistent")
        assert reg.available() == []

    def test_discovers_valid_dataset(self, datasets_dir: Path) -> None:
        _write_dataset(datasets_dir / "my_ds")
        assert _registry(datasets_dir).available() == ["my_ds"]

    def test_discovers_multiple_datasets(self, datasets_dir: Path) -> None:
        for name in ["alpha", "beta", "gamma"]:
            _write_dataset(datasets_dir / name)
        assert _registry(datasets_dir).available() == ["alpha", "beta", "gamma"]

    def test_skips_directory_without_index(self, datasets_dir: Path) -> None:
        _write_dataset(datasets_dir / "valid")
        bad = datasets_dir / "bad"
        bad.mkdir(parents=True)
        pd.DataFrame({"a": [1]}).to_parquet(bad / "metadata.parquet")
        assert _registry(datasets_dir).available() == ["valid"]

    def test_skips_non_directory_files(self, datasets_dir: Path) -> None:
        (datasets_dir / "readme.txt").write_text("ignore me")
        _write_dataset(datasets_dir / "real")
        assert _registry(datasets_dir).available() == ["real"]


class TestDatasetRegistryGet:
    @pytest.fixture()
    def datasets_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "datasets"
        _write_dataset(d / "ds1")
        return d

    def test_get_returns_meta_and_index(self, datasets_dir: Path) -> None:
        reg = _registry(datasets_dir)
        meta, idx = reg.get("ds1")
        assert meta.name == "ds1"
        assert idx.ntotal == 5

    def test_get_unknown_name_raises_key_error(self, datasets_dir: Path) -> None:
        reg = _registry(datasets_dir)
        with pytest.raises(KeyError, match="nonexistent"):
            reg.get("nonexistent")


class TestDatasetRegistryRescan:
    @pytest.fixture()
    def datasets_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "datasets"
        d.mkdir()
        return d

    def test_rescan_finds_new_dataset(self, datasets_dir: Path) -> None:
        reg = _registry(datasets_dir)
        assert reg.available() == []

        _write_dataset(datasets_dir / "new_ds")
        new_names = reg.rescan()
        assert "new_ds" in new_names
        assert "new_ds" in reg.available()

    def test_rescan_returns_only_new_names(self, datasets_dir: Path) -> None:
        _write_dataset(datasets_dir / "existing")
        reg = _registry(datasets_dir)
        _write_dataset(datasets_dir / "added")
        assert reg.rescan() == ["added"]

    def test_rescan_empty_when_nothing_new(self, datasets_dir: Path) -> None:
        _write_dataset(datasets_dir / "ds")
        reg = _registry(datasets_dir)
        assert reg.rescan() == []


class TestDatasetRegistryHotReload:
    def test_hot_reload_on_mtime_change(self, tmp_path: Path) -> None:
        """Modifying index.faiss triggers a transparent hot-reload."""
        datasets_dir = tmp_path / "datasets"
        ds_dir = datasets_dir / "hot"
        _write_dataset(ds_dir, n=5)

        reg = _registry(datasets_dir)
        _, idx1 = reg.get("hot")
        assert idx1.ntotal == 5

        # Rewrite the dataset with more vectors
        _write_dataset(ds_dir, n=8)
        # Force mtime to be newer (filesystem may have 1-second resolution)
        time.sleep(0.01)
        (ds_dir / "index.faiss").touch()

        _, idx2 = reg.get("hot")
        assert idx2.ntotal == 8

    def test_hot_reload_skipped_when_mtime_unchanged(self, tmp_path: Path) -> None:
        datasets_dir = tmp_path / "datasets"
        _write_dataset(datasets_dir / "stable", n=5)
        reg = _registry(datasets_dir)
        _, idx1 = reg.get("stable")
        _, idx2 = reg.get("stable")
        # Same object returned — no reload happened
        assert idx1 is idx2


def test_last_reload_info_returns_mtimes(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_dataset(datasets_dir / "ds")
    reg = _registry(datasets_dir)
    info = reg.last_reload_info("ds")
    assert info is not None
    idx_mtime, meta_mtime = info
    assert idx_mtime > 0
    assert meta_mtime > 0


def test_last_reload_info_none_for_unknown(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    assert _registry(datasets_dir).last_reload_info("nope") is None


class TestResolveImagesRoot:
    def test_default_to_dataset_dir(self, tmp_path: Path) -> None:
        assert _resolve_images_root(tmp_path) == tmp_path

    def test_reads_absolute_path_from_txt(self, tmp_path: Path) -> None:
        target = tmp_path / "images"
        target.mkdir()
        (tmp_path / "images_root.txt").write_text(str(target))
        assert _resolve_images_root(tmp_path) == target

    def test_reads_relative_path_from_txt(self, tmp_path: Path) -> None:
        (tmp_path / "imgs").mkdir()
        (tmp_path / "images_root.txt").write_text("imgs")
        assert _resolve_images_root(tmp_path) == (tmp_path / "imgs").resolve()

    def test_dataset_with_images_root_txt(self, tmp_path: Path) -> None:
        datasets_dir = tmp_path / "datasets"
        ds_dir = datasets_dir / "withroot"
        _write_dataset(ds_dir)
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (ds_dir / "images_root.txt").write_text(str(images_dir))

        meta, _ = _registry(datasets_dir).get("withroot")
        assert meta.images_root == images_dir
