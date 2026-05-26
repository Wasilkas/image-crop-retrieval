"""Tests for image_retrieval.embedder — TorchEmbedder with tiny in-memory models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image as PILImage

from image_retrieval.embedder import EmbedderProtocol, TorchEmbedder


class TinyModel(nn.Module):
    """A tiny 3-channel → 8-output linear model (no vision layers)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(start_dim=1))  # type: ignore[no-any-return]


class TinySpatialModel(nn.Module):
    """Returns a spatial output (N, C, H, W) to test flatten logic."""

    def __init__(self, out_channels: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=224, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # type: ignore[no-any-return]  # shape: (N, C, 1, 1)


@pytest.fixture()
def tiny_full_model_path(tmp_path: Path) -> Path:
    path = tmp_path / "full_model.pth"
    torch.save(TinyModel(), path)
    return path


@pytest.fixture()
def tiny_state_dict_path(tmp_path: Path) -> Path:
    path = tmp_path / "state_dict.pth"
    torch.save(TinyModel().state_dict(), path)
    return path


@pytest.fixture()
def spatial_model_path(tmp_path: Path) -> Path:
    path = tmp_path / "spatial.pth"
    torch.save(TinySpatialModel(), path)
    return path


class TestTorchEmbedderConstruction:
    def test_full_model_loads(self, tiny_full_model_path: Path) -> None:
        assert TorchEmbedder(tiny_full_model_path) is not None

    def test_state_dict_loads_with_class(self, tiny_state_dict_path: Path) -> None:
        assert TorchEmbedder(tiny_state_dict_path, model_class=TinyModel) is not None

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TorchEmbedder(tmp_path / "missing.pth")

    def test_state_dict_without_class_raises(self, tiny_state_dict_path: Path) -> None:
        """torch.load of a dict without model_class → RuntimeError."""
        with pytest.raises(RuntimeError, match="state-dict"):
            TorchEmbedder(tiny_state_dict_path)

    def test_full_model_with_wrong_class_raises(
        self, tiny_full_model_path: Path
    ) -> None:
        """torch.load of an nn.Module when model_class is given → RuntimeError."""
        with pytest.raises(RuntimeError, match="state-dict"):
            TorchEmbedder(tiny_full_model_path, model_class=TinyModel)


class TestTorchEmbedderEmbed:
    @pytest.fixture()
    def embedder(self, tiny_full_model_path: Path) -> TorchEmbedder:
        return TorchEmbedder(tiny_full_model_path)

    def test_embed_single_crop(self, embedder: TorchEmbedder) -> None:
        result = embedder.embed([PILImage.new("RGB", (32, 32), color=(100, 150, 200))])
        assert result.shape == (1, 8)
        assert result.dtype == np.float32

    def test_embed_batch(self, embedder: TorchEmbedder) -> None:
        crops = [PILImage.new("RGB", (16, 16), color=(i * 25, 0, 0)) for i in range(4)]
        assert embedder.embed(crops).shape == (4, 8)

    def test_output_is_l2_normalised(self, embedder: TorchEmbedder) -> None:
        result = embedder.embed([PILImage.new("RGB", (32, 32))])
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-5)

    def test_batch_all_rows_l2_normalised(self, embedder: TorchEmbedder) -> None:
        crops = [PILImage.new("RGB", (16, 16)) for _ in range(5)]
        result = embedder.embed(crops)
        np.testing.assert_allclose(
            np.linalg.norm(result, axis=1), np.ones(5), atol=1e-5
        )

    def test_empty_crops_raises(self, embedder: TorchEmbedder) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            embedder.embed([])

    def test_grayscale_image_converted(self, embedder: TorchEmbedder) -> None:
        result = embedder.embed([PILImage.new("L", (32, 32), color=128)])
        assert result.shape == (1, 8)

    def test_rgba_image_converted(self, embedder: TorchEmbedder) -> None:
        result = embedder.embed(
            [PILImage.new("RGBA", (32, 32), color=(10, 20, 30, 128))]
        )
        assert result.shape == (1, 8)


def test_spatial_output_flattened(spatial_model_path: Path) -> None:
    """Models returning (N, C, H, W) should have their output flattened."""
    result = TorchEmbedder(spatial_model_path, input_size=(224, 224)).embed(
        [PILImage.new("RGB", (32, 32))]
    )
    # TinySpatialModel with kernel_size=224 → output (1, 4, 1, 1) → flatten → (1, 4)
    assert result.ndim == 2
    assert result.shape[0] == 1


def test_torch_embedder_satisfies_protocol(tiny_full_model_path: Path) -> None:
    assert isinstance(TorchEmbedder(tiny_full_model_path), EmbedderProtocol)
