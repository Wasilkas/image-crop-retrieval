"""SSL embedding abstraction and PyTorch implementation.

Two public objects are exported:

* :class:`EmbedderProtocol` — a ``@runtime_checkable`` structural type that any
  embedder must satisfy.  Keeps the rest of the codebase decoupled from PyTorch.

* :class:`TorchEmbedder` — concrete implementation that loads a ``.pt`` / ``.pth``
  checkpoint and runs CPU inference.

Supported checkpoint formats
------------------------------
Full-model pickle
    ``torch.save(model, path)``  →  ``torch.load(path)`` returns an ``nn.Module``.
    Use when the model class definition is *not* available in the inference
    environment (the class is baked into the pickle).

State-dict
    ``torch.save(model.state_dict(), path)``  →  ``torch.load(path)`` returns a
    ``dict``.  Requires passing ``model_class`` to ``TorchEmbedder`` so the
    architecture can be instantiated before loading weights.

.. warning::
    ``torch.load`` with ``weights_only=False`` executes arbitrary Python code
    embedded in the pickle.  Only load checkpoints from **trusted sources**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Minimal interface that every embedder must implement.

    Implementors should return **L2-normalised** float32 vectors so that inner
    product search is equivalent to cosine similarity.
    """

    def embed(self, crops: list[PILImage.Image]) -> np.ndarray:
        """Compute embeddings for a batch of PIL images.

        Args:
            crops: Non-empty list of PIL images in any size / mode.
                   Implementations are expected to handle resizing and
                   colour-mode conversion internally.

        Returns:
            Float32 ndarray of shape ``(len(crops), embedding_dim)``.
            Rows are **L2-normalised** (unit vectors).

        Raises:
            ValueError: If ``crops`` is empty.
        """
        ...


class TorchEmbedder:
    """Wraps a custom PyTorch checkpoint for CPU-based SSL inference.

    Args:
        checkpoint_path: Path to the ``.pt`` or ``.pth`` file.
        model_class: Optional ``nn.Module`` *class* (not an instance).
            When provided the checkpoint is assumed to be a state-dict and
            the class is instantiated before loading weights.
            When ``None`` the checkpoint is assumed to be a full-model pickle.
        input_size: ``(height, width)`` to which every crop is resized before
            being fed to the model.  Defaults to ``(224, 224)``.
        device: PyTorch device string.  Defaults to ``"cpu"``.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
        RuntimeError: If the checkpoint format does not match the expected mode
            (e.g. a state-dict dict when no ``model_class`` was given, or vice-versa).
    """

    def __init__(
        self,
        checkpoint_path: Path,
        model_class: type[nn.Module] | None = None,
        input_size: tuple[int, int] = (224, 224),
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._model: nn.Module = self._load_model(checkpoint_path, model_class)
        self._model.eval()
        self._transform = _build_transform(input_size)

    def _load_model(
        self,
        path: Path,
        model_class: type[nn.Module] | None,
    ) -> nn.Module:
        """Load and return the model from *path*.

        Supports both full-model pickle and state-dict formats; the format is
        inferred from whether *model_class* is provided.

        Raises:
            FileNotFoundError: checkpoint file missing.
            RuntimeError: format mismatch between checkpoint and expected mode.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # weights_only=False is required for full-model pickles.
        # Only load checkpoints from trusted sources (see module docstring).
        payload = torch.load(path, map_location=self._device, weights_only=False)  # noqa: S614

        if model_class is not None:
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Expected a state-dict (dict) at '{path}', "
                    f"but torch.load() returned {type(payload).__name__}. "
                    "Remove the model_class argument to use full-model loading."
                )
            model = model_class()
            model.load_state_dict(payload)
            return model.to(self._device)

        if not isinstance(payload, nn.Module):
            raise RuntimeError(
                f"Expected an nn.Module at '{path}', "
                f"but torch.load() returned {type(payload).__name__}. "
                "Provide model_class= for state-dict checkpoints."
            )
        return payload.to(self._device)

    @torch.no_grad()
    def embed(self, crops: list[PILImage.Image]) -> np.ndarray:
        """Embed a batch of crops; returns L2-normalised float32 array ``(N, D)``.

        Args:
            crops: Non-empty list of PIL images.

        Returns:
            Float32 ndarray of shape ``(len(crops), embedding_dim)``,
            where every row is an L2-unit vector.

        Raises:
            ValueError: If *crops* is empty.
        """
        if not crops:
            raise ValueError("crops must be a non-empty list")

        # Convert to RGB, apply transforms, stack into a batch tensor
        tensors = torch.stack(
            [self._transform(crop.convert("RGB")) for crop in crops]
        )
        tensors = tensors.to(self._device)

        raw: torch.Tensor = self._model(tensors)

        # Some models return spatial feature maps (N, C, H, W) — flatten them
        if raw.dim() > 2:
            raw = raw.flatten(start_dim=1)

        # L2-normalise so that IndexFlatIP gives cosine similarity scores
        normed = torch.nn.functional.normalize(raw, p=2, dim=1)
        return normed.cpu().numpy().astype(np.float32)


def _build_transform(input_size: tuple[int, int]) -> torch.nn.Module:
    """Return a torchvision transform pipeline.

    The pipeline resizes to *input_size*, converts to a float tensor, and
    normalises using ImageNet mean/std — a sensible default for most SSL models
    trained on natural images.

    Args:
        input_size: ``(height, width)`` target size.

    Returns:
        A ``torchvision.transforms.Compose`` instance.
    """
    from torchvision import transforms  # deferred import: torchvision is heavy

    # torchvision has no py.typed stubs; cast to the expected type
    return cast(
        torch.nn.Module,
        transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
