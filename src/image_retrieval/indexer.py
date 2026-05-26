"""FAISS index wrapper for cosine-similarity nearest-neighbour search.

Design
------
We use ``faiss.IndexFlatIP`` (inner-product / dot-product index) together with
**L2-normalised** embedding vectors.  For unit-norm vectors, inner product is
mathematically equivalent to cosine similarity, so the scores returned are in
the range ``[-1, 1]`` (higher = more similar).

The index is **read-only** at app runtime.  Building and writing the index is
handled entirely by ``scripts/build_index.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SearchResult:
    """A single retrieved bounding-box hit.

    Attributes:
        box_id: Unique identifier for the bounding box (string).
        image_path: Relative (or absolute) path to the source image as stored
            in the metadata.  Resolve against ``DatasetMeta.images_root`` to
            get the absolute path.
        x1: Left edge of the bounding box (pixels).
        y1: Top edge of the bounding box (pixels).
        x2: Right edge of the bounding box (pixels).
        y2: Bottom edge of the bounding box (pixels).
        score: Cosine similarity score in ``[-1, 1]``; higher is more similar.
    """

    box_id: str
    image_path: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


# Required columns in the metadata Parquet file.
_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"image_path", "x1", "y1", "x2", "y2", "box_id"}
)


class FAISSIndex:
    """Wraps a pre-built ``faiss.IndexFlatIP`` for read-only nearest-neighbour search.

    The index and metadata are loaded from disk once at construction and kept
    in memory for the lifetime of the object.

    Args:
        index_path: Absolute path to a ``*.faiss`` file written by
            ``faiss.write_index()``.
        metadata_path: Absolute path to the ``metadata.parquet`` file.
            Must contain at minimum the columns: ``image_path``, ``x1``,
            ``y1``, ``x2``, ``y2``, ``box_id``.

    Raises:
        FileNotFoundError: If either *index_path* or *metadata_path* is missing.
        ValueError: If *metadata_path* is missing required columns, or if the
            number of metadata rows does not match the number of index vectors.
    """

    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata Parquet not found: {metadata_path}")

        # faiss.read_index returns faiss.Index (base class); narrowing to it is correct
        self._index: faiss.Index = faiss.read_index(str(index_path))
        self._metadata: pd.DataFrame = pd.read_parquet(metadata_path)

        self._validate()

    def _validate(self) -> None:
        """Check that required metadata columns exist and row counts match."""
        missing = _REQUIRED_COLUMNS - set(self._metadata.columns)
        if missing:
            raise ValueError(
                f"Metadata is missing required columns: {sorted(missing)}"
            )
        if len(self._metadata) != self._index.ntotal:
            raise ValueError(
                f"Metadata has {len(self._metadata)} rows but FAISS index has "
                f"{self._index.ntotal} vectors — they must match."
            )

    @property
    def ntotal(self) -> int:
        """Number of vectors stored in the index."""
        return int(self._index.ntotal)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the stored embedding vectors."""
        return int(self._index.d)

    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        """Find the *top_k* most similar bounding boxes to *query*.

        Args:
            query: Float32 ndarray of shape ``(1, D)`` — must be L2-normalised.
            top_k: Number of results to return.  Automatically capped at
                ``self.ntotal`` so as not to request more results than exist.

        Returns:
            List of :class:`SearchResult` ordered by **descending** cosine
            similarity (best match first).

        Raises:
            ValueError: If *query* is not shape ``(1, D)``, or *top_k* < 1,
                or the index is empty.
        """
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError(
                f"query must have shape (1, D), got {query.shape}"
            )
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.ntotal == 0:
            raise ValueError("The FAISS index is empty — run build_index.py first.")

        k = min(top_k, self.ntotal)
        distances, indices = self._index.search(query.astype(np.float32), k)
        # distances / indices have shape (1, k)

        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            # FAISS returns index -1 when fewer results exist than requested
            if idx < 0:
                continue
            row = self._metadata.iloc[int(idx)]
            results.append(
                SearchResult(
                    box_id=str(row["box_id"]),
                    image_path=str(row["image_path"]),
                    x1=int(row["x1"]),
                    y1=int(row["y1"]),
                    x2=int(row["x2"]),
                    y2=int(row["y2"]),
                    score=float(dist),
                )
            )
        return results
