"""Streamlit results grid: display top-K retrieved bounding-box crops.

Each cell shows the cropped image patch, cosine-similarity score, bounding-box
coordinates, source filename, and a **checkbox** for multi-selection.

Selected results are stored in ``st.session_state["selected_box_ids"]``
(a ``set[str]``).  The CVAT export panel in
:mod:`image_retrieval.ui.cvat_exporter` reads this set to determine which
crops to export.

Images are loaded from the local filesystem by default.  When *s3_client* is
passed and an image path starts with ``s3://``, the image is fetched from S3.

Individual image-load failures are shown as error tiles so the rest of the
grid continues to render normally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from PIL import Image as PILImage

from ..config import AppBlock
from ..indexer import SearchResult

if TYPE_CHECKING:
    from ..s3_client import S3Client

logger = logging.getLogger(__name__)

# Session-state key that holds the set of selected box IDs.
_SELECTED_KEY = "selected_box_ids"


def get_selected_box_ids() -> set[str]:
    """Return the current selection set, initialising it if absent."""
    if _SELECTED_KEY not in st.session_state:
        st.session_state[_SELECTED_KEY] = set()
    return st.session_state[_SELECTED_KEY]  # type: ignore[no-any-return]


def render_results(
    results: list[SearchResult],
    images_root: Path,
    config: AppBlock,
    s3_client: S3Client | None = None,
) -> None:
    """Render *results* as a responsive grid of cropped images with checkboxes.

    Each cell has a checkbox for multi-selection.  Selected items accumulate
    in ``st.session_state["selected_box_ids"]`` and persist across reruns.
    A **Clear selection** button resets the set.

    Args:
        results: Ordered list (best match first) from :meth:`FAISSIndex.search`.
        images_root: Base directory for resolving local ``image_path`` values.
            Ignored when *s3_client* is provided and paths start with ``s3://``.
        config: Application configuration; ``results_columns`` is used.
        s3_client: Optional S3 client.  When provided, image paths that start
            with ``s3://`` are loaded from S3 instead of the local filesystem.
    """
    if not results:
        st.warning("No results found.")
        return

    selected = get_selected_box_ids()

    hdr_col, clear_col = st.columns([6, 1])
    with hdr_col:
        n_sel = len(selected)
        label = f"Top {len(results)} results"
        if n_sel:
            label += f"  ·  **{n_sel} selected**"
        st.subheader(label)
    with clear_col:
        if selected and st.button(
            "✖ Clear",
            help="Deselect all results.",
            use_container_width=True,
        ):
            st.session_state[_SELECTED_KEY] = set()
            st.rerun()

    cols = st.columns(config.results_columns)
    for i, result in enumerate(results):
        with cols[i % config.results_columns]:
            _render_single_result(result, images_root, s3_client, selected)



def _render_single_result(
    result: SearchResult,
    images_root: Path,
    s3_client: S3Client | None,
    selected: set[str],
) -> None:
    """Render one result cell: checkbox + image + caption."""
    is_selected = result.box_id in selected
    checked = st.checkbox(
        label=result.box_id,
        value=is_selected,
        key=f"sel_{result.box_id}",
        label_visibility="collapsed",
    )
    # Sync checkbox state back to the shared selection set
    if checked and not is_selected:
        selected.add(result.box_id)
    elif not checked and is_selected:
        selected.discard(result.box_id)

    crop = _load_crop(result, images_root, s3_client)
    if crop is not None:
        st.image(crop, use_container_width=True)
    else:
        st.error(f"⚠️ Image not found:\n`{result.image_path}`")

    score_pct = result.score * 100
    st.caption(
        f"**Score:** {score_pct:.1f}%  \n"
        f"**Box:** [{result.x1}, {result.y1} – {result.x2}, {result.y2}]  \n"
        f"`{Path(result.image_path).name}`"
    )


def _load_crop(
    result: SearchResult,
    images_root: Path,
    s3_client: S3Client | None,
) -> PILImage.Image | None:
    """Load and return the cropped region for *result*.

    Routing logic:
    1. If ``result.image_path`` starts with ``s3://`` **and** *s3_client* is
       provided → fetch from S3.
    2. Otherwise → resolve against *images_root* (or as an absolute path) and
       load from local disk.

    Returns ``None`` on any failure so the grid cell degrades gracefully.
    """
    try:
        if result.image_path.startswith("s3://") and s3_client is not None:
            return _load_crop_from_s3(result, s3_client)
        return _load_crop_from_local(result, images_root)
    except Exception:
        logger.warning(
            "Failed to load/crop image for box_id='%s' (path='%s').",
            result.box_id,
            result.image_path,
            exc_info=True,
        )
        return None


def _load_crop_from_s3(
    result: SearchResult,
    s3_client: S3Client,
) -> PILImage.Image | None:
    """Fetch image from S3 and return the cropped region."""
    try:
        img = s3_client.load_image(result.image_path)
        return img.crop((result.x1, result.y1, result.x2, result.y2))
    except Exception:
        logger.warning(
            "S3 image load failed for box_id='%s', uri='%s'.",
            result.box_id,
            result.image_path,
            exc_info=True,
        )
        return None


def _load_crop_from_local(
    result: SearchResult,
    images_root: Path,
) -> PILImage.Image | None:
    """Load image from the local filesystem and return the cropped region."""
    candidates = [
        images_root / result.image_path,
        Path(result.image_path),
    ]
    for path in candidates:
        if path.exists():
            try:
                img = PILImage.open(path).convert("RGB")
                return img.crop((result.x1, result.y1, result.x2, result.y2))
            except Exception:
                logger.warning(
                    "Failed to open/crop '%s' for box_id='%s'.",
                    path,
                    result.box_id,
                    exc_info=True,
                )
                return None

    logger.warning(
        "Image not found for box_id='%s': tried %s",
        result.box_id,
        [str(p) for p in candidates],
    )
    return None
