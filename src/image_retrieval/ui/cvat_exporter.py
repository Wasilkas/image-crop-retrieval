"""Streamlit widget: export selected crops to a CVAT task.

Renders an expandable panel below the results grid.  When the user has
selected at least one result via the checkboxes in
:mod:`~image_retrieval.ui.results_viewer`, this panel shows:

* The count of selected crops and unique source images.
* An optional task-name text input.
* An **Export to CVAT** button that:
  1. Loads each unique source image (local or S3).
  2. Creates a new CVAT task with the configured label.
  3. Uploads the images and pushes bbox annotations.
  4. Displays a clickable link to the created task.

All CVAT interaction is delegated to :class:`~image_retrieval.cvat_client.CVATClient`.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from PIL import Image as PILImage

from ..config import CVATBlock
from ..cvat_client import CVATClient, prepare_export
from ..indexer import SearchResult
from .results_viewer import get_selected_box_ids

if TYPE_CHECKING:
    from ..s3_client import S3Client

logger = logging.getLogger(__name__)


def render_cvat_exporter(
    all_results: list[SearchResult],
    images_root: Path,
    cvat_config: CVATBlock,
    s3_client: S3Client | None = None,
) -> None:
    """Render the CVAT export panel below the results grid.

    The panel is only shown when at least one result is selected.  It
    collapses automatically to an expander so it doesn't crowd the page.

    Args:
        all_results: The full list of search results (same as passed to
            :func:`~results_viewer.render_results`).
        images_root: Base directory for resolving local image paths.
        cvat_config: CVAT connection settings.
        s3_client: Optional S3 client for ``s3://`` image paths.
    """
    selected_ids = get_selected_box_ids()
    if not selected_ids:
        return  # Nothing selected — don't show the panel

    selected_results = [r for r in all_results if r.box_id in selected_ids]
    n_images = len({r.image_path for r in selected_results})

    with st.expander(
        f"📤 Export {len(selected_results)} crops "
        f"from {n_images} image(s) to CVAT",
        expanded=True,
    ):
        task_name = st.text_input(
            label="Task name",
            placeholder="image-crop-retrieval-YYYYMMDD-HHMMSS",
            help=(
                "Name for the new CVAT task.  Leave blank to auto-generate "
                "from the current timestamp."
            ),
        )

        st.caption(
            f"**CVAT URL:** `{cvat_config.url}`  \n"
            f"**Label:** `{cvat_config.task_label}`"
            + (
                f"  \n**Project ID:** `{cvat_config.project_id}`"
                if cvat_config.project_id is not None
                else ""
            )
        )

        if st.button(
            "📤 Export to CVAT",
            type="primary",
            use_container_width=True,
        ):
            _run_export(
                selected_results=selected_results,
                images_root=images_root,
                cvat_config=cvat_config,
                s3_client=s3_client,
                task_name=task_name.strip(),
            )



def _image_loader(
    image_path: str,
    images_root: Path,
    s3_client: S3Client | None,
) -> bytes:
    """Load *image_path* and return raw JPEG bytes.

    Routes to S3 or local filesystem based on the URI scheme, exactly as
    :func:`~results_viewer._load_crop` does for crops.

    Args:
        image_path: Absolute path, relative path (resolved against
            *images_root*), or ``s3://bucket/key`` URI.
        images_root: Base directory for relative paths.
        s3_client: S3 client; required when ``image_path`` is an S3 URI.

    Returns:
        JPEG-encoded image bytes.

    Raises:
        FileNotFoundError: When a local image cannot be found.
        RuntimeError: When an S3 URI is given but no client is available.
    """
    img: PILImage.Image

    if image_path.startswith("s3://"):
        if s3_client is None:
            raise RuntimeError(
                f"Cannot load '{image_path}': S3 client not configured."
            )
        img = s3_client.load_image(image_path)
    else:
        candidates = [images_root / image_path, Path(image_path)]
        for p in candidates:
            if p.exists():
                img = PILImage.open(p).convert("RGB")
                break
        else:
            raise FileNotFoundError(
                f"Image not found: tried {[str(c) for c in candidates]}"
            )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _run_export(
    selected_results: list[SearchResult],
    images_root: Path,
    cvat_config: CVATBlock,
    s3_client: S3Client | None,
    task_name: str,
) -> None:
    """Orchestrate the full export flow with a progress spinner."""

    def load_bytes(path: str) -> bytes:
        return _image_loader(path, images_root, s3_client)

    with st.spinner("Preparing images…"):
        try:
            export_data = prepare_export(
                results=selected_results,
                load_image_bytes=load_bytes,
                task_name=task_name,
            )
        except Exception as exc:
            st.error(f"Failed to load images: {exc}", icon="🚫")
            logger.exception("Image load failed during CVAT export.")
            return

    with st.spinner(
        f"Uploading {len(export_data.images)} image(s) to CVAT…"
    ):
        try:
            client = CVATClient(cvat_config)
            result = client.export_to_task(
                task_name=export_data.task_name,
                images=export_data.images,
                annotations=export_data.annotations,
            )
        except Exception as exc:
            st.error(f"CVAT export failed: {exc}", icon="🚫")
            logger.exception("CVAT export failed.")
            return

    st.success(
        f"✅ CVAT task created: **{result.annotation_count}** annotations "
        f"across **{result.image_count}** image(s).",
        icon="✅",
    )
    st.link_button(
        label="🔗 Open task in CVAT",
        url=result.task_url,
        use_container_width=True,
    )
