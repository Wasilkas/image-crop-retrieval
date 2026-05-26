"""Streamlit widget: image upload + bounding-box crop selector.

The widget renders a ``streamlit-drawable-canvas`` over the uploaded image
and returns the user-drawn crop as a ``(full_image, crop)`` tuple, or
``None`` when the user has not yet drawn a valid rectangle.

Fabric.js coordinate quirks
-----------------------------
The canvas JSON encodes rectangles as Fabric.js objects with the following
fields that need special handling:

* ``left`` / ``top`` — top-left corner *before* any transforms.
* ``width`` / ``height`` — un-scaled dimensions; can be **negative** when the
  user draws right-to-left or bottom-to-top.
* ``scaleX`` / ``scaleY`` — scale transform applied on top of width/height
  (default ``1.0``).  The true pixel size of the rectangle is
  ``width * scaleX`` × ``height * scaleY``.

This module resolves all of the above into a canonical ``(x1, y1, x2, y2)``
bounding box with ``x1 < x2`` and ``y1 < y2``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from PIL import Image as PILImage
from streamlit_drawable_canvas import st_canvas

from ..config import AppConfig


def render_crop_selector(
    config: AppConfig,
) -> tuple[PILImage.Image, PILImage.Image] | None:
    """Render the file-uploader and drawable canvas; return the crop or ``None``.

    The function is **stateless** beyond Streamlit's own widget state — it reads
    widget results and returns them without storing anything in ``session_state``.

    Args:
        config: Application configuration, used for canvas dimensions and
            ``min_crop_px``.

    Returns:
        ``(full_image, crop)`` — the original PIL image at its native resolution
        and the cropped sub-image — when the user has uploaded a file **and**
        drawn a valid rectangle.  ``None`` otherwise.
    """
    uploaded_file = st.file_uploader(
        label="Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Supported formats: JPEG, PNG, BMP, TIFF, WebP",
    )
    if uploaded_file is None:
        return None

    full_image = PILImage.open(uploaded_file).convert("RGB")

    # Resize for display — never upscale, preserve aspect ratio
    display_image = _fit_to_canvas(
        full_image, config.canvas_width, config.canvas_height
    )

    st.markdown(
        "**Draw a rectangle** on the image to select the crop you want to search for."
    )
    st.caption(
        f"Original size: {full_image.width} × {full_image.height} px  ·  "
        f"Displayed at: {display_image.width} × {display_image.height} px"
    )

    canvas_result = st_canvas(
        background_image=display_image,
        drawing_mode="rect",
        height=display_image.height,
        width=display_image.width,
        stroke_color="#FF3333",
        stroke_width=2,
        fill_color="rgba(255, 51, 51, 0.10)",
        key="crop_canvas",
        update_streamlit=True,
    )

    bbox = _extract_rect(canvas_result)
    if bbox is None:
        st.info("⬆️ Draw a rectangle on the image above, then click **Search**.")
        return None

    x1_d, y1_d, x2_d, y2_d = bbox

    # Scale display-space coordinates back to original image space
    scale_x = full_image.width / display_image.width
    scale_y = full_image.height / display_image.height
    x1 = max(0, int(x1_d * scale_x))
    y1 = max(0, int(y1_d * scale_y))
    x2 = min(full_image.width, int(x2_d * scale_x))
    y2 = min(full_image.height, int(y2_d * scale_y))

    if (x2 - x1) < config.min_crop_px or (y2 - y1) < config.min_crop_px:
        st.warning(
            f"The drawn crop is too small "
            f"({x2 - x1} × {y2 - y1} px in the original image).  "
            f"Please draw a box that is at least {config.min_crop_px} px on each side."
        )
        return None

    crop = full_image.crop((x1, y1, x2, y2))

    # Show a small preview of the selected crop
    with st.expander("Selected crop preview", expanded=False):
        caption = (
            f"Crop: [{x1},{y1} – {x2},{y2}] "
            f"({crop.width}×{crop.height} px)"
        )
        st.image(crop, caption=caption)

    return full_image, crop


def _extract_rect(canvas_result: Any) -> tuple[int, int, int, int] | None:
    """Parse the canvas JSON and return the last drawn rectangle.

    Args:
        canvas_result: The object returned by ``st_canvas()``.

    Returns:
        ``(x1, y1, x2, y2)`` in display-space pixels with ``x1 < x2, y1 < y2``,
        or ``None`` if no rectangle has been drawn yet.
    """
    if canvas_result is None or canvas_result.json_data is None:
        return None

    objects: list[dict[str, Any]] = canvas_result.json_data.get("objects", [])
    rects = [obj for obj in objects if obj.get("type") == "rect"]
    if not rects:
        return None

    # Use the most recently drawn rectangle
    obj = rects[-1]

    left: float = float(obj.get("left", 0))
    top: float = float(obj.get("top", 0))
    # Apply scaleX/scaleY transforms (Fabric.js sometimes stores raw + scale)
    width: float = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
    height: float = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))

    # Normalise so x1 < x2 and y1 < y2 regardless of drawing direction
    x1 = int(min(left, left + width))
    y1 = int(min(top, top + height))
    x2 = int(max(left, left + width))
    y2 = int(max(top, top + height))

    return x1, y1, x2, y2


def _fit_to_canvas(
    image: PILImage.Image,
    max_width: int,
    max_height: int,
) -> PILImage.Image:
    """Resize *image* to fit within *(max_width, max_height)*, never upscaling.

    Args:
        image: Source image.
        max_width: Maximum display width in pixels.
        max_height: Maximum display height in pixels.

    Returns:
        A new PIL image at the target size, or the original if no resize needed.
    """
    w, h = image.size
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return image
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h), PILImage.LANCZOS)
