"""Streamlit-виджет: загрузка изображения и выбор кропа рамкой.

Виджет отображает ``streamlit-drawable-canvas`` поверх загруженного изображения
и возвращает нарисованный пользователем кроп в виде кортежа ``(полное_изображение,
кроп)`` или ``None``, если пользователь ещё не нарисовал корректный прямоугольник.

Особенности координат Fabric.js
---------------------------------
JSON холста кодирует прямоугольники как объекты Fabric.js со следующими полями,
требующими особой обработки:

* ``left`` / ``top`` — левый верхний угол *до* применения трансформаций.
* ``width`` / ``height`` — размеры без масштабирования; могут быть **отрицательными**
  при рисовании справа налево или снизу вверх.
* ``scaleX`` / ``scaleY`` — масштабная трансформация поверх width/height
  (по умолч. ``1.0``).  Фактический пиксельный размер:
  ``width * scaleX`` × ``height * scaleY``.

Модуль приводит всё перечисленное к каноническому ``(x1, y1, x2, y2)``
с гарантией ``x1 < x2`` и ``y1 < y2``.
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
    """Отображает загрузчик файлов и холст; возвращает кроп или ``None``.

    Функция **не имеет состояния** за пределами встроенного состояния виджетов
    Streamlit — читает результаты виджетов и возвращает их без сохранения в
    ``session_state``.

    Args:
        config: Конфигурация приложения; используются размеры холста и
            ``min_crop_px``.

    Returns:
        ``(полное_изображение, кроп)`` — оригинальное PIL-изображение в исходном
        разрешении и вырезанный фрагмент — когда пользователь загрузил файл
        **и** нарисовал корректный прямоугольник.  ``None`` в противном случае.
    """
    uploaded_file = st.file_uploader(
        label="Загрузите изображение",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Поддерживаемые форматы: JPEG, PNG, BMP, TIFF, WebP",
    )
    if uploaded_file is None:
        return None

    full_image = PILImage.open(uploaded_file).convert("RGB")

    # Изменяем размер для отображения — никогда не увеличиваем, сохраняем пропорции
    display_image = _fit_to_canvas(
        full_image, config.canvas_width, config.canvas_height
    )

    st.markdown(
        "**Нарисуйте прямоугольник** на изображении, чтобы выбрать кроп для поиска."
    )
    st.caption(
        f"Исходный размер: {full_image.width} × {full_image.height} пкс  ·  "
        f"Отображается как: {display_image.width} × {display_image.height} пкс"
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
        st.info("⬆️ Нарисуйте прямоугольник на изображении, затем нажмите **Найти**.")
        return None

    x1_d, y1_d, x2_d, y2_d = bbox

    # Масштабируем координаты холста обратно в пространство исходного изображения
    scale_x = full_image.width / display_image.width
    scale_y = full_image.height / display_image.height
    x1 = max(0, int(x1_d * scale_x))
    y1 = max(0, int(y1_d * scale_y))
    x2 = min(full_image.width, int(x2_d * scale_x))
    y2 = min(full_image.height, int(y2_d * scale_y))

    if (x2 - x1) < config.min_crop_px or (y2 - y1) < config.min_crop_px:
        st.warning(
            f"Нарисованный кроп слишком мал "
            f"({x2 - x1} × {y2 - y1} пкс в исходном изображении).  "
            f"Нарисуйте рамку не менее {config.min_crop_px} пкс с каждой стороны."
        )
        return None

    crop = full_image.crop((x1, y1, x2, y2))

    # Небольшой предпросмотр выбранного кропа
    with st.expander("Предпросмотр выбранного кропа", expanded=False):
        caption = (
            f"Кроп: [{x1},{y1} – {x2},{y2}] "
            f"({crop.width}×{crop.height} пкс)"
        )
        st.image(crop, caption=caption)

    return full_image, crop


def _extract_rect(canvas_result: Any) -> tuple[int, int, int, int] | None:
    """Разбирает JSON холста и возвращает последний нарисованный прямоугольник.

    Args:
        canvas_result: Объект, возвращённый ``st_canvas()``.

    Returns:
        ``(x1, y1, x2, y2)`` в пикселях пространства холста с ``x1 < x2, y1 < y2``,
        или ``None`` если прямоугольник ещё не нарисован.
    """
    if canvas_result is None or canvas_result.json_data is None:
        return None

    objects: list[dict[str, Any]] = canvas_result.json_data.get("objects", [])
    rects = [obj for obj in objects if obj.get("type") == "rect"]
    if not rects:
        return None

    # Используем последний нарисованный прямоугольник
    obj = rects[-1]

    left: float = float(obj.get("left", 0))
    top: float = float(obj.get("top", 0))
    # Применяем трансформации scaleX/scaleY (Fabric.js иногда хранит raw + scale)
    width: float = float(obj.get("width", 0)) * float(obj.get("scaleX", 1.0))
    height: float = float(obj.get("height", 0)) * float(obj.get("scaleY", 1.0))

    # Нормализуем: x1 < x2 и y1 < y2 независимо от направления рисования
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
    """Изменяет размер *image* до *(max_width, max_height)* без увеличения.

    Args:
        image: Исходное изображение.
        max_width: Максимальная ширина отображения в пикселях.
        max_height: Максимальная высота отображения в пикселях.

    Returns:
        Новое PIL-изображение нужного размера, или оригинал если изменение не нужно.
    """
    w, h = image.size
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return image
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h), PILImage.LANCZOS)
