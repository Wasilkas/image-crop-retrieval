"""Сетка результатов Streamlit: отображение топ-K найденных bounding-box кропов.

Каждая ячейка показывает вырезанный фрагмент изображения, оценку косинусного
сходства, координаты bounding-box, имя файла и **чекбокс** для множественного
выбора.

Выбранные результаты хранятся в ``st.session_state["selected_box_ids"]``
(``set[str]``).  Панель экспорта CVAT в
:mod:`image_retrieval.ui.cvat_exporter` читает этот набор для определения
кропов к экспорту.

Изображения по умолчанию загружаются из локальной файловой системы.  Если
передан *s3_client* и путь к изображению начинается с ``s3://`` — изображение
загружается из S3.

Ошибки загрузки отдельных изображений отображаются как плитки с ошибкой,
остальные ячейки сетки продолжают отрисовываться нормально.
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

# Ключ session_state, хранящий набор выбранных box ID
_SELECTED_KEY = "selected_box_ids"


def get_selected_box_ids() -> set[str]:
    """Возвращает текущий набор выбранных ID, инициализируя его при отсутствии."""
    if _SELECTED_KEY not in st.session_state:
        st.session_state[_SELECTED_KEY] = set()
    return st.session_state[_SELECTED_KEY]  # type: ignore[no-any-return]


def render_results(
    results: list[SearchResult],
    images_root: Path,
    config: AppBlock,
    s3_client: S3Client | None = None,
) -> None:
    """Отображает *results* как адаптивную сетку кропов с чекбоксами.

    Каждая ячейка содержит чекбокс для множественного выбора.  Выбранные
    элементы накапливаются в ``st.session_state["selected_box_ids"]`` и
    сохраняются между перезапусками.  Кнопка **Снять выбор** сбрасывает набор.

    Args:
        results: Упорядоченный список (лучшее совпадение первым) из
            :meth:`FAISSIndex.search`.
        images_root: Базовая директория для разрешения локальных ``image_path``.
            Игнорируется если *s3_client* задан и пути начинаются с ``s3://``.
        config: Конфигурация приложения; используется ``results_columns``.
        s3_client: Необязательный S3-клиент.  При наличии пути, начинающиеся
            с ``s3://``, загружаются из S3 вместо локальной файловой системы.
    """
    if not results:
        st.warning("Результаты не найдены.")
        return

    selected = get_selected_box_ids()

    hdr_col, clear_col = st.columns([6, 1])
    with hdr_col:
        n_sel = len(selected)
        label = f"Топ {len(results)} результатов"
        if n_sel:
            label += f"  ·  **{n_sel} выбрано**"
        st.subheader(label)
    with clear_col:
        if selected and st.button(
            "✖ Сбросить",
            help="Снять выбор со всех результатов.",
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
    """Отображает одну ячейку результата: чекбокс + изображение + подпись."""
    is_selected = result.box_id in selected
    checked = st.checkbox(
        label=result.box_id,
        value=is_selected,
        key=f"sel_{result.box_id}",
        label_visibility="collapsed",
    )
    # Синхронизируем состояние чекбокса с общим набором выбранных
    if checked and not is_selected:
        selected.add(result.box_id)
    elif not checked and is_selected:
        selected.discard(result.box_id)

    crop = _load_crop(result, images_root, s3_client)
    if crop is not None:
        st.image(crop, use_container_width=True)
    else:
        st.error(f"⚠️ Изображение не найдено:\n`{result.image_path}`")

    score_pct = result.score * 100
    st.caption(
        f"**Сходство:** {score_pct:.1f}%  \n"
        f"**Кроп:** [{result.x1}, {result.y1} – {result.x2}, {result.y2}]  \n"
        f"`{Path(result.image_path).name}`"
    )


def _load_crop(
    result: SearchResult,
    images_root: Path,
    s3_client: S3Client | None,
) -> PILImage.Image | None:
    """Загружает и возвращает вырезанный фрагмент для *result*.

    Логика маршрутизации:
    1. Если ``result.image_path`` начинается с ``s3://`` **и** *s3_client* задан
       → загрузка из S3.
    2. Иначе → разрешается относительно *images_root* (или как абсолютный путь)
       и загружается с локального диска.

    При любой ошибке возвращает ``None``, чтобы ячейка деградировала корректно.
    """
    try:
        if result.image_path.startswith("s3://") and s3_client is not None:
            return _load_crop_from_s3(result, s3_client)
        return _load_crop_from_local(result, images_root)
    except Exception:
        logger.warning(
            "Ошибка загрузки/вырезания изображения для box_id='%s' (path='%s').",
            result.box_id,
            result.image_path,
            exc_info=True,
        )
        return None


def _load_crop_from_s3(
    result: SearchResult,
    s3_client: S3Client,
) -> PILImage.Image | None:
    """Загружает изображение из S3 и возвращает вырезанный фрагмент."""
    try:
        img = s3_client.load_image(result.image_path)
        return img.crop((result.x1, result.y1, result.x2, result.y2))
    except Exception:
        logger.warning(
            "Ошибка загрузки S3-изображения для box_id='%s', uri='%s'.",
            result.box_id,
            result.image_path,
            exc_info=True,
        )
        return None


def _load_crop_from_local(
    result: SearchResult,
    images_root: Path,
) -> PILImage.Image | None:
    """Загружает изображение из локальной файловой системы
    и возвращает вырезанный фрагмент."""
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
                    "Ошибка открытия/вырезания '%s' для box_id='%s'.",
                    path,
                    result.box_id,
                    exc_info=True,
                )
                return None

    logger.warning(
        "Изображение не найдено для box_id='%s': проверены %s",
        result.box_id,
        [str(p) for p in candidates],
    )
    return None
