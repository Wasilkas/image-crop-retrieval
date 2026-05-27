"""Streamlit-виджет: экспорт выбранных кропов в задачу CVAT через cveta SDK.

Отображает разворачиваемую панель под сеткой результатов.  Когда пользователь
выбрал хотя бы один результат через чекбоксы в
:mod:`~image_retrieval.ui.results_viewer`, панель показывает:

* Количество выбранных кропов и уникальных исходных изображений.
* Необязательное поле ввода имени задачи.
* Кнопку **Экспортировать в CVAT**, которая инициализирует клиент cveta
  (``CVAT(cvat_name=...)``), формирует список аннотаций и вызывает экспорт.

Инициализация клиента::

    from cveta.cvat.cvat_tools import CVAT
    cvat = CVAT(cvat_name='sip')
"""

from __future__ import annotations

import logging
import time
from typing import Any

import streamlit as st

from ..indexer import SearchResult
from .results_viewer import get_selected_box_ids

logger = logging.getLogger(__name__)


def render_cvat_exporter(
    all_results: list[SearchResult],
    cvat_name: str = "sip",
) -> None:
    """Отображает панель экспорта CVAT под сеткой результатов.

    Панель отображается только при наличии хотя бы одного выбранного результата.
    Автоматически сворачивается в expander, чтобы не загромождать страницу.

    Args:
        all_results: Полный список результатов поиска.
        cvat_name: Имя CVAT-инстанса для ``CVAT(cvat_name=...)``.
    """
    selected_ids = get_selected_box_ids()
    if not selected_ids:
        return

    selected = [r for r in all_results if r.box_id in selected_ids]
    n_images = len({r.image_path for r in selected})

    with st.expander(
        f"📤 Экспортировать {len(selected)} кропов "
        f"из {n_images} изображения(-й) в CVAT",
        expanded=True,
    ):
        task_name = st.text_input(
            label="Имя задачи",
            placeholder="image-crop-retrieval-YYYYMMDD-HHMMSS",
            help=(
                "Имя новой задачи CVAT.  Оставьте пустым для автогенерации "
                "из текущей временной метки."
            ),
        )

        st.caption(f"**CVAT-инстанс:** `{cvat_name}`")

        if st.button(
            "📤 Экспортировать в CVAT",
            type="primary",
            use_container_width=True,
        ):
            _run_export(
                selected_results=selected,
                cvat_name=cvat_name,
                task_name=task_name.strip(),
            )


def _build_annotations(
    results: list[SearchResult],
) -> list[dict[str, Any]]:
    """Собирает список аннотаций из выбранных результатов."""
    return [
        {
            "image_path": r.image_path,
            "box_id": r.box_id,
            "x1": r.x1,
            "y1": r.y1,
            "x2": r.x2,
            "y2": r.y2,
            "label_class": r.label_class,
            "score": r.score,
        }
        for r in results
    ]


def _run_export(
    selected_results: list[SearchResult],
    cvat_name: str,
    task_name: str,
) -> None:
    """Выполняет экспорт через cveta SDK."""
    try:
        from cveta.cvat.cvat_tools import CVAT
    except ImportError:
        st.error(
            "cveta SDK не установлен. Выполните: `pip install cveta`",
            icon="🚫",
        )
        return

    if not task_name:
        task_name = f"image-crop-retrieval-{time.strftime('%Y%m%d-%H%M%S')}"

    annotations = _build_annotations(selected_results)

    with st.spinner(f"Экспорт {len(annotations)} аннотаций в CVAT..."):
        try:
            cvat = CVAT(cvat_name=cvat_name)
            cvat.export_annotations(
                task_name=task_name,
                annotations=annotations,
            )
        except Exception as exc:
            st.error(f"Ошибка экспорта в CVAT: {exc}", icon="🚫")
            logger.exception("Ошибка экспорта через cveta SDK.")
            return

    st.success(
        f"✅ Экспортировано **{len(annotations)}** аннотаций в задачу "
        f"**{task_name}**.",
        icon="✅",
    )
