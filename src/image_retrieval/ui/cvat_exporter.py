"""Streamlit-виджет: экспорт выбранных кропов в задачу CVAT.

Отображает разворачиваемую панель под сеткой результатов.  Когда пользователь
выбрал хотя бы один результат через чекбоксы в
:mod:`~image_retrieval.ui.results_viewer`, панель показывает:

* Количество выбранных кропов и уникальных исходных изображений.
* Необязательное поле ввода имени задачи.
* Кнопку **Экспортировать в CVAT**, которая:
  1. Загружает каждое уникальное исходное изображение (локально или из S3).
  2. Создаёт новую задачу CVAT с настроенной меткой.
  3. Загружает изображения и отправляет bbox-аннотации.
  4. Отображает кликабельную ссылку на созданную задачу.

Всё взаимодействие с CVAT делегируется :class:`~image_retrieval.cvat_client.CVATClient`.
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
    """Отображает панель экспорта CVAT под сеткой результатов.

    Панель отображается только при наличии хотя бы одного выбранного результата.
    Автоматически сворачивается в expander, чтобы не загромождать страницу.

    Args:
        all_results: Полный список результатов поиска (тот же, что передаётся в
            :func:`~results_viewer.render_results`).
        images_root: Базовая директория для разрешения локальных путей изображений.
        cvat_config: Настройки подключения к CVAT.
        s3_client: Необязательный S3-клиент для ``s3://``-путей изображений.
    """
    selected_ids = get_selected_box_ids()
    if not selected_ids:
        return  # ничего не выбрано — панель не показываем

    selected_results = [r for r in all_results if r.box_id in selected_ids]
    n_images = len({r.image_path for r in selected_results})

    with st.expander(
        f"📤 Экспортировать {len(selected_results)} кропов "
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

        st.caption(
            f"**URL CVAT:** `{cvat_config.url}`  \n"
            f"**Метка:** `{cvat_config.task_label}`"
            + (
                f"  \n**ID проекта:** `{cvat_config.project_id}`"
                if cvat_config.project_id is not None
                else ""
            )
        )

        if st.button(
            "📤 Экспортировать в CVAT",
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
    """Загружает *image_path* и возвращает сырые JPEG-байты.

    Маршрутизирует в S3 или локальную файловую систему по схеме URI — аналогично
    тому, как :func:`~results_viewer._load_crop` маршрутизирует кропы.

    Args:
        image_path: Абсолютный путь, относительный путь (разрешается относительно
            *images_root*) или ``s3://bucket/key``-URI.
        images_root: Базовая директория для относительных путей.
        s3_client: S3-клиент; обязателен для S3-URI.

    Returns:
        JPEG-закодированные байты изображения.

    Raises:
        FileNotFoundError: Если локальное изображение не найдено.
        RuntimeError: Если задан S3-URI, но клиент не предоставлен.
    """
    img: PILImage.Image

    if image_path.startswith("s3://"):
        if s3_client is None:
            raise RuntimeError(
                f"Невозможно загрузить '{image_path}': S3-клиент не настроен."
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
                f"Изображение не найдено: проверены {[str(c) for c in candidates]}"
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
    """Оркестрирует полный процесс экспорта со спиннером прогресса."""

    def load_bytes(path: str) -> bytes:
        return _image_loader(path, images_root, s3_client)

    with st.spinner("Подготовка изображений..."):
        try:
            export_data = prepare_export(
                results=selected_results,
                load_image_bytes=load_bytes,
                task_name=task_name,
            )
        except Exception as exc:
            st.error(f"Ошибка загрузки изображений: {exc}", icon="🚫")
            logger.exception("Ошибка загрузки изображений при экспорте в CVAT.")
            return

    with st.spinner(
        f"Загрузка {len(export_data.images)} изображения(-й) в CVAT..."
    ):
        try:
            client = CVATClient(cvat_config)
            result = client.export_to_task(
                task_name=export_data.task_name,
                images=export_data.images,
                annotations=export_data.annotations,
            )
        except Exception as exc:
            st.error(f"Ошибка экспорта в CVAT: {exc}", icon="🚫")
            logger.exception("Ошибка экспорта в CVAT.")
            return

    st.success(
        f"✅ Задача CVAT создана: **{result.annotation_count}** аннотаций "
        f"в **{result.image_count}** изображении(-ях).",
        icon="✅",
    )
    st.link_button(
        label="🔗 Открыть задачу в CVAT",
        url=result.task_url,
        use_container_width=True,
    )
