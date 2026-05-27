"""
🔍 Streamlit Image Crop Retrieval — SSL-поиск похожих кропов

Архитектура
-----------
Страница реализована как класс :class:`ImageRetriever` с методом :meth:`run`.
Точка входа — :func:`app`, принимающая ``(configuration, config_path)``
в точности как все прочие страницы приложения.

Конфигурация передаётся снаружи через ``app.py``; самостоятельной загрузки
конфигурации нет.

S3-интеграция
-------------
Прямой доступ к S3 через ``boto3.client("s3")`` без промежуточных адаптеров.
Клиент кэшируется через ``@st.cache_resource(max_entries=5, ttl=3600)``,
тяжёлые ресурсы (веса, индекс) — через ``@st.cache_resource(ttl=600)``.
Изображения результатов — через ``@st.cache_data(ttl=600)``.

Структура бакета
----------------
::

    weights/           # чекпоинты SSL-модели (*.pt / *.pth)
    ssl_index/         # FAISS-индексы (*.faiss) и метаданные (*.parquet)

Берётся последний файл каждого типа по ``LastModified``.

CVAT-экспорт
------------
Через ``cveta`` SDK: ``CVAT(cvat_name=...)``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import streamlit as st
from cveta.cvat.cvat_tools import CVAT
from PIL import Image, ImageDraw

from image_retrieval.config import Configuration
from image_retrieval.embedder import TorchEmbedder
from image_retrieval.indexer import FAISSIndex, SearchResult
from image_retrieval.ui.crop_selector import render_crop_selector

logger = logging.getLogger(__name__)

# ============================================================================
# 🔧 SESSION_DEFAULTS
# ============================================================================

SESSION_DEFAULTS_RETRIEVAL: dict[str, Any] = {
    "retrieval_bucket": None,
    "retrieval_results": None,
    "selected_result_ids": set(),
    "reset_counter": 0,
}

# ============================================================================
# 🔐 S3-утилиты (модульный уровень, аналогично dataset_viewer.py)
# ============================================================================


@st.cache_resource(max_entries=5, ttl=3600)
def _get_s3_client_cached() -> Any:
    """Кэшированный boto3 S3-клиент (TTL 1 час)."""
    return boto3.client("s3")


def _list_s3_objects(
    bucket: str,
    prefix: str,
    suffix: str = "",
) -> list[dict[str, Any]]:
    """Перечисляет объекты S3 под *prefix* (без «директорий»).

    Args:
        bucket: Имя бакета.
        prefix: Префикс ключей объектов (напр. ``"weights/"``).
        suffix: Если задан, возвращает только объекты с этим суффиксом.

    Returns:
        Список словарей с ключами ``Key``, ``Size``, ``LastModified``.
    """
    try:
        s3 = _get_s3_client_cached()
        paginator = s3.get_paginator("list_objects_v2")
        results: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key: str = obj["Key"]
                if key.endswith("/"):
                    continue
                if suffix and not key.endswith(suffix):
                    continue
                results.append(obj)
        return results
    except Exception as exc:
        logger.error("Ошибка списка объектов s3://%s/%s: %s", bucket, prefix, exc)
        return []


def _latest_by_modified(
    objects: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Возвращает объект с наибольшим ``LastModified``, или ``None``."""
    return max(objects, key=lambda o: o["LastModified"]) if objects else None


def _s3_newer_than_local(
    s3: Any,
    bucket: str,
    key: str,
    local: Path,
) -> bool:
    """Возвращает ``True`` если S3-объект новее локального файла."""
    if not local.exists():
        return True
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        return bool(head["LastModified"].timestamp() > local.stat().st_mtime)
    except Exception:
        return False


@st.cache_data(ttl=600, show_spinner="📥 Загрузка изображения...")
def _load_image_from_s3_cached(
    bucket: str,
    key: str,
) -> Image.Image | None:
    """Загружает PIL-изображение из S3; кэш 10 минут."""
    try:
        s3 = _get_s3_client_cached()
        response = s3.get_object(Bucket=bucket, Key=key)
        data: bytes = response["Body"].read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        logger.warning("Не удалось загрузить s3://%s/%s: %s", bucket, key, exc)
        return None


# ============================================================================
# 🏋️ Тяжёлые ресурсы (веса и индекс) — TTL 10 мин
# ============================================================================


@st.cache_resource(ttl=600)
def _load_latest_weights(bucket: str) -> Path:
    """Скачивает последний файл весов из ``{bucket}/weights/``; TTL 10 мин.

    При истечении TTL повторно проверяет ``LastModified`` и скачивает только
    если S3-версия новее локального кэша.

    Raises:
        FileNotFoundError: Если ``weights/`` пуста.
    """
    s3 = _get_s3_client_cached()
    objects = _list_s3_objects(bucket, "weights/")
    latest = _latest_by_modified(objects)
    if latest is None:
        raise FileNotFoundError(
            f"Файлы весов не найдены в s3://{bucket}/weights/"
        )
    key: str = latest["Key"]

    cache_dir = Path(tempfile.gettempdir()) / "image-retrieval-weights"
    cache_dir.mkdir(parents=True, exist_ok=True)
    uri_hash = hashlib.md5(f"{bucket}/{key}".encode()).hexdigest()[:16]
    local_path = cache_dir / f"{uri_hash}{Path(key).suffix or '.pth'}"

    if _s3_newer_than_local(s3, bucket, key, local_path):
        logger.info("Скачивание весов: s3://%s/%s → %s", bucket, key, local_path)
        s3.download_file(bucket, key, str(local_path))

    return local_path


@st.cache_resource(ttl=600)
def _load_latest_index(bucket: str) -> FAISSIndex:
    """Скачивает последние ``*.faiss`` + ``*.parquet`` из ``{bucket}/ssl_index/``.

    При истечении TTL проверяет ``LastModified`` и перекачивает только
    обновлённые файлы.

    Raises:
        FileNotFoundError: Если ``ssl_index/`` не содержит нужных файлов.
    """
    s3 = _get_s3_client_cached()
    all_objects = _list_s3_objects(bucket, "ssl_index/")

    latest_faiss = _latest_by_modified(
        [o for o in all_objects if o["Key"].endswith(".faiss")]
    )
    latest_parquet = _latest_by_modified(
        [o for o in all_objects if o["Key"].endswith(".parquet")]
    )

    if latest_faiss is None:
        raise FileNotFoundError(
            f"FAISS-индекс не найден в s3://{bucket}/ssl_index/"
        )
    if latest_parquet is None:
        raise FileNotFoundError(
            f"Parquet-метаданные не найдены в s3://{bucket}/ssl_index/"
        )

    cache_dir = Path(tempfile.gettempdir()) / "image-retrieval-index"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bucket_hash = hashlib.md5(bucket.encode()).hexdigest()[:8]
    faiss_local = cache_dir / f"{bucket_hash}_index.faiss"
    meta_local = cache_dir / f"{bucket_hash}_metadata.parquet"

    faiss_key: str = latest_faiss["Key"]
    parquet_key: str = latest_parquet["Key"]

    if _s3_newer_than_local(s3, bucket, faiss_key, faiss_local):
        logger.info(
            "Скачивание индекса: s3://%s/%s → %s", bucket, faiss_key, faiss_local
        )
        s3.download_file(bucket, faiss_key, str(faiss_local))

    if _s3_newer_than_local(s3, bucket, parquet_key, meta_local):
        logger.info(
            "Скачивание метаданных: s3://%s/%s → %s",
            bucket,
            parquet_key,
            meta_local,
        )
        s3.download_file(bucket, parquet_key, str(meta_local))

    return FAISSIndex(faiss_local, meta_local)


@st.cache_resource(ttl=600)
def _load_embedder_cached(weights_path: str, device: str) -> TorchEmbedder:
    """Загружает TorchEmbedder; TTL 10 мин."""
    return TorchEmbedder(checkpoint_path=Path(weights_path), device=device)


# ============================================================================
# 🎨 Отрисовка bbox на изображении результата
# ============================================================================

_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def _parse_image_path(image_path: str, default_bucket: str) -> tuple[str, str]:
    """Разбирает ``image_path`` в ``(bucket, key)``."""
    match = _S3_URI_RE.match(image_path)
    if match:
        return match.group(1), match.group(2)
    return default_bucket, image_path


def _draw_result_bbox(
    image: Image.Image,
    result: SearchResult,
) -> Image.Image:
    """Рисует bounding-box найденного кропа на полном изображении."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    color = (255, 69, 58)
    draw.rectangle(
        (result.x1, result.y1, result.x2, result.y2),
        outline=color,
        width=2,
    )
    if result.label_class:
        draw.text(
            (result.x1 + 4, result.y1 + 4),
            result.label_class,
            fill=color,
        )
    return annotated


# ============================================================================
# 🧩 ОСНОВНОЙ КЛАСС
# ============================================================================


class ImageRetriever:
    """🔍 SSL-поиск похожих кропов из S3-датасета.

    Args:
        configuration: Корневой объект конфигурации приложения.
        config_path: Путь/имя конфигурационного файла (передаётся из ``app.py``
            для совместимости с соглашением многостраничного приложения).
    """

    def __init__(self, configuration: Configuration, config_path: str) -> None:
        self.configuration = configuration
        self.config_path = config_path
        cvat_name = (
            configuration.cvat.cvat_name if configuration.cvat else "sip"
        )
        self.cvat2 = CVAT(cvat_name=cvat_name)
        self._state_initialization()

    def _state_initialization(self) -> None:
        """Инициализирует ключи session_state из ``SESSION_DEFAULTS_RETRIEVAL``."""
        for key, value in SESSION_DEFAULTS_RETRIEVAL.items():
            if key not in st.session_state:
                st.session_state[key] = value
        if not isinstance(st.session_state.selected_result_ids, set):
            st.session_state.selected_result_ids = set()

    # ── CVAT-экспорт ─────────────────────────────────────────────────────

    def _create_cvat_task(self, results: list[SearchResult]) -> None:
        """Создаёт задачу CVAT из выбранных результатов."""
        selected_ids: set[str] = st.session_state.selected_result_ids
        selected = [r for r in results if r.box_id in selected_ids]
        if not selected:
            st.warning("❌ Нет выбранных результатов. Отметьте чекбоксы.")
            return

        cvat_project_name: str | None = None
        if self.configuration.cvat and self.configuration.cvat.project_id is not None:
            cvat_project_name = str(self.configuration.cvat.project_id)

        bucket: str = st.session_state.retrieval_bucket or ""
        now = datetime.now()
        task_name = f"retrieval_{now.strftime('%Y%m%d_%H%M%S')}"

        # Строим DataFrame аннотаций в формате cveta
        rows = []
        for r in selected:
            img_bucket, img_key = _parse_image_path(r.image_path, bucket)
            s3_uri = f"s3://{img_bucket}/{img_key}"
            rows.append(
                {
                    "s3_image_path": s3_uri,
                    "image_name": Path(r.image_path).name,
                    "bbox_x_tl": r.x1,
                    "bbox_y_tl": r.y1,
                    "bbox_x_br": r.x2,
                    "bbox_y_br": r.y2,
                    "instance_label": r.label_class or "crop",
                    "confidence": r.score,
                }
            )
        annotations_df = pd.DataFrame(rows)
        s3_paths = annotations_df["s3_image_path"].unique().tolist()

        with st.spinner(f"⏳ Создание задачи '{task_name}' в CVAT..."):
            try:
                self.cvat2.create_task(
                    name=task_name,
                    labels=None,
                    content=s3_paths,
                    annotations=annotations_df,
                    assignee=None,
                    image_quality=100,
                    project_id=None,
                    project_name=cvat_project_name,
                    segment_size=100,
                    annotation_xml_path=None,
                )
                st.success(
                    f"✅ Задача **{task_name}** создана! "
                    f"{len(selected)} аннотаций.",
                    icon="✅",
                )
            except Exception as exc:
                logger.error("Ошибка создания задачи CVAT: %s", exc)
                st.error(f"❌ Ошибка: {exc}")

    # ── Отрисовка карточки результата ─────────────────────────────────────

    def _render_result_card(
        self,
        result: SearchResult,
        col: Any,
        idx: int,
        bucket: str,
    ) -> None:
        """Отображает карточку результата: изображение с bbox + чекбокс + подпись."""
        with col:
            img_bucket, img_key = _parse_image_path(result.image_path, bucket)
            image = _load_image_from_s3_cached(img_bucket, img_key)

            if image is not None:
                annotated = _draw_result_bbox(image, result)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                annotated.save(tmp_path, "JPEG", quality=95)
                st.image(tmp_path, use_container_width=True)
                try:
                    import os
                    os.unlink(tmp_path)
                except Exception:
                    pass
            else:
                st.error(f"⚠️ {Path(result.image_path).name}")

            uid = result.box_id
            safe_key = re.sub(r"[/:.]+", "_", uid)
            cb_key = f"res_{safe_key}_{idx}_{st.session_state.reset_counter}"

            current: bool = uid in st.session_state.selected_result_ids
            new_selected: bool = st.checkbox("✅ Выбрать", value=current, key=cb_key)
            if new_selected != current:
                if new_selected:
                    st.session_state.selected_result_ids.add(uid)
                else:
                    st.session_state.selected_result_ids.discard(uid)

            score_pct = result.score * 100
            class_text = f" · {result.label_class}" if result.label_class else ""
            st.caption(
                f"**{score_pct:.1f}%**{class_text}  \n"
                f"`{Path(result.image_path).name}`  \n"
                f"[{result.x1}, {result.y1} – {result.x2}, {result.y2}]"
            )

    # ── Сетка результатов ─────────────────────────────────────────────────

    def _render_results_grid(
        self,
        results: list[SearchResult],
        bucket: str,
    ) -> None:
        """Отображает результаты в сетке карточек с чекбоксами."""
        if not results:
            st.info("ℹ️ Результатов не найдено.")
            return

        n_sel: int = len(st.session_state.selected_result_ids)

        hdr_col, clear_col = st.columns([6, 1])
        with hdr_col:
            label = f"### 🖼️ Топ-{len(results)} результатов"
            if n_sel:
                label += f" · **{n_sel} выбрано**"
            st.markdown(label)
        with clear_col:
            if n_sel and st.button(
                "✖ Сбросить",
                help="Снять выбор со всех результатов.",
                use_container_width=True,
            ):
                st.session_state.selected_result_ids = set()
                st.session_state.reset_counter += 1
                st.rerun()

        cols_count = self.configuration.app.results_columns
        for i in range(0, len(results), cols_count):
            cols = st.columns(cols_count)
            for j in range(cols_count):
                idx = i + j
                if idx < len(results):
                    self._render_result_card(results[idx], cols[j], idx, bucket)

        # Кнопка экспорта в CVAT
        if n_sel > 0 and self.configuration.cvat is not None:
            st.divider()
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**Выбрано {n_sel} результатов для экспорта**")
            with c2:
                if st.button(
                    "📤 Экспорт в CVAT",
                    type="primary",
                    use_container_width=True,
                ):
                    self._create_cvat_task(results)

    # ── Главный метод ─────────────────────────────────────────────────────

    def run(self) -> None:
        """Отрисовывает страницу поиска."""
        st.markdown("""
        <style>
            .stImage > img { border-radius: 8px; }
            div[data-testid="stMetricValue"] { font-size: 1.5rem; }
        </style>
        """, unsafe_allow_html=True)

        st.title("🔍 Поиск похожих кропов")

        cfg = self.configuration
        buckets = cfg.app.buckets
        if not buckets:
            st.error(
                "Список бакетов не задан. Укажите `app.buckets` в конфигурации.",
                icon="🚫",
            )
            st.stop()

        # ── Боковая панель ────────────────────────────────────────────────
        with st.sidebar:
            st.header("⚙️ Настройки")

            bucket: str = st.selectbox(
                "☁️ S3-бакет",
                options=buckets,
                help=(
                    "Бакет с весами модели (`weights/`) "
                    "и FAISS-индексом (`ssl_index/`)."
                ),
            )

            # Сбрасываем результаты при смене бакета
            if st.session_state.retrieval_bucket != bucket:
                st.session_state.retrieval_bucket = bucket
                st.session_state.retrieval_results = None
                st.session_state.selected_result_ids = set()

            top_k: int = st.slider(
                "Топ-K результатов",
                min_value=1,
                max_value=50,
                value=cfg.app.top_k,
                help="Количество ближайших соседей для поиска.",
            )

            st.divider()
            st.caption(f"☁️ `s3://{bucket}`")

        # ── Выбор кропа ───────────────────────────────────────────────────
        crop_result = render_crop_selector(cfg.app)
        if crop_result is None:
            if st.session_state.retrieval_results:
                st.divider()
                self._render_results_grid(
                    st.session_state.retrieval_results, bucket
                )
            st.stop()
        _full_image, crop = crop_result

        # ── Кнопка поиска ─────────────────────────────────────────────────
        st.divider()
        search_clicked = st.button(
            "🔍 Найти", type="primary", use_container_width=False
        )

        if not search_clicked:
            if st.session_state.retrieval_results:
                self._render_results_grid(
                    st.session_state.retrieval_results, bucket
                )
            st.stop()

        # ── Загрузка ресурсов из S3 (с кэшем) ────────────────────────────
        with st.spinner("Загрузка весов модели из S3..."):
            try:
                weights_path = _load_latest_weights(bucket)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()

        with st.spinner("Загрузка FAISS-индекса из S3..."):
            try:
                faiss_index = _load_latest_index(bucket)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()

        try:
            embedder = _load_embedder_cached(str(weights_path), cfg.app.device)
        except Exception as exc:
            st.error(f"Ошибка загрузки модели: {exc}")
            st.stop()

        # ── Инференс + поиск ──────────────────────────────────────────────
        with st.spinner("Вычисляется эмбеддинг..."):
            query_vec: np.ndarray = embedder.embed([crop])

        with st.spinner(f"Поиск топ-{top_k}..."):
            results: list[SearchResult] = faiss_index.search(query_vec, top_k)

        st.session_state.retrieval_results = results
        st.toast(f"✅ Найдено {len(results)} результатов", icon="🎉")

        # ── Отображение результатов ───────────────────────────────────────
        self._render_results_grid(results, bucket)

        st.markdown("---")
        st.caption(
            "💡 **Подсказки:** Загрузите изображение → нарисуйте рамку → "
            "нажмите Найти → выберите результаты → экспорт в CVAT"
        )


# ============================================================================
# Точка входа страницы
# ============================================================================


def app(configuration: Configuration, config_path: str) -> None:
    """Entry point ✨

    Вызывается из ``app.py`` по соглашению
    ``selected_page_class_or_module.app(config, st.session_state.config_file)``.

    Args:
        configuration: Конфигурация приложения.
        config_path: Имя/путь конфигурационного файла.
    """
    ImageRetriever(configuration, config_path).run()
