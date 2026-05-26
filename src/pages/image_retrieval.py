"""Страница «Поиск похожих кропов».

Содержит кэшированные фабрики Streamlit и функцию :func:`render`,
вызываемую через ``st.Page()`` в ``app.py``.

Конфигурация загружается через :class:`~image_retrieval.config.Configuration`
с цепочкой приоритетов Consul → YAML → умолчания.  Переменные среды
применяются поверх через :func:`_apply_env_overrides`.

Переменные среды (Consul)::

    CONSUL_URL    URL агента Consul, напр. ``http://consul:8500``
    CONSUL_KEY    Ключ KV с YAML-конфигом (умолч.: ``config/image-crop-retrieval``)

Переменные среды (YAML-файл)::

    CONFIG_PATH   Путь к YAML-файлу конфигурации (умолч.: ``config.yaml``)

Быстрые переопределения::

    MODEL_PATH         Предзаполнить поле пути к чекпоинту
    S3_BUCKET          Включить S3-режим
    S3_PREFIX          Префикс ключей датасетов в S3
    S3_REGION          Регион AWS (умолч.: ``us-east-1``)
    S3_ENDPOINT_URL    Кастомный endpoint (MinIO, Yandex Cloud и др.)
    S3_CHECK_INTERVAL  Интервал опроса S3 в секундах (умолч.: 300)
    DATASETS_DIR       Директория датасетов (только локальный режим)

Подробнее о полной схеме YAML см. ``config.yaml.example``.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from image_retrieval.config import (
    AppBlock,
    Configuration,
    EncoderBlock,
    S3Block,
)
from image_retrieval.embedder import TorchEmbedder
from image_retrieval.registry import DatasetRegistry, S3DatasetRegistry
from image_retrieval.s3_client import S3Client
from image_retrieval.ui.crop_selector import render_crop_selector
from image_retrieval.ui.cvat_exporter import render_cvat_exporter
from image_retrieval.ui.results_viewer import render_results

logger = logging.getLogger(__name__)

_AnyRegistry = DatasetRegistry | S3DatasetRegistry

# TTL кэша для всех S3-ресурсов (модель и реестр датасетов): 10 минут.
# По истечении TTL Streamlit пересоздаёт ресурс: для S3-URI проверяется
# наличие более новой версии на S3, при необходимости файл скачивается заново.
_S3_CACHE_TTL = 600


@st.cache_resource
def _load_config() -> Configuration:
    """Загружает корневой :class:`Configuration` один раз на время жизни процесса.

    Автоматически читает переменные среды ``CONSUL_URL`` / ``CONFIG_PATH``.
    Быстрые переопределения (``S3_BUCKET``, ``DATASETS_DIR`` и др.) применяются
    поверх в :func:`_apply_env_overrides`.
    """
    return Configuration.load()


def _apply_env_overrides(cfg: Configuration) -> Configuration:
    """Применяет быстрые переопределения из переменных среды поверх *cfg*.

    Переменные среды имеют приоритет над YAML / Consul, что позволяет
    передавать секреты (токены, пароли) без хранения в файлах конфигурации.
    """
    overrides: dict[str, object] = {}

    # Блок S3 — любая S3_*-переменная включает S3-режим
    s3_bucket = os.environ.get("S3_BUCKET", "").strip()
    if s3_bucket:
        existing_s3 = cfg.s3 or S3Block(bucket=s3_bucket)
        overrides["s3"] = S3Block(
            bucket=s3_bucket,
            prefix=os.environ.get("S3_PREFIX", existing_s3.prefix),
            region=os.environ.get("S3_REGION", existing_s3.region),
            endpoint_url=(
                os.environ.get("S3_ENDPOINT_URL") or existing_s3.endpoint_url
            ),
            check_interval_seconds=int(
                os.environ.get(
                    "S3_CHECK_INTERVAL",
                    str(existing_s3.check_interval_seconds),
                )
            ),
        )

    # Блок App
    datasets_dir_env = os.environ.get("DATASETS_DIR", "").strip()
    top_k_env = os.environ.get("TOP_K", "").strip()
    device_env = os.environ.get("DEVICE", "").strip()
    if datasets_dir_env or top_k_env or device_env:
        overrides["app"] = AppBlock(
            datasets_dir=(
                Path(datasets_dir_env) if datasets_dir_env else cfg.app.datasets_dir
            ),
            top_k=int(top_k_env) if top_k_env else cfg.app.top_k,
            device=device_env or cfg.app.device,
            min_crop_px=cfg.app.min_crop_px,
            canvas_width=cfg.app.canvas_width,
            canvas_height=cfg.app.canvas_height,
            results_columns=cfg.app.results_columns,
        )

    # Блок Encoder — переменная среды MODEL_PATH
    model_path_env = os.environ.get("MODEL_PATH", "").strip()
    if model_path_env and cfg.encoder is None:
        overrides["encoder"] = EncoderBlock(checkpoint=model_path_env)

    if not overrides:
        return cfg
    return cfg.model_copy(update=overrides)


@st.cache_resource
def _load_local_registry(datasets_dir: str) -> DatasetRegistry:
    """Создаёт и кэширует локальный реестр датасетов на время жизни процесса."""
    return DatasetRegistry(AppBlock(datasets_dir=Path(datasets_dir)))


@st.cache_resource(ttl=_S3_CACHE_TTL)
def _load_s3_registry(
    bucket: str,
    prefix: str,
    region: str,
    endpoint_url: str,
    check_interval: int,
) -> S3DatasetRegistry:
    """Создаёт S3-реестр датасетов; кэш пересоздаётся каждые 10 минут.

    При пересоздании повторно скачивает обновлённые индексы из S3
    (если версия на S3 новее локального кэша).
    """
    s3_block = S3Block(
        bucket=bucket,
        prefix=prefix,
        region=region,
        endpoint_url=endpoint_url or None,
        check_interval_seconds=check_interval,
    )
    return S3DatasetRegistry(s3_block)


@st.cache_resource(ttl=_S3_CACHE_TTL)
def _load_s3_client(
    bucket: str,
    prefix: str,
    region: str,
    endpoint_url: str,
    check_interval: int,
) -> S3Client:
    """Создаёт и кэширует S3-клиент; кэш пересоздаётся каждые 10 минут."""
    s3_block = S3Block(
        bucket=bucket,
        prefix=prefix,
        region=region,
        endpoint_url=endpoint_url or None,
        check_interval_seconds=check_interval,
    )
    return S3Client(s3_block)


def _resolve_model_path(
    encoder_cfg: EncoderBlock,
    s3_client: S3Client | None,
) -> Path:
    """Возвращает локальный путь к чекпоинту модели.

    Если ``encoder_cfg.checkpoint`` — локальный путь, возвращает его напрямую.
    Если это ``s3://``-URI:
    * Вычисляет путь кэша в ``encoder_cfg.cache_dir`` (или системном tmp),
      используя MD5-хэш URI в качестве уникального ключа.
    * Скачивает файл при первом обращении.
    * При последующих вызовах (в т.ч. после сброса TTL-кэша) проверяет, не новее
      ли объект на S3, и перекачивает только при необходимости.

    Args:
        encoder_cfg: Конфигурационный блок энкодера.
        s3_client: S3-клиент (обязателен для ``s3://``-чекпоинтов).

    Raises:
        ValueError: Если задан S3-URI, но *s3_client* не предоставлен.
        FileNotFoundError: Если локальный путь не существует.
    """
    checkpoint = encoder_cfg.checkpoint
    if not checkpoint.startswith("s3://"):
        local = Path(checkpoint)
        if not local.exists():
            raise FileNotFoundError(
                f"Чекпоинт модели не найден: {checkpoint}"
            )
        return local

    if s3_client is None:
        raise ValueError(
            "Чекпоинт энкодера — S3-URI, но S3-конфигурация не найдена. "
            "Укажите блок 's3' в конфиге или переменную среды S3_BUCKET."
        )

    # Стабильный локальный путь кэша, производный от URI модели
    cache_dir = (
        encoder_cfg.cache_dir
        or Path(tempfile.gettempdir()) / "image-retrieval-models"
    )
    uri_hash = hashlib.md5(checkpoint.encode()).hexdigest()[:16]
    suffix = Path(checkpoint.rsplit("/", 1)[-1]).suffix or ".pth"
    local_path = cache_dir / f"model_{uri_hash}{suffix}"

    if s3_client.is_remote_newer(_s3_key_from_uri(checkpoint), local_path):
        logger.info("Скачивание чекпоинта модели с %s ...", checkpoint)
        s3_client.download_uri(checkpoint, local_path)
        logger.info("Модель сохранена в кэш: %s", local_path)

    return local_path


def _s3_key_from_uri(uri: str) -> str:
    """Извлекает ключ из URI формата ``s3://bucket/key``."""
    parts = uri.split("/", 3)
    return parts[3] if len(parts) >= 4 else uri


@st.cache_resource(ttl=_S3_CACHE_TTL)
def _load_embedder(
    checkpoint: str,
    device: str,
    s3_bucket: str = "",
    s3_region: str = "us-east-1",
    s3_endpoint_url: str = "",
    cache_dir_str: str = "",
) -> TorchEmbedder:
    """Загружает PyTorch-модель; кэш пересоздаётся каждые 10 минут.

    При истечении TTL Streamlit повторно вызывает эту функцию с теми же
    аргументами.  Для ``s3://``-чекпоинтов вызывается :func:`_resolve_model_path`,
    которая скачивает файл только если версия на S3 новее локального кэша.

    Args:
        checkpoint: Локальный путь или ``s3://``-URI к чекпоинту модели.
        device: Строка устройства PyTorch (напр. ``"cpu"``).
        s3_bucket: Имя бакета S3 (только для ``s3://``-чекпоинтов).
        s3_region: Регион AWS.
        s3_endpoint_url: Кастомный endpoint S3 (MinIO, Yandex Cloud и др.).
        cache_dir_str: Директория локального кэша для S3-моделей (пустая строка
            означает использование системной временной директории).
    """
    if checkpoint.startswith("s3://"):
        if not s3_bucket:
            raise ValueError(
                "Чекпоинт модели — S3-URI, но S3_BUCKET не задан. "
                "Укажите блок 's3' в конфиге или переменную среды S3_BUCKET."
            )
        s3_block = S3Block(
            bucket=s3_bucket,
            region=s3_region,
            endpoint_url=s3_endpoint_url or None,
        )
        s3_client = S3Client(s3_block)
        cache_dir: Path | None = Path(cache_dir_str) if cache_dir_str else None
        encoder_block = EncoderBlock(checkpoint=checkpoint, cache_dir=cache_dir)
        local_path = _resolve_model_path(encoder_block, s3_client)
    else:
        local_path = Path(checkpoint)
        if not local_path.exists():
            raise FileNotFoundError(
                f"Чекпоинт модели не найден: {checkpoint}"
            )

    return TorchEmbedder(checkpoint_path=local_path, device=device)


def _render_dataset_status(
    registry: _AnyRegistry,
    dataset_name: str,
) -> None:
    """Отображает дату построения индекса и кнопку «Пересканировать датасеты»."""
    reload_info = registry.last_reload_info(dataset_name)
    if reload_info is not None:
        index_mtime_ns, _ = reload_info
        ts = datetime.datetime.fromtimestamp(
            index_mtime_ns / 1_000_000_000, tz=datetime.UTC
        ).astimezone()
        st.caption(f"🕒 Индекс построен: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    if st.button(
        "🔄 Пересканировать датасеты",
        help=(
            "Локальный режим: повторно сканирует директорию датасетов.  "
            "S3-режим: заново перечисляет префиксы в бакете."
        ),
        use_container_width=True,
    ):
        new_names = registry.rescan()
        if new_names:
            st.success(f"Найдены новые датасеты: {', '.join(new_names)}")
            st.rerun()
        else:
            st.info("Новых датасетов не найдено.")


def render() -> None:
    """Отрисовывает страницу поиска похожих кропов."""
    base_cfg = _load_config()
    cfg = _apply_env_overrides(base_cfg)

    st.title("🔍 Поиск похожих кропов")
    st.caption(
        "Загрузите изображение, нарисуйте рамку и найдите визуально "
        "похожие кропы в вашем датасете."
    )

    with st.sidebar:
        st.header("⚙️ Настройки")

        default_checkpoint = cfg.encoder.checkpoint if cfg.encoder else ""
        model_input = st.text_input(
            label="Чекпоинт модели (.pt / .pth или s3://)",
            value=default_checkpoint,
            placeholder="s3://bucket/models/encoder.pth  или  /local/encoder.pth",
            help=(
                "Локальный путь или s3://-URI к чекпоинту SSL-модели.  "
                "S3-модели скачиваются в локальный кэш при первом запросе "
                "и обновляются каждые 10 минут."
            ),
        )

        st.divider()

        registry: _AnyRegistry
        s3_client: S3Client | None = None

        if cfg.s3 is not None:
            s3 = cfg.s3
            prefix_display = s3.prefix or "/"
            st.caption(f"☁️ S3: `s3://{s3.bucket}/{prefix_display}`")

            registry = _load_s3_registry(
                bucket=s3.bucket,
                prefix=s3.prefix,
                region=s3.region,
                endpoint_url=s3.endpoint_url or "",
                check_interval=s3.check_interval_seconds,
            )
            s3_client = _load_s3_client(
                bucket=s3.bucket,
                prefix=s3.prefix,
                region=s3.region,
                endpoint_url=s3.endpoint_url or "",
                check_interval=s3.check_interval_seconds,
            )
        else:
            registry = _load_local_registry(str(cfg.app.datasets_dir))

        available_datasets = registry.available()
        if not available_datasets:
            if cfg.s3 is not None:
                msg = (
                    f"Датасеты не найдены в S3-бакете `{cfg.s3.bucket}` "
                    f"по префиксу `{cfg.s3.prefix or '/'}`.  \n"
                    "Запустите `scripts/build_index.py s3 …` для построения индекса."
                )
            else:
                msg = (
                    f"Индексированные датасеты не найдены в "
                    f"`{cfg.app.datasets_dir}`.  \n"
                    "Запустите `scripts/build_index.py local …` для создания индекса."
                )
            st.error(msg, icon="🚫")
            st.stop()

        dataset_name = st.selectbox(
            label="Датасет",
            options=available_datasets,
            help="Выберите датасет для поиска.",
        )
        assert dataset_name is not None

        _render_dataset_status(registry, dataset_name)

        st.divider()

        top_k: int = st.slider(
            label="Топ-K результатов",
            min_value=1,
            max_value=50,
            value=cfg.app.top_k,
            help="Количество ближайших соседей для поиска.",
        )

        st.divider()
        if cfg.s3 is None:
            st.caption(f"📁 `{cfg.app.datasets_dir}`")

    if not model_input:
        st.info(
            "👈 Укажите путь к чекпоинту SSL-модели в боковой панели.",
            icon="ℹ️",
        )
        st.stop()

    crop_result = render_crop_selector(cfg.app)
    if crop_result is None:
        st.stop()
    _full_image, crop = crop_result

    st.divider()
    if not st.button("🔍 Найти", type="primary", use_container_width=False):
        st.stop()

    # Параметры S3 для скачивания s3://-чекпоинтов
    s3_bucket_str = cfg.s3.bucket if cfg.s3 else ""
    s3_region_str = cfg.s3.region if cfg.s3 else "us-east-1"
    s3_endpoint_str = (cfg.s3.endpoint_url or "") if cfg.s3 else ""
    cache_dir_str = (
        str(cfg.encoder.cache_dir)
        if cfg.encoder and cfg.encoder.cache_dir
        else ""
    )

    try:
        embedder = _load_embedder(
            checkpoint=model_input,
            device=cfg.app.device,
            s3_bucket=s3_bucket_str,
            s3_region=s3_region_str,
            s3_endpoint_url=s3_endpoint_str,
            cache_dir_str=cache_dir_str,
        )
    except FileNotFoundError as exc:
        st.error(f"Чекпоинт модели не найден: {exc}", icon="🚫")
        st.stop()
    except (ValueError, RuntimeError) as exc:
        st.error(f"Ошибка загрузки модели: {exc}", icon="🚫")
        st.stop()

    with st.spinner("Вычисляется эмбеддинг..."):
        query_vec: np.ndarray = embedder.embed([crop])

    dataset_meta, faiss_index = registry.get(dataset_name)
    with st.spinner(f"Поиск топ-{top_k} в «{dataset_name}»..."):
        search_results = faiss_index.search(query_vec, top_k)

    render_results(search_results, dataset_meta.images_root, cfg.app, s3_client)

    if cfg.cvat is not None:
        render_cvat_exporter(
            all_results=search_results,
            images_root=dataset_meta.images_root,
            cvat_config=cfg.cvat,
            s3_client=s3_client,
        )
