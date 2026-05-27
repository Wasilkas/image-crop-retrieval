"""Точка входа Streamlit-приложения с поддержкой нескольких страниц.

Запуск::

    uv run streamlit run app.py

Конфигурация загружается один раз (``@st.cache_resource``) и передаётся на
страницы как объект :class:`~image_retrieval.config.Configuration`.  Новые
страницы добавляются через ``st.Page()`` в списке ``st.navigation``.

Приоритет загрузки конфигурации:

1. **Consul KV** — если задана переменная среды ``CONSUL_URL``.
2. **YAML-файл** — ``CONFIG_PATH`` или ``config.yaml`` в рабочей директории.
3. **Умолчания** — все поля имеют разумные значения.

Быстрые переопределения через переменные среды::

    S3_BUCKET          Включить S3-режим; имя бакета
    S3_PREFIX          Префикс ключей датасетов в S3
    S3_REGION          Регион AWS (умолч.: ``us-east-1``)
    S3_ENDPOINT_URL    Кастомный endpoint (MinIO, Yandex Cloud и др.)
    DATASETS_DIR       Директория датасетов (только локальный режим)
    TOP_K              Количество результатов по умолчанию
    DEVICE             Устройство PyTorch
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from image_retrieval.config import AppBlock, Configuration, EncoderBlock, S3Block
from pages.image_retrieval import app as retrieval_app

st.set_page_config(
    page_title="Поиск похожих кропов",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def _load_config() -> Configuration:
    """Загружает и кэширует корневую конфигурацию на время жизни процесса."""
    return _apply_env_overrides(Configuration.load())


def _apply_env_overrides(cfg: Configuration) -> Configuration:
    """Применяет быстрые переопределения из переменных среды поверх *cfg*."""
    overrides: dict[str, object] = {}

    # Блок S3
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
                Path(datasets_dir_env)
                if datasets_dir_env
                else cfg.app.datasets_dir
            ),
            top_k=int(top_k_env) if top_k_env else cfg.app.top_k,
            device=device_env or cfg.app.device,
            min_crop_px=cfg.app.min_crop_px,
            canvas_width=cfg.app.canvas_width,
            canvas_height=cfg.app.canvas_height,
            results_columns=cfg.app.results_columns,
            buckets=cfg.app.buckets,
        )

    # Блок Encoder
    model_path_env = os.environ.get("MODEL_PATH", "").strip()
    if model_path_env and cfg.encoder is None:
        overrides["encoder"] = EncoderBlock(checkpoint=model_path_env)

    if not overrides:
        return cfg
    return cfg.model_copy(update=overrides)


cfg = _load_config()
_config_path = "config"

pg = st.navigation(
    [
        st.Page(
            lambda: retrieval_app(cfg, _config_path),
            title="Поиск похожих кропов",
            icon="🔍",
        )
    ]
)
pg.run()
