"""Конфигурация приложения — блочный дизайн на Pydantic v2.

Один объект :class:`Configuration` содержит все блоки подсистем.  Каждый блок —
замороженная Pydantic-модель: поля валидируются при создании и неизменны после.

Приоритет загрузки (от высшего к низшему):

1. **Consul KV** — если ``consul_url`` задан (или переменная среды ``CONSUL_URL``)
   и сервер доступен, из него читается YAML по указанному ключу.
2. **YAML-файл** — ``config_path`` (или ``CONFIG_PATH``, по умолч. ``config.yaml``
   в рабочей директории) если файл существует.
3. **Встроенные умолчания** — у всех полей есть разумные значения; объект
   :class:`Configuration` без аргументов полностью работоспособен.

Пример ``config.yaml``::

    app:
      top_k: 20
      device: cpu

    s3:
      bucket: my-datasets-bucket
      prefix: datasets/
      region: eu-west-1

    encoder:
      checkpoint: s3://my-datasets-bucket/models/encoder.pth

    cvat:
      url: https://cvat.example.com
      token: my-api-token
      project_id: 42

Обратно-совместимые псевдонимы :data:`AppConfig` и :data:`S3Config` сохранены,
чтобы существующий код продолжал компилироваться без изменений.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


def _default_datasets_dir() -> Path:
    """Возвращает ``<корень_проекта>/datasets`` как директорию датасетов
    по умолчанию."""
    return (Path(__file__).resolve().parents[3] / "datasets").resolve()


class AppBlock(BaseModel):
    """Настройки UI и среды выполнения Streamlit-приложения.

    Атрибуты:
        datasets_dir: Директория с поддиректориями-датасетами (только локальный
            режим; игнорируется, если задан :attr:`Configuration.s3`).
        top_k: Количество ближайших соседей для поиска по умолчанию.
        device: Строка устройства PyTorch (``"cpu"`` для CPU-инференса).
        min_crop_px: Минимальная сторона (в пикселях) нарисованного кропа.
        canvas_width: Максимальная ширина холста для рисования (пкс).
        canvas_height: Максимальная высота холста для рисования (пкс).
        results_columns: Количество колонок в сетке результатов.
    """

    model_config = ConfigDict(frozen=True)

    datasets_dir: Path = Field(default_factory=_default_datasets_dir)
    top_k: int = 10
    device: str = "cpu"
    min_crop_px: int = 8
    canvas_width: int = 800
    canvas_height: int = 600
    results_columns: int = 3
    buckets: list[str] = Field(default_factory=list)

    @field_validator("datasets_dir", mode="before")
    @classmethod
    def _resolve_datasets_dir(cls, v: Any) -> Path:  # noqa: ANN401
        return Path(v).resolve()


class S3Block(BaseModel):
    """Параметры подключения к S3-совместимому объектному хранилищу.

    Атрибуты:
        bucket: Имя S3-бакета.
        prefix: Префикс ключей, под которым хранятся все датасеты
            (напр. ``"datasets/"``). Может быть пустым для доступа от корня бакета.
        region: Строка региона AWS (напр. ``"us-east-1"``).
        endpoint_url: Кастомный endpoint для S3-совместимых хранилищ (MinIO,
            Yandex Cloud Object Storage и др.).  ``None`` → стандартные AWS-endpoint'ы.
        check_interval_seconds: Как часто (в секундах) приложение опрашивает S3
            для обнаружения нового индекса.
    """

    model_config = ConfigDict(frozen=True)

    bucket: str
    prefix: str = ""
    region: str = "us-east-1"
    endpoint_url: str | None = None
    check_interval_seconds: int = 300

    def dataset_prefix(self, dataset_name: str) -> str:
        """Возвращает S3-префикс ключей датасета *dataset_name*
        (с завершающим слэшем)."""
        base = self.prefix.rstrip("/")
        return f"{base}/{dataset_name}/" if base else f"{dataset_name}/"


class EncoderBlock(BaseModel):
    """Конфигурация SSL-модели для вычисления эмбеддингов кропов.

    Атрибуты:
        checkpoint: Путь к чекпоинту ``.pt`` / ``.pth``.  Принимает как локальные
            пути, так и ``s3://bucket/key``-URI.  При S3-URI файл скачивается
            в *cache_dir* при первом использовании и переиспользуется при
            последующих запусках (если объект на S3 не обновился).
        model_module: Необязательная строка ``"module.path:ClassName"`` для
            state-dict-чекпоинтов, требующих явный класс модели.
            ``None`` → загрузка полной модели из pickle.
        cache_dir: Директория для кэширования скачанных S3-моделей.
            По умолчанию ``<tmp>/image-retrieval-models/``.
    """

    model_config = ConfigDict(frozen=True)

    checkpoint: str
    model_module: str | None = None
    cache_dir: Path | None = None

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _resolve_cache_dir(cls, v: Any) -> Path | None:  # noqa: ANN401
        return Path(v).resolve() if v is not None else None


class CVATBlock(BaseModel):
    """Параметры подключения к платформе аннотирования CVAT (API v2.x).

    Приоритет аутентификации: *token* → ``username`` + ``password``.

    Атрибуты:
        url: Базовый URL инстанса CVAT (напр. ``"https://cvat.example.com"``).
        token: API-токен CVAT.  Имеет приоритет над username/password.
        username: Логин CVAT (используется только если *token* не задан).
        password: Пароль CVAT.
        project_id: Проект, к которому прикрепляются новые задачи.
            ``None`` → задачи создаются без проекта.
        task_label: Название метки bounding-box в экспортируемых задачах.
    """

    model_config = ConfigDict(frozen=True)

    url: str = ""
    token: str | None = None
    username: str | None = None
    password: str | None = None
    project_id: int | None = None
    task_label: str = "crop"
    cvat_name: str = "sip"

    @field_validator("url", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, v: Any) -> str:  # noqa: ANN401
        return str(v).rstrip("/")


class Configuration(BaseModel):
    """Корневой объект конфигурации, содержащий все блоки подсистем.

    Все блоки опциональны, кроме *app* — он всегда несёт умолчания.
    Используйте :meth:`load` как канонический способ создания объекта:
    он автоматически обрабатывает цепочку Consul → YAML → умолчания.
    """

    model_config = ConfigDict(frozen=True)

    app: AppBlock = Field(default_factory=AppBlock)
    s3: S3Block | None = None
    encoder: EncoderBlock | None = None
    cvat: CVATBlock | None = None

    @classmethod
    def load(
        cls,
        *,
        config_path: Path | None = None,
        consul_url: str | None = None,
        consul_key: str | None = None,
    ) -> Configuration:
        """Загружает конфигурацию по цепочке Consul → YAML → умолчания.

        Все параметры по умолчанию берутся из переменных среды, поэтому
        достаточно вызвать ``Configuration.load()`` и положиться на
        ``CONSUL_URL``, ``CONSUL_KEY`` и ``CONFIG_PATH``.

        Args:
            config_path: Путь к YAML-файлу конфигурации.  Если не задан —
                читается из ``CONFIG_PATH``, затем ``config.yaml`` в cwd.
            consul_url: URL агента Consul (схема + хост + порт).
                Если не задан — читается из ``CONSUL_URL``.
            consul_key: Ключ KV, значение которого — YAML-документ.
                Если не задан — читается из ``CONSUL_KEY``.
        """
        effective_consul_url = consul_url or os.environ.get("CONSUL_URL", "")
        effective_consul_key = (
            consul_key
            or os.environ.get("CONSUL_KEY", "config/image-crop-retrieval")
        )

        if effective_consul_url:
            try:
                cfg = cls.from_consul(effective_consul_url, effective_consul_key)
                logger.info(
                    "Конфигурация загружена из Consul, ключ '%s'.",
                    effective_consul_key,
                )
                return cfg
            except Exception as exc:
                logger.warning(
                    "Consul недоступен (%s) — переключаемся на YAML / умолчания.",
                    exc,
                )

        effective_path = config_path or Path(
            os.environ.get("CONFIG_PATH", "config.yaml")
        )
        if effective_path.exists():
            cfg = cls.from_yaml(effective_path)
            logger.info("Конфигурация загружена из '%s'.", effective_path)
            return cfg

        logger.info("Consul и файл конфигурации не найдены — используются умолчания.")
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> Configuration:
        """Разбирает *path* как YAML и возвращает валидированный :class:`Configuration`.

        Args:
            path: Путь к YAML-файлу конфигурации.

        Raises:
            FileNotFoundError: Если *path* не существует.
            pydantic.ValidationError: Если содержимое YAML не прошло валидацию.
        """
        import yaml  # локальный импорт: pyyaml опциональна до вызова этого метода

        with path.open(encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.model_validate(data)

    @classmethod
    def from_consul(cls, url: str, key: str) -> Configuration:
        """Читает YAML-документ из Consul KV и разбирает его.

        Args:
            url: Базовый URL агента Consul (схема + хост + порт).
            key: KV-ключ, значение которого — YAML-документ.

        Raises:
            KeyError: Если *key* не существует в KV-хранилище.
        """
        from urllib.parse import urlparse

        import consul
        import yaml

        parsed = urlparse(url)
        c = consul.Consul(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8500,
            scheme=parsed.scheme or "http",
        )
        _, kv_data = c.kv.get(key)
        if kv_data is None:
            raise KeyError(f"Consul-ключ '{key}' не найден по адресу {url}")
        raw: str = kv_data["Value"].decode("utf-8")
        data: dict[str, Any] = yaml.safe_load(raw) or {}
        return cls.model_validate(data)


@dataclass(frozen=True)
class DatasetMeta:
    """Resolved-пути для одного индексированного датасета.

    Создаётся :class:`~image_retrieval.registry.DatasetRegistry` при запуске;
    передаётся в :class:`~image_retrieval.indexer.FAISSIndex` и виджет
    отображения результатов.

    Атрибуты:
        name: Человекочитаемое имя датасета (имя поддиректории в ``datasets_dir``).
        index_path: Абсолютный путь к файлу ``index.faiss``.
        metadata_path: Абсолютный путь к файлу ``metadata.parquet``.
        images_root: Абсолютный путь к директории с исходными изображениями.
    """

    name: str
    index_path: Path
    metadata_path: Path
    images_root: Path


#: Псевдоним :class:`AppBlock` — сохраняет работоспособность импортов ``AppConfig``.
AppConfig = AppBlock

#: Псевдоним :class:`S3Block` — сохраняет работоспособность импортов ``S3Config``.
S3Config = S3Block
