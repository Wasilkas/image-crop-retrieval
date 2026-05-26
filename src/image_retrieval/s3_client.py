"""Низкоуровневые S3-утилиты для проекта image-crop-retrieval.

Модуль предоставляет тонкую типизированную обёртку над ``boto3`` для:

* Обнаружения датасетов через перечисление префиксов бакета.
* Поиска **последнего** файла ``split_<date>.csv`` по дате в имени файла
  (лексикографическая сортировка по ISO-8601 дате эквивалентна хронологической).
* Скачивания / загрузки артефактов индекса (``index.faiss``, ``metadata.parquet``)
  между S3 и локальным кэшем.
* Загрузки изображений из ``s3://bucket/key``-URI по запросу для отображения.

Учётные данные разрешаются через стандартную цепочку boto3:
переменные среды (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``),
``~/.aws/credentials`` или IAM-роль инстанса / задачи.

Для S3-совместимых хранилищ (MinIO, Yandex Cloud Object Storage и др.) укажите
``S3Config.endpoint_url``.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from PIL import Image as PILImage

from .config import S3Block as S3Config  # псевдоним S3Config оставлен для ясности

if TYPE_CHECKING:
    # boto3-stubs[s3] — dev-зависимость; импорт только для проверки типов
    from mypy_boto3_s3 import S3Client as BotoS3Client

logger = logging.getLogger(__name__)

# Совпадает с именами файлов вида split_2024-01-15.csv или split_20240115.csv
_SPLIT_RE = re.compile(r"split_(\d{4}-?\d{2}-?\d{2})\.csv$")

# Совпадает с URI вида s3://bucket/key
_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")


class S3Client:
    """Типизированная обёртка над ``boto3`` для S3-операций image-crop-retrieval.

    Args:
        config: Параметры подключения к S3.
    """

    def __init__(self, config: S3Config) -> None:
        self._config = config
        # Явная передача kwargs позволяет mypy использовать типизированную перегрузку
        # "s3" из boto3-stubs вместо непреодолимого пути через **kwargs.
        self._s3: BotoS3Client = boto3.client(
            "s3",
            region_name=config.region,
            endpoint_url=config.endpoint_url,
        )

    def list_dataset_names(self) -> list[str]:
        """Возвращает имена датасетов-поддиректорий под ``config.prefix``.

        Датасеты определяются как *общие префиксы* (виртуальные директории)
        непосредственно под настроенным префиксом.  Датасет ``"my_dataset"``
        должен соответствовать префиксу ``{config.prefix}my_dataset/`` в бакете.

        Returns:
            Отсортированный список имён датасетов.
        """
        response = self._s3.list_objects_v2(
            Bucket=self._config.bucket,
            Prefix=self._config.prefix,
            Delimiter="/",
        )
        common_prefixes = response.get("CommonPrefixes") or []
        names: list[str] = []
        for entry in common_prefixes:
            full_prefix: str = entry.get("Prefix", "")
            # Убираем родительский префикс и завершающий слэш для получения имени
            name = full_prefix.removeprefix(self._config.prefix).rstrip("/")
            if name:
                names.append(name)
        return sorted(names)

    def find_latest_split_key(self, dataset_name: str) -> str | None:
        """Находит S3-ключ самого свежего файла ``split_<date>.csv``.

        Файлы сравниваются по строке даты в имени.  ISO-8601-даты (``YYYY-MM-DD``)
        сортируются лексикографически в хронологическом порядке, поэтому
        разбор дат не требуется.

        Args:
            dataset_name: Имя поддиректории датасета.

        Returns:
            Полный S3-ключ самого свежего split-файла, или ``None`` если таких
            файлов нет под префиксом датасета.
        """
        prefix = self._config.dataset_prefix(dataset_name)
        response = self._s3.list_objects_v2(
            Bucket=self._config.bucket,
            Prefix=prefix,
        )
        contents = response.get("Contents") or []

        candidates: list[tuple[str, str]] = []  # (строка_даты, ключ)
        for obj in contents:
            key: str = obj.get("Key", "")
            filename = key.split("/")[-1]
            match = _SPLIT_RE.match(filename)
            if match:
                # Нормализуем: убираем тире, чтобы YYYYMMDD и YYYY-MM-DD
                # корректно сравнивались между собой
                date_str = match.group(1).replace("-", "")
                candidates.append((date_str, key))

        if not candidates:
            logger.warning(
                "Файлы split_<date>.csv не найдены под s3://%s/%s",
                self._config.bucket,
                prefix,
            )
            return None

        candidates.sort(key=lambda t: t[0], reverse=True)
        latest_key = candidates[0][1]
        logger.info("Последний split для '%s': %s", dataset_name, latest_key)
        return latest_key

    def download_file(self, s3_key: str, local_path: Path) -> None:
        """Скачивает *s3_key* из настроенного бакета в *local_path*.

        Родительская директория создаётся автоматически.

        Args:
            s3_key: Ключ объекта внутри ``config.bucket``.
            local_path: Путь назначения в локальной файловой системе.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "Скачивание s3://%s/%s → %s", self._config.bucket, s3_key, local_path
        )
        self._s3.download_file(self._config.bucket, s3_key, str(local_path))

    def download_uri(self, uri: str, local_path: Path) -> None:
        """Скачивает ``s3://bucket/key``-URI (или голый ключ) в *local_path*.

        В отличие от :meth:`download_file`, разбирает бакет и ключ из URI,
        поэтому работает с URI, ссылающимися на другой бакет (напр. чекпоинты
        модели из отдельного бакета).

        Родительская директория создаётся автоматически.

        Args:
            uri: ``s3://bucket/key``-URI или голый ключ относительно ``config.bucket``.
            local_path: Путь назначения в локальной файловой системе.
        """
        bucket, key = self._parse_uri(uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Скачивание %s → %s", uri, local_path)
        self._s3.download_file(bucket, key, str(local_path))

    def upload_file(self, local_path: Path, s3_key: str) -> None:
        """Загружает *local_path* в *s3_key* настроенного бакета.

        Args:
            local_path: Исходный файл в локальной файловой системе.
            s3_key: Ключ объекта назначения внутри ``config.bucket``.
        """
        logger.debug(
            "Загрузка %s → s3://%s/%s", local_path, self._config.bucket, s3_key
        )
        self._s3.upload_file(str(local_path), self._config.bucket, s3_key)

    def upload_file_atomic(self, local_path: Path, s3_key: str) -> None:
        """Загружает *local_path* во временный ``*.tmp``-ключ,
        затем переименовывает в *s3_key*.

        S3 не поддерживает атомарный rename нативно, но шаблон *copy + delete*
        гарантирует: читатели никогда не видят частично записанный объект.
        Окно между завершением загрузки и завершением copy/delete минимально.

        Args:
            local_path: Исходный файл.
            s3_key: Итоговый ключ в настроенном бакете.
        """
        tmp_key = s3_key + ".tmp"
        self.upload_file(local_path, tmp_key)
        self._s3.copy_object(
            Bucket=self._config.bucket,
            CopySource={"Bucket": self._config.bucket, "Key": tmp_key},
            Key=s3_key,
        )
        self._s3.delete_object(Bucket=self._config.bucket, Key=tmp_key)
        logger.debug(
            "Атомарная загрузка завершена: s3://%s/%s", self._config.bucket, s3_key
        )

    def load_image(self, uri: str) -> PILImage.Image:
        """Загружает PIL-изображение из S3-URI или голого ключа.

        Поддерживает два формата URI:
        * ``s3://bucket/path/to/image.jpg`` — явный бакет в URI.
        * ``path/to/image.jpg`` — ключ относительно ``config.bucket``.

        Args:
            uri: S3-URI или строка голого ключа.

        Returns:
            RGB PIL-изображение.

        Raises:
            ValueError: Если URI не удаётся разобрать.
            botocore.exceptions.ClientError: Если объект не существует.
        """
        bucket, key = self._parse_uri(uri)
        response = self._s3.get_object(Bucket=bucket, Key=key)
        data: bytes = response["Body"].read()
        return PILImage.open(io.BytesIO(data)).convert("RGB")

    def load_image_to_tmp(self, uri: str) -> Path:
        """Скачивает изображение во временный файл и возвращает его путь.

        Полезно при пакетной индексации, когда одно изображение используется
        несколько раз — вызывающий код кэширует путь по URI.

        Args:
            uri: S3-URI или голый ключ.

        Returns:
            Путь к временному файлу (вызывающий отвечает за удаление).
        """
        bucket, key = self._parse_uri(uri)
        suffix = Path(key).suffix or ".jpg"
        fd, tmp_str = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        tmp_path = Path(tmp_str)
        self._s3.download_file(bucket, key, str(tmp_path))
        return tmp_path

    def get_last_modified(self, s3_key: str) -> float | None:
        """Возвращает метку времени ``LastModified`` объекта *s3_key* как POSIX float.

        Возвращает ``None`` если объект не существует.

        Args:
            s3_key: Ключ объекта внутри ``config.bucket``.
        """
        try:
            head = self._s3.head_object(Bucket=self._config.bucket, Key=s3_key)
            return head["LastModified"].timestamp()
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            # botocore.exceptions.ClientError для 404
            if _is_not_found(exc):
                return None
            raise

    def is_remote_newer(self, s3_key: str, local_path: Path) -> bool:
        """Возвращает ``True`` если S3-объект новее *local_path*.

        Также возвращает ``True`` если *local_path* не существует (считается
        бесконечно старым) или если *s3_key* не существует на S3 (обновлений нет,
        возвращается ``False``).

        Args:
            s3_key: Ключ объекта внутри ``config.bucket``.
            local_path: Локальный файл для сравнения.
        """
        if not local_path.exists():
            return True
        remote_ts = self.get_last_modified(s3_key)
        if remote_ts is None:
            return False
        local_ts = local_path.stat().st_mtime
        return remote_ts > local_ts

    def index_key(self, dataset_name: str) -> str:
        """Возвращает S3-ключ файла ``index.faiss`` датасета *dataset_name*."""
        return self._config.dataset_prefix(dataset_name) + "index.faiss"

    def metadata_key(self, dataset_name: str) -> str:
        """Возвращает S3-ключ файла ``metadata.parquet`` датасета *dataset_name*."""
        return self._config.dataset_prefix(dataset_name) + "metadata.parquet"

    def _parse_uri(self, uri: str) -> tuple[str, str]:
        """Разбирает ``s3://bucket/key`` или голый ``key`` в ``(bucket, key)``.

        Args:
            uri: URI или строка голого ключа.

        Returns:
            Кортеж ``(bucket, key)``.
        """
        match = _S3_URI_RE.match(uri)
        if match:
            return match.group(1), match.group(2)
        # Голый ключ — используем настроенный бакет
        return self._config.bucket, uri


def _is_not_found(exc: Exception) -> bool:
    """Возвращает ``True`` если *exc* — boto3 ClientError для 404/NoSuchKey."""
    try:
        # botocore.exceptions.ClientError несёт словарь ``response``.
        code: str = exc.response["Error"]["Code"]  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return False
    return code in {"404", "NoSuchKey"}
