"""Реестр датасетов: обнаружение и загрузка FAISS-индексированных датасетов.

Ожидаемая структура директорий (создаётся ``scripts/build_index.py``)::

    datasets/
    └── {dataset_name}/
        ├── index.faiss        ← FAISS IndexFlatIP
        ├── metadata.parquet   ← image_path, x1, y1, x2, y2, box_id
        └── images_root.txt    ← (опционально) путь к директории изображений

Горячая перезагрузка
--------------------
Реестр отслеживает ``mtime`` (метку времени изменения) файлов каждого датасета
в момент загрузки.  При каждом вызове :meth:`get` проверяется, не изменились ли
файлы на диске.  Если да — только этот датасет перезагружается без перезапуска
приложения.

Это безопасно, когда ``scripts/build_index.py`` пишет файлы **атомарно** через
``tmp → rename``.  Атомарный rename гарантирует, что приложение читает либо
старый полный файл, либо новый полный файл, но никогда — частично записанный.

Потокобезопасность
------------------
:class:`threading.RLock` сериализует конкурентные попытки перезагрузки одного
датасета.  Последовательность проверка-перезагрузка использует двойную проверку
блокировки, чтобы несколько потоков Streamlit не перезагружали один датасет
одновременно.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import AppBlock, DatasetMeta, S3Block
from .indexer import FAISSIndex

# Обратно-совместимые псевдонимы
AppConfig = AppBlock
S3Config = S3Block

logger = logging.getLogger(__name__)


@dataclass
class _DatasetEntry:
    """Внутреннее хранилище для одного загруженного датасета.

    Атрибуты:
        meta: Resolved-пути и имя датасета.
        index: Загруженный FAISS-индекс + метаданные.
        index_mtime_ns: Время изменения ``index.faiss`` в момент загрузки
            (наносекунды, из ``Path.stat().st_mtime_ns``).
        metadata_mtime_ns: Время изменения ``metadata.parquet`` в момент загрузки
            (наносекунды).
    """

    meta: DatasetMeta
    index: FAISSIndex
    index_mtime_ns: int
    metadata_mtime_ns: int


class DatasetRegistry:
    """Обнаруживает, загружает и горячо перезагружает FAISS-индексированные датасеты.

    Предназначен для создания один раз при запуске Streamlit и кэширования
    через ``@st.cache_resource``.

    Args:
        config: Блок конфигурации приложения.  Используется только ``datasets_dir``.
    """

    def __init__(self, config: AppBlock) -> None:
        self._config = config
        self._datasets: dict[str, _DatasetEntry] = {}
        self._lock = threading.RLock()
        self._scan()

    def available(self) -> list[str]:
        """Возвращает отсортированный список успешно загруженных датасетов."""
        with self._lock:
            return sorted(self._datasets.keys())

    def get(self, name: str) -> tuple[DatasetMeta, FAISSIndex]:
        """Возвращает ``(DatasetMeta, FAISSIndex)`` для *name*,
        перезагружая при устаревании.

        Проверяет ``mtime`` файлов при каждом вызове.  Если ``index.faiss`` или
        ``metadata.parquet`` изменились с момента последней загрузки — датасет
        прозрачно перезагружается перед возвратом.  Перезагрузка выполняется под
        блокировкой, чтобы конкурентные вызовы не дублировали её.

        Args:
            name: Имя датасета из :meth:`available`.

        Raises:
            KeyError: Если *name* отсутствует в реестре.
        """
        with self._lock:
            if name not in self._datasets:
                raise KeyError(
                    f"Датасет '{name}' не найден. "
                    f"Доступные: {self.available()}"
                )
            self._check_and_reload(name)
            entry = self._datasets[name]
            return entry.meta, entry.index

    def rescan(self) -> list[str]:
        """Повторно сканирует ``datasets_dir`` и загружает новые датасеты.

        Уже загруженные датасеты здесь *не* перезагружаются — они горячо
        перезагружаются лениво в :meth:`get` при изменении файлов.

        Returns:
            Отсортированный список **вновь обнаруженных** датасетов (пустой если нет).
        """
        with self._lock:
            before = set(self._datasets.keys())
            self._scan()
            after = set(self._datasets.keys())
            new_names = sorted(after - before)
            if new_names:
                logger.info("Rescan: найдены новые датасеты: %s", new_names)
            return new_names

    def last_reload_info(self, name: str) -> tuple[int, int] | None:
        """Возвращает ``(index_mtime_ns, metadata_mtime_ns)`` для *name*, или None.

        Используется для отображения информации «последнее обновление» в UI.
        """
        with self._lock:
            entry = self._datasets.get(name)
            if entry is None:
                return None
            return entry.index_mtime_ns, entry.metadata_mtime_ns

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)

    def _scan(self) -> None:
        """Обходит ``datasets_dir`` и загружает ещё не загруженные поддиректории."""
        datasets_dir = self._config.datasets_dir
        if not datasets_dir.exists():
            logger.warning(
                "datasets_dir не существует: %s — датасеты не загружены.", datasets_dir
            )
            return

        candidates = sorted(p for p in datasets_dir.iterdir() if p.is_dir())
        for candidate in candidates:
            if candidate.name not in self._datasets:
                self._try_load(candidate)

    def _try_load(self, dataset_dir: Path) -> None:
        """Первичная загрузка директории датасета; при ошибке — пропуск."""
        index_path = dataset_dir / "index.faiss"
        metadata_path = dataset_dir / "metadata.parquet"

        if not index_path.exists() or not metadata_path.exists():
            logger.debug(
                "Пропуск '%s': отсутствует index.faiss или metadata.parquet.",
                dataset_dir.name,
            )
            return

        try:
            images_root = _resolve_images_root(dataset_dir)
            meta = DatasetMeta(
                name=dataset_dir.name,
                index_path=index_path,
                metadata_path=metadata_path,
                images_root=images_root,
            )
            faiss_index = FAISSIndex(index_path, metadata_path)
            entry = _DatasetEntry(
                meta=meta,
                index=faiss_index,
                index_mtime_ns=index_path.stat().st_mtime_ns,
                metadata_mtime_ns=metadata_path.stat().st_mtime_ns,
            )
            self._datasets[dataset_dir.name] = entry
            logger.info(
                "Загружен датасет '%s': %d векторов, dim=%d.",
                dataset_dir.name,
                faiss_index.ntotal,
                faiss_index.embedding_dim,
            )
        except Exception:
            logger.exception("Ошибка загрузки датасета '%s'.", dataset_dir.name)

    def _check_and_reload(self, name: str) -> None:
        """Перезагружает *name* если его файлы изменились с момента последней загрузки.

        Должен вызываться при уже захваченном ``self._lock`` (RLock реентерабелен,
        поэтому вызов :meth:`get` внутри блокировки безопасен).

        Перезагрузка молча пропускается, если файлы нельзя stat-нуть (напр.
        cron-задача пишет на не-атомарной файловой системе).
        """
        entry = self._datasets[name]

        try:
            current_index_mtime = entry.meta.index_path.stat().st_mtime_ns
            current_meta_mtime = entry.meta.metadata_path.stat().st_mtime_ns
        except FileNotFoundError:
            # Файлы исчезли (датасет пересобирается) — оставляем старую версию
            logger.warning(
                "Датасет '%s': файлы индекса отсутствуют при проверке mtime; "
                "используется кэшированная версия.",
                name,
            )
            return

        if (
            current_index_mtime == entry.index_mtime_ns
            and current_meta_mtime == entry.metadata_mtime_ns
        ):
            return  # ничего не изменилось

        logger.info(
            "Датасет '%s': файлы изменились на диске, горячая перезагрузка...", name
        )
        self._try_reload(name, entry)

    def _try_reload(self, name: str, old_entry: _DatasetEntry) -> None:
        """Заменяет кэшированную запись для *name* свежезагруженной.

        При ошибке старая запись сохраняется и логируется, чтобы приложение
        продолжало отдавать устаревшие результаты, а не падать.
        """
        meta = old_entry.meta
        try:
            new_index = FAISSIndex(meta.index_path, meta.metadata_path)
            new_entry = _DatasetEntry(
                meta=meta,
                index=new_index,
                index_mtime_ns=meta.index_path.stat().st_mtime_ns,
                metadata_mtime_ns=meta.metadata_path.stat().st_mtime_ns,
            )
            self._datasets[name] = new_entry
            logger.info(
                "Горячая перезагрузка датасета '%s': %d векторов.",
                name,
                new_index.ntotal,
            )
        except Exception:
            logger.exception(
                "Ошибка горячей перезагрузки датасета '%s'; "
                "используется устаревшая версия.",
                name,
            )


@dataclass
class _S3DatasetEntry:
    """Внутреннее хранилище для одного S3-датасета.

    Атрибуты:
        meta: Resolved-пути — ``index_path`` и ``metadata_path`` указывают на
            локальный кэш, ``images_root`` не используется (изображения из S3).
        index: Загруженный FAISS-индекс + метаданные.
        s3_index_mtime: Метка времени ``LastModified`` объекта ``index.faiss`` на S3
            в момент последнего скачивания (POSIX float, секунды).
        last_s3_check: Значение ``time.monotonic()`` последней проверки S3.
    """

    meta: DatasetMeta
    index: FAISSIndex
    s3_index_mtime: float
    last_s3_check: float


class S3DatasetRegistry:
    """Реестр датасетов, читающий индексы из S3-бакета.

    При создании обнаруживаются все доступные датасеты, их файлы индексов
    скачиваются в локальный кэш и загружаются в память.

    S3 опрашивается на обновления не чаще одного раза в
    ``config.check_interval_seconds``
    (по умолч. 300 с) на датасет.  При обнаружении нового ``index.faiss`` только
    файлы этого датасета перескачиваются и FAISSIndex перезагружается.
    Перезапуск приложения не требуется.

    Изображения **не кэшируются** локально — они загружаются из S3 по запросу
    через ``s3://``-URI в ``metadata.parquet``.

    Args:
        s3_config: Параметры подключения к S3.
        cache_dir: Локальная директория для скачанных файлов индексов.
            По умолчанию ``<tmp>/image-retrieval-s3/``.
    """

    def __init__(
        self,
        s3_config: S3Block,
        cache_dir: Path | None = None,
    ) -> None:
        from .s3_client import S3Client  # локальный импорт во избежание цикла

        self._s3_config = s3_config
        self._s3 = S3Client(s3_config)
        self._cache_dir = (
            cache_dir
            if cache_dir is not None
            else Path(tempfile.gettempdir()) / "image-retrieval-s3"
        )
        self._datasets: dict[str, _S3DatasetEntry] = {}
        self._lock = threading.RLock()
        self._sync_all()

    def available(self) -> list[str]:
        """Возвращает отсортированный список загруженных датасетов."""
        with self._lock:
            return sorted(self._datasets.keys())

    def get(self, name: str) -> tuple[DatasetMeta, FAISSIndex]:
        """Возвращает ``(DatasetMeta, FAISSIndex)`` для *name*.

        Опрашивает S3 на обновления не чаще одного раза в ``check_interval_seconds``.
        Если на S3 есть более новый ``index.faiss`` — перескачивает и перезагружает
        перед возвратом.

        Raises:
            KeyError: Если *name* отсутствует в реестре.
        """
        with self._lock:
            if name not in self._datasets:
                raise KeyError(
                    f"Датасет '{name}' не найден. Доступные: {self.available()}"
                )
            self._maybe_sync_dataset(name)
            entry = self._datasets[name]
            return entry.meta, entry.index

    def rescan(self) -> list[str]:
        """Повторно обнаруживает датасеты в S3; загружает вновь найденные.

        Returns:
            Отсортированный список **вновь добавленных** датасетов.
        """
        with self._lock:
            before = set(self._datasets.keys())
            self._sync_all()
            after = set(self._datasets.keys())
            new_names = sorted(after - before)
            if new_names:
                logger.info("S3 rescan: найдены новые датасеты: %s", new_names)
            return new_names

    def last_reload_info(self, name: str) -> tuple[int, int] | None:
        """Возвращает ``(index_mtime_ns, 0)`` совместимо с DatasetRegistry API.

        Второе значение всегда 0 (S3 не даёт наносекундной точности).
        """
        with self._lock:
            entry = self._datasets.get(name)
            if entry is None:
                return None
            return int(entry.s3_index_mtime * 1_000_000_000), 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)

    def _sync_all(self) -> None:
        """Скачивает файлы индексов для всех S3-датасетов, ещё не загруженных."""
        try:
            names = self._s3.list_dataset_names()
        except Exception:
            logger.exception(
                "Ошибка перечисления датасетов S3 в бакете '%s'.",
                self._s3_config.bucket,
            )
            return

        for name in names:
            if name not in self._datasets:
                self._try_load_from_s3(name)

    def _try_load_from_s3(self, dataset_name: str) -> None:
        """Скачивает и загружает один S3-датасет; при ошибке — пропуск."""
        local_dir = self._cache_dir / dataset_name
        index_key = self._s3.index_key(dataset_name)
        meta_key = self._s3.metadata_key(dataset_name)

        try:
            index_path = local_dir / "index.faiss"
            meta_path = local_dir / "metadata.parquet"

            if self._s3.is_remote_newer(index_key, index_path):
                logger.info("Скачивание индекса для '%s'...", dataset_name)
                self._s3.download_file(index_key, index_path)

            if self._s3.is_remote_newer(meta_key, meta_path):
                logger.info("Скачивание метаданных для '%s'...", dataset_name)
                self._s3.download_file(meta_key, meta_path)

            s3_mtime = self._s3.get_last_modified(index_key) or 0.0

            meta = DatasetMeta(
                name=dataset_name,
                index_path=index_path,
                metadata_path=meta_path,
                images_root=local_dir,  # не используется — изображения из S3
            )
            faiss_index = FAISSIndex(index_path, meta_path)

            self._datasets[dataset_name] = _S3DatasetEntry(
                meta=meta,
                index=faiss_index,
                s3_index_mtime=s3_mtime,
                last_s3_check=time.monotonic(),
            )
            logger.info(
                "Загружен S3-датасет '%s': %d векторов, dim=%d.",
                dataset_name,
                faiss_index.ntotal,
                faiss_index.embedding_dim,
            )
        except Exception:
            logger.exception("Ошибка загрузки S3-датасета '%s'.", dataset_name)

    def _bump_s3_check(self, name: str, entry: _S3DatasetEntry) -> None:
        """Обновляет метку времени последней проверки без перескачивания."""
        self._datasets[name] = _S3DatasetEntry(
            meta=entry.meta,
            index=entry.index,
            s3_index_mtime=entry.s3_index_mtime,
            last_s3_check=time.monotonic(),
        )

    def _maybe_sync_dataset(self, name: str) -> None:
        """Проверяет S3 на новый индекс, если TTL истёк.

        Вызывается при захваченном ``self._lock``.
        """
        entry = self._datasets[name]
        elapsed = time.monotonic() - entry.last_s3_check
        if elapsed < self._s3_config.check_interval_seconds:
            return  # TTL не истёк — пропускаем опрос S3

        # TTL истёк — проверяем S3 на обновление
        index_key = self._s3.index_key(name)
        try:
            remote_mtime = self._s3.get_last_modified(index_key)
        except Exception:
            logger.warning(
                "Не удалось опросить S3 для датасета '%s'; используется кэш.", name
            )
            # Сбрасываем таймер, чтобы не спамить неудачными запросами
            self._bump_s3_check(name, entry)
            return

        if remote_mtime is None or remote_mtime <= entry.s3_index_mtime:
            # Обновлений нет — обновляем метку проверки и возвращаемся
            self._bump_s3_check(name, entry)
            return

        # На S3 новая версия — перескачиваем и перезагружаем
        logger.info("S3-датасет '%s' обновлён; горячая перезагрузка...", name)
        old_entry = self._datasets[name]
        self._try_load_from_s3(name)
        if name in self._datasets and self._datasets[name] is not old_entry:
            logger.info("Горячая перезагрузка '%s' завершена.", name)


def _resolve_images_root(dataset_dir: Path) -> Path:
    """Возвращает директорию изображений для *dataset_dir*.

    Читает ``images_root.txt`` если файл существует; иначе возвращает *dataset_dir*.
    Относительные пути в файле разрешаются относительно *dataset_dir*.
    """
    txt_path = dataset_dir / "images_root.txt"
    if txt_path.exists():
        raw = txt_path.read_text(encoding="utf-8").strip()
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        return (dataset_dir / candidate).resolve()
    return dataset_dir
