"""Dataset registry: discovers and loads all available FAISS-indexed datasets.

Expected on-disk layout per dataset (created by ``scripts/build_index.py``)::

    datasets/
    └── {dataset_name}/
        ├── index.faiss        ← FAISS IndexFlatIP
        ├── metadata.parquet   ← image_path, x1, y1, x2, y2, box_id
        └── images_root.txt    ← (optional) one line: path to images folder

Hot-reload
----------
The registry tracks the ``mtime`` (modification timestamp) of each dataset's
files at load time.  Every call to :meth:`get` checks whether the files on
disk have been updated since the last load.  If so, only that dataset is
reloaded — the app does not need to restart.

This is safe when ``scripts/build_index.py`` writes files **atomically** via
``tmp → rename``.  An atomic rename guarantees the app reads either the old
complete file or the new complete file, never a partially-written one.

Thread safety
-------------
A :class:`threading.RLock` serialises concurrent reload attempts for the same
dataset.  The check-then-reload sequence uses double-checked locking so that
multiple Streamlit threads don't reload the same dataset simultaneously.
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

# Backward-compatible aliases
AppConfig = AppBlock
S3Config = S3Block

logger = logging.getLogger(__name__)


@dataclass
class _DatasetEntry:
    """Internal storage for one loaded dataset.

    Attributes:
        meta: Resolved paths and name for the dataset.
        index: Loaded FAISS index + metadata.
        index_mtime_ns: Modification time of ``index.faiss`` at load time
            (nanoseconds, from ``Path.stat().st_mtime_ns``).
        metadata_mtime_ns: Modification time of ``metadata.parquet`` at load
            time (nanoseconds).
    """

    meta: DatasetMeta
    index: FAISSIndex
    index_mtime_ns: int
    metadata_mtime_ns: int


class DatasetRegistry:
    """Discovers, loads, and hot-reloads FAISS-indexed datasets.

    Intended to be instantiated once at Streamlit startup and cached with
    ``@st.cache_resource``.

    Args:
        config: Application block.  Only ``datasets_dir`` is used.
    """

    def __init__(self, config: AppBlock) -> None:
        self._config = config
        self._datasets: dict[str, _DatasetEntry] = {}
        self._lock = threading.RLock()
        self._scan()

    def available(self) -> list[str]:
        """Return a sorted list of successfully loaded dataset names."""
        with self._lock:
            return sorted(self._datasets.keys())

    def get(self, name: str) -> tuple[DatasetMeta, FAISSIndex]:
        """Return ``(DatasetMeta, FAISSIndex)`` for *name*, reloading if stale.

        Checks file ``mtime`` on every call.  If ``index.faiss`` or
        ``metadata.parquet`` was updated since the last load, the dataset is
        transparently reloaded before returning.  The reload is performed under
        a lock so concurrent calls do not trigger duplicate reloads.

        Args:
            name: Dataset name as returned by :meth:`available`.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        with self._lock:
            if name not in self._datasets:
                raise KeyError(
                    f"Dataset '{name}' not found. "
                    f"Available: {self.available()}"
                )
            self._check_and_reload(name)
            entry = self._datasets[name]
            return entry.meta, entry.index

    def rescan(self) -> list[str]:
        """Re-scan ``datasets_dir`` and load any new datasets.

        Existing (already loaded) datasets are *not* reloaded here — they are
        hot-reloaded lazily in :meth:`get` when their files change.

        Returns:
            Sorted list of **newly discovered** dataset names (empty if none).
        """
        with self._lock:
            before = set(self._datasets.keys())
            self._scan()
            after = set(self._datasets.keys())
            new_names = sorted(after - before)
            if new_names:
                logger.info("rescan: found new datasets: %s", new_names)
            return new_names

    def last_reload_info(self, name: str) -> tuple[int, int] | None:
        """Return ``(index_mtime_ns, metadata_mtime_ns)`` for *name*, or None.

        Useful for displaying "last updated" information in the UI.
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
        """Walk ``datasets_dir`` and load any sub-directory not yet loaded."""
        datasets_dir = self._config.datasets_dir
        if not datasets_dir.exists():
            logger.warning(
                "datasets_dir does not exist: %s — no datasets loaded.", datasets_dir
            )
            return

        candidates = sorted(p for p in datasets_dir.iterdir() if p.is_dir())
        for candidate in candidates:
            if candidate.name not in self._datasets:
                self._try_load(candidate)

    def _try_load(self, dataset_dir: Path) -> None:
        """Load a dataset directory for the first time; skip on any error."""
        index_path = dataset_dir / "index.faiss"
        metadata_path = dataset_dir / "metadata.parquet"

        if not index_path.exists() or not metadata_path.exists():
            logger.debug(
                "Skipping '%s': missing index.faiss or metadata.parquet.",
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
                "Loaded dataset '%s': %d vectors, dim=%d.",
                dataset_dir.name,
                faiss_index.ntotal,
                faiss_index.embedding_dim,
            )
        except Exception:
            logger.exception("Failed to load dataset '%s'.", dataset_dir.name)

    def _check_and_reload(self, name: str) -> None:
        """Reload *name* in-place if its files have changed since last load.

        Must be called with ``self._lock`` already held (RLock is reentrant,
        so calling :meth:`get` from within the lock is safe).

        The reload is skipped silently if the files cannot be stat-ed (e.g.
        because the cron job is mid-write on a non-atomic filesystem).
        """
        entry = self._datasets[name]

        try:
            current_index_mtime = entry.meta.index_path.stat().st_mtime_ns
            current_meta_mtime = entry.meta.metadata_path.stat().st_mtime_ns
        except FileNotFoundError:
            # Files disappeared (dataset being rebuilt) — keep old version
            logger.warning(
                "Dataset '%s': index files missing during mtime check; "
                "keeping cached version.",
                name,
            )
            return

        if (
            current_index_mtime == entry.index_mtime_ns
            and current_meta_mtime == entry.metadata_mtime_ns
        ):
            return  # nothing changed

        logger.info(
            "Dataset '%s': files changed on disk, hot-reloading…", name
        )
        self._try_reload(name, entry)

    def _try_reload(self, name: str, old_entry: _DatasetEntry) -> None:
        """Replace the cached entry for *name* with a freshly loaded one.

        On failure the old entry is kept and an error is logged, so the app
        continues to serve stale results rather than crashing.
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
                "Hot-reloaded dataset '%s': %d vectors.",
                name,
                new_index.ntotal,
            )
        except Exception:
            logger.exception(
                "Failed to hot-reload dataset '%s'; keeping stale version.",
                name,
            )


@dataclass
class _S3DatasetEntry:
    """Internal storage for one S3-backed dataset.

    Attributes:
        meta: Resolved paths — ``index_path`` and ``metadata_path`` point to
            the local cache, ``images_root`` is unused (images come from S3).
        index: Loaded FAISS index + metadata.
        s3_index_mtime: S3 ``LastModified`` timestamp of ``index.faiss`` at
            the time of the last download (POSIX float seconds).
        last_s3_check: ``time.monotonic()`` value of the last S3 poll.
    """

    meta: DatasetMeta
    index: FAISSIndex
    s3_index_mtime: float
    last_s3_check: float


class S3DatasetRegistry:
    """Dataset registry that reads indexes from an S3 bucket.

    On construction all available datasets are discovered, their index files
    downloaded to a local cache directory, and loaded into memory.

    S3 is polled for updates at most once per ``config.check_interval_seconds``
    (default: 300 s) per dataset.  When a newer ``index.faiss`` is found, only
    that dataset's files are re-downloaded and the FAISSIndex reloaded.
    The app never needs to restart.

    Images are **not** cached locally — they are fetched from S3 on-demand by
    :func:`~image_retrieval.ui.results_viewer.render_results` using the
    ``s3://`` URIs stored in ``metadata.parquet``.

    Args:
        s3_config: S3 connection and bucket parameters.
        cache_dir: Local directory for downloaded index files.  Defaults to
            ``<tmp>/image-retrieval-s3/``.
    """

    def __init__(
        self,
        s3_config: S3Block,
        cache_dir: Path | None = None,
    ) -> None:
        from .s3_client import S3Client  # local import to avoid circular

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
        """Return a sorted list of loaded dataset names."""
        with self._lock:
            return sorted(self._datasets.keys())

    def get(self, name: str) -> tuple[DatasetMeta, FAISSIndex]:
        """Return ``(DatasetMeta, FAISSIndex)`` for *name*.

        Polls S3 for updates at most once per ``check_interval_seconds``.
        If a newer ``index.faiss`` exists on S3, re-downloads and reloads
        before returning.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        with self._lock:
            if name not in self._datasets:
                raise KeyError(
                    f"Dataset '{name}' not found. Available: {self.available()}"
                )
            self._maybe_sync_dataset(name)
            entry = self._datasets[name]
            return entry.meta, entry.index

    def rescan(self) -> list[str]:
        """Re-discover datasets on S3; load newly found ones.

        Returns:
            Sorted list of **newly added** dataset names.
        """
        with self._lock:
            before = set(self._datasets.keys())
            self._sync_all()
            after = set(self._datasets.keys())
            new_names = sorted(after - before)
            if new_names:
                logger.info("S3 rescan: new datasets found: %s", new_names)
            return new_names

    def last_reload_info(self, name: str) -> tuple[int, int] | None:
        """Return ``(index_mtime_ns, 0)`` compatible with DatasetRegistry API.

        The second value is always 0 (S3 doesn't give nanosecond precision).
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
        """Download index files for all S3 datasets not yet loaded."""
        try:
            names = self._s3.list_dataset_names()
        except Exception:
            logger.exception("Failed to list S3 datasets under bucket '%s'.",
                             self._s3_config.bucket)
            return

        for name in names:
            if name not in self._datasets:
                self._try_load_from_s3(name)

    def _try_load_from_s3(self, dataset_name: str) -> None:
        """Download and load one S3 dataset; skip on any error."""
        local_dir = self._cache_dir / dataset_name
        index_key = self._s3.index_key(dataset_name)
        meta_key = self._s3.metadata_key(dataset_name)

        try:
            index_path = local_dir / "index.faiss"
            meta_path = local_dir / "metadata.parquet"

            if self._s3.is_remote_newer(index_key, index_path):
                logger.info("Downloading index for '%s'…", dataset_name)
                self._s3.download_file(index_key, index_path)

            if self._s3.is_remote_newer(meta_key, meta_path):
                logger.info("Downloading metadata for '%s'…", dataset_name)
                self._s3.download_file(meta_key, meta_path)

            s3_mtime = self._s3.get_last_modified(index_key) or 0.0

            meta = DatasetMeta(
                name=dataset_name,
                index_path=index_path,
                metadata_path=meta_path,
                images_root=local_dir,  # unused — images served from S3
            )
            faiss_index = FAISSIndex(index_path, meta_path)

            self._datasets[dataset_name] = _S3DatasetEntry(
                meta=meta,
                index=faiss_index,
                s3_index_mtime=s3_mtime,
                last_s3_check=time.monotonic(),
            )
            logger.info(
                "Loaded S3 dataset '%s': %d vectors, dim=%d.",
                dataset_name,
                faiss_index.ntotal,
                faiss_index.embedding_dim,
            )
        except Exception:
            logger.exception("Failed to load S3 dataset '%s'.", dataset_name)

    def _bump_s3_check(self, name: str, entry: _S3DatasetEntry) -> None:
        """Update the last-checked timestamp without re-downloading."""
        self._datasets[name] = _S3DatasetEntry(
            meta=entry.meta,
            index=entry.index,
            s3_index_mtime=entry.s3_index_mtime,
            last_s3_check=time.monotonic(),
        )

    def _maybe_sync_dataset(self, name: str) -> None:
        """Check S3 for a newer index if the TTL has expired.

        Called with ``self._lock`` held.
        """
        entry = self._datasets[name]
        elapsed = time.monotonic() - entry.last_s3_check
        if elapsed < self._s3_config.check_interval_seconds:
            return  # still within TTL — skip the S3 poll

        # TTL expired — check S3 for a newer version
        index_key = self._s3.index_key(name)
        try:
            remote_mtime = self._s3.get_last_modified(index_key)
        except Exception:
            logger.warning(
                "Could not poll S3 for dataset '%s'; keeping cached version.", name
            )
            # Reset timer so we don't spam failing requests
            self._bump_s3_check(name, entry)
            return

        if remote_mtime is None or remote_mtime <= entry.s3_index_mtime:
            # Nothing new — update the check timestamp and return
            self._bump_s3_check(name, entry)
            return

        # S3 has a newer version — re-download and reload
        logger.info("S3 dataset '%s' has been updated; hot-reloading…", name)
        old_entry = self._datasets[name]
        self._try_load_from_s3(name)
        if name in self._datasets and self._datasets[name] is not old_entry:
            logger.info("Hot-reload of '%s' complete.", name)


def _resolve_images_root(dataset_dir: Path) -> Path:
    """Return the images root for *dataset_dir*.

    Reads ``images_root.txt`` if present; otherwise defaults to *dataset_dir*.
    Relative paths in the file are resolved relative to *dataset_dir*.
    """
    txt_path = dataset_dir / "images_root.txt"
    if txt_path.exists():
        raw = txt_path.read_text(encoding="utf-8").strip()
        candidate = Path(raw)
        if candidate.is_absolute():
            return candidate
        return (dataset_dir / candidate).resolve()
    return dataset_dir
