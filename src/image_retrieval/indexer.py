"""Обёртка над FAISS-индексом для поиска ближайших соседей по косинусному сходству.

Дизайн
------
Используется ``faiss.IndexFlatIP`` (inner-product / dot-product индекс) совместно
с **L2-нормализованными** векторами эмбеддингов.  Для единичных векторов inner
product математически эквивалентен косинусному сходству, поэтому возвращаемые
оценки находятся в диапазоне ``[-1, 1]`` (больше = более похоже).

Индекс **только для чтения** во время работы приложения.  Построение и запись
индекса полностью выполняются в ``scripts/build_index.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SearchResult:
    """Один найденный bounding-box.

    Атрибуты:
        box_id: Уникальный идентификатор bounding-box (строка).
        image_path: Относительный (или абсолютный) путь к исходному изображению
            из метаданных.  Разрешается относительно ``DatasetMeta.images_root``
            для получения абсолютного пути.
        x1: Левый край bounding-box (пикселей).
        y1: Верхний край bounding-box (пикселей).
        x2: Правый край bounding-box (пикселей).
        y2: Нижний край bounding-box (пикселей).
        score: Оценка косинусного сходства в ``[-1, 1]``; больше = более похоже.
    """

    box_id: str
    image_path: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


# Обязательные колонки в Parquet-файле метаданных.
_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"image_path", "x1", "y1", "x2", "y2", "box_id"}
)


class FAISSIndex:
    """Обёртка над предпостроенным ``faiss.IndexFlatIP`` для поиска ближайших соседей.

    Индекс и метаданные загружаются с диска один раз при создании объекта
    и хранятся в памяти всё время жизни объекта.

    Args:
        index_path: Абсолютный путь к ``*.faiss``-файлу, записанному через
            ``faiss.write_index()``.
        metadata_path: Абсолютный путь к файлу ``metadata.parquet``.
            Должен содержать как минимум колонки: ``image_path``, ``x1``,
            ``y1``, ``x2``, ``y2``, ``box_id``.

    Raises:
        FileNotFoundError: Если *index_path* или *metadata_path* не существует.
        ValueError: Если в *metadata_path* отсутствуют обязательные колонки, или
            количество строк метаданных не совпадает с количеством векторов индекса.
    """

    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS-индекс не найден: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Parquet-метаданные не найдены: {metadata_path}")

        # faiss.read_index возвращает faiss.Index (базовый класс) — корректное сужение
        self._index: faiss.Index = faiss.read_index(str(index_path))
        self._metadata: pd.DataFrame = pd.read_parquet(metadata_path)

        self._validate()

    def _validate(self) -> None:
        """Проверяет наличие обязательных колонок и совпадение количества строк."""
        missing = _REQUIRED_COLUMNS - set(self._metadata.columns)
        if missing:
            raise ValueError(
                f"В метаданных отсутствуют обязательные колонки: {sorted(missing)}"
            )
        if len(self._metadata) != self._index.ntotal:
            raise ValueError(
                f"Метаданные содержат {len(self._metadata)} строк, "
                f"а FAISS-индекс — {self._index.ntotal} векторов: должны совпадать."
            )

    @property
    def ntotal(self) -> int:
        """Количество векторов в индексе."""
        return int(self._index.ntotal)

    @property
    def embedding_dim(self) -> int:
        """Размерность хранимых векторов эмбеддингов."""
        return int(self._index.d)

    def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        """Находит *top_k* наиболее похожих bounding-box'ов на *query*.

        Args:
            query: float32-ndarray формы ``(1, D)`` — должен быть L2-нормализован.
            top_k: Количество результатов.  Автоматически ограничивается
                ``self.ntotal``, чтобы не запрашивать больше, чем есть.

        Returns:
            Список :class:`SearchResult` в порядке убывания косинусного сходства
            (лучшее совпадение первым).

        Raises:
            ValueError: Если *query* не имеет формы ``(1, D)``, *top_k* < 1,
                или индекс пуст.
        """
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError(
                f"query должен иметь форму (1, D), получено {query.shape}"
            )
        if top_k < 1:
            raise ValueError("top_k должен быть не меньше 1")
        if self.ntotal == 0:
            raise ValueError("FAISS-индекс пуст — сначала запустите build_index.py.")

        k = min(top_k, self.ntotal)
        distances, indices = self._index.search(query.astype(np.float32), k)
        # distances / indices имеют форму (1, k)

        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0], strict=True):
            # FAISS возвращает индекс -1 если результатов меньше запрошенного
            if idx < 0:
                continue
            row = self._metadata.iloc[int(idx)]
            results.append(
                SearchResult(
                    box_id=str(row["box_id"]),
                    image_path=str(row["image_path"]),
                    x1=int(row["x1"]),
                    y1=int(row["y1"]),
                    x2=int(row["x2"]),
                    y2=int(row["y2"]),
                    score=float(dist),
                )
            )
        return results
