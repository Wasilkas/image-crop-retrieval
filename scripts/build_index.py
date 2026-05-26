#!/usr/bin/env python3
"""CLI-скрипт: построение FAISS-индекса из аннотированного датасета изображений.

Поддерживает два режима: **local** и **s3**.

Локальный режим
---------------
Читает аннотации из локального CSV/Parquet-файла, изображения — из локальной
директории.  Записывает индекс в ``datasets/{dataset_name}/``.

::

    uv run python scripts/build_index.py local \\
        --annotations /data/annotations.csv \\
        --images-root /data/images/ \\
        --checkpoint   /models/encoder.pth \\
        --dataset-name my_dataset

S3-режим
--------
Находит последний ``split_<date>.csv`` под заданным S3-префиксом датасета,
скачивает его, загружает изображения из S3 (через колонку ``s3_image_path``),
строит индекс и загружает ``index.faiss`` + ``metadata.parquet`` обратно в S3.

::

    uv run python scripts/build_index.py s3 \\
        --bucket    my-bucket \\
        --dataset   my_dataset \\
        --checkpoint /models/encoder.pth

Требования к колонкам
---------------------
*Локальный режим* — аннотации должны содержать: ``image_path, x1, y1, x2, y2``.

*S3-режим* — split CSV должен содержать: ``s3_image_path, x1, y1, x2, y2``.
Значения ``s3_image_path`` должны быть валидными ``s3://bucket/key``-URI.

Колонка ``box_id`` автоматически генерируется если отсутствует в обоих режимах.

State-dict-чекпоинты
--------------------
Передайте ``--model-module module.path:ClassName`` для state-dict-чекпоинта
(в любом режиме)::

    uv run python scripts/build_index.py s3 \\
        --bucket    my-bucket \\
        --dataset   my_dataset \\
        --checkpoint /models/weights.pt \\
        --model-module mypackage.models:MyEncoder

.. warning::
    ``torch.load(weights_only=False)`` выполняет произвольный pickle-код.
    Загружайте чекпоинты только из **доверенных источников**.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image as PILImage
from tqdm import tqdm

from image_retrieval.embedder import TorchEmbedder

logger = logging.getLogger(__name__)


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    """Добавляет аргументы эмбеддера, общие для обоих режимов."""
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        metavar="FILE",
        help="Путь к чекпоинту SSL-модели (.pt / .pth)",
    )
    p.add_argument(
        "--model-module",
        default=None,
        metavar="MODULE:CLASS",
        help=(
            "Для state-dict-чекпоинтов: 'module.path:ClassName'.  "
            "Пример: mypackage.models:ResNetEncoder"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Количество кропов за один forward pass",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Строка устройства PyTorch",
    )
    p.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("H", "W"),
        help="Размер кропа для подачи в модель (высота ширина)",
    )


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="build_index",
        description="Построение FAISS-индекса для image-crop-retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = root.add_subparsers(dest="mode", required=True)

    local = sub.add_parser(
        "local",
        help="Построение из локального CSV/Parquet + локальной директории изображений.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    local.add_argument(
        "--annotations",
        required=True,
        type=Path,
        metavar="FILE",
        help="CSV или Parquet с колонками: image_path, x1, y1, x2, y2",
    )
    local.add_argument(
        "--images-root",
        required=True,
        type=Path,
        metavar="DIR",
        help="Корневая директория изображений из --annotations",
    )
    local.add_argument(
        "--dataset-name",
        required=True,
        metavar="NAME",
        help="Имя выходной поддиректории датасета",
    )
    local.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("datasets"),
        metavar="DIR",
        help="Корневая директория датасетов (вывод: DATASETS_DIR/DATASET_NAME/)",
    )
    _add_shared_args(local)

    s3p = sub.add_parser(
        "s3",
        help=(
            "Построение из S3: читает последний split_<date>.csv, "
            "записывает индекс в S3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    s3p.add_argument(
        "--bucket",
        required=True,
        metavar="BUCKET",
        help="Имя S3-бакета",
    )
    s3p.add_argument(
        "--dataset",
        required=True,
        metavar="NAME",
        help="Имя датасета (префикс внутри бакета)",
    )
    s3p.add_argument(
        "--prefix",
        default="",
        metavar="PREFIX",
        help="Префикс ключей для хранения датасетов (напр. 'datasets/')",
    )
    s3p.add_argument(
        "--region",
        default="us-east-1",
        metavar="REGION",
        help="Регион AWS",
    )
    s3p.add_argument(
        "--endpoint-url",
        default=None,
        metavar="URL",
        help="Кастомный S3-совместимый endpoint (напр. для MinIO)",
    )
    s3p.add_argument(
        "--image-cache-dir",
        default=None,
        type=Path,
        metavar="DIR",
        help=(
            "Локальная директория для кэширования скачанных изображений "
            "во время индексации.  По умолчанию — временная директория, "
            "удаляемая после завершения."
        ),
    )
    _add_shared_args(s3p)

    return root


def _resolve_model_class(module_spec: str) -> type[nn.Module]:
    """Импортирует и возвращает подкласс ``nn.Module`` из ``'module:Class'``.

    Raises:
        ValueError: Если *module_spec* не содержит ``:``.
        TypeError: Если разрешённый объект не является подклассом ``nn.Module``.
    """
    if ":" not in module_spec:
        raise ValueError(
            "--model-module должен быть 'module.path:ClassName', "
            f"получено: '{module_spec}'"
        )
    module_path, class_name = module_spec.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise TypeError(
            f"'{module_spec}' разрешился в {cls!r}, а не в подкласс nn.Module."
        )
    return cls


def _validate_and_fill_box_id(
    df: pd.DataFrame,
    required: set[str],
    context: str = "Аннотации",
) -> pd.DataFrame:
    """Поднимает ValueError при отсутствии обязательных колонок;
    автозаполняет box_id."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{context}: отсутствуют колонки: {sorted(missing)}")
    if "box_id" not in df.columns:
        df["box_id"] = [f"box_{i}" for i in range(len(df))]
    return df.reset_index(drop=True)


def _load_annotations_local(path: Path) -> pd.DataFrame:
    """Загружает локальный CSV или Parquet файл аннотаций.

    Обязательные колонки: ``image_path, x1, y1, x2, y2``.
    ``box_id`` автогенерируется если отсутствует.

    Returns:
        Валидированный DataFrame с ``image_path``, указывающим на локальные файлы.
    """
    if not path.exists():
        raise FileNotFoundError(f"Файл аннотаций не найден: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = _validate_and_fill_box_id(df, {"image_path", "x1", "y1", "x2", "y2"})
    logger.info("Загружено %d локальных аннотаций из '%s'.", len(df), path)
    return df


def _load_annotations_s3(
    s3_client: object,  # S3Client, слабая типизация во избежание цикла
    dataset_name: str,
    tmp_dir: Path,
) -> pd.DataFrame:
    """Скачивает последний ``split_<date>.csv`` из S3 и загружает его.

    Обязательные колонки CSV: ``s3_image_path, x1, y1, x2, y2``.

    Значения ``s3_image_path`` копируются в ``image_path``, чтобы метаданные
    индекса хранили ``s3://``-URI — приложение обнаруживает их при запросе
    и маршрутизирует загрузку через S3.

    Returns:
        Валидированный DataFrame, готовый для :func:`_embed_all_s3`.
    """
    from image_retrieval.s3_client import S3Client

    assert isinstance(s3_client, S3Client)

    split_key = s3_client.find_latest_split_key(dataset_name)
    if split_key is None:
        raise FileNotFoundError(
            f"Файлы split_<date>.csv не найдены для датасета '{dataset_name}' "
            f"в бакете '{s3_client._config.bucket}'."
        )

    local_csv = tmp_dir / "annotations.csv"
    logger.info(
        "Скачивание аннотаций: s3://%s/%s", s3_client._config.bucket, split_key
    )
    s3_client.download_file(split_key, local_csv)

    df = pd.read_csv(local_csv)
    df = df.copy()
    df["image_path"] = df["s3_image_path"]
    df = _validate_and_fill_box_id(
        df,
        {"s3_image_path", "x1", "y1", "x2", "y2"},
        f"Split CSV '{split_key}'",
    )
    logger.info(
        "Загружено %d аннотаций из S3-split '%s'.", len(df), split_key
    )
    return df


def _open_local(path: Path) -> PILImage.Image:
    """Открывает локальный файл изображения как RGB PIL Image."""
    return PILImage.open(path).convert("RGB")


def _embed_all_local(
    df: pd.DataFrame,
    images_root: Path,
    embedder: TorchEmbedder,
    batch_size: int,
) -> np.ndarray:
    """Вырезает кропы и вычисляет эмбеддинги для локальных изображений.

    Строки, для которых не удаётся загрузить изображение, заполняются нулями
    с предупреждением.
    """
    return _run_embedding_loop(
        df=df,
        load_image_fn=lambda img_path_str: _open_local(images_root / img_path_str),
        image_col="image_path",
        embedder=embedder,
        batch_size=batch_size,
    )


def _embed_all_s3(
    df: pd.DataFrame,
    s3_client: object,
    embedder: TorchEmbedder,
    batch_size: int,
    cache_dir: Path,
) -> np.ndarray:
    """Вырезает кропы и вычисляет эмбеддинги для S3-изображений
    с кэшированием в *cache_dir*.

    Каждый уникальный ``s3_image_path`` скачивается один раз и хранится в
    *cache_dir* на время выполнения.  Для 100K боксов из ~10K изображений
    это исключает повторное скачивание одного изображения для каждого бокса.
    """
    from image_retrieval.s3_client import S3Client

    assert isinstance(s3_client, S3Client)

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_cache: dict[str, Path] = {}

    def load_s3_image(uri: str) -> PILImage.Image:
        if uri not in local_cache:
            local_cache[uri] = s3_client.load_image_to_tmp(uri)
        cached = local_cache[uri]
        return PILImage.open(cached).convert("RGB")

    try:
        return _run_embedding_loop(
            df=df,
            load_image_fn=load_s3_image,
            image_col="s3_image_path",
            embedder=embedder,
            batch_size=batch_size,
        )
    finally:
        # Очищаем временные файлы изображений
        for tmp_path in local_cache.values():
            with contextlib.suppress(Exception):
                tmp_path.unlink(missing_ok=True)


def _run_embedding_loop(
    df: pd.DataFrame,
    load_image_fn: object,  # Callable[[str], PILImage.Image]
    image_col: str,
    embedder: TorchEmbedder,
    batch_size: int,
) -> np.ndarray:
    """Основной цикл эмбеддинга, общий для локального и S3-режимов.

    Args:
        df: DataFrame аннотаций.
        load_image_fn: Callable, принимающий строку пути/URI изображения и
            возвращающий PIL Image (RGB).  Может бросать исключения при ошибке.
        image_col: Имя колонки в *df* с путями/URI изображений.
        embedder: Экземпляр эмбеддера.
        batch_size: Количество кропов за один forward pass.

    Returns:
        float32-ndarray формы ``(len(df), embedding_dim)``.
    """
    from collections.abc import Callable

    assert callable(load_image_fn)
    loader: Callable[[str], PILImage.Image] = load_image_fn

    all_vecs: list[np.ndarray] = []
    placeholder: PILImage.Image | None = None

    progress = tqdm(total=len(df), desc="Вычисление эмбеддингов", unit="бокс")
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        crops: list[PILImage.Image] = []
        valid: list[bool] = []

        for _, row in batch.iterrows():
            try:
                img = loader(str(row[image_col]))
                crops.append(
                    img.crop(
                        (int(row["x1"]), int(row["y1"]),
                         int(row["x2"]), int(row["y2"]))
                    )
                )
                valid.append(True)
            except Exception:
                logger.warning(
                    "Пропуск бокса (image='%s'): не удалось загрузить/вырезать.",
                    row[image_col],
                    exc_info=True,
                )
                if placeholder is None:
                    placeholder = PILImage.new("RGB", (16, 16), color=0)
                crops.append(placeholder)
                valid.append(False)

        vecs = embedder.embed(crops)
        for i, ok in enumerate(valid):
            if not ok:
                vecs[i] = 0.0

        all_vecs.append(vecs)
        progress.update(len(batch))

    progress.close()
    return np.vstack(all_vecs).astype(np.float32)


def _build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Создаёт и заполняет IndexFlatIP (предварительно L2-нормализуя векторы)."""
    faiss.normalize_L2(embeddings)
    _, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    logger.info("Построен FAISS-индекс: %d векторов, dim=%d.", index.ntotal, d)
    return index


def _write_outputs_local(
    index: faiss.IndexFlatIP,
    df: pd.DataFrame,
    out_dir: Path,
    images_root: Path,
) -> None:
    """Атомарно записывает индекс, метаданные и images_root в *out_dir*."""
    index_path = out_dir / "index.faiss"
    metadata_path = out_dir / "metadata.parquet"

    tmp_index: Path | None = None
    tmp_meta: Path | None = None
    try:
        fd, tmp_str = tempfile.mkstemp(suffix=".faiss.tmp", dir=out_dir)
        tmp_index = Path(tmp_str)
        os.close(fd)
        faiss.write_index(index, str(tmp_index))

        fd, tmp_str = tempfile.mkstemp(suffix=".parquet.tmp", dir=out_dir)
        tmp_meta = Path(tmp_str)
        os.close(fd)
        df.to_parquet(tmp_meta, index=False)

        tmp_index.replace(index_path)
        tmp_index = None
        tmp_meta.replace(metadata_path)
        tmp_meta = None

        (out_dir / "images_root.txt").write_text(
            str(images_root), encoding="utf-8"
        )
    except Exception:
        if tmp_index is not None:
            tmp_index.unlink(missing_ok=True)
        if tmp_meta is not None:
            tmp_meta.unlink(missing_ok=True)
        raise


def _write_outputs_s3(
    index: faiss.IndexFlatIP,
    df: pd.DataFrame,
    s3_client: object,
    dataset_name: str,
) -> None:
    """Сериализует индекс и метаданные во временные файлы,
    затем атомарно загружает в S3."""
    from image_retrieval.s3_client import S3Client

    assert isinstance(s3_client, S3Client)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        index_tmp = tmp_dir / "index.faiss"
        meta_tmp = tmp_dir / "metadata.parquet"

        faiss.write_index(index, str(index_tmp))
        df.to_parquet(meta_tmp, index=False)

        logger.info("Загрузка индекса в S3...")
        s3_client.upload_file_atomic(
            index_tmp, s3_client.index_key(dataset_name)
        )

        logger.info("Загрузка метаданных в S3...")
        s3_client.upload_file_atomic(
            meta_tmp, s3_client.metadata_key(dataset_name)
        )


def main() -> int:
    """Точка входа; возвращает код выхода."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _build_parser().parse_args()

    model_class: type[nn.Module] | None = None
    if args.model_module:
        model_class = _resolve_model_class(args.model_module)

    input_size: tuple[int, int] = (args.input_size[0], args.input_size[1])
    embedder = TorchEmbedder(
        checkpoint_path=args.checkpoint,
        model_class=model_class,
        input_size=input_size,
        device=args.device,
    )

    if args.mode == "local":
        df = _load_annotations_local(args.annotations)

        logger.info(
            "Начало прохода эмбеддинга (batch_size=%d, device=%s)...",
            args.batch_size, args.device,
        )
        embeddings = _embed_all_local(
            df, args.images_root, embedder, args.batch_size
        )
        index = _build_faiss_index(embeddings)

        out_dir: Path = args.datasets_dir / args.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_outputs_local(index, df, out_dir, args.images_root.resolve())

        logger.info("✓ индекс    → %s", out_dir / "index.faiss")
        logger.info("✓ метаданные → %s", out_dir / "metadata.parquet")
        return 0

    from image_retrieval.config import S3Config
    from image_retrieval.s3_client import S3Client

    s3_config = S3Config(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        endpoint_url=args.endpoint_url,
    )
    s3_client = S3Client(s3_config)

    # Используем управляемую temp-директорию если постоянный кэш не указан
    if args.image_cache_dir is not None:
        cache_dir = args.image_cache_dir
        _tmp_mgr = None
    else:
        _tmp_mgr = tempfile.TemporaryDirectory()
        cache_dir = Path(_tmp_mgr.name)

    try:
        with tempfile.TemporaryDirectory() as anno_tmp:
            df = _load_annotations_s3(
                s3_client, args.dataset, Path(anno_tmp)
            )

        logger.info(
            "Начало S3-прохода эмбеддинга (batch_size=%d, device=%s)...",
            args.batch_size, args.device,
        )
        embeddings = _embed_all_s3(
            df, s3_client, embedder, args.batch_size, cache_dir
        )
        index = _build_faiss_index(embeddings)

        _write_outputs_s3(index, df, s3_client, args.dataset)

        logger.info(
            "✓ индекс    → s3://%s/%s",
            args.bucket, s3_client.index_key(args.dataset),
        )
        logger.info(
            "✓ метаданные → s3://%s/%s",
            args.bucket, s3_client.metadata_key(args.dataset),
        )
    finally:
        if _tmp_mgr is not None:
            _tmp_mgr.cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
