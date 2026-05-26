"""REST API-клиент CVAT 2.x.

Предоставляет тонкую типизированную обёртку над API CVAT v2, достаточную
для рабочего процесса экспорта из image-crop-retrieval:

1. Создать задачу с одной меткой bounding-box.
2. Загрузить исходные изображения в задачу.
3. Дождаться обработки загруженных данных в CVAT.
4. Получить автоматически созданный job ID.
5. Отправить прямоугольные аннотации (по одной на каждый выбранный кроп).

Аутентификация
--------------
Предпочтителен токен (``CVATBlock.token``).  Если указаны только
``username`` / ``password`` — клиент выполняет логин при первом использовании
и кэширует токен сессии в памяти.

Поддерживаемые версии CVAT: 2.x (протестировано с 2.4+).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import httpx

from .config import CVATBlock

logger = logging.getLogger(__name__)

# Таймауты ожидания готовности задачи CVAT
_TASK_READY_TIMEOUT = 120
_TASK_POLL_INTERVAL = 2


@dataclass(frozen=True)
class CVATAnnotation:
    """Одна прямоугольная аннотация для отправки в CVAT.

    Атрибуты:
        frame: Нулевой индекс кадра (изображения) внутри задачи.  Должен
            совпадать с порядком загрузки изображений.
        xtl: X-координата верхнего левого угла (пикселей).
        ytl: Y-координата верхнего левого угла (пикселей).
        xbr: X-координата нижнего правого угла (пикселей).
        ybr: Y-координата нижнего правого угла (пикселей).
        label_id: Внутренний ID метки CVAT, полученный из созданной задачи.
    """

    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    label_id: int


@dataclass
class ExportResult:
    """Сводка, возвращаемая :meth:`CVATClient.export_to_task`.

    Атрибуты:
        task_id: Идентификатор задачи CVAT.
        task_url: Прямая ссылка на задачу в веб-интерфейсе CVAT.
        image_count: Количество загруженных изображений.
        annotation_count: Количество созданных аннотаций.
    """

    task_id: int
    task_url: str
    image_count: int
    annotation_count: int


class CVATClient:
    """Тонкая обёртка над REST API CVAT 2.x.

    Args:
        config: Параметры подключения и аутентификации CVAT.
    """

    def __init__(self, config: CVATBlock) -> None:
        self._config = config
        self._token: str | None = config.token
        self._http = httpx.Client(
            base_url=config.url,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )

    def export_to_task(
        self,
        task_name: str,
        images: list[tuple[str, bytes]],
        annotations: list[tuple[str, int, int, int, int]],
    ) -> ExportResult:
        """Создаёт задачу CVAT, загружает изображения и отправляет аннотации.

        Args:
            task_name: Человекочитаемое имя новой задачи CVAT.
            images: Список пар ``(имя_файла, байты_изображения)``.  Порядок
                определяет индексы кадров в *annotations*.
            annotations: Список кортежей ``(имя_файла, x1, y1, x2, y2)``.
                *имя_файла* должно совпадать с одним из имён в *images*.
                Координаты — в пикселях относительно исходного изображения.

        Returns:
            :class:`ExportResult` с деталями задачи.

        Raises:
            httpx.HTTPStatusError: При любом не-2xx ответе CVAT.
            TimeoutError: Если CVAT не завершил обработку за
                :data:`_TASK_READY_TIMEOUT` секунд.
        """
        self._ensure_auth()

        task_id, label_id = self._create_task(task_name)
        logger.info("Создана задача CVAT id=%d name='%s'", task_id, task_name)

        self._upload_images(task_id, images)
        logger.info(
            "Загружено %d изображений в задачу %d", len(images), task_id
        )

        job_id = self._wait_for_job(task_id)
        logger.info("Задача %d готова, job_id=%d", task_id, job_id)

        frame_map = {name: idx for idx, (name, _) in enumerate(images)}

        cvat_annotations = [
            CVATAnnotation(
                frame=frame_map[filename],
                xtl=float(x1),
                ytl=float(y1),
                xbr=float(x2),
                ybr=float(y2),
                label_id=label_id,
            )
            for filename, x1, y1, x2, y2 in annotations
            if filename in frame_map
        ]
        if cvat_annotations:
            self._push_annotations(job_id, cvat_annotations)

        task_url = f"{self._config.url}/tasks/{task_id}"
        return ExportResult(
            task_id=task_id,
            task_url=task_url,
            image_count=len(images),
            annotation_count=len(cvat_annotations),
        )

    def _ensure_auth(self) -> None:
        """Гарантирует установку ``self._token``; выполняет логин при необходимости."""
        if self._token:
            return
        if not self._config.username or not self._config.password:
            raise ValueError(
                "Аутентификация CVAT требует 'token' или "
                "'username' + 'password' в CVATBlock."
            )
        resp = self._http.post(
            "/api/auth/login",
            json={
                "username": self._config.username,
                "password": self._config.password,
            },
        )
        resp.raise_for_status()
        self._token = resp.json()["key"]
        logger.debug(
            "CVAT: вход выполнен для пользователя '%s'.", self._config.username
        )

    def _auth_headers(self) -> dict[str, str]:
        """Возвращает HTTP-заголовки с токеном аутентификации."""
        if not self._token:
            raise RuntimeError(
                "Не аутентифицирован — сначала вызовите _ensure_auth()."
            )
        return {"Authorization": f"Token {self._token}"}

    def _create_task(self, name: str) -> tuple[int, int]:
        """Создаёт задачу CVAT и возвращает ``(task_id, label_id)``.

        Задача создаётся с одной меткой :attr:`CVATBlock.task_label`.
        Если задан :attr:`CVATBlock.project_id` — задача прикрепляется к проекту.
        """
        payload: dict[str, Any] = {
            "name": name,
            "labels": [{"name": self._config.task_label, "color": "#ff8c00"}],
        }
        if self._config.project_id is not None:
            payload["project_id"] = self._config.project_id

        resp = self._http.post(
            "/api/tasks",
            json=payload,
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        task_id: int = data["id"]

        # Ищем ID метки в ответе
        labels: list[dict[str, Any]] = data.get("labels", [])
        label_id: int = labels[0]["id"] if labels else 0

        return task_id, label_id

    def _upload_images(
        self, task_id: int, images: list[tuple[str, bytes]]
    ) -> None:
        """Загружает *images* в data-endpoint задачи *task_id*.

        CVAT ожидает multipart/form-data запрос с одним или несколькими полями
        ``client_files[]``.  ``image_quality`` управляет JPEG-рекомпрессией
        (100 = без потерь).
        """
        files = [
            ("client_files[]", (name, io.BytesIO(data), "image/jpeg"))
            for name, data in images
        ]
        # httpx автоматически обрабатывает Content-Type boundary для multipart
        resp = self._http.post(
            f"/api/tasks/{task_id}/data",
            data={"image_quality": "95"},
            files=files,
            headers=self._auth_headers(),
            timeout=120.0,  # большая загрузка может занять время
        )
        resp.raise_for_status()

    def _wait_for_job(self, task_id: int) -> int:
        """Ожидает обработки данных задачи и возвращает job ID.

        CVAT обрабатывает загруженные изображения асинхронно.  Метод опрашивает
        статус задачи до ``state == "completed"`` или до истечения таймаута
        :data:`_TASK_READY_TIMEOUT`.

        Returns:
            ID первого job'а, связанного с *task_id*.

        Raises:
            TimeoutError: Если задача не готова в течение таймаута.
        """
        deadline = time.monotonic() + _TASK_READY_TIMEOUT
        while time.monotonic() < deadline:
            resp = self._http.get(
                f"/api/tasks/{task_id}/status",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            state: str = resp.json().get("state", "")
            if state.lower() in {"completed", "finished"}:
                break
            logger.debug(
                "Задача %d статус: %s — ожидание...", task_id, state
            )
            time.sleep(_TASK_POLL_INTERVAL)
        else:
            raise TimeoutError(
                f"Задача CVAT {task_id} не стала готова за "
                f"{_TASK_READY_TIMEOUT} секунд."
            )

        # Получаем список jobs для извлечения job ID
        resp = self._http.get(
            "/api/jobs",
            params={"task_id": task_id},
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        jobs: list[dict[str, Any]] = resp.json().get("results", [])
        if not jobs:
            raise RuntimeError(
                f"Для задачи CVAT {task_id} не найдено ни одного job'а."
            )
        job_id: int = jobs[0]["id"]
        return job_id

    def _push_annotations(
        self, job_id: int, annotations: list[CVATAnnotation]
    ) -> None:
        """Отправляет прямоугольные аннотации в *job_id*.

        Использует ``PATCH /api/jobs/{id}/annotations?action=create``.
        """
        shapes = [
            {
                "type": "rectangle",
                "label_id": ann.label_id,
                "frame": ann.frame,
                "points": [ann.xtl, ann.ytl, ann.xbr, ann.ybr],
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0.0,
                "attributes": [],
                "source": "manual",
            }
            for ann in annotations
        ]
        resp = self._http.patch(
            f"/api/jobs/{job_id}/annotations",
            params={"action": "create"},
            json={"shapes": shapes, "tags": [], "tracks": []},
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        logger.info(
            "Отправлено %d аннотаций в job CVAT %d.", len(shapes), job_id
        )


@dataclass
class PreparedExport:
    """Подготовленные данные экспорта, сгруппированные по исходному изображению.

    Строится :func:`prepare_export` из выбранных пользователем
    :class:`~image_retrieval.indexer.SearchResult`.

    Атрибуты:
        task_name: Предлагаемое имя задачи CVAT.
        images: Пары ``(имя_файла, байты_изображения)`` в порядке загрузки.
        annotations: ``(имя_файла, x1, y1, x2, y2)`` для каждого выбранного кропа.
    """

    task_name: str
    images: list[tuple[str, bytes]] = field(default_factory=list)
    annotations: list[tuple[str, int, int, int, int]] = field(default_factory=list)


def prepare_export(
    results: list[Any],
    load_image_bytes: Any,  # Callable[[str], bytes]
    task_name: str = "",
) -> PreparedExport:
    """Строит :class:`PreparedExport` из выбранных результатов поиска.

    Загружает каждое уникальное исходное изображение ровно один раз и
    собирает bounding-box аннотации.

    Args:
        results: Выбранные экземпляры :class:`~image_retrieval.indexer.SearchResult`.
        load_image_bytes: ``Callable[[image_path], bytes]``, возвращающий
            сырые байты изображения по заданному пути (локальному или S3).
        task_name: Необязательное имя задачи.  Если пусто — генерируется
            автоматически из временной метки.

    Returns:
        :class:`PreparedExport`, готовый для передачи в
        :meth:`CVATClient.export_to_task`.
    """
    from collections.abc import Callable

    assert callable(load_image_bytes)
    loader: Callable[[str], bytes] = load_image_bytes

    if not task_name:
        task_name = f"image-crop-retrieval-{time.strftime('%Y%m%d-%H%M%S')}"

    seen_paths: dict[str, str] = {}  # image_path → имя файла в задаче
    images: list[tuple[str, bytes]] = []
    annotations: list[tuple[str, int, int, int, int]] = []

    for result in results:
        img_path: str = result.image_path
        if img_path not in seen_paths:
            filename = (
                PurePosixPath(img_path).name or f"image_{len(seen_paths)}.jpg"
            )
            # Обеспечиваем уникальность имён в задаче
            if any(name == filename for name, _ in images):
                stem = PurePosixPath(filename).stem
                ext = PurePosixPath(filename).suffix
                filename = f"{stem}_{len(seen_paths)}{ext}"
            seen_paths[img_path] = filename
            img_bytes = loader(img_path)
            images.append((filename, img_bytes))

        filename = seen_paths[img_path]
        annotations.append(
            (filename, result.x1, result.y1, result.x2, result.y2)
        )

    return PreparedExport(
        task_name=task_name,
        images=images,
        annotations=annotations,
    )
