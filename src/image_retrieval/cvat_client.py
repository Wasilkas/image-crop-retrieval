"""CVAT 2.x REST API client.

Provides a thin, typed wrapper around the CVAT v2 API sufficient for the
image-crop-retrieval export workflow:

1. Create a task with a single bounding-box label.
2. Upload source images to the task.
3. Wait until CVAT has processed the uploaded data.
4. Fetch the auto-created job ID.
5. Push rectangle annotations (one per selected crop).

Authentication
--------------
Token-based authentication is preferred (``CVATBlock.token``).  If only
``username`` / ``password`` are supplied the client performs the login flow
on first use and caches the session token in memory.

Supported CVAT versions: 2.x (tested against 2.4+).
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

# How long (seconds) to wait for CVAT to process uploaded data.
_TASK_READY_TIMEOUT = 120
_TASK_POLL_INTERVAL = 2


@dataclass(frozen=True)
class CVATAnnotation:
    """One rectangle annotation to be pushed to CVAT.

    Attributes:
        frame: Zero-based index of the frame (image) within the task. Must
            match the order in which images were uploaded.
        xtl: X-coordinate of the top-left corner (pixels).
        ytl: Y-coordinate of the top-left corner (pixels).
        xbr: X-coordinate of the bottom-right corner (pixels).
        ybr: Y-coordinate of the bottom-right corner (pixels).
        label_id: CVAT internal label ID, obtained from the created task.
    """

    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    label_id: int


@dataclass
class ExportResult:
    """Summary returned by :meth:`CVATClient.export_to_task`.

    Attributes:
        task_id: CVAT task identifier.
        task_url: Direct URL to the task in the CVAT web UI.
        image_count: Number of images uploaded.
        annotation_count: Number of annotations created.
    """

    task_id: int
    task_url: str
    image_count: int
    annotation_count: int


class CVATClient:
    """Thin wrapper around the CVAT 2.x REST API.

    Args:
        config: CVAT connection and authentication parameters.
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
        """Create a CVAT task, upload images, and push bounding-box annotations.

        Args:
            task_name: Human-readable name for the new CVAT task.
            images: List of ``(filename, image_bytes)`` pairs.  The order
                determines frame indices used in *annotations*.
            annotations: List of ``(filename, x1, y1, x2, y2)`` tuples.
                *filename* must match one of the names in *images*.
                Coordinates are in pixels relative to the source image.

        Returns:
            :class:`ExportResult` with task details.

        Raises:
            httpx.HTTPStatusError: On any non-2xx CVAT response.
            TimeoutError: If CVAT does not finish processing within
                :data:`_TASK_READY_TIMEOUT` seconds.
        """
        self._ensure_auth()

        task_id, label_id = self._create_task(task_name)
        logger.info("Created CVAT task id=%d name='%s'", task_id, task_name)

        self._upload_images(task_id, images)
        logger.info("Uploaded %d images to task %d", len(images), task_id)

        job_id = self._wait_for_job(task_id)
        logger.info("Task %d is ready, job_id=%d", task_id, job_id)

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
        """Guarantee ``self._token`` is set; perform login if necessary."""
        if self._token:
            return
        if not self._config.username or not self._config.password:
            raise ValueError(
                "CVAT authentication requires either 'token' or "
                "'username' + 'password' in CVATBlock."
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
        logger.debug("CVAT login successful for user '%s'.", self._config.username)

    def _auth_headers(self) -> dict[str, str]:
        """Return HTTP headers carrying the authentication token."""
        if not self._token:
            raise RuntimeError("Not authenticated — call _ensure_auth() first.")
        return {"Authorization": f"Token {self._token}"}

    def _create_task(self, name: str) -> tuple[int, int]:
        """Create a CVAT task and return ``(task_id, label_id)``.

        The task is created with a single label named
        :attr:`CVATBlock.task_label`.  If :attr:`CVATBlock.project_id` is
        set, the task is attached to that project.
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

        # Locate label ID in the response
        labels: list[dict[str, Any]] = data.get("labels", [])
        label_id: int = labels[0]["id"] if labels else 0

        return task_id, label_id

    def _upload_images(
        self, task_id: int, images: list[tuple[str, bytes]]
    ) -> None:
        """Upload *images* to the data endpoint of *task_id*.

        CVAT expects a multipart/form-data request with one or more
        ``client_files[]`` fields.  ``image_quality`` controls JPEG
        re-compression (100 = lossless).
        """
        files = [
            ("client_files[]", (name, io.BytesIO(data), "image/jpeg"))
            for name, data in images
        ]
        # httpx handles Content-Type boundary automatically for multipart
        resp = self._http.post(
            f"/api/tasks/{task_id}/data",
            data={"image_quality": "95"},
            files=files,
            headers=self._auth_headers(),
            timeout=120.0,  # large upload may take time
        )
        resp.raise_for_status()

    def _wait_for_job(self, task_id: int) -> int:
        """Poll until the task data is processed and return the job ID.

        CVAT processes uploaded images asynchronously.  This method polls
        the task status until ``state == "completed"`` or the timeout
        :data:`_TASK_READY_TIMEOUT` is exceeded.

        Returns:
            ID of the first job associated with *task_id*.

        Raises:
            TimeoutError: If the task is not ready within the timeout.
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
            logger.debug("Task %d status: %s — waiting…", task_id, state)
            time.sleep(_TASK_POLL_INTERVAL)
        else:
            raise TimeoutError(
                f"CVAT task {task_id} was not ready within "
                f"{_TASK_READY_TIMEOUT}s."
            )

        resp = self._http.get(
            "/api/jobs",
            params={"task_id": task_id},
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        jobs: list[dict[str, Any]] = resp.json().get("results", [])
        if not jobs:
            raise RuntimeError(f"No jobs found for CVAT task {task_id}.")
        job_id: int = jobs[0]["id"]
        return job_id

    def _push_annotations(
        self, job_id: int, annotations: list[CVATAnnotation]
    ) -> None:
        """Push rectangle annotations to *job_id*.

        Uses ``PATCH /api/jobs/{id}/annotations?action=create``.
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
            "Pushed %d annotations to CVAT job %d.", len(shapes), job_id
        )


@dataclass
class PreparedExport:
    """Prepared export data grouped by source image.

    Built by :func:`prepare_export` from the user's selected
    :class:`~image_retrieval.indexer.SearchResult` list.

    Attributes:
        task_name: Suggested CVAT task name.
        images: ``(filename, image_bytes)`` pairs in upload order.
        annotations: ``(filename, x1, y1, x2, y2)`` for each selected crop.
    """

    task_name: str
    images: list[tuple[str, bytes]] = field(default_factory=list)
    annotations: list[tuple[str, int, int, int, int]] = field(default_factory=list)


def prepare_export(
    results: list[Any],
    load_image_bytes: Any,  # Callable[[str], bytes]
    task_name: str = "",
) -> PreparedExport:
    """Build a :class:`PreparedExport` from selected search results.

    Loads each unique source image exactly once and collects bounding-box
    annotations.

    Args:
        results: Selected :class:`~image_retrieval.indexer.SearchResult`
            instances.
        load_image_bytes: ``Callable[[image_path], bytes]`` that returns the
            raw image bytes for a given path (local or S3).
        task_name: Optional task name override.  Auto-generated from timestamp
            if empty.

    Returns:
        A :class:`PreparedExport` ready to pass to
        :meth:`CVATClient.export_to_task`.
    """
    from collections.abc import Callable

    assert callable(load_image_bytes)
    loader: Callable[[str], bytes] = load_image_bytes

    if not task_name:
        task_name = f"image-crop-retrieval-{time.strftime('%Y%m%d-%H%M%S')}"

    seen_paths: dict[str, str] = {}  # image_path → filename used in task
    images: list[tuple[str, bytes]] = []
    annotations: list[tuple[str, int, int, int, int]] = []

    for result in results:
        img_path: str = result.image_path
        if img_path not in seen_paths:
            filename = PurePosixPath(img_path).name or f"image_{len(seen_paths)}.jpg"
            # Ensure uniqueness in the task image list
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
