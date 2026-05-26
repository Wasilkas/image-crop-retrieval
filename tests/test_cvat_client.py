"""Tests for image_retrieval.cvat_client using respx HTTP mocks."""

from __future__ import annotations

import dataclasses
import io
from unittest.mock import patch

import httpx
import pytest
import respx

from image_retrieval.config import CVATBlock
from image_retrieval.cvat_client import (
    CVATAnnotation,
    CVATClient,
    ExportResult,
    prepare_export,
)
from image_retrieval.indexer import SearchResult

CVAT_URL = "http://cvat.test"


@pytest.fixture()
def cvat_cfg() -> CVATBlock:
    return CVATBlock(url=CVAT_URL, token="test-token")


@pytest.fixture()
def cvat_cfg_userpass() -> CVATBlock:
    return CVATBlock(url=CVAT_URL, username="user", password="pass")


def _make_jpeg_bytes() -> bytes:
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (4, 4), color=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_export_routes(
    task_id: int, job_id: int, label_id: int = 1
) -> respx.Route:
    """Register all CVAT API mocks for a successful export flow.

    Returns the annotations PATCH route so callers can inspect whether it was called.
    Must be called within an active ``@respx.mock`` context.
    """
    respx.post(f"{CVAT_URL}/api/tasks").mock(
        return_value=httpx.Response(
            201,
            json={"id": task_id, "labels": [{"id": label_id, "name": "crop"}]},
        )
    )
    respx.post(f"{CVAT_URL}/api/tasks/{task_id}/data").mock(
        return_value=httpx.Response(202)
    )
    respx.get(f"{CVAT_URL}/api/tasks/{task_id}/status").mock(
        return_value=httpx.Response(200, json={"state": "completed"})
    )
    respx.get(f"{CVAT_URL}/api/jobs").mock(
        return_value=httpx.Response(200, json={"results": [{"id": job_id}]})
    )
    return respx.patch(f"{CVAT_URL}/api/jobs/{job_id}/annotations").mock(
        return_value=httpx.Response(200)
    )


@respx.mock
def test_export_uses_token_auth(cvat_cfg: CVATBlock) -> None:
    """Token is set in CVATBlock — no login call expected."""
    _mock_export_routes(task_id=42, job_id=7)
    result = CVATClient(cvat_cfg).export_to_task(
        task_name="test-task",
        images=[("img.jpg", _make_jpeg_bytes())],
        annotations=[("img.jpg", 0, 0, 10, 10)],
    )
    assert result.task_id == 42
    assert result.image_count == 1
    assert result.annotation_count == 1


@respx.mock
def test_login_flow_on_first_call(cvat_cfg_userpass: CVATBlock) -> None:
    """When only username+password are given, POST /api/auth/login is called."""
    task_id, job_id = 1, 2
    respx.post(f"{CVAT_URL}/api/auth/login").mock(
        return_value=httpx.Response(200, json={"key": "session-token"})
    )
    _mock_export_routes(task_id=task_id, job_id=job_id, label_id=3)

    result = CVATClient(cvat_cfg_userpass).export_to_task(
        task_name="login-test",
        images=[("img.jpg", _make_jpeg_bytes())],
        annotations=[("img.jpg", 0, 0, 5, 5)],
    )
    assert result.task_id == task_id


def test_ensure_auth_raises_without_credentials() -> None:
    """No token and no username/password → ValueError."""
    client = CVATClient(CVATBlock(url=CVAT_URL))
    with pytest.raises(ValueError, match="authentication"):
        client._ensure_auth()


@respx.mock
def test_wait_for_job_timeout(cvat_cfg: CVATBlock) -> None:
    """Task never reaches 'completed' → TimeoutError."""
    task_id = 99
    respx.get(f"{CVAT_URL}/api/tasks/{task_id}/status").mock(
        return_value=httpx.Response(200, json={"state": "processing"})
    )

    with (
        patch("image_retrieval.cvat_client._TASK_READY_TIMEOUT", 0.1),
        patch("image_retrieval.cvat_client._TASK_POLL_INTERVAL", 0.05),
        pytest.raises(TimeoutError, match=str(task_id)),
    ):
        CVATClient(cvat_cfg)._wait_for_job(task_id)


@respx.mock
def test_wait_for_job_raises_when_no_jobs(cvat_cfg: CVATBlock) -> None:
    """Task completes but no jobs returned → RuntimeError."""
    task_id = 55
    respx.get(f"{CVAT_URL}/api/tasks/{task_id}/status").mock(
        return_value=httpx.Response(200, json={"state": "completed"})
    )
    respx.get(f"{CVAT_URL}/api/jobs").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    with pytest.raises(RuntimeError, match="No jobs"):
        CVATClient(cvat_cfg)._wait_for_job(task_id)


@respx.mock
def test_export_skips_unknown_filename(cvat_cfg: CVATBlock) -> None:
    """Annotation for a filename not in the images list is silently skipped."""
    push_route = _mock_export_routes(task_id=10, job_id=11)

    result = CVATClient(cvat_cfg).export_to_task(
        task_name="skip-test",
        images=[("real.jpg", _make_jpeg_bytes())],
        annotations=[("unknown.jpg", 0, 0, 5, 5)],  # filename not in images
    )
    assert result.annotation_count == 0
    assert not push_route.called


def test_export_result_fields() -> None:
    er = ExportResult(
        task_id=1,
        task_url="http://cvat.test/tasks/1",
        image_count=3,
        annotation_count=5,
    )
    assert er.task_id == 1
    assert er.image_count == 3
    assert er.annotation_count == 5


def test_cvat_annotation_frozen() -> None:
    ann = CVATAnnotation(frame=0, xtl=1.0, ytl=2.0, xbr=10.0, ybr=20.0, label_id=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ann.frame = 99  # type: ignore[misc]


def _make_result(
    box_id: str,
    image_path: str,
    x1: int = 0,
    y1: int = 0,
    x2: int = 50,
    y2: int = 50,
) -> SearchResult:
    return SearchResult(
        box_id=box_id, image_path=image_path, x1=x1, y1=y1, x2=x2, y2=y2, score=0.9
    )


class TestPrepareExport:
    def _loader(self) -> tuple[list[str], object]:
        """Return (call_log, loader_fn) pair for tracking image loads."""
        calls: list[str] = []

        def loader(path: str) -> bytes:
            calls.append(path)
            return _make_jpeg_bytes()

        return calls, loader

    def test_basic(self) -> None:
        calls, loader = self._loader()
        results = [_make_result("b1", "img_a.jpg"), _make_result("b2", "img_b.jpg")]
        export = prepare_export(results, loader, task_name="test-task")
        assert export.task_name == "test-task"
        assert len(export.images) == 2
        assert len(export.annotations) == 2
        assert len(calls) == 2

    def test_deduplicates_source_images(self) -> None:
        calls, loader = self._loader()
        results = [
            _make_result("b1", "shared.jpg", x1=0, x2=20),
            _make_result("b2", "shared.jpg", x1=30, x2=60),
        ]
        export = prepare_export(results, loader)
        assert len(export.images) == 1
        assert len(export.annotations) == 2
        assert len(calls) == 1  # loaded once

    def test_auto_generates_task_name(self) -> None:
        _, loader = self._loader()
        export = prepare_export([_make_result("b", "img.jpg")], loader)
        assert "image-crop-retrieval-" in export.task_name

    def test_filename_deduplication(self) -> None:
        """Two different paths with the same basename get unique filenames."""
        _, loader = self._loader()
        results = [
            _make_result("b1", "/dir_a/photo.jpg"),
            _make_result("b2", "/dir_b/photo.jpg"),
        ]
        export = prepare_export(results, loader)
        assert len(export.images) == 2
        filenames = [name for name, _ in export.images]
        assert len(set(filenames)) == 2

    def test_empty_results_gives_empty_export(self) -> None:
        _, loader = self._loader()
        export = prepare_export([], loader)
        assert export.images == []
        assert export.annotations == []
