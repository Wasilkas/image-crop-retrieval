"""Tests for image_retrieval.s3_client using moto S3 mocks."""

from __future__ import annotations

import io
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from PIL import Image as PILImage

from image_retrieval.config import S3Block
from image_retrieval.s3_client import S3Client, _is_not_found

BUCKET = "test-bucket"
REGION = "us-east-1"


@pytest.fixture()
def s3_block() -> S3Block:
    return S3Block(bucket=BUCKET, prefix="datasets/", region=REGION)


def _make_bucket() -> None:
    """Create the test bucket inside an active @mock_aws context."""
    boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)


def _make_jpeg_bytes(color: tuple[int, int, int] = (100, 150, 200)) -> bytes:
    img = PILImage.new("RGB", (16, 16), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@mock_aws
def test_list_dataset_names_empty(s3_block: S3Block) -> None:
    _make_bucket()
    assert S3Client(s3_block).list_dataset_names() == []


@mock_aws
def test_list_dataset_names_multiple(s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    for name in ["alpha", "beta", "gamma"]:
        s3.put_object(Bucket=BUCKET, Key=f"datasets/{name}/index.faiss", Body=b"data")

    assert S3Client(s3_block).list_dataset_names() == ["alpha", "beta", "gamma"]


@mock_aws
def test_list_dataset_names_respects_prefix() -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="other/ds1/file.txt", Body=b"x")
    s3.put_object(Bucket=BUCKET, Key="datasets/ds2/file.txt", Body=b"x")

    client = S3Client(S3Block(bucket=BUCKET, prefix="datasets/"))
    assert client.list_dataset_names() == ["ds2"]


@mock_aws
def test_find_latest_split_key_returns_most_recent(s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    for date_str in ["2024-01-01", "2024-03-15", "2024-02-10"]:
        s3.put_object(
            Bucket=BUCKET, Key=f"datasets/myds/split_{date_str}.csv", Body=b"data"
        )

    key = S3Client(s3_block).find_latest_split_key("myds")
    assert key is not None
    assert "2024-03-15" in key


@mock_aws
def test_find_latest_split_key_none_when_absent(s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="datasets/myds/annotations.csv", Body=b"x")

    assert S3Client(s3_block).find_latest_split_key("myds") is None


@mock_aws
def test_download_file(tmp_path: Path, s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="datasets/ds/index.faiss", Body=b"faiss_data")

    dest = tmp_path / "out" / "index.faiss"
    S3Client(s3_block).download_file("datasets/ds/index.faiss", dest)
    assert dest.exists()
    assert dest.read_bytes() == b"faiss_data"


@mock_aws
def test_upload_file(tmp_path: Path, s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)

    src = tmp_path / "data.bin"
    src.write_bytes(b"hello world")
    S3Client(s3_block).upload_file(src, "datasets/ds/data.bin")

    resp = s3.get_object(Bucket=BUCKET, Key="datasets/ds/data.bin")
    assert resp["Body"].read() == b"hello world"


@mock_aws
def test_upload_file_atomic(tmp_path: Path, s3_block: S3Block) -> None:
    """Atomic upload: final key exists, .tmp key is deleted."""
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)

    src = tmp_path / "payload.bin"
    src.write_bytes(b"atomic")
    S3Client(s3_block).upload_file_atomic(src, "datasets/ds/index.faiss")

    body = s3.get_object(Bucket=BUCKET, Key="datasets/ds/index.faiss")["Body"].read()
    assert body == b"atomic"
    listed = s3.list_objects_v2(Bucket=BUCKET, Prefix="datasets/ds/index.faiss.tmp")
    assert listed.get("KeyCount", 0) == 0


@mock_aws
def test_download_uri_s3_scheme(tmp_path: Path, s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="models/enc.pth", Body=b"weights")

    dest = tmp_path / "enc.pth"
    S3Client(s3_block).download_uri(f"s3://{BUCKET}/models/enc.pth", dest)
    assert dest.read_bytes() == b"weights"


@mock_aws
def test_download_uri_bare_key(tmp_path: Path, s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="datasets/ds/metadata.parquet", Body=b"parquet")

    dest = tmp_path / "metadata.parquet"
    S3Client(s3_block).download_uri("datasets/ds/metadata.parquet", dest)
    assert dest.read_bytes() == b"parquet"


@mock_aws
def test_load_image_returns_pil_image(s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="datasets/ds/img.jpg", Body=_make_jpeg_bytes())

    img = S3Client(s3_block).load_image(f"s3://{BUCKET}/datasets/ds/img.jpg")
    assert isinstance(img, PILImage.Image)
    assert img.mode == "RGB"


@mock_aws
def test_load_image_bare_key(s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="datasets/ds/img.jpg", Body=_make_jpeg_bytes())

    img = S3Client(s3_block).load_image("datasets/ds/img.jpg")
    assert isinstance(img, PILImage.Image)


@mock_aws
def test_get_last_modified_returns_float(s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="datasets/ds/f.bin", Body=b"x")

    ts = S3Client(s3_block).get_last_modified("datasets/ds/f.bin")
    assert ts is not None
    assert isinstance(ts, float)
    assert ts > 0


@mock_aws
def test_get_last_modified_missing_key(s3_block: S3Block) -> None:
    boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
    assert S3Client(s3_block).get_last_modified("nonexistent/key") is None


@mock_aws
def test_is_remote_newer_when_local_missing(tmp_path: Path, s3_block: S3Block) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key="datasets/ds/idx.faiss", Body=b"x")

    assert S3Client(s3_block).is_remote_newer(
        "datasets/ds/idx.faiss", tmp_path / "nope.faiss"
    )


@mock_aws
def test_is_remote_newer_s3_missing_returns_false(
    tmp_path: Path, s3_block: S3Block
) -> None:
    boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
    local_file = tmp_path / "local.faiss"
    local_file.write_bytes(b"local")
    assert not S3Client(s3_block).is_remote_newer("nonexistent/key", local_file)


def test_index_key(s3_block: S3Block) -> None:
    assert S3Client(s3_block).index_key("myds") == "datasets/myds/index.faiss"


def test_metadata_key(s3_block: S3Block) -> None:
    assert S3Client(s3_block).metadata_key("myds") == "datasets/myds/metadata.parquet"


def _exc_with_code(code: str) -> Exception:
    exc = Exception("oops")
    exc.response = {"Error": {"Code": code}}  # type: ignore[attr-defined]
    return exc


def test_is_not_found_true_for_404() -> None:
    assert _is_not_found(_exc_with_code("404")) is True


def test_is_not_found_true_for_no_such_key() -> None:
    assert _is_not_found(_exc_with_code("NoSuchKey")) is True


def test_is_not_found_false_for_other_error() -> None:
    assert _is_not_found(_exc_with_code("403")) is False


def test_is_not_found_false_no_response_attr() -> None:
    assert _is_not_found(ValueError("plain")) is False
