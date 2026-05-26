"""Tests for image_retrieval.config — blocks, Configuration, load priority."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import ValidationError

from image_retrieval.config import (
    AppBlock,
    AppConfig,
    Configuration,
    CVATBlock,
    DatasetMeta,
    EncoderBlock,
    S3Block,
    S3Config,
)


class TestAppBlock:
    def test_defaults(self) -> None:
        app = AppBlock()
        assert app.top_k == 10
        assert app.device == "cpu"
        assert app.min_crop_px == 8
        assert app.canvas_width == 800
        assert app.canvas_height == 600
        assert app.results_columns == 3

    def test_datasets_dir_resolved_from_str(self, tmp_path: Path) -> None:
        app = AppBlock(datasets_dir=str(tmp_path))  # type: ignore[arg-type]
        assert app.datasets_dir == tmp_path.resolve()

    def test_datasets_dir_resolved_from_path(self, tmp_path: Path) -> None:
        app = AppBlock(datasets_dir=tmp_path)
        assert app.datasets_dir == tmp_path.resolve()

    def test_frozen(self) -> None:
        app = AppBlock()
        with pytest.raises(ValidationError):
            app.top_k = 99  # pydantic frozen raises at runtime


class TestS3Block:
    def test_required_bucket(self) -> None:
        with pytest.raises(ValidationError):
            S3Block()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        s3 = S3Block(bucket="my-bucket")
        assert s3.prefix == ""
        assert s3.region == "us-east-1"
        assert s3.endpoint_url is None
        assert s3.check_interval_seconds == 300

    def test_dataset_prefix_with_prefix(self) -> None:
        s3 = S3Block(bucket="b", prefix="datasets/")
        assert s3.dataset_prefix("mnist") == "datasets/mnist/"

    def test_dataset_prefix_without_prefix(self) -> None:
        s3 = S3Block(bucket="b", prefix="")
        assert s3.dataset_prefix("mnist") == "mnist/"

    def test_dataset_prefix_trailing_slash_stripped(self) -> None:
        s3 = S3Block(bucket="b", prefix="data")
        assert s3.dataset_prefix("cifar") == "data/cifar/"


class TestEncoderBlock:
    def test_local_path(self) -> None:
        enc = EncoderBlock(checkpoint="/models/encoder.pth")
        assert enc.checkpoint == "/models/encoder.pth"
        assert enc.model_module is None
        assert enc.cache_dir is None

    def test_s3_uri(self) -> None:
        enc = EncoderBlock(checkpoint="s3://bucket/models/encoder.pth")
        assert enc.checkpoint.startswith("s3://")

    def test_cache_dir_resolved(self, tmp_path: Path) -> None:
        enc = EncoderBlock(checkpoint="x.pth", cache_dir=str(tmp_path))  # type: ignore[arg-type]
        assert enc.cache_dir == tmp_path.resolve()

    def test_cache_dir_none(self) -> None:
        enc = EncoderBlock(checkpoint="x.pth")
        assert enc.cache_dir is None


class TestCVATBlock:
    def test_trailing_slash_stripped(self) -> None:
        cvat = CVATBlock(url="https://cvat.example.com/")
        assert cvat.url == "https://cvat.example.com"

    def test_no_slash_unchanged(self) -> None:
        cvat = CVATBlock(url="https://cvat.example.com")
        assert cvat.url == "https://cvat.example.com"

    def test_defaults(self) -> None:
        cvat = CVATBlock(url="https://cvat.example.com")
        assert cvat.token is None
        assert cvat.username is None
        assert cvat.password is None
        assert cvat.project_id is None
        assert cvat.task_label == "crop"

    def test_with_token(self) -> None:
        cvat = CVATBlock(url="https://cvat.example.com", token="tok123")
        assert cvat.token == "tok123"


class TestConfiguration:
    def test_defaults(self) -> None:
        cfg = Configuration()
        assert isinstance(cfg.app, AppBlock)
        assert cfg.s3 is None
        assert cfg.encoder is None
        assert cfg.cvat is None

    def test_from_yaml(self, tmp_path: Path) -> None:
        content: dict[str, Any] = {
            "app": {"top_k": 25, "device": "cuda"},
            "s3": {"bucket": "test-bucket", "prefix": "data/"},
        }
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml.dump(content))

        cfg = Configuration.from_yaml(yaml_file)
        assert cfg.app.top_k == 25
        assert cfg.app.device == "cuda"
        assert cfg.s3 is not None
        assert cfg.s3.bucket == "test-bucket"
        assert cfg.s3.prefix == "data/"

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = Configuration.from_yaml(yaml_file)
        assert cfg.app.top_k == 10

    def test_from_yaml_partial(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text("app:\n  top_k: 5\n")
        cfg = Configuration.from_yaml(yaml_file)
        assert cfg.app.top_k == 5
        assert cfg.s3 is None

    def test_load_defaults_no_file(self, tmp_path: Path) -> None:
        """When no Consul URL and no YAML file, returns defaults."""
        cfg = Configuration.load(config_path=tmp_path / "nonexistent.yaml")
        assert cfg.app.top_k == 10
        assert cfg.s3 is None

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("app:\n  top_k: 15\n")
        cfg = Configuration.load(config_path=yaml_file)
        assert cfg.app.top_k == 15

    def test_load_consul_fallback_on_connection_error(self, tmp_path: Path) -> None:
        """If Consul is unreachable, falls back to YAML."""
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("app:\n  top_k: 7\n")

        cfg = Configuration.load(
            config_path=yaml_file,
            consul_url="http://localhost:19999",  # nothing running here
            consul_key="test/key",
        )
        assert cfg.app.top_k == 7

    def test_load_consul_success(self, tmp_path: Path) -> None:
        """If Consul returns valid YAML, it is used (YAML file ignored)."""
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("app:\n  top_k: 1\n")

        mock_kv = MagicMock()
        mock_kv.get.return_value = (None, {"Value": b"app:\n  top_k: 99\n"})
        mock_consul = MagicMock()
        mock_consul.kv = mock_kv

        with patch("consul.Consul", return_value=mock_consul):
            cfg = Configuration.load(
                config_path=yaml_file,
                consul_url="http://consul:8500",
                consul_key="config/app",
            )
        assert cfg.app.top_k == 99

    def test_load_consul_key_missing_falls_back(self, tmp_path: Path) -> None:
        """Consul returns None (key not found) → falls back to YAML."""
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("app:\n  top_k: 3\n")

        mock_kv = MagicMock()
        mock_kv.get.return_value = (None, None)
        mock_consul = MagicMock()
        mock_consul.kv = mock_kv

        with patch("consul.Consul", return_value=mock_consul):
            cfg = Configuration.load(
                config_path=yaml_file,
                consul_url="http://consul:8500",
                consul_key="missing/key",
            )
        assert cfg.app.top_k == 3

    def test_model_copy_update(self) -> None:
        cfg = Configuration()
        updated = cfg.model_copy(update={"s3": S3Block(bucket="new-bucket")})
        assert updated.s3 is not None
        assert updated.s3.bucket == "new-bucket"
        assert cfg.s3 is None

    def test_config_path_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yaml_file = tmp_path / "env_cfg.yaml"
        yaml_file.write_text("app:\n  top_k: 42\n")
        monkeypatch.setenv("CONFIG_PATH", str(yaml_file))
        monkeypatch.delenv("CONSUL_URL", raising=False)
        cfg = Configuration.load()
        assert cfg.app.top_k == 42


class TestDatasetMeta:
    @pytest.fixture()
    def meta(self, tmp_path: Path) -> DatasetMeta:
        return DatasetMeta(
            name="test",
            index_path=tmp_path / "index.faiss",
            metadata_path=tmp_path / "metadata.parquet",
            images_root=tmp_path,
        )

    def test_creation(self, meta: DatasetMeta) -> None:
        assert meta.name == "test"

    def test_frozen(self, meta: DatasetMeta) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            meta.name = "other"  # type: ignore[misc]


def test_appconfig_alias() -> None:
    assert AppConfig is AppBlock


def test_s3config_alias() -> None:
    assert S3Config is S3Block
