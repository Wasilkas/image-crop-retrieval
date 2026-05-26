# CLAUDE.md — image-crop-retrieval

Streamlit application for SSL-based image crop retrieval: user uploads an image,
draws a bounding box, and the app finds the top-K most visually similar crops from
a pre-indexed dataset using FAISS cosine similarity search over SSL embeddings.

---

## Essential commands

```bash
# Install / sync dependencies (requires uv)
uv sync --all-groups

# Run the app
uv run streamlit run app.py

# Run all tests
uv run pytest

# Lint
uv run ruff check src/ app.py scripts/

# Auto-fix lint
uv run ruff check --fix src/ app.py scripts/

# Type-check
uv run mypy src/ app.py scripts/

# Build a dataset index (offline CLI)
uv run python scripts/build_index.py \
    --annotations /data/boxes.csv \
    --images-root /data/images/ \
    --checkpoint  /models/encoder.pth \
    --dataset-name my_dataset
```

Tests pass in about 5 seconds. Currently: **133 passed**.

---

## Project layout

```
app.py                              # Streamlit entry point
scripts/build_index.py              # Offline indexing CLI (cron job)
src/image_retrieval/
    config.py       # Pydantic v2 config blocks; Consul → YAML → defaults chain
    embedder.py     # EmbedderProtocol + TorchEmbedder (full-model & state-dict)
    indexer.py      # FAISSIndex + SearchResult dataclass
    registry.py     # DatasetRegistry (local) + S3DatasetRegistry
    s3_client.py    # boto3 wrapper: download, upload, is_remote_newer
    cvat_client.py  # CVAT 2.x REST API client + prepare_export helper
    ui/
        crop_selector.py    # Image upload + streamlit-drawable-canvas widget
        results_viewer.py   # Top-K results grid
        cvat_exporter.py    # UI widget that calls CVATClient.export_to_task
tests/
    conftest.py             # Shared fixtures: sample_embeddings, dataset_dir, …
    test_*.py               # One file per source module
```

**Dataset on-disk layout** (created by `build_index.py`, read by `registry.py`):
```
datasets/{name}/
    index.faiss          # faiss.IndexFlatIP (L2-normalised vectors)
    metadata.parquet     # image_path, x1, y1, x2, y2, box_id
    images_root.txt      # optional — one line: path to images folder
```

---

## Architecture

### Data flow

```
Upload image → drawable canvas (rect) → crop = image.crop(x1,y1,x2,y2)
    → TorchEmbedder.embed([crop])  → np.ndarray (1, D), L2-normalised
    → FAISSIndex.search(query, top_k)  → list[SearchResult]
    → render_results() grid + optional CVAT export
```

### Why IndexFlatIP + L2-normalise?

FAISS `IndexFlatIP` computes inner products. For unit-norm vectors, inner product
equals cosine similarity. `TorchEmbedder.embed()` always L2-normalises its output,
and `build_index.py` normalises vectors before adding to the index. Do not break
this invariant — scores will become meaningless.

### Configuration chain (config.py)

`Configuration.load()` tries in order:
1. **Consul KV** — if `CONSUL_URL` env var is set
2. **YAML file** — `CONFIG_PATH` env var or `config.yaml`
3. **Built-in defaults** — all fields have sensible values

After loading, `app.py:_apply_env_overrides()` applies `S3_BUCKET`, `DATASETS_DIR`,
`MODEL_PATH`, etc. on top (for secrets / Docker-compose overrides).

All config blocks are **frozen Pydantic v2 BaseModel** — treat them as immutable.
Use `model_copy(update=...)` when you need to patch a config.

### Registry hot-reload

`DatasetRegistry` tracks `mtime_ns` of `index.faiss` and `metadata.parquet`.
Every `registry.get(name)` call checks mtimes and reloads the dataset in-place if
files changed. `build_index.py` writes via **atomic rename** (tmp → final) so the
app never reads a partially-written file.

`S3DatasetRegistry` polls S3 at most once per `check_interval_seconds` (default 300 s)
using `S3Client.is_remote_newer()` + `S3Client.get_last_modified()`.

### Streamlit caching

`@st.cache_resource` is used for:
- `DatasetRegistry` / `S3DatasetRegistry` — keyed by `datasets_dir` path or S3 params
- `S3Client` — keyed by bucket + prefix + region + endpoint
- `TorchEmbedder` — keyed by `(model_path, device)`

These objects are created once per process. Never call `st.cache_resource`-decorated
functions inside a loop — they are singletons by design.

---

## Key conventions

### Types
- All public functions and methods have full type annotations.
- `mypy --strict` must pass. Override only for third-party libs without stubs
  (see `[[tool.mypy.overrides]]` in `pyproject.toml`).
- `ANN401` (typing.Any) is suppressed globally — `Any` is unavoidable for
  `streamlit_drawable_canvas` results and some Pydantic validators.

### Linting
- ruff: line length 88, rules `E F I N UP ANN B SIM`.
- No `print()` in `src/` or `app.py` — use `logging.getLogger(__name__)`.
- `scripts/` may use `print()` (T201 is suppressed for scripts).

### Error handling
- **Hard errors** (missing checkpoint, bad format) → raise exception → `st.error()` + `st.stop()` in `app.py`.
- **Soft errors** (one dataset fails to load) → `logger.exception()`, skip dataset, app continues.
- **UI validation** (crop too small, no dataset selected) → `st.warning()` / `st.error()`, `return None`.

### Tests
- `conftest.py` provides shared fixtures: `sample_embeddings`, `sample_meta_df`,
  `dataset_dir`, `small_image`, `small_image_bytes`.
- `DIM = 64` — small embedding dimension used across all test fixtures.
- S3 tests use `@mock_aws` (moto). CVAT tests use `respx` (httpx transport mock).
- Each test class owns an `out_dir` fixture via `@pytest.fixture()` if it needs
  a temp output directory (pattern in `TestWriteOutputsLocal`).

### Atomicity
- Local files: write to `*.tmp`, then `Path.rename()`.
- S3 files: upload to `key.tmp`, then `copy_object` + `delete_object`
  (`S3Client.upload_file_atomic()`).
- Never write directly to the final path — the registry may be reading it concurrently.

---

## Adding a new feature

**New config field** → add to the relevant block in `config.py`; add a test in
`test_config.py` covering YAML round-trip.

**New dataset mode** → implement the same interface as `DatasetRegistry`
(`available()`, `get()`, `rescan()`, `last_reload_info()`); add to `_AnyRegistry`
union type in `app.py`.

**New UI panel** → add a function in `src/image_retrieval/ui/`, call it from
`app.py:main()` after the search results block.

**New CLI subcommand** → add a subparser in `scripts/build_index.py:_build_parser()`
and a `_cmd_*` handler function following the existing pattern.

---

## Checkpoint formats

| Mode | How to save | How to use |
|------|-------------|-----------|
| Full-model pickle | `torch.save(model, path)` | Just `--checkpoint path` |
| State-dict | `torch.save(model.state_dict(), path)` | `--checkpoint path --model-module pkg.mod:ClassName` |

`TorchEmbedder` infers the mode from whether `model_class` is provided.
The model must accept `(N, 3, H, W)` float tensors and return `(N, D)` or `(N, C, H, W)`.
Spatial outputs are auto-flattened. ImageNet normalisation is applied automatically.

> **Security**: `torch.load(weights_only=False)` executes arbitrary pickle code.
> Only load checkpoints from trusted sources.

---

## S3 / MinIO setup

Set these env vars or add an `s3:` block to `config.yaml`:

```bash
S3_BUCKET=my-bucket
S3_PREFIX=datasets/          # optional prefix
S3_REGION=us-east-1
S3_ENDPOINT_URL=http://localhost:9000   # MinIO / Yandex Cloud
```

Build S3 index:
```bash
uv run python scripts/build_index.py s3 \
    --dataset-name my_dataset \
    --bucket my-bucket \
    --prefix datasets/ \
    --split-key datasets/my_dataset/split_2024-01-15.csv \
    --checkpoint /models/encoder.pth
```

## CVAT export

Add a `cvat:` block to `config.yaml` (or `config.yaml.example` for reference):

```yaml
cvat:
  url: https://cvat.example.com
  token: my-api-token      # preferred over username/password
  project_id: 42           # optional
  task_label: crop         # default
```

When configured, a CVAT export panel appears below search results.
