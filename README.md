# image-crop-retrieval

Streamlit application for **SSL-based image crop retrieval**.

Upload an image, draw a bounding box, and the app finds the top-K most visually
similar bounding boxes from a pre-indexed dataset using cosine similarity search
over SSL embeddings.

---

## Quick start

### 1. Create the virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv sync --all-groups
```

### 2. Build a dataset index (one-time / cron job)

Your annotations file must be a CSV or Parquet with at least these columns:

| column | description |
|--------|-------------|
| `image_path` | Path to the image **relative to `--images-root`** |
| `x1`, `y1` | Top-left corner of the bounding box (pixels) |
| `x2`, `y2` | Bottom-right corner of the bounding box (pixels) |

```bash
uv run python scripts/build_index.py \
    --annotations /data/boxes.csv \
    --images-root /data/images/ \
    --checkpoint  /models/encoder.pth \
    --dataset-name my_dataset
```

For **state-dict** checkpoints (requires the model class to be importable):

```bash
uv run python scripts/build_index.py \
    --annotations /data/boxes.parquet \
    --images-root /data/images/ \
    --checkpoint  /models/weights.pt \
    --model-module mypackage.models:MyEncoder \
    --dataset-name my_dataset
```

All options:

```
--annotations   FILE      CSV or Parquet annotations file  [required]
--images-root   DIR       Root directory for images         [required]
--checkpoint    FILE      Path to .pt / .pth checkpoint     [required]
--dataset-name  NAME      Output sub-directory name         [required]
--model-module  MOD:CLS   module.path:ClassName for state-dict mode
--datasets-dir  DIR       Root datasets dir  (default: datasets/)
--batch-size    N         Embedding batch size  (default: 64)
--device        STR       PyTorch device  (default: cpu)
--input-size    H W       Resize crops to H×W before the model  (default: 224 224)
```

### 3. Run the Streamlit app

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Optional environment variables:

| variable | default | description |
|----------|---------|-------------|
| `DATASETS_DIR` | `datasets/` | Path to the datasets directory |
| `TOP_K` | `10` | Default number of search results |
| `DEVICE` | `cpu` | PyTorch device |
| `MODEL_PATH` | _(empty)_ | Pre-fill the model path field |

---

## Project structure

```
image-crop-retrieval/
├── app.py                              # Streamlit entry point
├── pyproject.toml                      # uv project config, ruff, mypy
├── scripts/
│   └── build_index.py                  # Offline indexing CLI
├── src/
│   └── image_retrieval/
│       ├── config.py                   # AppConfig, DatasetMeta
│       ├── embedder.py                 # EmbedderProtocol, TorchEmbedder
│       ├── indexer.py                  # FAISSIndex, SearchResult
│       ├── registry.py                 # DatasetRegistry
│       └── ui/
│           ├── crop_selector.py        # Upload + canvas crop widget
│           └── results_viewer.py       # Top-K results grid
└── datasets/                           # Created by build_index.py
    └── {dataset_name}/
        ├── index.faiss
        ├── metadata.parquet
        └── images_root.txt
```

---

## Model requirements

Your SSL model must:

1. Accept a float tensor of shape `(N, 3, H, W)` (standard ImageNet normalisation is applied automatically).
2. Return a float tensor of shape `(N, D)` — embedding vectors.  
   Spatial outputs `(N, C, H, W)` are flattened automatically.

The model checkpoint can be:

- **Full-model pickle** (`torch.save(model, path)`) — just pass `--checkpoint`.
- **State-dict** (`torch.save(model.state_dict(), path)`) — also pass `--model-module module.path:ClassName`.

> ⚠️ **Security**: `torch.load(weights_only=False)` executes arbitrary pickle code.  
> Only load checkpoints from trusted sources.

---

## Development

```bash
# Lint
uv run ruff check src/ app.py scripts/

# Type check
uv run mypy src/ app.py scripts/

# Auto-fix lint issues
uv run ruff check --fix src/ app.py scripts/
```
