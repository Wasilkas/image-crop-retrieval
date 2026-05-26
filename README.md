# image-crop-retrieval

Streamlit-приложение для **SSL-поиска похожих кропов**.

Загрузите изображение, нарисуйте рамку — и приложение найдёт топ-K наиболее
визуально похожих bounding-box'ов из предварительно проиндексированного датасета,
используя косинусное сходство по SSL-эмбеддингам.

---

## Быстрый старт

### 1. Создание виртуального окружения

```bash
uv venv --python 3.11
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv sync --all-groups
```

### 2. Построение индекса датасета (разово / cron-задача)

Файл аннотаций должен быть CSV или Parquet и содержать как минимум следующие колонки:

| колонка | описание |
|---------|----------|
| `image_path` | Путь к изображению **относительно `--images-root`** |
| `x1`, `y1` | Левый верхний угол bounding-box (пикселей) |
| `x2`, `y2` | Правый нижний угол bounding-box (пикселей) |

```bash
uv run python scripts/build_index.py local \
    --annotations /data/boxes.csv \
    --images-root /data/images/ \
    --checkpoint  /models/encoder.pth \
    --dataset-name my_dataset
```

Для **state-dict** чекпоинтов (класс модели должен быть импортируемым):

```bash
uv run python scripts/build_index.py local \
    --annotations /data/boxes.parquet \
    --images-root /data/images/ \
    --checkpoint  /models/weights.pt \
    --model-module mypackage.models:MyEncoder \
    --dataset-name my_dataset
```

Все параметры:

```
--annotations   FILE      CSV или Parquet с аннотациями          [обязательно]
--images-root   DIR       Корневая директория изображений         [обязательно]
--checkpoint    FILE      Путь к чекпоинту .pt / .pth            [обязательно]
--dataset-name  NAME      Имя выходной поддиректории              [обязательно]
--model-module  MOD:CLS   module.path:ClassName для state-dict
--datasets-dir  DIR       Корень датасетов  (по умолч.: datasets/)
--batch-size    N         Размер батча эмбеддинга  (по умолч.: 64)
--device        STR       Устройство PyTorch  (по умолч.: cpu)
--input-size    H W       Размер кропа для модели  (по умолч.: 224 224)
```

### 3. Запуск приложения

```bash
uv run streamlit run app.py
```

Откройте [http://localhost:8501](http://localhost:8501) в браузере.

Необязательные переменные среды:

| переменная | по умолчанию | описание |
|------------|--------------|----------|
| `DATASETS_DIR` | `datasets/` | Путь к директории датасетов |
| `TOP_K` | `10` | Количество результатов по умолчанию |
| `DEVICE` | `cpu` | Устройство PyTorch |
| `MODEL_PATH` | _(пусто)_ | Предзаполнить поле пути к модели |

---

## Структура проекта

```
image-crop-retrieval/
├── app.py                              # Точка входа Streamlit (навигация)
├── pyproject.toml                      # Конфигурация uv, ruff, mypy
├── scripts/
│   └── build_index.py                  # CLI для офлайн-индексации
├── src/
│   ├── image_retrieval/
│   │   ├── config.py                   # AppConfig, DatasetMeta
│   │   ├── embedder.py                 # EmbedderProtocol, TorchEmbedder
│   │   ├── indexer.py                  # FAISSIndex, SearchResult
│   │   ├── registry.py                 # DatasetRegistry, S3DatasetRegistry
│   │   ├── s3_client.py                # S3Client (boto3-обёртка)
│   │   ├── cvat_client.py              # CVATClient (REST API 2.x)
│   │   └── ui/
│   │       ├── crop_selector.py        # Виджет загрузки + выбора кропа
│   │       ├── results_viewer.py       # Сетка топ-K результатов
│   │       └── cvat_exporter.py        # Панель экспорта в CVAT
│   └── pages/
│       └── image_retrieval.py          # Логика страницы поиска
└── datasets/                           # Создаётся build_index.py
    └── {dataset_name}/
        ├── index.faiss
        ├── metadata.parquet
        └── images_root.txt
```

---

## Требования к модели

SSL-модель должна:

1. Принимать float-тензор формы `(N, 3, H, W)` (нормализация ImageNet применяется автоматически).
2. Возвращать float-тензор формы `(N, D)` — векторы эмбеддингов.  
   Пространственные выходы `(N, C, H, W)` выравниваются автоматически.

Чекпоинт модели может быть:

- **Полная модель (pickle)** (`torch.save(model, path)`) — достаточно `--checkpoint`.
- **State-dict** (`torch.save(model.state_dict(), path)`) — также укажите `--model-module module.path:ClassName`.

> ⚠️ **Безопасность**: `torch.load(weights_only=False)` выполняет произвольный pickle-код.  
> Загружайте чекпоинты только из **доверенных источников**.

---

## S3-режим

Для работы с S3 укажите переменные среды или добавьте блок `s3:` в `config.yaml`:

```bash
S3_BUCKET=my-bucket
S3_PREFIX=datasets/          # необязательный префикс
S3_REGION=us-east-1
S3_ENDPOINT_URL=http://localhost:9000   # MinIO / Yandex Cloud
```

Построение индекса в S3:

```bash
uv run python scripts/build_index.py s3 \
    --bucket    my-bucket \
    --dataset   my_dataset \
    --checkpoint /models/encoder.pth
```

Индексы датасетов и модель кэшируются локально и **обновляются каждые 10 минут**:
при истечении TTL приложение проверяет S3 и перескачивает файл только если версия на S3 новее.

---

## CVAT-экспорт

Добавьте блок `cvat:` в `config.yaml`:

```yaml
cvat:
  url: https://cvat.example.com
  token: my-api-token      # предпочтительнее username/password
  project_id: 42           # необязательно
  task_label: crop         # по умолчанию
```

При наличии конфигурации под сеткой результатов появляется панель экспорта в CVAT.

---

## Разработка

```bash
# Линтер
uv run ruff check src/ app.py scripts/

# Проверка типов
uv run mypy src/ app.py scripts/

# Автофикс
uv run ruff check --fix src/ app.py scripts/

# Тесты
uv run pytest
```
