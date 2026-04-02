# JWST Image Pipeline

Local data pipeline on macOS (M4 MacBook Air) that ingests James Webb Space Telescope photos from Flickr, stores metadata and image embeddings in DuckDB, and trains image classifiers.

## Architecture

```
Flickr API (nasawebbtelescope account)
        │
        ▼
  Airflow DAG (Astro CLI)
        │
        ├─── Download images → include/images/
        ├─── Extract metadata → DuckDB (photos table)
        └─── Extract ResNet embeddings → DuckDB (photos.embedding column)
                │
                ▼
        ML Training
        ├─── ResNet features → XGBoost classifier
        └─── Fine-tuned ResNet (end-to-end, MPS backend)
```

## Tech Stack

- **Orchestration:** Apache Airflow via [Astro CLI](https://docs.astronomer.io/astro/cli/overview) (`astro dev start/stop`)
- **Storage:** DuckDB at `include/jwst.duckdb`
- **ML:** PyTorch (MPS backend for Apple Silicon), XGBoost
- **Feature extraction:** ResNet (torchvision) — embeddings stored in DuckDB
- **API:** Flickr API, account `nasawebbtelescope`
- **Python:** 3.12 (Astro CLI container)

## Environment

The Flickr API key must be set in `.env` at the project root (Astro CLI loads this automatically into containers):

```
FLICKR_API_KEY=<your_key>
```

`.env` is gitignored. Never commit it.

## Key Paths

| Path | Purpose |
|------|---------|
| `include/jwst.duckdb` | Primary database |
| `include/images/` | Downloaded raw images |
| `include/models/` | Saved model checkpoints |
| `include/.torch_cache/` | Cached ResNet50 weights (persists across container restarts) |
| `dags/` | Airflow DAGs |
| `include/` | Shared Python modules + data (on Python path, mounted by Astro CLI) |

> **Why `include/` for data?** Astro CLI only mounts `dags/`, `plugins/`, `include/`, and `tests/` into containers. `data/` is not mounted, so the database and images live in `include/` to persist to the host filesystem.

## DuckDB Schema

```sql
CREATE TABLE photos (
    photo_id        TEXT PRIMARY KEY,
    title           TEXT,
    description     TEXT,
    tags            TEXT[],           -- list of tag strings
    image_path      TEXT,             -- absolute path to local file
    date_taken      TIMESTAMP,
    date_ingested   TIMESTAMP DEFAULT current_timestamp,
    embedding       FLOAT[],          -- ResNet feature vector (2048-dim)
    canonical_label TEXT,             -- tag-based label set by tag_consolidation.py
    predicted_label TEXT
);

CREATE TABLE training_runs (
    run_id      TEXT PRIMARY KEY,
    ts          TIMESTAMP DEFAULT current_timestamp,
    model_type  TEXT,                 -- 'xgboost' or 'resnet_finetune'
    accuracy    DOUBLE,
    f1_score    DOUBLE,
    model_path  TEXT
);
```

## Airflow DAGs

### `jwst_flickr_ingest` — Daily ingestion
1. `ensure_schema` — create DuckDB tables if missing
2. `get_watermark` — read MAX(date_taken) from photos table (0 on first run)
3. `fetch_photos_metadata` — paginate Flickr API for photos newer than watermark
4. `download_images` — fetch largest available size → `include/images/{photo_id}.jpg`
5. `insert_records` — upsert metadata into `photos` table
6. `predict_labels` — run inference on photos with embeddings but no predicted_label

**Key constants in the DAG:**
- `MAX_PHOTOS = 1000` — cap per run; set to `None` for full backfill
- `PER_PAGE = 500` — Flickr page size
- Sort order: `date-taken-desc` (newest first)

### `jwst_train` — Model training (triggered manually or weekly)
1. Load embeddings + labels from DuckDB
2. Train XGBoost classifier on ResNet features
3. Fine-tune ResNet end-to-end (MPS backend)
4. Save checkpoints to `models/`

## Common Commands

```bash
# Start local Airflow environment
astro dev start

# Stop
astro dev stop

# Trigger the ingest DAG manually
astro dev run airflow dags trigger jwst_flickr_ingest

# Open Airflow UI
open http://localhost:8080   # admin / admin

# Query DuckDB (install with: brew install duckdb)
duckdb include/jwst.duckdb "SELECT count(*) FROM photos"
duckdb include/jwst.duckdb "SELECT photo_id, title, date_taken FROM photos LIMIT 5"

# Reset watermark (re-ingest from scratch)
duckdb include/jwst.duckdb "DELETE FROM photos"
```

## Flickr API Notes

- Account: `nasawebbtelescope`
- User lookup: `flickr.urls.lookupUser` with `https://www.flickr.com/photos/nasawebbtelescope/` (path alias — do NOT use `flickr.people.findByUsername`)
- Relevant endpoints: `flickr.people.getPublicPhotos`, `flickr.photos.getSizes`, `flickr.photos.getInfo`
- Rate limit: 3600 req/hour — DAG sleeps 0.3s between pages

## PyTorch / MPS Notes

Always check MPS availability before falling back to CPU:

```python
import torch

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
```

ResNet feature extraction (no grad, eval mode):

```python
model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
model.fc = torch.nn.Identity()   # strip classifier head → 2048-dim output
model = model.to(device).eval()

with torch.no_grad():
    embedding = model(img_tensor.unsqueeze(0).to(device))
```

## Development Tips

- DuckDB is single-writer. Serialize DB writes through a single task; use `read_only=True` for reads.
- Store embeddings as `FLOAT[]` in DuckDB; use `.fetchnumpy()` to get numpy arrays for XGBoost.
- Keep heavy ML dependencies (torch, xgboost) out of `requirements.txt` if they conflict with Airflow's Docker image — use a custom Dockerfile instead.
- `docker-compose.override.yml` does NOT work with Astro CLI — Astro generates its compose config internally and does not write a base `docker-compose.yml` to the project directory, so overrides fail to load.
