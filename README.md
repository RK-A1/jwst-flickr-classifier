# JWST Image Pipeline

A local data pipeline that ingests James Webb Space Telescope photos from Flickr, stores metadata and feature embeddings in DuckDB, and trains image classifiers to categorise JWST imagery by subject type.

**Author:** RK

---

## Overview

```
Flickr API (nasawebbtelescope account)
        │
        ▼
  Airflow DAG (Astro CLI)
        │
        ├─── Download images → include/images/
        ├─── Extract metadata → DuckDB (photos table)
        └─── Extract ResNet50 embeddings → DuckDB (photos.embedding column)
                │
                ▼
        ML Training
        ├─── ResNet features → XGBoost classifier
        └─── Fine-tuned ResNet50 (end-to-end, MPS backend)
```

Photos are classified into six subject categories: **galaxy**, **galaxy cluster**, **nebula**, **star**, **solar system**, and **exoplanet**. Labels are derived from Flickr tags via a consolidation script, then used to train classifiers. Inference is gated by a confidence threshold (0.6) — low-confidence predictions are stored as `unclassified` rather than forced into a wrong class.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Orchestration | Apache Airflow via [Astro CLI](https://docs.astronomer.io/astro/cli/overview) |
| Storage | [DuckDB](https://duckdb.org/) |
| Feature extraction | ResNet50 (torchvision, `IMAGENET1K_V2` weights) |
| Classifier | XGBoost (on ResNet features) + fine-tuned ResNet50 |
| Compute | PyTorch with MPS backend (Apple Silicon) |
| Image source | Flickr API — `nasawebbtelescope` account |

---

## Prerequisites

- macOS with Apple Silicon (MPS acceleration)
- [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli)
- Docker Desktop
- A Flickr API key ([apply here](https://www.flickr.com/services/api/misc.api_keys.html))

---

## Setup

**1. Clone the repo**

```bash
git clone <repo-url>
cd JWST2
```

**2. Add your Flickr API key**

Create a `.env` file in the project root:

```
FLICKR_API_KEY=your_key_here
```

This file is gitignored and loaded automatically by Astro CLI.

**3. Start Airflow**

```bash
astro dev start
```

Airflow UI is available at [http://localhost:8080](http://localhost:8080) (admin / admin).

---

## Running the Pipeline

### Step 1 — Ingest photos

Trigger the ingest DAG to download photos from Flickr and populate DuckDB:

```bash
astro dev run airflow dags trigger jwst_flickr_ingest
```

By default, the DAG fetches the latest 1000 photos. To ingest the full archive (~4000 photos), set `MAX_PHOTOS = None` in `dags/jwst_flickr_ingest.py`.

### Step 2 — Extract embeddings

```bash
astro dev run airflow dags trigger jwst_feature_extraction
```

Runs ResNet50 (MPS-accelerated) over all downloaded images and stores 2048-dim feature vectors in DuckDB. Batched at 32 images, max 2 concurrent batches to avoid OOM.

### Step 3 — Consolidate labels

```bash
python include/tag_consolidation.py
```

Maps raw Flickr tags to canonical subject labels (`galaxy`, `nebula`, etc.) and writes them to the `canonical_label` column.

### Step 4 — Train classifiers

```bash
astro dev run airflow dags trigger jwst_train_classifiers
```

Trains an XGBoost classifier on ResNet features and fine-tunes ResNet50 end-to-end. Checkpoints are saved to `include/models/`. Results are logged to the `training_runs` table.

### Step 5 — Run inference

Inference runs automatically at the end of each `jwst_flickr_ingest` run (the `predict_labels` task). It loads the best model from `training_runs` and writes predictions to `photos.predicted_label`.

---

## Project Structure

```
JWST2/
├── dags/
│   ├── jwst_flickr_ingest.py        # Ingest + inference DAG
│   ├── jwst_feature_extraction.py   # ResNet50 embedding DAG
│   └── jwst_train_classifiers.py    # XGBoost + ResNet fine-tune DAG
├── include/
│   ├── db.py                        # DuckDB connection helper + schema
│   ├── tag_consolidation.py         # Tag → canonical label mapping
│   ├── images/                      # Downloaded images (gitignored)
│   ├── models/                      # Saved model checkpoints (gitignored)
│   └── jwst.duckdb                  # Primary database (gitignored)
├── Dockerfile                       # Astro Runtime base image
├── requirements.txt                 # Python dependencies
└── .env                             # API keys — never commit (gitignored)
```

---

## Database Schema

```sql
CREATE TABLE photos (
    photo_id        TEXT PRIMARY KEY,
    title           TEXT,
    description     TEXT,
    tags            TEXT[],
    image_path      TEXT,
    date_taken      TIMESTAMP,
    date_ingested   TIMESTAMP DEFAULT current_timestamp,
    embedding       FLOAT[],          -- ResNet50 2048-dim vector
    canonical_label TEXT,             -- ground truth from tag consolidation
    predicted_label TEXT              -- model prediction (or 'unclassified')
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

---

## Useful Commands

```bash
# Stop/start Airflow
astro dev stop
astro dev start

# Query the database directly
duckdb include/jwst.duckdb "SELECT count(*) FROM photos"
duckdb include/jwst.duckdb "SELECT canonical_label, count(*) FROM photos GROUP BY 1 ORDER BY 2 DESC"
duckdb include/jwst.duckdb "SELECT canonical_label, predicted_label, count(*) FROM photos GROUP BY 1, 2 ORDER BY 1, 3 DESC"

# Reset and re-ingest from scratch
duckdb include/jwst.duckdb "DELETE FROM photos"
```

---

## Notes

- **DuckDB is single-writer.** All writes go through a single Airflow task; reads use `read_only=True`.
- **`include/` only.** Astro CLI mounts `dags/`, `plugins/`, `include/`, and `tests/` into containers. All persistent data lives in `include/` for this reason.
- **`docker-compose.override.yml` does not work** with Astro CLI — Astro generates its compose config internally.
- **JWST images exceed PIL's default decompression bomb limit.** `Image.MAX_IMAGE_PIXELS = None` is set wherever images are opened.
- **MPS fallback.** All torch code checks `torch.backends.mps.is_available()` and falls back to CPU gracefully.
