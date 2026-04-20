# 🍎 Dianalysis

Calibrated diabetes-risk scoring for packaged foods, combining nutrition-based rules with trainable models (`logreg` or `xgboost`) and recommendation ranking.

Roadmap:
- `docs/release_plan_v1.2.md` (v1.1.0 release boundary + v1.2.0 evaluation upgrade plan)

## Installation

```bash
git clone https://github.com/edebbyi/dianalysis.git
cd dianalysis

python -m venv venv
source venv/bin/activate  # Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```

## Run
Recommendation: for XGBoost workflows, use Python `3.14` virtual environment.

### Train the model
```bash
python train.py
```
Generates synthetic data, trains `LogisticRegression` with CV + holdout diagnostics, and writes artifacts to `artifacts/`.

Use config-driven defaults (with CLI override support):
```bash
python train.py --config configs/base.toml
```

### Run model quality gate
```bash
python experiments/model_quality_gate.py
```
Runs a pass/fail gate on `F1`, `AUPRC`, `Brier`, and ranking metrics, then writes
`reports/model_quality_report.json`.

### Start the demo UI
```bash
streamlit run app.py
```
Serves an interactive UI on `http://localhost:8501` for scoring items and browsing alternatives.

Run app with explicit config/profile:
```bash
streamlit run app.py -- --config configs/base.toml --profile configs/profiles/xgboost.toml
```

### Refresh recommendation assets (single script)
Use one command to refresh model artifacts, pre-scored candidates, and optional Qdrant index.

Clean source CSV duplicates first (recommended):
```bash
python experiments/dedupe_source_csv.py
```

Rescore only (use existing artifacts):
```bash
python experiments/rescore_candidates.py
```

Auto-train only if artifacts are missing:
```bash
python experiments/rescore_candidates.py --auto-train-if-missing
```

Force retrain + full index recreate (release workflow):
```bash
DIANALYSIS_RETRIEVAL_BACKEND=qdrant python experiments/rescore_candidates.py \
  --train \
  --model-type xgboost \
  --qdrant-mode recreate
```

Data-only update (keep model, upsert new index points):
```bash
DIANALYSIS_RETRIEVAL_BACKEND=qdrant python experiments/rescore_candidates.py \
  --qdrant-mode upsert
```

Data sync with stale-point cleanup:
```bash
DIANALYSIS_RETRIEVAL_BACKEND=qdrant python experiments/rescore_candidates.py \
  --qdrant-mode prune
```

### Makefile shortcuts
```bash
make train-xgb       # train using xgboost profile (dockerized)
make train-xgb-local # train using xgboost profile on host python
make app             # run streamlit app in docker (detached)
make app-logs        # follow docker app logs
make app-stop        # stop docker app
make app-local       # run streamlit app on host python
make refresh          # auto-train if missing + rescore
make refresh-upsert   # rescore + qdrant upsert
make prune            # rescore + qdrant upsert + prune missing
make refresh-release  # force retrain + full qdrant recreate
```

Optional Makefile config overrides:
```bash
make train CONFIG=configs/base.toml PROFILE=configs/profiles/xgboost.toml
```

Artifact promotion workflow (scratch -> eval -> prod):
```bash
RUN=xgb_$(date +%Y%m%d_%H%M%S)
make train-xgb RUN=$RUN
make eval RUN=$RUN
make prod RUN=$RUN
```

Notes:
- `RUN` is a run ID, and artifacts are stored in `artifacts_tmp/<RUN>`.
- Use `xgb_...` or `logreg_...` prefixes for run IDs.
- `make eval` auto-applies the XGBoost profile for `xgb_...` runs (override with `PROFILE=...` if needed).
- `make prod` validates that run prefix matches the artifact `model_type` before promotion.

### Config files
Run scripts can load TOML defaults from `configs/base.toml` and optional profile overlays:

- `python train.py --config configs/base.toml`
- `python experiments/rescore_candidates.py --config configs/base.toml`
- `python experiments/model_quality_gate.py --config configs/base.toml`
- `python train.py --config configs/base.toml --profile configs/profiles/xgboost.toml`

Profile examples:
- `configs/profiles/xgboost.toml`
- `configs/profiles/release_xgboost_qdrant.toml`

Precedence is: `CLI flags > profile config > base config`.

Each run also writes resolved config snapshots for traceability:
- `artifacts/run_configs/train_resolved_config.json`
- `reports/rescore_resolved_config.json`
- `reports/model_quality_resolved_config.json`

### DVC (Data + Experiments)
This repo includes a DVC pipeline scaffold in:
- `dvc.yaml`
- `params.yaml`

One-time setup:
```bash
dvc init
git add .dvc .dvcignore dvc.yaml params.yaml
git commit -m "Initialize DVC pipeline"
```

Run pipeline:
```bash
dvc repro
```

Run tracked experiments (works well with the VS Code DVC extension):
```bash
dvc exp run
dvc exp show
```

Change parameters from `params.yaml` or with CLI overrides, for example:
```bash
dvc exp run -S train.cv_folds=3 -S quality_gate.sample_size=40
```

#### Configure shared DVC remote (Backblaze B2 / S3-compatible)
This project uses a Backblaze B2 bucket as the shared DVC remote.

1. Install the S3 plugin (one-time):
```bash
python3.11 -m pip install dvc-s3
```

2. Configure the shared remote (safe to commit):
```bash
dvc remote add -d b2 s3://dianalysis
dvc remote modify b2 endpointurl https://s3.us-east-005.backblazeb2.com
```

3. Configure credentials locally only (never commit):
```bash
dvc remote modify --local b2 access_key_id <B2_KEY_ID>
dvc remote modify --local b2 secret_access_key <B2_APPLICATION_KEY>
```

4. Push tracked data:
```bash
dvc push
```

Public-repo safety:
- Commit `.dvc/config` (remote name, URL, endpoint).
- Do not commit `.dvc/config.local` (contains secrets).
- `.dvc/config.local` is already ignored via `.dvc/.gitignore`.

Troubleshooting (`dvc push` returns `403 Forbidden`):
- Confirm remote config:
  - `dvc remote list`
  - `dvc remote modify b2 endpointurl https://s3.us-east-005.backblazeb2.com`
- Confirm credentials are in `.dvc/config.local` (not `.dvc/config`):
  - `dvc remote modify --local b2 access_key_id <B2_KEY_ID>`
  - `dvc remote modify --local b2 secret_access_key <B2_APPLICATION_KEY>`
- Regenerate the B2 Application Key and scope it to bucket `dianalysis` with capabilities:
  - `listBuckets`, `listFiles`, `readFiles`, `writeFiles` (and `deleteFiles` if you use `dvc gc`)
- Wait ~1 minute after key changes, then retry:
  - `dvc push -v`

### Optional: Enable Qdrant Semantic Retrieval
Use this when you want semantic candidate pooling instead of only heuristic same-group pooling.

1. Start Qdrant:
```bash
docker run --rm -p 6333:6333 qdrant/qdrant
```

2. Build the vector index directly (optional low-level command):
```bash
DIANALYSIS_RETRIEVAL_BACKEND=qdrant python experiments/build_qdrant_index.py --recreate
```

3. Run app with Qdrant retrieval enabled:
```bash
DIANALYSIS_RETRIEVAL_BACKEND=qdrant streamlit run app.py
```

Useful environment variables:
- `DIANALYSIS_RETRIEVAL_BACKEND=heuristic|qdrant` (default: `heuristic`)
- `QDRANT_URL` (default: `http://localhost:6333`)
- `DIANALYSIS_QDRANT_COLLECTION` (default: `dianalysis_products`)
- `DIANALYSIS_EMBED_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)

## Docker

Build image:

```bash
docker build -t dianalysis:latest .
```

Default Docker build uses `requirements-docker.txt` (lightweight CPU-friendly dependencies).

To build with full dependencies from `requirements.txt`:

```bash
docker build --build-arg REQUIREMENTS_FILE=requirements.txt -t dianalysis:full .
```

Run the app:

```bash
docker run --rm -p 8501:8501 dianalysis:latest
```

Run training and quality gate inside the container:

```bash
docker run --rm dianalysis:latest python train.py
docker run --rm dianalysis:latest python experiments/model_quality_gate.py
```

Or use compose:

```bash
docker compose up --build
```

## How To Reproduce Results
Use this exact sequence to reproduce the current results:

```bash
python train.py
python experiments/rescore_candidates.py --qdrant-mode none
python experiments/model_quality_gate.py
python experiments/mlflow_missing_indicator_experiment.py
```

Then execute the notebook (`dianalysis/dianalysis.ipynb`) top-to-bottom to refresh plots and tables.

Key output files to keep:
- `reports/model_quality_report.json`
- `reports/mlflow_missing_indicator_summary.csv`

## Model Decision
- Chosen config: `class_weight='balanced'`, `C=0.3`, `add_indicator=True`.
- Chosen threshold: `0.50` for final binary classification.
- This setup passed the quality gate on test `F1`, test `AUPRC`, test `Brier`, and ranking (`NDCG@3` + coverage).
- I track `AUPRC` because the dataset is imbalanced and I care most about how well the model handles the positive class.
- I track `Brier` to make sure predicted risk probabilities are meaningful, not just the final 0/1 label.
- I track ranking metrics so the alternative recommendations are actually useful in practice.

## 60-Second Interview Narrative
This project predicts diabetes risk for packaged foods and recommends lower-risk alternatives.  
I used Open Food Facts as the base dataset, but I had to clean it carefully because of API issues, including a nut-filter bug I kept running into.  
Since the data is imbalanced, I focused on `F1` and `AUPRC`, and I also tracked `Brier` so I could check whether the risk probabilities were well-behaved.  
I compared thresholds and kept `0.50` because it gave the best overall balance on the test results in this version.  
For the recommendation side, I added `NDCG@k` and coverage so I could measure whether alternatives were both relevant and consistently available.  
To make handoff and deployment safer, I added a model quality gate script with clear pass/fail checks.

## Configuration

- `artifacts/`: saved model plus metadata consumed by `score_item()` and `score_by_barcode()`.
- `data/products_off_clean.csv`: purified [Open Food Facts](https://world.openfoodfacts.org) catalog used for alternatives; refresh via `dianalysis/off_pipeline.py`.
- `data/products_off_clean_scored.csv`: pre-scored candidate pool generated by `experiments/rescore_candidates.py`.
- `dianalysis/scoring.py`: customize risk explanations, alternative filtering/ranking rules, and NDCG proxy helpers.
- `experiments/model_quality_gate.py`: pass/fail thresholds and gate settings for a quick model quality check.
- `experiments/rescore_candidates.py`: one-script refresh workflow for train/rescore/index.
- MLflow experiments default to `sqlite:///mlflow.db` (override with `MLFLOW_TRACKING_URI` if needed).
- Streamlit respects `STREAMLIT_SERVER_PORT` / `STREAMLIT_SERVER_HEADLESS` environment overrides when running `app.py`.

## Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-topic`).
3. Commit your changes and push the branch.
4. Open a pull request with a concise summary.

## Author

Esosa Deborah Imafidon — https://www.kaggle.com/code/deborahimafidon/dianalysis

## License

MIT
