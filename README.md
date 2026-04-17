# 🍎 Dianalysis

Calibrated diabetes-risk scoring for packaged foods, combining nutrition-based rules with trainable models (`logreg` or `xgboost`) and recommendation ranking.

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
make refresh          # auto-train if missing + rescore
make refresh-upsert   # rescore + qdrant upsert
make prune            # rescore + qdrant upsert + prune missing
make refresh-release  # force retrain + full qdrant recreate
```

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
