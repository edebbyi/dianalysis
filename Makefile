PYTHON ?= python3
ifneq ("$(wildcard .venv312/bin/python)","")
PYTHON := .venv312/bin/python
endif
QDRANT_ENV = DIANALYSIS_RETRIEVAL_BACKEND=qdrant

.PHONY: train refresh refresh-upsert refresh-release prune quality-gate mlflow-exp eval-retrieval app

train:
	$(PYTHON) train.py

refresh:
	PYTHONPATH=. $(PYTHON) experiments/rescore_candidates.py --auto-train-if-missing --qdrant-mode none

refresh-upsert:
	PYTHONPATH=. $(QDRANT_ENV) $(PYTHON) experiments/rescore_candidates.py --auto-train-if-missing --qdrant-mode upsert

refresh-release:
	PYTHONPATH=. $(QDRANT_ENV) $(PYTHON) experiments/rescore_candidates.py --train --model-type xgboost --qdrant-mode recreate

prune:
	PYTHONPATH=. $(QDRANT_ENV) $(PYTHON) experiments/rescore_candidates.py --auto-train-if-missing --qdrant-mode prune

quality-gate:
	$(PYTHON) experiments/model_quality_gate.py

mlflow-exp:
	$(PYTHON) experiments/mlflow_missing_indicator_experiment.py

eval-retrieval:
	PYTHONPATH=. $(QDRANT_ENV) QDRANT_URL=http://localhost:6335 $(PYTHON) experiments/retrieval_ab_test.py --input-csv data/products_off_clean_scored_labeled_v2.csv --out-csv reports/retrieval_ab_test_latest.csv --out-json reports/retrieval_ab_test_latest.json

app:
	streamlit run app.py
