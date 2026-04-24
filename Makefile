PYTHON ?= python3
ifneq ("$(wildcard .venv312/bin/python)","")
PYTHON := .venv312/bin/python
endif
QDRANT_ENV = DIANALYSIS_RETRIEVAL_BACKEND=qdrant
CONFIG ?= configs/base.toml
PROFILE ?=
RELEASE_PROFILE ?= configs/profiles/release_xgboost_qdrant.toml
XGB_PROFILE ?= configs/profiles/xgboost.toml
SCRATCH_ROOT ?= artifacts_tmp
RUN ?=
RUN_ARTIFACTS_DIR = $(SCRATCH_ROOT)/$(RUN)
RUN_REPORT_PATH = reports/model_quality_$(RUN).json
DOCKER_COMPOSE ?= docker compose
DOCKER_SERVICE ?= app
DOCKER_OPS_SERVICE ?= ops
DOCKER_SYNC_BUILD ?=
DOCKER_VOLUME ?= $(CURDIR):/app
DOCKER_WORKDIR ?= /app
NORMALIZE_INPUT ?= data/products_off_clean.csv
NORMALIZE_OUTPUT ?=
SYNC_QDRANT_MODE ?= recreate
VERIFY_REQUIRE_QDRANT ?= true
CONFIG_ARGS = --config $(CONFIG)
ifneq ($(strip $(PROFILE)),)
CONFIG_ARGS += --profile $(PROFILE)
endif

.PHONY: train train-logreg train-xgb train-xgb-local train-xgb-active refresh refresh-upsert refresh-release prune quality-gate mlflow-exp eval eval-retrieval threshold-compare normalize-labels normalize-labels-write normalize-labels-inplace prod sync-retrieval sync-retrieval-docker verify-sync verify-sync-docker prod-sync prod-sync-docker qdrant-up qdrant-down qdrant-check test-live-qdrant-docker test-edges app app-dev app-logs app-dev-logs app-stop app-dev-stop app-local

train:
	$(PYTHON) train.py $(CONFIG_ARGS)

train-logreg:
	@if [ -n "$(RUN)" ]; then \
		OUT="$(SCRATCH_ROOT)/$(RUN)"; \
		echo "Training logreg scratch artifacts -> $$OUT"; \
		mkdir -p "$$OUT"; \
		$(PYTHON) train.py --config $(CONFIG) --model-type logreg --artifacts-dir "$$OUT"; \
		echo "Done. Evaluate with: make eval RUN=$(RUN)"; \
	else \
		$(PYTHON) train.py $(CONFIG_ARGS) --model-type logreg; \
	fi

train-xgb:
	@$(DOCKER_COMPOSE) run --rm --build --no-deps \
		--volume "$(DOCKER_VOLUME)" \
		--workdir "$(DOCKER_WORKDIR)" \
		$(DOCKER_SERVICE) \
		make train-xgb-local CONFIG="$(CONFIG)" XGB_PROFILE="$(XGB_PROFILE)" RUN="$(RUN)" SCRATCH_ROOT="$(SCRATCH_ROOT)"

train-xgb-local:
	@if [ -n "$(RUN)" ]; then \
		OUT="$(SCRATCH_ROOT)/$(RUN)"; \
		echo "Training xgboost scratch artifacts -> $$OUT"; \
		mkdir -p "$$OUT"; \
		$(PYTHON) train.py --config $(CONFIG) --profile $(XGB_PROFILE) --model-type xgboost --artifacts-dir "$$OUT"; \
		echo "Done. Evaluate with: make eval RUN=$(RUN) PROFILE=$(XGB_PROFILE)"; \
	else \
		$(PYTHON) train.py --config $(CONFIG) --profile $(XGB_PROFILE); \
	fi

train-xgb-active:
	$(PYTHON) train.py --config $(CONFIG) --profile $(XGB_PROFILE)

refresh:
	PYTHONPATH=. $(PYTHON) experiments/rescore_candidates.py $(CONFIG_ARGS) --auto-train-if-missing --qdrant-mode none

refresh-upsert:
	PYTHONPATH=. $(QDRANT_ENV) $(PYTHON) experiments/rescore_candidates.py $(CONFIG_ARGS) --auto-train-if-missing --qdrant-mode upsert

refresh-release:
	PYTHONPATH=. $(QDRANT_ENV) $(PYTHON) experiments/rescore_candidates.py --config $(CONFIG) --profile $(RELEASE_PROFILE) --train --model-type xgboost --qdrant-mode recreate

prune:
	PYTHONPATH=. $(QDRANT_ENV) $(PYTHON) experiments/rescore_candidates.py $(CONFIG_ARGS) --auto-train-if-missing --qdrant-mode prune

quality-gate:
	$(PYTHON) experiments/model_quality_gate.py $(CONFIG_ARGS)

eval:
	@if [ -z "$(RUN)" ]; then \
		echo "Usage: make eval RUN=<run_id> [PROFILE=<profile.toml>]"; \
		exit 1; \
	fi
	@if [ ! -d "$(RUN_ARTIFACTS_DIR)" ]; then \
		echo "Run artifacts not found: $(RUN_ARTIFACTS_DIR)"; \
		echo "Train first: make train-xgb RUN=$(RUN) or make train-logreg RUN=$(RUN)"; \
		exit 1; \
	fi
	@mkdir -p reports
	@PROFILE_ARG=""; \
	if [ -n "$(PROFILE)" ]; then \
		PROFILE_ARG="--profile $(PROFILE)"; \
	elif echo "$(RUN)" | grep -q '^xgb_'; then \
		PROFILE_ARG="--profile $(XGB_PROFILE)"; \
	fi; \
	echo "Evaluating run $(RUN) with: --config $(CONFIG) $$PROFILE_ARG"; \
	$(PYTHON) experiments/model_quality_gate.py --config $(CONFIG) $$PROFILE_ARG --artifacts-dir $(RUN_ARTIFACTS_DIR) --report-path $(RUN_REPORT_PATH)
	@echo "Saved eval report -> $(RUN_REPORT_PATH)"

mlflow-exp:
	$(PYTHON) experiments/mlflow_missing_indicator_experiment.py

eval-retrieval:
	PYTHONPATH=. $(QDRANT_ENV) QDRANT_URL=http://localhost:6335 $(PYTHON) experiments/retrieval_ab_test.py --input-csv data/products_off_clean_scored_labeled_v2.csv --out-csv reports/retrieval_ab_test_latest.csv --out-json reports/retrieval_ab_test_latest.json

threshold-compare:
	$(PYTHON) experiments/threshold_comparison.py --input-csv data/products_off_clean.csv

normalize-labels:
	$(PYTHON) experiments/normalize_category_labels.py --input "$(NORMALIZE_INPUT)"

normalize-labels-write:
	$(PYTHON) experiments/normalize_category_labels.py --input "$(NORMALIZE_INPUT)" --write $(if $(strip $(NORMALIZE_OUTPUT)),--output "$(NORMALIZE_OUTPUT)",)

normalize-labels-inplace:
	$(PYTHON) experiments/normalize_category_labels.py --input "$(NORMALIZE_INPUT)" --write --in-place

prod:
	@if [ -z "$(RUN)" ]; then \
		echo "Usage: make prod RUN=<run_id>"; \
		exit 1; \
	fi
	@if [ ! -d "$(RUN_ARTIFACTS_DIR)" ]; then \
		echo "Run artifacts not found: $(RUN_ARTIFACTS_DIR)"; \
		exit 1; \
	fi
	@if [ ! -f "$(RUN_ARTIFACTS_DIR)/meta.joblib" ]; then \
		echo "meta.joblib not found in run artifacts: $(RUN_ARTIFACTS_DIR)/meta.joblib"; \
		exit 1; \
	fi
	@EXPECTED=""; \
	case "$(RUN)" in \
		xgb_*) EXPECTED="xgboost" ;; \
		logreg_*) EXPECTED="logreg" ;; \
		*) EXPECTED="" ;; \
	esac; \
	if [ -z "$$EXPECTED" ]; then \
		echo "RUN must start with 'xgb_' or 'logreg_' (got: $(RUN))"; \
		exit 1; \
	fi; \
	ACTUAL=$$($(PYTHON) -c "import joblib,sys; m=joblib.load(sys.argv[1]); print(str(m.get('model_type','')).strip().lower())" "$(RUN_ARTIFACTS_DIR)/meta.joblib"); \
	if [ -z "$$ACTUAL" ]; then \
		echo "model_type missing in $(RUN_ARTIFACTS_DIR)/meta.joblib"; \
		exit 1; \
	fi; \
	if [ "$$EXPECTED" != "$$ACTUAL" ]; then \
		echo "RUN prefix expects '$$EXPECTED' but artifacts report '$$ACTUAL' in $(RUN_ARTIFACTS_DIR)/meta.joblib"; \
		exit 1; \
	fi; \
	echo "Validated run/model match: $(RUN) -> $$ACTUAL"
	@mkdir -p artifacts_backup
	@if [ -d artifacts ]; then \
		BACKUP="artifacts_backup/$$(date +%Y%m%d_%H%M%S)"; \
		cp -R artifacts "$$BACKUP"; \
		echo "Backed up current artifacts -> $$BACKUP"; \
	fi
	@rm -rf artifacts
	@cp -R "$(RUN_ARTIFACTS_DIR)" artifacts
	@echo "Promoted $(RUN_ARTIFACTS_DIR) -> artifacts/"

sync-retrieval:
	PYTHONPATH=. $(QDRANT_ENV) $(PYTHON) experiments/rescore_candidates.py $(CONFIG_ARGS) --artifacts-dir artifacts --qdrant-mode $(SYNC_QDRANT_MODE)

qdrant-up:
	@$(DOCKER_COMPOSE) up -d qdrant

qdrant-down:
	@$(DOCKER_COMPOSE) stop qdrant

qdrant-check:
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	PYTHONPATH=. $(PYTHON) experiments/check_qdrant_connection.py --require-collection

test-live-qdrant-docker:
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	$(DOCKER_COMPOSE) run --rm --no-deps \
		-e QDRANT_URL \
		-e QDRANT_API_KEY \
		-e DIANALYSIS_QDRANT_COLLECTION \
		-e DIANALYSIS_EMBED_MODEL \
		-e DIANALYSIS_RETRIEVAL_BACKEND=qdrant \
		-e DIANALYSIS_LIVE_QDRANT_TEST=1 \
		$(DOCKER_OPS_SERVICE) \
		python -m unittest tests.test_qdrant_live_connection

sync-retrieval-docker: qdrant-up
	@$(DOCKER_COMPOSE) run --rm $(DOCKER_SYNC_BUILD) \
		--volume "$(DOCKER_VOLUME)" \
		--workdir "$(DOCKER_WORKDIR)" \
		$(DOCKER_OPS_SERVICE) \
		python experiments/rescore_candidates.py $(CONFIG_ARGS) --artifacts-dir artifacts --qdrant-mode $(SYNC_QDRANT_MODE)

verify-sync:
	PYTHONPATH=. $(QDRANT_ENV) $(PYTHON) experiments/verify_retrieval_sync.py $(CONFIG_ARGS) --artifacts-dir artifacts $(if $(filter true,$(VERIFY_REQUIRE_QDRANT)),,--no-require-qdrant)

verify-sync-docker: qdrant-up
	@$(DOCKER_COMPOSE) run --rm $(DOCKER_SYNC_BUILD) \
		--volume "$(DOCKER_VOLUME)" \
		--workdir "$(DOCKER_WORKDIR)" \
		$(DOCKER_OPS_SERVICE) \
		python experiments/verify_retrieval_sync.py $(CONFIG_ARGS) --artifacts-dir artifacts $(if $(filter true,$(VERIFY_REQUIRE_QDRANT)),,--no-require-qdrant)

prod-sync: prod sync-retrieval verify-sync
	@echo "Promotion + retrieval sync complete."

prod-sync-docker: prod sync-retrieval-docker verify-sync-docker
	@echo "Promotion + retrieval sync complete (docker path)."

test-edges:
	DIANALYSIS_RETRIEVAL_BACKEND=heuristic HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -m unittest tests.test_scoring_pipeline_edges tests.test_candidate_filters_edges tests.test_barcode_edges

app:
	@$(DOCKER_COMPOSE) up --build -d $(DOCKER_SERVICE)
	@echo "App running at http://localhost:8501"
	@echo "View logs: make app-logs"
	@echo "Stop app: make app-stop"

app-dev:
	@$(DOCKER_COMPOSE) up --build -d app-dev
	@echo "App dev mode running at http://localhost:8501"
	@echo "Live code mount enabled from $(CURDIR) -> /app"
	@echo "View logs: make app-dev-logs"
	@echo "Stop app dev: make app-dev-stop"

app-logs:
	@$(DOCKER_COMPOSE) logs -f $(DOCKER_SERVICE)

app-dev-logs:
	@$(DOCKER_COMPOSE) logs -f app-dev

app-stop:
	@$(DOCKER_COMPOSE) down

app-dev-stop:
	@$(DOCKER_COMPOSE) stop app-dev

app-local:
	$(PYTHON) -m streamlit run app.py
