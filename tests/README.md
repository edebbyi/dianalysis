# Tests

This folder contains unit and integration tests for scoring, retrieval, and barcode flows.

## Standard Test Run

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## Live Qdrant Integration Test (Optional)

This test uses real Qdrant Cloud credentials from `.env` and runs barcode smoke checks end-to-end.

```bash
make test-live-qdrant-docker
```

Notes:
- The live test file is `tests/test_qdrant_live_connection.py`.
- It is skipped by default unless `DIANALYSIS_LIVE_QDRANT_TEST=1`.
- The Docker target sets that flag automatically and forwards Qdrant env vars.
