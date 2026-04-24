# Qdrant Setup

This document explains how Dianalysis connects to Qdrant in both local and cloud setups.

## Supports Both Local and Cloud

Dianalysis supports:

- Local Qdrant (Docker on your machine)
- Qdrant Cloud (hosted cluster + API key)

The app chooses which one to use from environment variables.

## Required Environment Variables

- `DIANALYSIS_RETRIEVAL_BACKEND`
  - `qdrant` to use vector retrieval
  - `heuristic` to skip Qdrant
- `QDRANT_URL`
  - Local example: `http://localhost:6333`
  - Cloud example: `https://<cluster-endpoint>`
- `DIANALYSIS_QDRANT_COLLECTION`
  - Default: `dianalysis_products`
- `DIANALYSIS_EMBED_MODEL`
  - Default: `sentence-transformers/all-MiniLM-L6-v2`
- `QDRANT_API_KEY` (cloud only)
  - Not needed for default local Docker Qdrant
- `DIANALYSIS_RETRIEVAL_CANDIDATE_LIMIT` (optional)
  - How many Qdrant candidates to retrieve before ranking (default: `30`)
- `DIANALYSIS_RETRIEVAL_TOPUP_MULTIPLIER` (optional)
  - How much to widen category fallback retrieval (default: `2`)
- `DIANALYSIS_WARMUP_RETRIEVAL` (optional)
  - Startup warm-up for embedder + tiny Qdrant query (default: `1`)

## Local Setup

Start local Qdrant:

```bash
docker compose up -d qdrant
```

Use local env values:

```bash
DIANALYSIS_RETRIEVAL_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
DIANALYSIS_QDRANT_COLLECTION=dianalysis_products
DIANALYSIS_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Cloud Setup

Use your cluster endpoint and API key:

```bash
DIANALYSIS_RETRIEVAL_BACKEND=qdrant
QDRANT_URL=https://<your-qdrant-cloud-endpoint>
QDRANT_API_KEY=<your-qdrant-api-key>
DIANALYSIS_QDRANT_COLLECTION=dianalysis_products
DIANALYSIS_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Local Secret Storage

This repo includes:

- `.env.example` (template, safe to commit)
- `.env` (ignored by git, store real secrets here)

Quick start:

```bash
cp .env.example .env
```

Then edit `.env` with your real values.

## Verify Connection

Check URL, auth, and collection:

```bash
make qdrant-check
```

Expected result:

- `Connection OK`
- target collection exists

## Collection Dimensions and Index Sync

For the default embed model (`sentence-transformers/all-MiniLM-L6-v2`), the vector size must be **384**.

You do not need to enter dimensions manually in Qdrant UI if you use the scripts below.  
The app computes embedding size from the model and creates the collection with the correct dimension.

Create/recreate collection with correct dimensions:

```bash
set -a; . ./.env; set +a
PYTHONPATH=. python3 experiments/build_qdrant_index.py --data-path data/products_off_clean.csv --recreate
```

Docker equivalent:

```bash
set -a; . ./.env; set +a
docker compose run --rm --no-deps \
  -e QDRANT_URL -e QDRANT_API_KEY -e DIANALYSIS_QDRANT_COLLECTION \
  -e DIANALYSIS_EMBED_MODEL -e DIANALYSIS_RETRIEVAL_BACKEND=qdrant \
  ops python experiments/build_qdrant_index.py --data-path data/products_off_clean.csv --recreate
```

Recreate index from scored candidates:

```bash
python experiments/rescore_candidates.py --config configs/base.toml --qdrant-mode recreate
```

If local Python has dependency mismatch (for example Torch/NumPy), use Docker:

```bash
docker compose run --rm --no-deps \
  -e QDRANT_URL -e QDRANT_API_KEY -e DIANALYSIS_QDRANT_COLLECTION \
  -e DIANALYSIS_EMBED_MODEL -e DIANALYSIS_RETRIEVAL_BACKEND=qdrant \
  ops python experiments/rescore_candidates.py --config configs/base.toml --qdrant-mode recreate
```

## Live Retrieval Smoke Test

Run saved live test (uses `.env` values via Docker):

```bash
make test-live-qdrant-docker
```

This checks:

- Qdrant connection + collection
- Barcode scoring returns alternatives from indexed data

## Performance Tips

If retrieval feels slow, these usually help:

- Keep the app process warm (`make app-dev`) so model and clients stay cached.
- Keep startup warm-up enabled (`DIANALYSIS_WARMUP_RETRIEVAL=1`) to reduce first-query cold-start lag.
- Lower candidate fan-out for faster response:
  - `DIANALYSIS_RETRIEVAL_CANDIDATE_LIMIT=24`
  - `DIANALYSIS_RETRIEVAL_TOPUP_MULTIPLIER=1`
- Use the same `DIANALYSIS_EMBED_MODEL` for indexing and runtime queries.
