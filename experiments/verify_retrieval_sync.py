"""
Verify that promoted model artifacts, scored CSV, and Qdrant index are in sync.

Why:
- Prevent stale pre-scored candidate data after promoting a new model.
- Catch artifact/index drift before release traffic hits the app.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from dianalysis.model import model_identity
from dianalysis.recommendation.vector_client import collection_name, qdrant_client, retrieval_enabled
from dianalysis.run_config import cfg_get, load_runtime_config


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    bootstrap.add_argument("--profile", type=Path, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    cfg = load_runtime_config(bootstrap_args.config, bootstrap_args.profile)

    default_artifacts = str(cfg_get(cfg, "paths", "artifacts_dir", default="artifacts"))
    default_scored = Path(str(cfg_get(cfg, "paths", "scored_csv", default="data/products_off_clean_scored.csv")))

    parser = argparse.ArgumentParser(description="Verify retrieval sync between artifacts, CSV, and Qdrant.")
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)
    parser.add_argument("--profile", type=Path, default=bootstrap_args.profile)
    parser.add_argument("--artifacts-dir", type=str, default=default_artifacts)
    parser.add_argument("--scored-csv", type=Path, default=default_scored)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--sample-points", type=int, default=500)
    parser.add_argument(
        "--require-qdrant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when retrieval backend is not qdrant or index cannot be queried.",
    )
    return parser.parse_args()


def _single_nonempty_value(series: pd.Series, label: str) -> str:
    vals = {
        str(v).strip()
        for v in series.fillna("").astype(str).tolist()
        if str(v).strip()
    }
    if len(vals) != 1:
        raise SystemExit(f"{label} expected exactly one non-empty value, found {sorted(vals)}")
    return next(iter(vals))


def verify_scored_csv(scored_csv: Path, *, expected_model_type: str, expected_fingerprint: str) -> None:
    """Assert scored CSV identity columns match promoted model identity."""
    if not scored_csv.exists():
        raise SystemExit(f"Scored CSV not found: {scored_csv}")

    cols = ["risk_prob", "risk_score", "risk_display", "model_type", "model_fingerprint"]
    df = pd.read_csv(scored_csv, usecols=lambda c: c in cols, dtype={"model_type": str, "model_fingerprint": str})
    required = {"risk_prob", "risk_score", "risk_display", "model_type", "model_fingerprint"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Scored CSV missing required columns: {sorted(missing)}")

    csv_model_type = _single_nonempty_value(df["model_type"], "model_type")
    csv_fingerprint = _single_nonempty_value(df["model_fingerprint"], "model_fingerprint")

    if csv_model_type != expected_model_type:
        raise SystemExit(
            f"Scored CSV model_type mismatch: csv={csv_model_type} artifacts={expected_model_type}"
        )
    if csv_fingerprint != expected_fingerprint:
        raise SystemExit(
            f"Scored CSV model_fingerprint mismatch: csv={csv_fingerprint} artifacts={expected_fingerprint}"
        )

    print(
        f"[ok] scored CSV matches artifacts ({len(df)} rows, model_type={csv_model_type}, "
        f"fingerprint={csv_fingerprint[:12]}...)"
    )


def verify_qdrant_payload(
    *,
    expected_model_type: str,
    expected_fingerprint: str,
    target_collection: str,
    sample_points: int,
    require_qdrant: bool,
) -> None:
    """Assert sampled Qdrant points carry the same model identity payload."""
    if not retrieval_enabled():
        msg = "Qdrant retrieval backend not enabled in this environment."
        if require_qdrant:
            raise SystemExit(msg)
        print(f"[skip] {msg}")
        return

    client = qdrant_client()
    checked = 0
    offset: Any | None = None
    page_size = 128
    try:
        while checked < sample_points:
            points, next_offset = client.scroll(
                collection_name=target_collection,
                with_payload=True,
                with_vectors=False,
                limit=page_size,
                offset=offset,
            )
            if not points:
                break
            for p in points:
                payload = getattr(p, "payload", None) or {}
                got_type = str(payload.get("model_type", "") or "").strip().lower()
                got_fp = str(payload.get("model_fingerprint", "") or "").strip()
                if got_type != expected_model_type:
                    raise SystemExit(
                        f"Qdrant payload model_type mismatch at point {p.id}: got={got_type} expected={expected_model_type}"
                    )
                if got_fp != expected_fingerprint:
                    raise SystemExit(
                        f"Qdrant payload model_fingerprint mismatch at point {p.id}: "
                        f"got={got_fp} expected={expected_fingerprint}"
                    )
                checked += 1
                if checked >= sample_points:
                    break
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        msg = f"Qdrant could not be queried: {exc}"
        if require_qdrant:
            raise SystemExit(msg) from exc
        print(f"[skip] {msg}")
        return

    if checked == 0:
        msg = f"Qdrant collection '{target_collection}' has no points to verify."
        if require_qdrant:
            raise SystemExit(msg)
        print(f"[skip] {msg}")
        return
    print(
        f"[ok] qdrant payload sample matches artifacts ({checked} points, collection={target_collection}, "
        f"fingerprint={expected_fingerprint[:12]}...)"
    )


def main() -> None:
    args = parse_args()
    cfg = load_runtime_config(args.config, args.profile)
    os.environ["DIANALYSIS_CONFIG"] = str(args.config)
    if args.profile:
        os.environ["DIANALYSIS_PROFILE"] = str(args.profile)
    else:
        os.environ.pop("DIANALYSIS_PROFILE", None)

    if "DIANALYSIS_RETRIEVAL_BACKEND" not in os.environ:
        os.environ["DIANALYSIS_RETRIEVAL_BACKEND"] = str(cfg_get(cfg, "retrieval", "backend", default="qdrant"))
    if "QDRANT_URL" not in os.environ:
        os.environ["QDRANT_URL"] = str(cfg_get(cfg, "retrieval", "qdrant_url", default="http://localhost:6333"))
    if "DIANALYSIS_QDRANT_COLLECTION" not in os.environ:
        os.environ["DIANALYSIS_QDRANT_COLLECTION"] = str(
            cfg_get(cfg, "retrieval", "qdrant_collection", default="dianalysis_products")
        )

    identity = model_identity(args.artifacts_dir)
    expected_model_type = identity["model_type"]
    expected_fingerprint = identity["model_fingerprint"]
    print(
        f"[info] artifacts identity model_type={expected_model_type} "
        f"fingerprint={expected_fingerprint[:12]}..."
    )

    verify_scored_csv(
        args.scored_csv,
        expected_model_type=expected_model_type,
        expected_fingerprint=expected_fingerprint,
    )
    verify_qdrant_payload(
        expected_model_type=expected_model_type,
        expected_fingerprint=expected_fingerprint,
        target_collection=(args.collection or collection_name()),
        sample_points=max(1, int(args.sample_points)),
        require_qdrant=bool(args.require_qdrant),
    )
    print("[ok] retrieval sync verification passed")


if __name__ == "__main__":
    main()
