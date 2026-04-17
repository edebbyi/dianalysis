"""
One-time migration utility from MLflow file store (`./mlruns`) to SQLite backend.

Why:
- Preserve historical runs when standardizing tracking on `sqlite:///mlflow.db`.
- Avoid losing experiment lineage during local tracking backend changes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from mlflow.tracking import MlflowClient

SOURCE_URI = "file:./mlruns"
TARGET_URI = "sqlite:///mlflow.db"
MIGRATION_TAG = "migration.source_run_id"


def _get_or_create_experiment(target: MlflowClient, name: str) -> str:
    existing = target.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    return target.create_experiment(name)


def _already_migrated(target: MlflowClient, experiment_id: str, source_run_id: str) -> bool:
    runs = target.search_runs(
        [experiment_id],
        filter_string=f"tags.`{MIGRATION_TAG}` = '{source_run_id}'",
        max_results=1,
    )
    return bool(runs)


def _copy_artifacts(source: MlflowClient, target: MlflowClient, source_run_id: str, target_run_id: str) -> None:
    # Download all source artifacts to a temp dir, then upload as-is to target run.
    with tempfile.TemporaryDirectory() as tmp:
        local_root = Path(tmp)
        source.download_artifacts(source_run_id, "", str(local_root))
        for p in local_root.rglob("*"):
            if p.is_file():
                rel_parent = p.relative_to(local_root).parent
                artifact_path = "" if str(rel_parent) == "." else str(rel_parent)
                target.log_artifact(target_run_id, str(p), artifact_path=artifact_path)


def migrate() -> None:
    source = MlflowClient(tracking_uri=SOURCE_URI)
    target = MlflowClient(tracking_uri=TARGET_URI)

    migrated_runs = 0
    skipped_runs = 0

    for exp in source.search_experiments(view_type=1):  # active only
        source_runs = source.search_runs([exp.experiment_id], max_results=50000)
        if not source_runs:
            continue

        target_exp_id = _get_or_create_experiment(target, exp.name)

        for run in source_runs:
            source_run_id = run.info.run_id
            if _already_migrated(target, target_exp_id, source_run_id):
                skipped_runs += 1
                continue

            tags = dict(run.data.tags)
            tags[MIGRATION_TAG] = source_run_id
            tags["migration.source_tracking_uri"] = SOURCE_URI
            if "mlflow.runName" not in tags:
                tags["mlflow.runName"] = run.info.run_name or source_run_id

            target_run = target.create_run(
                experiment_id=target_exp_id,
                tags=tags,
                start_time=run.info.start_time,
            )
            target_run_id = target_run.info.run_id

            for key, value in run.data.params.items():
                target.log_param(target_run_id, key, value)

            for key in run.data.metrics.keys():
                for m in source.get_metric_history(source_run_id, key):
                    target.log_metric(
                        target_run_id,
                        key,
                        m.value,
                        timestamp=m.timestamp,
                        step=m.step,
                    )

            _copy_artifacts(source, target, source_run_id, target_run_id)
            target.set_terminated(
                target_run_id,
                status=run.info.status,
                end_time=run.info.end_time,
            )
            migrated_runs += 1

    print(f"Migrated runs: {migrated_runs}")
    print(f"Skipped runs (already migrated): {skipped_runs}")
    print(f"Source URI: {SOURCE_URI}")
    print(f"Target URI: {TARGET_URI}")


if __name__ == "__main__":
    migrate()
