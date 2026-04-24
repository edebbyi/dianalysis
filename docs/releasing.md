# Releasing

`main` is the live development branch.  
Releases are snapshots of `main` saved as Git tags.

## Versioning Plan (Keep v1.0.0, move main to v1.1.0)

1. Tag the old stable commit as `v1.0.0` (this preserves it forever).
2. Merge new work into `main`.
3. Bump `setup.py` to `1.1.0`.
4. Tag updated `main` as `v1.1.0`.

Important:
- You do not rename branches for releases.
- `main` keeps moving forward.
- Tags (`v1.0.0`, `v1.1.0`) are your fixed release points.

## Commands

Tag current stable state as `v1.0.0`:

```bash
git checkout main
git pull
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

After v1.1.0 changes are merged to `main`, tag `v1.1.0`:

```bash
git checkout main
git pull
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0
```

Optional maintenance branch for hotfixes on `v1.0.x`:

```bash
git checkout -b release/1.0 v1.0.0
git push -u origin release/1.0
```

## Quality Gate Retrieval Metrics (v1.1.0)

For v1.1.0, retrieval gate metrics were updated so "no alternatives found" is only counted as a miss when lower-risk options actually exist.

- Coverage gate metric:
  - Old: `coverage_with_alternatives`
  - New: `coverage_given_eligible`
- Ranking gate metric:
  - Old: `ndcg_at_k_mean`
  - New: `ndcg_given_non_empty`

Current thresholds in `configs/base.toml`:
- `min_ndcg = 0.80`
- `min_coverage = 0.80`
