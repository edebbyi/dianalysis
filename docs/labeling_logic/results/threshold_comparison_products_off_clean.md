# Threshold Comparison Results (Reproducible Run)

Generated (UTC): `2026-04-21T00:13:08.794730+00:00`

Dataset: `data/products_off_clean.csv` (`n=431`)

Dataset SHA256: `ec8862f2c92d4184a36e5093696159eff486e980d12247767ccf6c4e60f8400b`

Re-run commands:
- `make threshold-compare`
- `python experiments/threshold_comparison.py --input-csv data/products_off_clean.csv --output-json docs/labeling_logic/results/threshold_comparison_products_off_clean.json --output-md docs/labeling_logic/results/threshold_comparison_products_off_clean.md`

Note: rows labeled as "same thresholds, stricter trigger" use identical nutrient rules,
but raise the positive cutoff from `>= 2` to `>= 3`.

| Rule set | Positive trigger | Positive label rate | Positive count | Total |
|---|---:|---:|---:|---:|
| Legacy rules (`net>20,+2`; `add>=8,+2`; `sod>=500,+1`; `fiber>=5,-2`; `protein>=12,-1`) | `>= 2` | 41.30% | 178 | 431 |
| Balanced rules (`net>=25,+1`; `add>=10,+2`; `sod>=460,+1`; `fiber>=5.6,-2`; `protein>=10,-1`) | `>= 2` | 8.35% | 36 | 431 |
| Balanced rules (same thresholds, stricter trigger) | `>= 3` | 3.94% | 17 | 431 |
| Current adopted rules (`carbs>=30,+2`; `add>=10,+2`; `sod>=460,+1`; `fiber>=5.6,-2`; `protein>=10,-1`) | `>= 2` | 24.59% | 106 | 431 |
| Current adopted rules (same thresholds, stricter trigger) | `>= 3` | 5.57% | 24 | 431 |
