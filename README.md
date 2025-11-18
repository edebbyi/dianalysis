# 🍎 Dianalysis

Calibrated diabetes-risk scoring for packaged foods, combining nutrition-based rules with a logistic regression model and curated alternatives.

## Installation

```bash
git clone https://github.com/edebbyi/dianalysis.git
cd dianalysis

python -m venv venv
source venv/bin/activate  # Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```

## Run

### Train the model
```bash
python train.py
```
Generates synthetic data, trains the calibrated `LogisticRegression`, and writes artifacts to `artifacts/`.

### Start the demo UI
```bash
streamlit run app.py
```
Serves an interactive UI on `http://localhost:8501` for scoring items and browsing alternatives.

## Configuration

- `artifacts/`: saved model plus metadata consumed by `score_item()` and `score_by_barcode()`.
- `data/products_off_clean.csv`: purified [Open Food Facts](https://world.openfoodfacts.org) catalog used for alternatives; refresh via `dianalysis/off_pipeline.py`.
- `dianalysis/scoring.py`: customize `ALT_GROUP_PRIORITIES`, risk thresholds, or alternative ranking rules here.
- Streamlit respects `STREAMLIT_SERVER_PORT` / `STREAMLIT_SERVER_HEADLESS` environment overrides when running `app.py`.

## Manual / Wiki

See the `docs/` directory or the GitHub wiki for deeper architecture notes, data-pipeline details, and troubleshooting guidance.

## Contributions

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-topic`).
3. Commit your changes and push the branch.
4. Open a pull request with a concise summary.

## Author

Deborah Imafidon — https://www.kaggle.com/code/deborahimafidon/dianalysis

## License

MIT
