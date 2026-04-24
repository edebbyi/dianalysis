# Contributing

## Workflow

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes and run relevant checks locally.
4. Commit with clear messages.
5. Push your branch to your fork.
6. Open a pull request to `main`.

## Recommended Local Checks

```bash
ruff check dianalysis experiments train.py app.py
python -m unittest discover -s tests -p "test_*.py"
```

If your change affects model behavior, also run:

```bash
python experiments/model_quality_gate.py --config configs/base.toml
```
