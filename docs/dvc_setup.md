# DVC (Optional, Maintainer-Only)

Contributors do not need DVC to run this project.
The active dataset is already in Git at:

- `data/products_off_clean.csv`

Use DVC only if you want local dataset versioning with Backblaze B2.

## Optional Local Setup

```bash
python3 -m pip install "dvc[s3]"
dvc remote modify --local b2 access_key_id "<key_id>"
dvc remote modify --local b2 secret_access_key "<app_key>"
dvc remote modify b2 endpointurl "https://s3.us-east-005.backblazeb2.com"
```

## Optional Pull/Push

```bash
dvc pull data/products_off_clean.csv.dvc --force
dvc push
```
