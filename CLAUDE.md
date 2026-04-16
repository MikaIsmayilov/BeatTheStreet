# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Earnings Surprise Predictor** — a multi-page Streamlit web app that predicts whether a company will beat, meet, or miss analyst consensus EPS estimates ahead of earnings. Built for BA870/AC820 (Financial and Accounting Analytics, Prof. Peter Wysocki, BU Questrom Spring 2026). Demo: April 24, 2026.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Pull WRDS data (one-time, requires BU WRDS credentials)
python src/wrds_pull.py

# Phase 1: Merge datasets and engineer features
python src/feature_engineering.py

# Phase 2: Train models
python src/train_model.py

# Phase 3: Run the Streamlit app locally
streamlit run app.py
```

## Architecture

The project follows a linear pipeline:

```
WRDS (cloud) → data/raw/*.csv → data/processed/features.csv → models/*.joblib → Streamlit app
```

**Phase 1 — Data (`src/`)**
- `wrds_pull.py`: Pulls 4 datasets from WRDS via SQL and saves to `data/raw/`. Requires interactive WRDS login (`mikaismayilli`). WRDS credentials are stored in `~/.pgpass` after first run — never hardcode them.
- `feature_engineering.py`: Merges Compustat + I/B/E/S + CRSP via the CCM link table, engineers predictive features, assigns beat/meet/miss labels, and saves `data/processed/features.csv`.

**Phase 2 — Models (`src/train_model.py`)**
- Reads `features.csv`, trains logistic regression (baseline) + XGBoost/LightGBM (main), saves artifacts to `models/`.

**Phase 3 — Streamlit App**
- `app.py`: Entry point and navigation.
- `pages/1_Prediction.py`: Core demo — user enters ticker, sees beat/meet/miss probability + SHAP chart.
- `pages/2_Home.py`: Upcoming earnings calendar.
- `pages/3_Backtesting.py`: Historical model accuracy, confusion matrix.
- `pages/4_Sector_Overview.py`: Aggregate predictions by sector.

## Data Notes

- `data/raw/` is gitignored (WRDS data is licensed and large).
- `data/processed/features.csv` is the single source of truth for model training (104,938 samples; beat 47.8%, miss 27.4%, meet 24.8%).
- Label thresholds: beat ≥ +$0.02 vs. mean estimate, miss ≤ -$0.02, meet in between.
- Class imbalance is handled with `class_weight='balanced'` in models.
- WRDS live queries will NOT work on Streamlit Community Cloud — all WRDS data must remain as pre-pulled CSVs. Real-time inputs use `yfinance` and FRED API instead.

## Key Data Relationships

| Join | Keys | Notes |
|---|---|---|
| Compustat → CRSP | `gvkey` → `permno` via CCM link table | Filter `linktype IN ('LU','LC')`, validate date ranges |
| Compustat → I/B/E/S | `tic` + `datadate` ≈ `fpedats` | Use `merge_asof` with 15-day tolerance |
| CRSP → merged | `permno` + month-end date before `rdq` | Align on month prior to earnings announcement date |

## I/B/E/S Query Parameter

`fpi = '6'` = next fiscal **quarter** EPS forecast. Do not change to `'1'` (that is next fiscal year).

## Deployment

Deployed to Streamlit Community Cloud from GitHub. `data/raw/` and `models/` are gitignored — `data/processed/features.csv` and trained model files must either be committed (if small enough) or loaded from an alternative source at runtime.
