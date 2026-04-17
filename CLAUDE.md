# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Earnings Surprise Predictor** — a multi-page Streamlit web app that predicts whether a company will beat, meet, or miss analyst consensus EPS estimates ahead of earnings. Built for BA870/AC820 (Financial and Accounting Analytics, Prof. Peter Wysocki, BU Questrom Spring 2026). Demo: April 24, 2026.

## Common Commands

```bash
# Install dependencies (also requires: pip install setuptools to fix distutils on Python 3.13)
pip install -r requirements.txt

# Phase 1: Pull WRDS data (one-time, requires BU WRDS credentials — interactive login)
python src/wrds_pull.py

# Phase 1: Merge datasets, engineer features, and join macro data from FRED
python src/feature_engineering.py

# Phase 1b: Refresh macro features only (re-pulls FRED, updates macro_monthly.csv)
python src/macro_features.py

# Phase 2: Train models (outputs 8 .joblib artifacts to models/)
python src/train_model.py

# Phase 3: Run the Streamlit app locally
streamlit run app.py
```

## Architecture

The project follows a linear pipeline:

```
WRDS (cloud) ──► data/raw/*.csv ──► data/processed/features.csv ──► models/*.joblib ──► Streamlit app
                                           ▲
                              FRED API (macro_features.py)
```

**Phase 1 — Data (`src/`)**
- `wrds_pull.py`: Pulls 4 datasets from WRDS via SQL → `data/raw/`. Requires interactive WRDS login (`mikaismayilli`). Credentials stored in `~/.pgpass` after first run.
- `feature_engineering.py`: Merges Compustat + I/B/E/S + CRSP via CCM link table, engineers features, assigns beat/meet/miss labels, calls `macro_features.py` to join FRED data, saves `data/processed/features.csv`.
- `macro_features.py`: Pulls 6 FRED series (WTI oil, VIX, 10yr yield, HY spread, GDP growth, unemployment), engineers 11 monthly macro features, saves `data/processed/macro_monthly.csv`. Can run standalone to refresh macro data.

**Phase 2 — Models (`src/train_model.py`)**

Time splits: train 2005–2019 | val 2020–2021 (early stopping) | test 2022–2024.

Preprocessing pipeline fitted on train only: winsorize (1st/99th pct) → median impute → StandardScaler (LR only). Saves 8 artifacts to `models/`:
- `lightgbm_model.joblib`, `logistic_regression.joblib`
- `imputer.joblib`, `scaler.joblib`, `win_low.joblib`, `win_high.joblib`
- `label_encoder.joblib` (beat=0, meet=1, miss=2), `feature_cols.joblib`

**Phase 3 — Streamlit App**
- `app.py`: Landing page with key stats and navigation.
- `pages/1_Prediction.py`: Core demo — ticker input → `live_features.py` fetches 25/27 features live → LightGBM prediction → SHAP waterfall chart.
- `pages/2_Home.py`: Upcoming earnings calendar for a curated watchlist, with model predictions per company.
- `pages/3_Backtesting.py`: Confusion matrix, quarterly accuracy over time, per-class F1 on the 2022–2024 test set.
- `pages/4_Sector_Overview.py`: Historical beat/miss rates by company, accuracy breakdown by label.
- `src/live_features.py`: Fetches live features for a ticker at inference time. yfinance → price/momentum + financials + analyst estimates; FRED → macro snapshot. Returns 25/27 features (sue_lag1/2 unavailable from yfinance; imputer fills medians).

## Feature Set (27 total)

| Group | Features |
|---|---|
| I/B/E/S | `meanest`, `numest`, `est_dispersion`, `sue_lag1`, `sue_lag2` |
| Compustat | `revenue_growth`, `roa`, `accruals`, `current_ratio`, `asset_growth`, `op_margin` |
| CRSP | `ret_1m`, `ret_3m`, `ret_6m`, `vol_ratio`, `prc` |
| Macro (FRED) | `oil_1m_ret`, `oil_3m_ret`, `vix_level`, `vix_1m_chg`, `gs10_level`, `gs10_1m_chg`, `hy_spread`, `hy_spread_chg`, `gdp_growth`, `unrate`, `unrate_chg` |

`est_revision_1m` and `est_revision_2m` exist in `features.csv` but are 100% NaN — excluded from `FEATURE_COLS` in both `train_model.py` and `live_features.py`.

**`FEATURE_COLS` must be identical in `train_model.py` and `live_features.py`.** Changing one requires changing the other and retraining.

## Data Notes

- `data/raw/` is gitignored (WRDS data is licensed).
- `data/processed/features.csv`: 104,938 samples × 36 columns (25 features + 11 macro + identifiers + label). Beat 47.8%, miss 27.4%, meet 24.8%.
- `data/processed/macro_monthly.csv`: 268 months (2004–2026) × 11 macro features. Refreshed by running `python src/macro_features.py`.
- Label thresholds: beat ≥ +$0.02 vs. mean estimate, miss ≤ -$0.02, meet in between.
- WRDS live queries will NOT work on Streamlit Community Cloud — all WRDS data stays as pre-pulled CSVs. Live inputs use `yfinance` and FRED only.

## Key Data Relationships

| Join | Keys | Notes |
|---|---|---|
| Compustat → CRSP | `gvkey` → `permno` via CCM link table | Filter `linktype IN ('LU','LC')`, validate date ranges |
| Compustat → I/B/E/S | `tic` + `datadate` ≈ `fpedats` | `merge_asof` with 15-day tolerance |
| CRSP → merged | `permno` + month-end date before `rdq` | Align on month prior to earnings announcement date |
| Macro → merged | month prior to `rdq` | `(rdq.to_period('M') - 1).to_timestamp('M')` |

## I/B/E/S Query Parameter

`fpi = '6'` = next fiscal **quarter** EPS forecast. Do not change to `'1'` (that is next fiscal year).

## Python / Dependency Notes

- `pandas_datareader` requires `setuptools` on Python 3.13 (`distutils` was removed). Run `pip install setuptools` if FRED pulls fail with `ModuleNotFoundError: No module named 'distutils'`.
- LightGBM requires `libomp` on macOS: `brew install libomp`.

## Deployment

Target: Streamlit Community Cloud from GitHub (`MikaIsmayilov/BeatTheStreet`). `data/raw/` and `models/` are gitignored — `data/processed/features.csv`, `data/processed/macro_monthly.csv`, and all `models/*.joblib` files must be committed or provided via an alternative source at runtime.
