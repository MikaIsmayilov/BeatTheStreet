# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**BeatTheStreet** — a multi-page Streamlit web app that predicts whether a company will beat, meet, or miss analyst consensus EPS estimates ahead of earnings. Deployed at [beatthestreet.streamlit.app](https://beatthestreet.streamlit.app). GitHub: `MikaIsmayilov/BeatTheStreet`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Pull WRDS data (one-time, requires BU WRDS credentials — interactive login)
python src/wrds_pull.py

# Phase 1: Merge datasets, engineer features, join macro data from FRED
python src/feature_engineering.py

# Phase 1b: Refresh macro features only (re-pulls FRED, updates macro_monthly.csv)
python src/macro_features.py

# Phase 2: Train models (outputs 8 .joblib artifacts to models/)
python src/train_model.py

# Phase 3: Run the Streamlit app locally
streamlit run app.py
```

## Architecture

```
WRDS (cloud) ──► data/raw/*.csv ──► data/processed/features.csv ──► models/*.joblib
                                           ▲
                              FRED REST API (macro_features.py)

                                                    ┌── yfinance (live prices + financials)
Streamlit app (app.py) ──► pages/ ──► src/live_features.py ──┤
                                                    └── FRED REST API (live macro snapshot)
```

**Phase 1 — Data**
- `src/wrds_pull.py` — Pulls Compustat, I/B/E/S (`fpi='6'` = next fiscal quarter), CRSP, and CCM link table via SQL. `fpi='6'` must not change to `'1'` (annual).
- `src/feature_engineering.py` — Joins the four WRDS tables and calls `macro_features.py`. Key joins: Compustat→CRSP via CCM (`linktype IN ('LU','LC')`), Compustat→I/B/E/S via `merge_asof` (±15-day tolerance on ticker+date), macro joined on the month *prior* to `rdq` using `(rdq.to_period('M') - 1).to_timestamp('M')`.
- `src/macro_features.py` — Pulls 6 FRED series via direct CSV endpoint (`https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}`) using `requests`. No API key needed. Resamples to month-end; GDP forward-filled quarterly→monthly.

**Phase 2 — Models**

Time split: train 2005–2019 | val 2020–2021 (early stopping) | test 2022–2024.

`src/train_model.py` fits a preprocessing pipeline on train only: winsorize (1st/99th pct) → median impute → StandardScaler (LR only). Saves 8 artifacts to `models/`: `lightgbm_model.joblib`, `logistic_regression.joblib`, `imputer.joblib`, `scaler.joblib`, `win_low.joblib`, `win_high.joblib`, `label_encoder.joblib` (beat=0, meet=1, miss=2), `feature_cols.joblib`.

**Phase 3 — Streamlit App**

- `app.py` — Navigation shell only. Sets `page_config`, calls `inject_sidebar()` once (so sidebar CSS/logo applies to every page), then runs `st.navigation()`. No page content lives here.
- `src/ui.py` — `inject_sidebar()` injects global CSS (font size bump to 18px, sidebar reordering so logo appears above nav links, larger nav pill sizing) and embeds the nav icon as base64 HTML. Also renders a fixed GitHub link at the sidebar bottom.
- `pages/0_Home.py` — Landing page: logo, key stats, "How it works", navigation guide.
- `pages/1_Earnings_Predictor.py` — Core page. Ticker input → `live_features.py` → LightGBM → Beat/Meet/Miss prediction + probability bars + interactive candlestick chart + SHAP waterfall.
- `pages/2_Earnings_Calendar.py` — Upcoming earnings for a curated watchlist with model predictions.
- `pages/3_Backtesting.py` — Confusion matrix, quarterly accuracy, per-class F1 on 2022–2024 test set.
- `pages/4_Sector_Overview.py` — Historical beat/miss rates by sector/company.

## Feature Set (27 total)

`FEATURE_COLS` is defined in `src/live_features.py` and must exactly match what `src/train_model.py` uses. Changing one requires changing the other and retraining.

| Group | Features |
|---|---|
| I/B/E/S | `meanest`, `numest`, `est_dispersion`, `sue_lag1`, `sue_lag2` |
| Compustat | `revenue_growth`, `roa`, `accruals`, `current_ratio`, `asset_growth`, `op_margin` |
| CRSP | `ret_1m`, `ret_3m`, `ret_6m`, `vol_ratio`, `prc` |
| Macro (FRED) | `oil_1m_ret`, `oil_3m_ret`, `vix_level`, `vix_1m_chg`, `gs10_level`, `gs10_1m_chg`, `hy_spread`, `hy_spread_chg`, `gdp_growth`, `unrate`, `unrate_chg` |

`sue_lag1` / `sue_lag2` are always `NaN` at live inference (yfinance has no historical actuals). The imputer fills training-set medians. `est_revision_1m` / `est_revision_2m` exist in `features.csv` but are 100% NaN — excluded from `FEATURE_COLS`.

## Key Implementation Details

**Session state caching on the Prediction page** — All prediction results are stored in `st.session_state["pred_cache"]` after a form submit. Every rerun (e.g. clicking a chart timeframe pill) reads from cache rather than re-running the model. This prevents the page from resetting.

**SHAP version handling** — `shap.TreeExplainer.shap_values()` returns either a list of `(n_samples, n_features)` arrays (older SHAP) or a single `(n_samples, n_features, n_classes)` ndarray (SHAP ≥ 0.41). Both are handled in `compute_shap()`:
```python
if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    return shap_vals[0, :, class_idx]
return shap_vals[class_idx][0]
```

**Candlestick chart** — Always fetches `MAX_PERIOD` for the selected interval (so zoom-out has data), then uses `xaxis.range` to set the initial visible window. `rangebreaks` skips weekends; intraday intervals also skip overnight hours (16:00–09:30). The `_last_tf_{ticker}` session state key tracks the previous timeframe to auto-reset the interval pill when the timeframe changes.

**`load_explainer()` uses underscore prefix** — `def load_explainer(_lgbm_model)` — the underscore tells Streamlit's `@st.cache_resource` to skip hashing that argument (LightGBM models are not hashable).

**Ticker resolution** — `resolve_ticker()` in `pages/1_Earnings_Predictor.py`: `$NVDA` or all-uppercase ≤5-char no-space inputs are used directly as tickers; everything else goes through `yf.Search()` to resolve company names. Only US equity exchanges are preferred.

**Macro fetch at inference** — `src/live_features.py` calls `build_macro_monthly()` live (pulls FRED via `requests`). This is the same function used during training, ensuring consistency. The most recent non-null value per column is used since some FRED series lag by 1–2 months.

## Data Notes

- `data/raw/` is gitignored (WRDS licensed data).
- `data/processed/features.csv` — 104,938 rows × 36 columns. Beat 47.8% / Meet 24.8% / Miss 27.4%.
- Label thresholds: beat ≥ +$0.02 vs. mean estimate, miss ≤ -$0.02, meet in between.
- `models/` artifacts are committed to the repo (required for Streamlit Cloud; WRDS pulls are not possible at runtime on cloud).

## Assets

- `assets/beatthestreet_logo.png` — horizontal logo, used on the Home page (780px wide) and in README.
- `assets/beatthestreet_nav_icon.png` — square icon, used in sidebar (150px) and browser tab.
- Two logos exist: one for dark mode. **A light-mode variant is still needed** (`beatthestreet_logo_light.png` / `beatthestreet_nav_icon_light.png`) — detect with `st.get_option("theme.base")` or `st_theme()` and swap accordingly.

## Pending Work

- **Light-mode logo** — Create light-mode variants of both logo assets. The app currently renders dark logos on light backgrounds.
- **Separate chart tab** — Move the candlestick chart out of `pages/1_Earnings_Predictor.py` into its own tab (e.g. using `st.tabs(["Prediction", "Chart"])`) to reduce clutter on the prediction page.

## Deployment

Streamlit Community Cloud from `MikaIsmayilov/BeatTheStreet` (main branch, `app.py` entrypoint). `data/raw/` is gitignored; `data/processed/` and `models/` are committed and served directly.
