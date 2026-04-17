# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**BeatTheStreet** — a multi-page Streamlit web app (BA870/AC820, BU Questrom, Spring 2026) that predicts whether a company will **beat**, **meet**, or **miss** analyst consensus EPS estimates ahead of earnings. Live at [beatthestreet.streamlit.app](https://beatthestreet.streamlit.app) · GitHub: `MikaIsmayilov/BeatTheStreet`.

---

## Commands

```bash
pip install -r requirements.txt

# Phase 1 — data (one-time, BU WRDS credentials required — interactive login)
python src/wrds_pull.py          # pull Compustat, I/B/E/S, CRSP, CCM from WRDS
python src/feature_engineering.py  # merge tables + compute features → data/processed/features.csv
python src/macro_features.py     # refresh FRED macro data only

# Phase 2 — train
python src/train_model.py        # outputs 8 .joblib artifacts to models/

# Phase 3 — run locally
streamlit run app.py
```

---

## Architecture

```
WRDS (SQL) ──► data/raw/*.csv ──► feature_engineering.py ──► data/processed/features.csv
                                            ▲
                                 FRED REST API (macro_features.py)
                                            │
                                            ▼
                                 train_model.py ──► models/*.joblib

Streamlit (app.py)
  └─► pages/
        ├── 0_Home.py
        ├── 1_Chart.py
        ├── 1_Earnings_Predictor.py ──► src/live_features.py ──► yfinance + FRED
        ├── 2_Earnings_Calendar.py  ──► src/live_features.py
        ├── 3_Backtesting.py        ──► data/processed/features.csv + models/
        └── 4_Sector_Overview.py    ──► data/processed/features.csv + models/
```

---

## Phase 1 — Data Pipeline

**`src/wrds_pull.py`**  
Connects to WRDS via the `wrds` Python library. Username `mikaismayilli` is hardcoded in `connect()`; password is prompted interactively on first run and cached in `~/.pgpass` — never stored in the repo. Pulls four tables via SQL and saves to `data/raw/`:
- `compustat_quarterly.csv` — quarterly financials (`comp.fundq`)
- `ibes_summary.csv` — analyst EPS estimates (`ibes.statsum_epsus`, `fpi='6'` = next fiscal quarter — do **not** change to `'1'`)
- `crsp_monthly.csv` — monthly stock returns (`crsp.msf`)
- `ccm_links.csv` — CRSP–Compustat link table (`crsp.ccmxpf_linktable`)

**`src/feature_engineering.py`**  
Merges the four WRDS tables and appends macro data:
- Compustat → CRSP via CCM (`linktype IN ('LU','LC')`, `linkprim IN ('P','C')`)
- Compustat → I/B/E/S via `pd.merge_asof` on ticker + date (±15-day tolerance)
- Macro joined on the month *prior* to earnings date: `(rdq.to_period('M') - 1).to_timestamp('M')`
- Output: `data/processed/features.csv` — 104,938 rows × 36 columns

**`src/macro_features.py`**  
Fetches 6 FRED series via direct CSV endpoint (no API key): oil price, VIX, 10-yr Treasury, HY credit spread, GDP, unemployment. Resamples to month-end; GDP forward-filled quarterly → monthly. Called both at training time (via `feature_engineering.py`) and at live inference (via `live_features.py`) to ensure consistency.

---

## Phase 2 — Model Training (`src/train_model.py`)

**Time splits** (strict, no lookahead):
- Train: 2005–2019
- Validation: 2020–2021 (LightGBM early stopping only)
- Test: 2022–2024 (never touched during training)

**Preprocessing pipeline** (fitted on train only, applied identically at inference):
1. Replace ±inf → NaN
2. Winsorize at 1st/99th percentile (`win_low.joblib`, `win_high.joblib`)
3. Median imputation (`imputer.joblib`)
4. StandardScaler for Logistic Regression only (`scaler.joblib`)

**Models trained:**
- Logistic Regression (baseline, `solver='saga'`, `class_weight='balanced'`, `C=0.5`)
- LightGBM (main model, `objective='multiclass'`, 2000 tree ceiling, early stopping at 75 rounds with no improvement on val)

**Label encoding** (`label_encoder.joblib`): beat=0, meet=1, miss=2  
**Test accuracy**: LightGBM 60.9% vs. 33% random baseline

**Saved artifacts** (`models/`):
```
lightgbm_model.joblib   logistic_regression.joblib
imputer.joblib          scaler.joblib
win_low.joblib          win_high.joblib
label_encoder.joblib    feature_cols.joblib
```
These are committed to the repo — Streamlit Cloud cannot run WRDS pulls at runtime.

---

## Phase 3 — Streamlit App

### Navigation & Shared UI

**`app.py`** — Navigation shell only. Calls `inject_sidebar()` once so CSS and sidebar logo apply globally, then runs `st.navigation()`. Nav order: Home → Price Chart → Earnings Predictor → Earnings Calendar → Sector Overview → Backtesting.

**`src/ui.py`** — `inject_sidebar()`: injects CSS (18px global font, sidebar logo above nav links, larger pill sizing), renders the `beatthestreet_nav_icon.png` as base64, and pins a GitHub link to the sidebar bottom.

**`.streamlit/config.toml`** — Forces dark mode for all users (`base = "dark"`).

### Pages

**`pages/0_Home.py`** — Logo, key stats, "How it works", navigation links. Logo: `st.image("assets/beatthestreet_logo_dark.png", width=780)`.

**`pages/1_Chart.py`** — Standalone price chart. Ticker input pre-fills from `st.session_state["pred_cache"]` if the user came from the Predictor. Features: timeframe pills (1D–All), interval pills (5m–1W) with auto-default per timeframe, measure tool (date range → $ and % change), draw tools (line, path, circle, rect, erase) via Plotly modebar, weekend rangebreaks + overnight hour skipping for intraday.

**`pages/1_Earnings_Predictor.py`** — Core prediction page:
1. Ticker input → `resolve_ticker()` (supports `$NVDA`, `NVDA`, or plain-English names via `yf.Search()`)
2. `fetch_live_features()` → 27 features
3. Winsorize → impute → LightGBM → `predict_proba()`
4. Prediction box (color-coded Beat/Meet/Miss), probability bar chart, SHAP waterfall
5. Expandable feature-value table
All results stored in `st.session_state["pred_cache"]` so reruns (timeframe changes, etc.) do not re-invoke the model.

**`pages/2_Earnings_Calendar.py`** — Curated 23-ticker watchlist. Pulls next earnings date via `yf.Ticker.calendar`, runs model predictions, groups by week, renders colored cards.

**`pages/3_Backtesting.py`** — Runs LightGBM + Logistic Regression on 2022–2024 test set from `features.csv`. Three tabs: confusion matrix (row-normalized), quarterly accuracy line chart, per-class precision/recall/F1 table + bar chart.

**`pages/4_Sector_Overview.py`** — Historical beat/miss/meet rates by year (full dataset), top consistent beaters/missers by company (min 20 quarters), per-class model accuracy on test set, label distribution pie.

### Live Feature Construction (`src/live_features.py`)

Called at inference for each ticker. Builds the same 27-feature vector used during training:

| Group | Source | Method |
|---|---|---|
| Price / momentum | yfinance `.history()` | 1/3/6-month returns, volume ratio, price |
| Analyst estimates | yfinance `.earnings_estimate` or `.info` | Mean EPS estimate, analyst count, dispersion |
| Quarterly financials | yfinance `.quarterly_financials`, `.quarterly_balance_sheet`, `.quarterly_cashflow` | Revenue growth, ROA, op margin, accruals, current ratio, asset growth |
| Macro | FRED via `build_macro_monthly()` | Most recent non-null value per series (some lag 1–2 months) |
| SUE lags | Not available via yfinance | Always NaN — imputer fills training-set median |

---

## Feature Set (27 total)

`FEATURE_COLS` is defined in both `src/live_features.py` and `src/train_model.py` and must be identical. Any change requires retraining.

| Group | Features |
|---|---|
| I/B/E/S | `meanest`, `numest`, `est_dispersion`, `sue_lag1`, `sue_lag2` |
| Compustat | `revenue_growth`, `roa`, `accruals`, `current_ratio`, `asset_growth`, `op_margin` |
| CRSP | `ret_1m`, `ret_3m`, `ret_6m`, `vol_ratio`, `prc` |
| Macro (FRED) | `oil_1m_ret`, `oil_3m_ret`, `vix_level`, `vix_1m_chg`, `gs10_level`, `gs10_1m_chg`, `hy_spread`, `hy_spread_chg`, `gdp_growth`, `unrate`, `unrate_chg` |

`est_revision_1m` / `est_revision_2m` exist in `features.csv` but are 100% NaN — excluded from `FEATURE_COLS`.

---

## Key Implementation Details

**Session-state prediction cache** — `st.session_state["pred_cache"]` stores all prediction outputs after a form submit. Every subsequent rerun reads from cache instead of re-running the model, so interacting with the chart or any widget doesn't reset the prediction.

**SHAP version compatibility** — `shap.TreeExplainer.shap_values()` returns a list of 2D arrays (SHAP < 0.41) or a single 3D ndarray (SHAP ≥ 0.41). `compute_shap()` handles both:
```python
if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    return shap_vals[0, :, class_idx]
return shap_vals[class_idx][0]
```

**`load_explainer()` underscore prefix** — `def load_explainer(_lgbm_model)` — underscore tells `@st.cache_resource` to skip hashing the argument (LightGBM models are not hashable by Streamlit).

**Chart interval auto-reset** — `st.session_state[f"_last_tf_{ticker}"]` tracks the previous timeframe. When it changes, the interval pill is reset to the sensible default for that timeframe (e.g. 1M → 1D, 1D → 5m).

**Macro at inference** — `src/live_features.py` calls `build_macro_monthly()` live at prediction time, which re-pulls FRED. This guarantees training-inference feature consistency. Most recent non-null value is used because some series (GDP, HY spread) lag by 1–2 months.

---

## Data & Assets

- `data/raw/` — gitignored (WRDS licensed). Must be re-pulled locally with `src/wrds_pull.py`.
- `data/processed/features.csv` — 104,938 rows × 36 columns. Label split: Beat 47.8% / Meet 24.8% / Miss 27.4%. Label thresholds: beat ≥ +$0.02 vs. mean estimate; miss ≤ −$0.02; meet in between.
- `models/` — committed to repo (8 `.joblib` files). Required for Streamlit Cloud.
- `assets/beatthestreet_logo_dark.png` — used on Home page (780px wide).
- `assets/beatthestreet_logo_light.png` — kept for future use if theme policy changes.
- `assets/beatthestreet_nav_icon.png` — used in sidebar (150px) and browser tab favicon.

---

## Deployment

Streamlit Community Cloud from `MikaIsmayilov/BeatTheStreet` (main branch, `app.py` entrypoint). Every push to `main` triggers an automatic redeploy. `data/raw/` is gitignored; `data/processed/` and `models/` are committed and served directly — WRDS is not accessible at runtime on Streamlit Cloud.
