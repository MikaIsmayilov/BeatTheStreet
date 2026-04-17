# BeatTheStreet

![BeatTheStreet](assets/beatthestreet_logo_dark.png)

### [beatthestreet.streamlit.app](https://beatthestreet.streamlit.app/)

An earnings surprise predictor that uses machine learning to forecast whether a company will **beat**, **meet**, or **miss** Wall Street's EPS consensus estimate — ahead of the announcement.

---

## Overview

Analysts and portfolio managers spend enormous effort forecasting whether a company will beat or miss earnings. BeatTheStreet treats this as a supervised classification problem: given a snapshot of analyst sentiment, company fundamentals, price momentum, and macroeconomic conditions available *before* the announcement, can a model predict the outcome better than chance?

**Answer: yes — 60.9% accuracy on held-out 2022–2024 data, versus a 47.8% naive baseline (always predicting "beat").**

---

## Data Sources

| Source | Access | Contents |
|---|---|---|
| **Compustat** (via WRDS) | Institutional license | Quarterly financials: revenue, operating income, total assets, current assets/liabilities, accruals |
| **I/B/E/S** (via WRDS) | Institutional license | Analyst EPS estimates: mean consensus, number of analysts, dispersion, prior-quarter SUE |
| **CRSP** (via WRDS) | Institutional license | Daily stock returns and prices |
| **FRED** (St. Louis Fed) | Public API | Macro series: WTI oil, VIX, 10-yr Treasury yield, HY credit spread, real GDP growth, unemployment |

**Training set size:** 104,938 earnings observations spanning 2005–2024.

**Label definition:**
- **Beat** — actual EPS ≥ consensus + $0.02 (47.8% of sample)
- **Meet** — within ±$0.02 of consensus (24.8%)
- **Miss** — actual EPS ≤ consensus − $0.02 (27.4%)

---

## Pipeline

```
WRDS (cloud)
  ├── Compustat  ─┐
  ├── I/B/E/S    ─┼─► feature_engineering.py ─► data/processed/features.csv ─► train_model.py ─► models/*.joblib
  ├── CRSP       ─┘            ▲
  └── CCM link                 │
                         FRED REST API
                         (macro_features.py)
                                              Streamlit app (app.py + pages/)
                                                     ▲           ▲
                                               yfinance      FRED API
                                            (live prices)  (live macro)
```

**Phase 1 — Data Collection**

`src/wrds_pull.py` queries four WRDS datasets via SQL and saves them to `data/raw/`:
- Compustat quarterly fundamentals
- I/B/E/S summary-level analyst estimates (`fpi='6'` for next fiscal quarter)
- CRSP monthly returns and prices
- CCM link table for Compustat–CRSP identifier mapping

**Phase 1b — Macro Features**

`src/macro_features.py` pulls six FRED series via their public CSV endpoint and engineers 11 monthly macro features. Each feature is computed so that at any given month, only data available through that month is used — no lookahead.

**Phase 2 — Feature Engineering**

`src/feature_engineering.py` joins the four WRDS tables, engineers company-level features, and left-joins macro features using the month *prior* to the earnings announcement date (`rdq`) to maintain strict temporal integrity.

Key joins:
| Join | Method |
|---|---|
| Compustat → CRSP | `gvkey` → `permno` via CCM link table (`linktype IN ('LU','LC')`) |
| Compustat → I/B/E/S | `merge_asof` on ticker + date (±15-day tolerance) |
| CRSP → merged | Month-end return aligned to month before `rdq` |
| Macro → merged | Month prior to `rdq`: `(rdq.to_period('M') − 1).to_timestamp('M')` |

**Phase 3 — Model Training**

`src/train_model.py` trains on a strict time-based split:

| Split | Period | Samples |
|---|---|---|
| Train | 2005–2019 | ~82,000 |
| Validation | 2020–2021 | Early stopping only |
| Test | 2022–2024 | ~14,000 (never seen during training) |

Preprocessing pipeline (fitted on train only, applied consistently to val/test and live inference):
1. **Winsorize** at 1st/99th percentile — clips extreme outliers without dropping rows
2. **Median imputation** — fills missing values with training-set medians
3. **StandardScaler** — applied for Logistic Regression only; tree models skip this step

Two models are trained and saved:
- **LightGBM** (gradient-boosted trees) — primary model, 60.9% test accuracy
- **Logistic Regression** — baseline for comparison

Eight `.joblib` artifacts are saved to `models/`: the two models, imputer, scaler, winsorization bounds, label encoder, and feature column list.

---

## Feature Set (27 total)

| Group | Feature | Description |
|---|---|---|
| **Analyst (I/B/E/S)** | `meanest` | Mean analyst EPS estimate |
| | `numest` | Number of analysts covering the stock |
| | `est_dispersion` | Standard deviation of estimates (disagreement) |
| | `sue_lag1` | Standardized unexpected earnings, 1 quarter ago |
| | `sue_lag2` | Standardized unexpected earnings, 2 quarters ago |
| **Fundamentals (Compustat)** | `revenue_growth` | YoY revenue growth |
| | `roa` | Return on assets |
| | `accruals` | Accruals ratio (earnings quality signal) |
| | `current_ratio` | Current assets / current liabilities |
| | `asset_growth` | YoY total asset growth |
| | `op_margin` | Operating income / revenue |
| **Price/Momentum (CRSP)** | `ret_1m` | 1-month stock return |
| | `ret_3m` | 3-month stock return |
| | `ret_6m` | 6-month stock return |
| | `vol_ratio` | Recent vs. historical volume ratio |
| | `prc` | Stock price level |
| **Macro (FRED)** | `oil_1m_ret` | WTI crude oil 1-month return |
| | `oil_3m_ret` | WTI crude oil 3-month return |
| | `vix_level` | CBOE VIX level |
| | `vix_1m_chg` | VIX 1-month change |
| | `gs10_level` | 10-year Treasury yield |
| | `gs10_1m_chg` | 10-year yield 1-month change |
| | `hy_spread` | ICE BofA HY credit spread |
| | `hy_spread_chg` | HY spread 1-month change |
| | `gdp_growth` | Real GDP QoQ growth rate |
| | `unrate` | Unemployment rate |
| | `unrate_chg` | Unemployment 1-month change |

---

## Explainability — SHAP

The app uses **SHAP (SHapley Additive exPlanations)** to decompose each prediction into the contribution of each individual feature. For any given ticker, a waterfall chart shows exactly which factors pushed the model toward beat, meet, or miss — and by how much.

This makes the model auditable: rather than a black-box probability, users can see whether the prediction is driven by strong analyst consensus, a recent price surge, a high VIX environment, or something else entirely.

---

## Live Inference

At prediction time, `src/live_features.py` fetches a real-time feature snapshot for any ticker:
- **yfinance** — current price, 1/3/6-month returns, volume ratio, analyst estimates, and recent financials
- **FRED REST API** — latest macro snapshot (oil, VIX, yields, spreads, GDP, unemployment)

25 of 27 features are available live. The two lagged SUE features (`sue_lag1`, `sue_lag2`) require historical actuals not available from yfinance — the imputer fills these with training-set medians.

---

## App Pages

| Page | Description |
|---|---|
| **Home** | Overview, model stats, navigation guide |
| **Price Chart** | Interactive candlestick chart for any ticker — timeframe/interval pills, draw tools, price-change measure tool |
| **Earnings Predictor** | Enter any ticker or company name → live prediction, probability bars, and SHAP waterfall explanation |
| **Earnings Calendar** | Upcoming earnings for a curated watchlist with pre-run model predictions |
| **Sector Overview** | Historical beat/miss rates broken down by sector |
| **Backtesting** | Confusion matrix, per-class F1 scores, and quarterly accuracy on the 2022–2024 test set |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | LightGBM, Scikit-learn (Logistic Regression, preprocessing) |
| Explainability | SHAP |
| Data (training) | WRDS (Compustat, I/B/E/S, CRSP) |
| Data (live) | yfinance, FRED REST API |
| Web App | Streamlit |
| Visualization | Plotly, Matplotlib |
| Deployment | Streamlit Community Cloud |

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Refresh macro data from FRED
python src/macro_features.py

# Launch the app
streamlit run app.py
```

> **Note:** `data/raw/` is gitignored — WRDS data is under an institutional license and cannot be redistributed. The processed feature matrix (`data/processed/features.csv`) and trained model artifacts (`models/*.joblib`) are committed and used directly by the app.

---

> For informational and educational purposes only. Not financial advice. Predictions are probabilistic and may be incorrect — do not use as the sole basis for any investment decision.
