# Earnings Surprise Predictor — Project Roadmap
**Course:** BA870/AC820, Prof. Peter Wysocki, BU Questrom Spring 2026
**Team:** Mika Ismayilli + Aishik
**Presentation Deadline:** April 24, 2026 (mandatory)

---

## App Overview
Predict whether a company will **beat / meet / miss** analyst consensus EPS estimates before earnings are announced. Multi-page Streamlit web app deployed on Streamlit Community Cloud.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Framework | Streamlit (multi-page) |
| ML Models | Logistic Regression (baseline) + XGBoost/LightGBM |
| Explainability | SHAP |
| Historical data | WRDS: Compustat + I/B/E/S + CRSP (pre-pulled to CSVs) |
| Real-time data | yfinance, FRED API, Alpha Vantage |
| Deployment | Streamlit Community Cloud (free, via GitHub) |

---

## App Pages (4 total)

1. **Home** — Upcoming earnings calendar with predicted outcome for each company
2. **Prediction Page** — User enters a ticker → see beat/meet/miss probability + SHAP feature importance chart
3. **Backtesting Page** — Historical model accuracy, confusion matrix, precision/recall by quarter
4. **Sector Overview** — Aggregate predictions grouped by GICS sector

---

## Phase 1 — Data (Days 1–2, ~Apr 15–16)
- [ ] Run WRDS Python pull script → save 4 CSVs to `data/raw/`
  - `compustat_quarterly.csv` — quarterly financials (Compustat `comp.fundq`)
  - `ibes_summary.csv` — analyst EPS estimates (I/B/E/S `ibes.statsum_epsus`)
  - `ccm_links.csv` — CRSP-Compustat link table
  - `crsp_monthly.csv` — monthly stock returns (CRSP `crsp.msf`)
- [ ] Feature engineering script → merge datasets, compute model features, save `features.csv` to `data/processed/`

**Key features to engineer:**
- Earnings momentum (prior quarter surprise direction)
- Estimate revision direction (last 30/60 days)
- Analyst dispersion (std dev of estimates)
- Financial ratio changes (margins, accruals, asset growth)
- SUE (Standardized Unexpected Earnings) history

---

## Phase 2 — Model (Days 3–4, ~Apr 17–18)
- [ ] Train logistic regression (baseline) — 3-class: beat / meet / miss
- [ ] Train XGBoost classifier — hyperparameter tuning via cross-validation
- [ ] Evaluate: confusion matrix, precision/recall, feature importance
- [ ] Save model artifacts to `models/` (`.joblib` format)
- [ ] Generate SHAP values for top features

---

## Phase 3 — Streamlit App (Days 5–8, ~Apr 19–22)
- [ ] `app.py` — Streamlit entry point + navigation
- [ ] `pages/1_Prediction.py` — core demo page (ticker input → prediction + SHAP)
- [ ] `pages/2_Home.py` — earnings calendar
- [ ] `pages/3_Backtesting.py` — historical accuracy charts
- [ ] `pages/4_Sector_Overview.py` — sector-level predictions

---

## Phase 4 — Deploy (Day 9, ~Apr 23)
- [ ] Push full repo to GitHub
- [ ] Connect to Streamlit Community Cloud
- [ ] Test live public URL
- [ ] Final polish / demo run-through

---

## File Structure

```
FinancialAnalytics/
├── app.py                  # Streamlit entry point
├── pages/
│   ├── 1_Prediction.py
│   ├── 2_Home.py
│   ├── 3_Backtesting.py
│   └── 4_Sector_Overview.py
├── data/
│   ├── raw/                # WRDS CSVs (not committed to GitHub)
│   └── processed/          # Engineered features
├── models/                 # Trained model artifacts (.joblib)
├── src/
│   ├── wrds_pull.py        # Phase 1: WRDS data pull script
│   ├── feature_engineering.py  # Phase 1: merge + feature computation
│   ├── train_model.py      # Phase 2: model training
│   └── utils.py            # Shared helpers
├── .prelim/                # Planning docs (not deployed)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Critical Constraints
- **WRDS live queries will NOT work on Streamlit Community Cloud** — all WRDS data must be pre-pulled and bundled or served externally
- WRDS credentials: username `mikaismayilli`, password entered interactively (NEVER hardcoded)
- `fpi = '1'` for I/B/E/S = next quarter EPS (correct for our use case, not `'6'` which is annual)
- CRSP pull may be large — filter to firms with analyst coverage to keep file sizes manageable
