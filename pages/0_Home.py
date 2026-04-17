"""
Home Page — BeatTheStreet landing page.
"""

import os
import streamlit as st

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# ── Header ────────────────────────────────────────────────────────────────────
st.image(os.path.join(ASSETS, "beatthestreet_logo_dark.png"), width=780)

st.markdown("""
Will a company **beat**, **meet**, or **miss** Wall Street's EPS estimates?
BeatTheStreet uses machine learning trained on 20 years of earnings data to generate
probability forecasts ahead of each announcement — combining analyst estimates,
company fundamentals, price momentum, live macroeconomic signals, and real-time news sentiment.
""")

st.divider()

# ── Key stats ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Training Samples",    "104,938")
c2.metric("Model Accuracy",      "60.9%")
c3.metric("Years of History",    "2005 – 2024")
c4.metric("Predictive Features", "27")
c5.metric("Best Model",          "LightGBM")

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### How it works")
    st.markdown("""
1. **Data** — 20 years of WRDS earnings data (Compustat + I/B/E/S + CRSP) forms
   the training set. Live inputs at inference time come from Yahoo Finance and FRED.
2. **Features** — 27 model inputs spanning analyst consensus, momentum signals,
   financial ratios, and macro indicators (VIX, oil, credit spreads, yields).
3. **Model** — LightGBM trained on a strict time-based split (train 2005–2019,
   test 2022–2024) to prevent lookahead bias. 60.9% accuracy vs. 47.8% naive baseline.
4. **Prediction** — Beat / Meet / Miss probabilities with a SHAP waterfall chart
   showing exactly which factors drove the forecast.
5. **News Sentiment** — Recent headlines scored with VADER sentiment analysis,
   displayed alongside each prediction as additional context.
""")

with col_b:
    st.markdown("### Navigate")
    st.page_link("pages/1_Chart.py",              label="Price Chart",
                 help="Interactive candlestick chart with draw tools and measure tool")
    st.page_link("pages/1_Earnings_Predictor.py", label="Earnings Predictor",
                 help="Live beat/meet/miss forecast + SHAP explanation + news sentiment")
    st.page_link("pages/2_Earnings_Calendar.py",  label="Earnings Calendar",
                 help="Upcoming earnings with pre-run model predictions for major companies")
    st.page_link("pages/4_Sector_Overview.py",    label="Sector Overview",
                 help="Historical beat/miss rates by sector and top consistent beaters/missers")
    st.page_link("pages/3_Backtesting.py",        label="Backtesting",
                 help="Confusion matrix, quarterly accuracy, and per-class F1 on 2022–2024 test set")

st.divider()

# ── Data sources ──────────────────────────────────────────────────────────────
st.markdown("### Data Sources")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Training data (WRDS)**
- **Compustat** — Quarterly financials: revenue, operating income, assets, accruals
- **I/B/E/S** — Analyst EPS estimates: consensus, analyst count, dispersion, lagged SUE
- **CRSP** — Monthly stock returns and prices
- **CCM Link** — Compustat–CRSP identifier mapping
""")

with col2:
    st.markdown("""
**Live inference**
- **Yahoo Finance** — Current price, momentum, analyst estimates, recent financials
- **FRED** — Latest macro snapshot: VIX, oil, Treasury yields, HY credit spread, GDP, unemployment
- **VADER** — News sentiment scored on recent headlines (no API key required)
""")

st.divider()
st.caption(
    "For informational and educational purposes only. Not financial advice. "
    "Predictions are probabilistic and may be incorrect — do not use as the sole basis for any investment decision."
)
