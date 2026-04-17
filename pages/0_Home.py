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
company fundamentals, price momentum, and live macroeconomic signals.
""")

st.divider()

# ── Key stats ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Training Samples",   "104,938")
c2.metric("Model Accuracy",     "60.9%")
c3.metric("Years of History",   "2005 – 2024")
c4.metric("Predictive Features","27")
c5.metric("Best Model",         "LightGBM")

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### How it works")
    st.markdown("""
1. **Data** — 20 years of WRDS earnings data (Compustat + I/B/E/S + CRSP) forms
   the training set. Live inputs come from Yahoo Finance and FRED.
2. **Features** — Analyst consensus, estimate revisions, momentum signals,
   financial ratios, and macro indicators (VIX, oil, credit spreads, yields).
3. **Model** — LightGBM trained on a strict time-based split to prevent lookahead bias.
   Test set: 2022–2024 (never seen during training).
4. **Output** — Beat / Meet / Miss probabilities + SHAP chart explaining the drivers.
""")

with col_b:
    st.markdown("### Navigate")
    st.page_link("pages/1_Chart.py",              label="Price Chart",        help="Interactive candlestick chart")
    st.page_link("pages/1_Earnings_Predictor.py", label="Earnings Predictor", help="Enter a ticker for a live forecast")
    st.page_link("pages/2_Earnings_Calendar.py",  label="Earnings Calendar",  help="Upcoming earnings this week")
    st.page_link("pages/4_Sector_Overview.py",    label="Sector Overview",    help="Beat/miss rates by sector")
    st.page_link("pages/3_Backtesting.py",        label="Backtesting",        help="Historical model performance")

st.divider()
st.caption(
    "For informational and educational purposes only. Not financial advice. "
    "Predictions are probabilistic and may be incorrect — do not use as the sole basis for any investment decision."
)
