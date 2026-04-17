"""
Earnings Calendar Page
=======================
Shows upcoming earnings announcements for major S&P 500 companies
and their model predictions.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from live_features import fetch_live_features, build_feature_df, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LABEL_COLORS = {"beat": "#2ecc71", "meet": "#f39c12", "miss": "#e74c3c"}

# Curated watchlist of major companies across sectors
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "NFLX",
    "JPM",  "BAC",  "GS",   "MS",
    "JNJ",  "PFE",  "UNH",
    "XOM",  "CVX",
    "WMT",  "HD",   "MCD",
    "BA",   "CAT",  "GE",
]


@st.cache_resource(show_spinner=False)
def load_artifacts():
    return {
        "lgbm":    joblib.load(f"{MODELS_DIR}/lightgbm_model.joblib"),
        "imputer": joblib.load(f"{MODELS_DIR}/imputer.joblib"),
        "le":      joblib.load(f"{MODELS_DIR}/label_encoder.joblib"),
        "win_low": joblib.load(f"{MODELS_DIR}/win_low.joblib"),
        "win_high":joblib.load(f"{MODELS_DIR}/win_high.joblib"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_dates(tickers: list) -> pd.DataFrame:
    """Fetch next earnings date and basic info for each ticker."""
    rows = []
    for sym in tickers:
        try:
            t = yf.Ticker(sym)
            info = t.info
            cal  = t.calendar
            next_e = None
            if cal is not None:
                if isinstance(cal, dict):
                    dates = cal.get("Earnings Date", [])
                    next_e = pd.Timestamp(dates[0]) if dates else None
                elif not cal.empty and "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"]
                    next_e = pd.Timestamp(ed.iloc[0]) if hasattr(ed, "iloc") else pd.Timestamp(ed)
            rows.append({
                "ticker":      sym,
                "name":        info.get("shortName", sym),
                "sector":      info.get("sector", "N/A"),
                "next_earnings": next_e,
                "price":       info.get("currentPrice") or info.get("regularMarketPrice"),
            })
        except Exception:
            rows.append({"ticker": sym, "name": sym, "sector": "N/A",
                         "next_earnings": None, "price": None})
    df = pd.DataFrame(rows)
    df = df[df["next_earnings"].notna()].copy()
    df = df.sort_values("next_earnings")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_prediction(sym: str, _artifacts: dict) -> tuple[str, float]:
    """Run model for a single ticker. Returns (label, confidence)."""
    try:
        features, _ = fetch_live_features(sym)
        df = build_feature_df(features).replace([np.inf, -np.inf], np.nan)
        df = df.clip(lower=_artifacts["win_low"], upper=_artifacts["win_high"], axis=1)
        arr = _artifacts["imputer"].transform(df)
        proba = _artifacts["lgbm"].predict_proba(arr)[0]
        le    = _artifacts["le"]
        idx   = int(np.argmax(proba))
        return le.classes_[idx], float(proba[idx])
    except Exception:
        return "N/A", 0.0


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("📅 Upcoming Earnings Calendar")
st.markdown("Model predictions for major companies reporting in the next 30 days.")

artifacts = load_artifacts()

with st.spinner("Fetching earnings dates…"):
    cal_df = get_earnings_dates(WATCHLIST)

today = pd.Timestamp.today().normalize()
cal_df = cal_df[cal_df["next_earnings"] >= today]
cal_df = cal_df[cal_df["next_earnings"] <= today + pd.Timedelta(days=30)]

if cal_df.empty:
    st.info("No earnings found in the next 30 days for the watchlist. "
            "Try the Prediction page to look up any ticker.")
    st.stop()

# Group by week
cal_df["week"] = cal_df["next_earnings"].dt.to_period("W")

for week, group in cal_df.groupby("week"):
    week_start = week.start_time.strftime("%b %d")
    week_end   = week.end_time.strftime("%b %d, %Y")
    st.subheader(f"Week of {week_start} – {week_end}")

    cols = st.columns(min(len(group), 4))
    for i, (_, row) in enumerate(group.iterrows()):
        with cols[i % 4]:
            with st.spinner(f"{row['ticker']}…"):
                label, conf = get_prediction(row["ticker"], artifacts)

            color = LABEL_COLORS.get(label, "#888")
            price_str = f"${row['price']:,.2f}" if row["price"] else "N/A"
            date_str  = row["next_earnings"].strftime("%b %d")

            st.markdown(f"""
<div style="
    border: 1px solid {color};
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
    background: {color}11;
">
    <div style="font-size: 1.3rem; font-weight: 700;">{row['ticker']}</div>
    <div style="font-size: 0.8rem; color: #aaa; margin-bottom: 8px;">
        {row['name'][:28]} · {date_str}
    </div>
    <div style="font-size: 1rem; color: #ccc;">{price_str} · {row['sector']}</div>
    <div style="font-size: 1.4rem; font-weight: 700; color: {color}; margin-top: 8px;">
        {label.upper() if label != 'N/A' else '—'}
        <span style="font-size: 0.85rem; font-weight: 400; color: #aaa;">
        {f'{conf*100:.0f}% conf.' if label != 'N/A' else ''}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()
st.caption("Predictions are model estimates, not financial advice. "
           "Earnings dates from Yahoo Finance — verify before trading.")
