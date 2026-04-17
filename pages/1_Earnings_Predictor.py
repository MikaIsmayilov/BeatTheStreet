"""
Prediction Page
================
User enters a ticker → live features fetched via yfinance + FRED →
LightGBM outputs beat/meet/miss probabilities + SHAP explanation.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import shap
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from live_features import fetch_live_features, get_company_info, build_feature_df, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

LABEL_COLORS = {"beat": "#2ecc71", "meet": "#f39c12", "miss": "#e74c3c"}

FEATURE_LABELS = {
    # I/B/E/S
    "meanest":           "Mean Analyst EPS Estimate",
    "numest":            "Number of Analyst Estimates",
    "est_dispersion":    "Analyst Estimate Dispersion",
    "sue_lag1":          "Earnings Surprise (1 qtr ago)",
    "sue_lag2":          "Earnings Surprise (2 qtrs ago)",
    # Compustat
    "revenue_growth":    "Revenue Growth (YoY)",
    "roa":               "Return on Assets",
    "accruals":          "Accruals",
    "current_ratio":     "Current Ratio",
    "asset_growth":      "Asset Growth (YoY)",
    "op_margin":         "Operating Margin",
    # CRSP
    "ret_1m":            "1-Month Stock Return",
    "ret_3m":            "3-Month Stock Return",
    "ret_6m":            "6-Month Stock Return",
    "vol_ratio":         "Volume Ratio (recent vs avg)",
    "prc":               "Stock Price",
    # Macro
    "oil_1m_ret":        "Oil Price Change (1-Month)",
    "oil_3m_ret":        "Oil Price Change (3-Month)",
    "vix_level":         "VIX (Market Fear Index)",
    "vix_1m_chg":        "VIX Change (1-Month)",
    "gs10_level":        "10-Year Treasury Yield",
    "gs10_1m_chg":       "Treasury Yield Change (1-Month)",
    "hy_spread":         "High-Yield Credit Spread",
    "hy_spread_chg":     "Credit Spread Change",
    "gdp_growth":        "GDP Growth Rate",
    "unrate":            "Unemployment Rate",
    "unrate_chg":        "Unemployment Rate Change",
}

def label(col):
    """Return plain-English name for a feature column."""
    return FEATURE_LABELS.get(col, col)


# ── Load model artifacts (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    return {
        "lgbm":         joblib.load(f"{MODELS_DIR}/lightgbm_model.joblib"),
        "imputer":      joblib.load(f"{MODELS_DIR}/imputer.joblib"),
        "scaler":       joblib.load(f"{MODELS_DIR}/scaler.joblib"),
        "le":           joblib.load(f"{MODELS_DIR}/label_encoder.joblib"),
        "win_low":      joblib.load(f"{MODELS_DIR}/win_low.joblib"),
        "win_high":     joblib.load(f"{MODELS_DIR}/win_high.joblib"),
        "feature_cols": joblib.load(f"{MODELS_DIR}/feature_cols.joblib"),
    }


@st.cache_resource(show_spinner="Loading SHAP explainer…")
def load_explainer(_lgbm_model):
    return shap.TreeExplainer(_lgbm_model)


def preprocess(features_dict, artifacts):
    """Apply winsorization → imputation to raw feature dict; return numpy array."""
    df = build_feature_df(features_dict)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.clip(lower=artifacts["win_low"], upper=artifacts["win_high"], axis=1)
    arr = artifacts["imputer"].transform(df)
    return arr, df


def predict(arr, artifacts):
    """Return (predicted_label, proba_dict)."""
    model = artifacts["lgbm"]
    le    = artifacts["le"]
    proba = model.predict_proba(arr)[0]
    classes = le.classes_                    # e.g. ['beat', 'meet', 'miss']
    pred_idx = int(np.argmax(proba))
    return classes[pred_idx], {c: float(p) for c, p in zip(classes, proba)}


def compute_shap(arr, pred_label, artifacts):
    """SHAP values for the predicted class."""
    explainer = load_explainer(artifacts["lgbm"])
    shap_vals = explainer.shap_values(arr)
    le        = artifacts["le"]
    class_idx = list(le.classes_).index(pred_label)
    # SHAP >= 0.41 returns ndarray (n_samples, n_features, n_classes)
    # older SHAP returns list of (n_samples, n_features) arrays
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        return shap_vals[0, :, class_idx]
    return shap_vals[class_idx][0]


# ── Helper: format market cap ─────────────────────────────────────────────────
def fmt_cap(cap):
    if cap is None:
        return "N/A"
    if cap >= 1e12:
        return f"${cap/1e12:.1f}T"
    if cap >= 1e9:
        return f"${cap/1e9:.1f}B"
    return f"${cap/1e6:.0f}M"


# ── Probability bars (plotly) ─────────────────────────────────────────────────
def probability_chart(proba: dict) -> go.Figure:
    labels = list(proba.keys())
    values = [v * 100 for v in proba.values()]
    colors = [LABEL_COLORS[l] for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=18, color="white"),
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 110], showticklabels=False, showgrid=False, zeroline=False),
        xaxis=dict(tickfont=dict(size=16, color="white")),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        showlegend=False,
    )
    return fig


# ── SHAP waterfall chart (plotly) ────────────────────────────────────────────��
def shap_chart(shap_values, feature_names, feature_vals, pred_label, top_n=12) -> go.Figure:
    idx   = np.argsort(np.abs(shap_values))[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals  = [shap_values[i]   for i in idx]
    fvals = [feature_vals[i]  for i in idx]

    # Sort by value for display
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals  = [vals[i]  for i in order]
    fvals = [fvals[i] for i in order]

    colors = [LABEL_COLORS["beat"] if v > 0 else LABEL_COLORS["miss"] for v in vals]
    labels = [f"{label(n)}  = {fv:.3g}" if not np.isnan(fv) else f"{label(n)}  (imputed)"
              for n, fv in zip(names, fvals)]

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=17),
    ))
    fig.update_layout(
        title=dict(
            text=f"Why <b>{pred_label.upper()}</b>? — Top {top_n} feature contributions",
            font=dict(size=20)
        ),
        xaxis=dict(title="SHAP value (impact on predicted probability)",
                   title_font=dict(size=15),
                   tickfont=dict(size=14),
                   zeroline=True, zerolinecolor="#555"),
        yaxis=dict(tickfont=dict(size=17)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20, l=10, r=80),
        height=52 * top_n + 80,
    )
    return fig


def resolve_ticker(query: str) -> tuple:
    """
    Resolve a user query to a ticker symbol.
    - $NVDA or NVDA (all-caps, ≤5 chars, no spaces) → used directly as ticker
    - Everything else (intel, NVIDIA, Apple Inc, etc.) → yf.Search name lookup
    Returns (ticker, info_msg_or_None).
    """
    raw = query.strip()

    # $TICKER — explicit ticker prefix, strip $ and use directly
    if raw.startswith("$"):
        return raw.lstrip("$").strip().upper(), None

    # All-uppercase, short, no spaces → user typed a ticker (e.g. AAPL, NVDA, META)
    if raw == raw.upper() and len(raw) <= 5 and " " not in raw:
        return raw.upper(), None

    # Anything else → name search
    try:
        results = yf.Search(raw, max_results=10).quotes
        # Prefer US equity exchanges
        us_exchanges = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "NASDAQ", "NYSE", "BTS"}
        equities = [r for r in results
                    if r.get("quoteType", "").upper() == "EQUITY"
                    and r.get("exchange", "") in us_exchanges]
        if not equities:
            equities = [r for r in results if r.get("quoteType", "").upper() == "EQUITY"]
        if equities:
            ticker = equities[0]["symbol"]
            name   = equities[0].get("shortname") or equities[0].get("longname", ticker)
            return ticker.upper(), f'Matched **{name}** → using ticker **{ticker}**'
    except Exception:
        pass

    # Last resort: uppercase and hope for the best
    return raw.upper(), None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_sentiment(ticker: str) -> list[dict]:
    """Fetch up to 20 recent headlines via yfinance and score with VADER."""
    sia  = SentimentIntensityAnalyzer()
    news = yf.Ticker(ticker).news or []
    results = []
    for item in news[:20]:
        # yfinance ≥0.2.50 wraps content; older builds expose fields at top level
        content   = item.get("content", item)
        title     = content.get("title", "")
        if not title:
            continue
        compound  = sia.polarity_scores(title)["compound"]
        sentiment = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
        provider  = content.get("provider", {})
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else item.get("publisher", "")
        results.append({"title": title, "compound": compound,
                         "sentiment": sentiment, "publisher": publisher})
    return results


def sentiment_bar_html(pos: int, neu: int, neg: int) -> str:
    total = pos + neu + neg or 1
    pw, nuw, nw = pos / total * 100, neu / total * 100, neg / total * 100
    return (
        f'<div style="display:flex;height:10px;border-radius:6px;overflow:hidden;margin:6px 0 14px">'
        f'<div style="width:{pw:.1f}%;background:#2ecc71"></div>'
        f'<div style="width:{nuw:.1f}%;background:#888"></div>'
        f'<div style="width:{nw:.1f}%;background:#e74c3c"></div>'
        f'</div>'
    )


# ── Main page ─────────────────────────────────────────────────────────────────
st.title("Earnings Surprise Prediction")

artifacts = load_artifacts()

# ── Larger pills CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    div[data-testid="stPills"] button {
        font-size: 1rem !important;
        padding: 6px 16px !important;
        min-width: 52px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PRED_CACHE = "pred_cache"

# ── Ticker input ──────────────────────────────────────────────────────────────
with st.form("ticker_form"):
    col_in, col_btn = st.columns([3, 1])
    raw_input = col_in.text_input(
        "Enter a ticker symbol or company name",
        placeholder="e.g. AAPL, MSFT, NVDA, Apple, NVIDIA",
        label_visibility="collapsed",
    ).strip()
    submitted = col_btn.form_submit_button("Predict", use_container_width=True,
                                           type="primary")

# On a fresh submit, run the prediction and cache the results
if submitted and raw_input:
    ticker_input, match_msg = resolve_ticker(raw_input)

    with st.spinner(f"Fetching live data for **{ticker_input}**…"):
        info     = get_company_info(ticker_input)
        features, debug = fetch_live_features(ticker_input)

    if info["current_price"] is None and all(np.isnan(v) for v in features.values()
                                              if isinstance(v, float)):
        st.error(f"Could not find data for **{ticker_input}**. Try searching by ticker symbol directly.")
        st.stop()

    with st.spinner("Running model…"):
        arr, df_raw  = preprocess(features, artifacts)
        pred_label, proba = predict(arr, artifacts)
        shap_vals    = compute_shap(arr, pred_label, artifacts)

    st.session_state[PRED_CACHE] = dict(
        ticker=ticker_input, match_msg=match_msg,
        info=info, features=features, debug=debug,
        arr=arr, df_raw=df_raw,
        pred_label=pred_label, proba=proba, shap_vals=shap_vals,
    )

# Nothing cached yet — show placeholder and stop
if PRED_CACHE not in st.session_state:
    st.info("Enter a ticker or company name above and click **Predict** to get a forecast.")
    st.stop()

# Restore from cache (works across reruns from pills / date pickers)
c          = st.session_state[PRED_CACHE]
ticker_input = c["ticker"]
match_msg  = c["match_msg"]
info       = c["info"]
features   = c["features"]
debug      = c["debug"]
arr        = c["arr"]
df_raw     = c["df_raw"]
pred_label = c["pred_label"]
proba      = c["proba"]
shap_vals  = c["shap_vals"]

if match_msg:
    st.info(match_msg)

# ── Company header ────────────────────────────────────────────────────────────
st.subheader(f"{info['name']}  ·  {ticker_input}")

m1, m2, m3, m4 = st.columns(4)
price  = info["current_price"]
next_e = info["next_earnings"]
m1.metric("Current Price", f"${price:,.2f}" if price else "N/A")
m2.metric("Sector",        info["sector"])
m3.metric("Market Cap",    fmt_cap(info["market_cap"]))
m4.metric("Next Earnings", next_e.strftime("%b %d, %Y") if next_e else "N/A")

st.divider()

# ── Coverage warning ──────────────────────────────────────────────────────────
missing = debug["missing"]
macro_missing   = [m for m in missing if m in
                   ["oil_1m_ret","oil_3m_ret","vix_level","vix_1m_chg",
                    "gs10_level","gs10_1m_chg","hy_spread","hy_spread_chg",
                    "gdp_growth","unrate","unrate_chg"]]
company_missing = [m for m in missing if m not in macro_missing]

if company_missing:
    st.warning(
        f"{len(company_missing)} company features unavailable "
        f"({', '.join(company_missing[:5])}{'…' if len(company_missing) > 5 else ''}). "
        "Training medians used for these — prediction is still valid."
    )

# ── Prediction output ─────────────────────────────────────────────────────────
pred_color = LABEL_COLORS[pred_label]

st.markdown(f"""
<div style="
    background: {pred_color}22;
    border: 2px solid {pred_color};
    border-radius: 12px;
    padding: 18px 28px;
    text-align: center;
    margin-bottom: 8px;
">
    <div style="font-size: 1rem; color: #aaa; margin-bottom: 4px;">Model Prediction</div>
    <div style="font-size: 3rem; font-weight: 800; color: {pred_color};">
        {pred_label.upper()}
    </div>
    <div style="font-size: 1.1rem; color: #ccc;">
        {proba[pred_label]*100:.1f}% confidence
    </div>
</div>
""", unsafe_allow_html=True)

st.plotly_chart(probability_chart(proba), use_container_width=True)

# ── SHAP chart ────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='font-size:1.1rem;'>"
    "<strong>What drives this prediction?</strong> The chart below uses "
    "<a href='https://shap.readthedocs.io/' target='_blank'>SHAP</a> "
    "(SHapley Additive exPlanations) to show how each factor pushed the model "
    "toward or away from this outcome. "
    "<strong>Green bars</strong> increase the predicted probability; "
    "<strong>red bars</strong> decrease it. Longer bars have more influence."
    "</p>",
    unsafe_allow_html=True,
)
feature_vals_arr = df_raw.values[0]
fig_shap = shap_chart(shap_vals, FEATURE_COLS, feature_vals_arr, pred_label)
st.plotly_chart(fig_shap, use_container_width=True)

# ── News sentiment ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent News Sentiment")
with st.spinner("Fetching headlines…"):
    articles = fetch_news_sentiment(ticker_input)

if not articles:
    st.caption("No recent headlines found.")
else:
    pos_n = sum(1 for a in articles if a["sentiment"] == "positive")
    neu_n = sum(1 for a in articles if a["sentiment"] == "neutral")
    neg_n = sum(1 for a in articles if a["sentiment"] == "negative")
    total = len(articles)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles Analyzed", total)
    c2.metric("Positive", f"{pos_n/total*100:.0f}%")
    c3.metric("Neutral",  f"{neu_n/total*100:.0f}%")
    c4.metric("Negative", f"{neg_n/total*100:.0f}%")
    st.markdown(sentiment_bar_html(pos_n, neu_n, neg_n), unsafe_allow_html=True)

    SENT_COLOR = {"positive": "#2ecc71", "neutral": "#aaa", "negative": "#e74c3c"}
    SENT_LABEL = {"positive": "POS", "neutral": "NEU", "negative": "NEG"}
    for a in articles:
        color = SENT_COLOR[a["sentiment"]]
        badge = SENT_LABEL[a["sentiment"]]
        pub   = f" · {a['publisher']}" if a["publisher"] else ""
        score = f"{a['compound']:+.2f}"
        st.markdown(
            f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:6px">'
            f'<span style="background:{color}33;color:{color};border:1px solid {color};'
            f'border-radius:4px;padding:1px 6px;font-size:0.75rem;font-weight:700;'
            f'white-space:nowrap">{badge} {score}</span>'
            f'<span style="font-size:0.9rem">{a["title"]}'
            f'<span style="color:#666;font-size:0.8rem">{pub}</span></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Feature table (expandable) ────────────────────────────────────────────────
with st.expander("Feature values used in this prediction"):
    feat_df = pd.DataFrame({
        "Feature":    [label(c) for c in FEATURE_COLS],
        "Value":      [f"{features.get(c, np.nan):.4f}"
                       if not pd.isna(features.get(c, np.nan)) else "imputed"
                       for c in FEATURE_COLS],
        "SHAP":       [f"{v:+.4f}" for v in shap_vals],
    })
    feat_df = feat_df.sort_values("SHAP", key=lambda x: x.str.replace("+", "").astype(float),
                                  ascending=False)
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

st.caption(
    "This tool is for informational and educational purposes only. Predictions are probabilistic "
    "and may be incorrect. Past model performance does not guarantee future results. "
    "Do not use this as the sole basis for any investment decision.\n\n"
    "Model: LightGBM · Trained on 2005–2019 · Validated 2020–2021 · Test accuracy 60.9% (2022–2024)"
)
