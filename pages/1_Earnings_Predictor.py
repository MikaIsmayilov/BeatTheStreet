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

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from live_features import fetch_live_features, get_company_info, build_feature_df, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

LABEL_COLORS = {"beat": "#2ecc71", "meet": "#f39c12", "miss": "#e74c3c"}
LABEL_EMOJI  = {"beat": "🟢", "meet": "🟡", "miss": "🔴"}

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


# ── Main page ─────────────────────────────────────────────────────────────────
st.title("🔮 Earnings Surprise Prediction")

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
    st.info(match_msg, icon="🔍")

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
        "Training medians used for these — prediction is still valid.",
        icon="⚠️"
    )

# ── Prediction output ─────────────────────────────────────────────────────────
pred_color = LABEL_COLORS[pred_label]
pred_emoji = LABEL_EMOJI[pred_label]

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
        {pred_emoji} {pred_label.upper()}
    </div>
    <div style="font-size: 1.1rem; color: #ccc;">
        {proba[pred_label]*100:.1f}% confidence
    </div>
</div>
""", unsafe_allow_html=True)

st.plotly_chart(probability_chart(proba), use_container_width=True)

# ── Candlestick chart ─────────────────────────────────────────────────────────
st.divider()
try:
    TF_OPTIONS  = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "All"]
    INT_OPTIONS = ["5m", "15m", "30m", "1h", "1D", "1W"]

    # yfinance interval strings
    INT_YF = {"5m": "5m", "15m": "15m", "30m": "30m",
              "1h": "60m", "1D": "1d", "1W": "1wk"}

    # Best default interval per timeframe
    INT_DEFAULT = {"1D": "5m", "5D": "30m", "1M": "1D",
                   "3M": "1D", "6M": "1D", "YTD": "1D",
                   "1Y": "1D", "All": "1W"}

    # Max data yfinance allows per interval (always fetch the max so zoom-out works)
    MAX_PERIOD = {"5m": "60d", "15m": "60d", "30m": "60d",
                  "1h": "730d", "1D": "max", "1W": "max"}

    # Chart slot (renders above controls)
    chart_ph = st.empty()

    # ── Controls below chart ──────────────────────────────────────────────────
    tf_col, int_col, erase_col, measure_col = st.columns([2.5, 1.8, 0.55, 2])

    with tf_col:
        st.markdown("<span style='font-size:0.85rem; color:#aaa; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>Timeframe</span>", unsafe_allow_html=True)
        tf_key = f"candle_tf_{ticker_input}"
        tf = st.pills("Timeframe", TF_OPTIONS, default="1M",
                      key=tf_key, label_visibility="collapsed")

    with int_col:
        st.markdown("<span style='font-size:0.85rem; color:#aaa; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>Interval</span>", unsafe_allow_html=True)
        # Auto-pick default when tf changes
        int_key     = f"candle_int_{ticker_input}"
        int_default = INT_DEFAULT.get(tf, "1D")
        if int_key not in st.session_state or \
                st.session_state.get(f"_last_tf_{ticker_input}") != tf:
            st.session_state[int_key] = int_default
        st.session_state[f"_last_tf_{ticker_input}"] = tf

        interval = st.pills("Interval", INT_OPTIONS,
                            default=st.session_state[int_key],
                            key=int_key, label_visibility="collapsed")

    with erase_col:
        st.markdown("<span style='font-size:0.85rem; color:#aaa; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>Clear</span>", unsafe_allow_html=True)
        st.button("🗑️", key=f"erase_{ticker_input}",
                  help="Erase all drawn shapes", use_container_width=True)

    # ── Fetch maximum available data for the interval ─────────────────────────
    yf_interval = INT_YF.get(interval, "1d")
    yf_period   = MAX_PERIOD.get(interval, "max")
    hist = yf.Ticker(ticker_input).history(period=yf_period, interval=yf_interval)
    hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index

    # Separate daily history for the measure-range date pickers
    hist_daily = yf.Ticker(ticker_input).history(period="2y", interval="1d")
    hist_daily.index = hist_daily.index.tz_localize(None) if hist_daily.index.tz else hist_daily.index

    with measure_col:
        st.markdown(
            "<span style='font-size:0.95rem; color:#aaa;'>📐 Measure price change</span>",
            unsafe_allow_html=True,
        )
        mc1, mc2 = st.columns(2)
        first_date = hist_daily.index[0].date()
        last_date  = hist_daily.index[-1].date()
        range_start = mc1.date_input(
            "From", value=first_date, min_value=first_date, max_value=last_date,
            label_visibility="collapsed", key=f"rs_{ticker_input}",
        )
        range_end = mc2.date_input(
            "To", value=last_date, min_value=first_date, max_value=last_date,
            label_visibility="collapsed", key=f"re_{ticker_input}",
        )
        rs_rows = hist_daily[hist_daily.index.date == range_start]
        re_rows = hist_daily[hist_daily.index.date == range_end]
        if not rs_rows.empty and not re_rows.empty:
            p0, p1 = rs_rows["Open"].iloc[0], re_rows["Close"].iloc[-1]
            chg = p1 - p0
            pct = chg / p0 * 100
            clr = "#2ecc71" if chg >= 0 else "#e74c3c"
            st.markdown(
                f"<div style='font-size:1rem; font-weight:700; color:{clr};'>"
                f"{'▲' if chg >= 0 else '▼'} ${abs(chg):.2f} ({pct:+.2f}%)"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Compute initial x-axis view window from TF selection ─────────────────
    if not hist.empty:
        today = pd.Timestamp.now()
        tf_start = {
            "1D":  today - pd.Timedelta(days=1),
            "5D":  today - pd.Timedelta(days=5),
            "1M":  today - pd.DateOffset(months=1),
            "3M":  today - pd.DateOffset(months=3),
            "6M":  today - pd.DateOffset(months=6),
            "YTD": pd.Timestamp(today.year, 1, 1),
            "1Y":  today - pd.DateOffset(years=1),
        }.get(tf, None)

        x0 = max(tf_start, hist.index[0]) if tf_start is not None else hist.index[0]
        x1 = hist.index[-1]

        # y range scoped to the visible window for a clean initial render
        visible = hist[hist.index >= x0]
        if visible.empty:
            visible = hist
        y_pad = (visible["High"].max() - visible["Low"].min()) * 0.05
        y0 = visible["Low"].min()  - y_pad
        y1 = visible["High"].max() + y_pad

        # ── Build chart ───────────────────────────────────────────────────────
        fig_candle = go.Figure(go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"],
            low=hist["Low"],   close=hist["Close"],
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
            name=ticker_input,
        ))
        fig_candle.update_layout(
            title=dict(
                text=f"{ticker_input} — Price History  <span style='font-size:14px; color:#aaa'>({tf} · {interval})</span>",
                font=dict(size=20),
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=14),
                range=[x0, x1],
                rangeslider=dict(visible=True, thickness=0.05),
                type="date",
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    *([dict(bounds=[16, 9.5], pattern="hour")]
                      if interval in ("5m", "15m", "30m", "1h") else []),
                ],
            ),
            yaxis=dict(
                showgrid=True, gridcolor="#2a2a2a",
                title="Price (USD)", title_font=dict(size=14),
                tickfont=dict(size=13),
                range=[y0, y1],
                fixedrange=False,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=55, b=10, l=10, r=10),
            height=460,
            dragmode="zoom",
            newshape=dict(line_color="#FFD700", line_width=2),
            shapes=[],
        )

        with chart_ph.container():
            st.plotly_chart(
                fig_candle,
                use_container_width=True,
                config={
                    "modeBarButtonsToAdd": [
                        "drawline", "drawopenpath",
                        "drawcircle", "drawrect", "eraseshape",
                    ],
                    "displayModeBar": True,
                    "scrollZoom": True,
                },
            )

except Exception:
    pass

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

# ── Feature table (expandable) ────────────────────────────────────────────────
with st.expander("📋 Feature values used in this prediction"):
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
    "⚠️ This tool is for informational and educational purposes only. Predictions are probabilistic "
    "and may be incorrect. Past model performance does not guarantee future results. "
    "Do not use this as the sole basis for any investment decision.\n\n"
    "Model: LightGBM · Trained on 2005–2019 · Validated 2020–2021 · Test accuracy 60.9% (2022–2024)"
)
