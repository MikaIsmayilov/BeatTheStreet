"""
Price Chart Page
=================
Interactive candlestick chart for any ticker.
Pre-fills from the prediction cache when navigating from the Predictor.
"""

import warnings
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

warnings.filterwarnings("ignore")

TF_OPTIONS  = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "All"]
INT_OPTIONS = ["5m", "15m", "30m", "1h", "1D", "1W"]
INT_YF      = {"5m": "5m", "15m": "15m", "30m": "30m",
               "1h": "60m", "1D": "1d", "1W": "1wk"}
INT_DEFAULT = {"1D": "5m", "5D": "30m", "1M": "1D", "3M": "1D",
               "6M": "1D", "YTD": "1D", "1Y": "1D", "All": "1W"}
MAX_PERIOD  = {"5m": "60d", "15m": "60d", "30m": "60d",
               "1h": "730d", "1D": "max", "1W": "max"}

st.title("Price Chart")

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

CHART_CACHE = "chart_ticker"


def resolve_ticker(query: str) -> tuple[str, str | None]:
    raw = query.strip()
    if raw.startswith("$"):
        return raw.lstrip("$").strip().upper(), None
    if raw == raw.upper() and len(raw) <= 5 and " " not in raw:
        return raw.upper(), None
    try:
        results = yf.Search(raw, max_results=10).quotes
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
    return raw.upper(), None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="2y", interval="1d")
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df

# Pre-fill from prediction cache if navigating from the Predictor
if CHART_CACHE not in st.session_state:
    st.session_state[CHART_CACHE] = st.session_state.get("pred_cache", {}).get("ticker", "")

with st.form("chart_form"):
    col_in, col_btn = st.columns([3, 1])
    ticker_raw = col_in.text_input(
        "Ticker",
        value=st.session_state[CHART_CACHE],
        placeholder="e.g. AAPL, MSFT, NVDA",
        label_visibility="collapsed",
    ).strip()
    submitted = col_btn.form_submit_button("Load", use_container_width=True, type="primary")

if submitted and ticker_raw:
    resolved, match_msg = resolve_ticker(ticker_raw)
    st.session_state[CHART_CACHE] = resolved
    if match_msg:
        st.session_state["chart_match_msg"] = match_msg
    else:
        st.session_state.pop("chart_match_msg", None)

ticker_input = st.session_state[CHART_CACHE]

if "chart_match_msg" in st.session_state:
    st.info(st.session_state["chart_match_msg"])

if not ticker_input:
    st.info("Enter a ticker symbol above to load the chart.")
    st.stop()

try:
    chart_ph = st.empty()

    tf_col, int_col, erase_col, measure_col = st.columns([2.5, 1.8, 0.55, 2])

    with tf_col:
        st.markdown(
            "<span style='font-size:0.85rem; color:#aaa; font-weight:600; "
            "text-transform:uppercase; letter-spacing:0.05em;'>Timeframe</span>",
            unsafe_allow_html=True,
        )
        tf_key = f"candle_tf_{ticker_input}"
        tf = st.pills("Timeframe", TF_OPTIONS, default="1M",
                      key=tf_key, label_visibility="collapsed")

    with int_col:
        st.markdown(
            "<span style='font-size:0.85rem; color:#aaa; font-weight:600; "
            "text-transform:uppercase; letter-spacing:0.05em;'>Interval</span>",
            unsafe_allow_html=True,
        )
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
        st.markdown(
            "<span style='font-size:0.85rem; color:#aaa; font-weight:600; "
            "text-transform:uppercase; letter-spacing:0.05em;'>Clear</span>",
            unsafe_allow_html=True,
        )
        st.button("Clear", key=f"erase_{ticker_input}",
                  help="Erase all drawn shapes", use_container_width=True)

    yf_interval = INT_YF.get(interval, "1d")
    yf_period   = MAX_PERIOD.get(interval, "max")
    hist       = fetch_ohlcv(ticker_input, yf_period, yf_interval)
    hist_daily = fetch_daily(ticker_input)

    with measure_col:
        st.markdown(
            "<span style='font-size:0.95rem; color:#aaa;'>Measure price change</span>",
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

    if not hist.empty:
        today  = pd.Timestamp.now()
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

        visible = hist[hist.index >= x0]
        if visible.empty:
            visible = hist
        y_pad = (visible["High"].max() - visible["Low"].min()) * 0.05
        y0 = visible["Low"].min()  - y_pad
        y1 = visible["High"].max() + y_pad

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
                text=f"{ticker_input} — Price History  "
                     f"<span style='font-size:14px; color:#aaa'>({tf} · {interval})</span>",
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
            height=520,
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
    st.error(f"Could not load chart for **{ticker_input}**. Check the ticker and try again.")
