"""
Live Feature Fetching
======================
Fetches real-time features for a ticker using yfinance + FRED.
Used by the Streamlit Prediction page to construct the model input vector.

All features mirror those the model was trained on. NaN is returned for
anything unavailable — the pre-trained imputer fills medians at inference.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from macro_features import build_macro_monthly

# Must match train_model.py FEATURE_COLS exactly
FEATURE_COLS = [
    "meanest", "numest", "est_dispersion",
    "sue_lag1", "sue_lag2",
    "revenue_growth", "roa", "accruals",
    "current_ratio", "asset_growth", "op_margin",
    "ret_1m", "ret_3m", "ret_6m", "vol_ratio", "prc",
    "oil_1m_ret", "oil_3m_ret",
    "vix_level", "vix_1m_chg",
    "gs10_level", "gs10_1m_chg",
    "hy_spread", "hy_spread_chg",
    "gdp_growth",
    "unrate", "unrate_chg",
]


def _safe(a, b=None):
    """Safe division or single-value float cast."""
    try:
        if b is None:
            v = float(a)
            return np.nan if np.isinf(v) else v
        if float(b) == 0 or pd.isna(b):
            return np.nan
        v = float(a) / float(b)
        return np.nan if np.isinf(v) else v
    except Exception:
        return np.nan


# ── Price / momentum ──────────────────────────────────────────────────────────

def _price_features(t: yf.Ticker) -> dict:
    f = {}
    try:
        hist = t.history(period="9mo", interval="1mo")
        if hist.empty:
            return f
        c = hist["Close"]
        v = hist["Volume"]

        f["prc"] = _safe(c.iloc[-1])
        if len(c) >= 2:
            f["ret_1m"] = _safe(c.iloc[-1] - c.iloc[-2], c.iloc[-2])
        if len(c) >= 4:
            f["ret_3m"] = _safe(c.iloc[-1] - c.iloc[-4], c.iloc[-4])
        if len(c) >= 7:
            f["ret_6m"] = _safe(c.iloc[-1] - c.iloc[-7], c.iloc[-7])
        if len(v) >= 7:
            avg = float(v.iloc[-7:-1].mean())
            f["vol_ratio"] = _safe(v.iloc[-1], avg)
    except Exception:
        pass
    return f


# ── Analyst estimates ─────────────────────────────────────────────────────────

def _estimate_features(t: yf.Ticker) -> dict:
    f = {}
    # Try structured earnings_estimate table first
    try:
        est = t.earnings_estimate
        if est is not None and not est.empty:
            for period in ["0q", "+1q", "0y"]:
                if period in est.index:
                    row = est.loc[period]
                    avg = row.get("avg", np.nan)
                    if not pd.isna(avg):
                        f["meanest"] = _safe(avg)
                        n = row.get("numberOfAnalysts", np.nan)
                        if not pd.isna(n):
                            f["numest"] = _safe(n)
                        hi = row.get("high", np.nan)
                        lo = row.get("low", np.nan)
                        if not pd.isna(hi) and not pd.isna(lo) and abs(float(avg)) > 1e-6:
                            f["est_dispersion"] = _safe(float(hi) - float(lo),
                                                         4 * abs(float(avg)))
                        break
    except Exception:
        pass

    # Fallback to info dict
    if "meanest" not in f:
        try:
            info = t.info
            feps = info.get("forwardEps") or info.get("trailingEps")
            if feps:
                f["meanest"] = _safe(feps)
            n = info.get("numberOfAnalystOpinions")
            if n:
                f["numest"] = _safe(n)
        except Exception:
            pass

    # SUE lags need I/B/E/S history — not available via yfinance, imputer fills median
    f["sue_lag1"] = np.nan
    f["sue_lag2"] = np.nan

    return f


# ── Quarterly financials ──────────────────────────────────────────────────────

def _financial_features(t: yf.Ticker) -> dict:
    f = {}
    try:
        fin = t.quarterly_financials
        bs  = t.quarterly_balance_sheet
        cf  = t.quarterly_cashflow

        if fin is None or fin.empty:
            return f

        def find(df, must, exclude=None):
            """Find first row whose index contains all 'must' keywords."""
            for idx in df.index:
                s = str(idx).lower()
                if all(k.lower() in s for k in must):
                    if exclude and any(e.lower() in s for e in exclude):
                        continue
                    return df.loc[idx]
            return None

        rev = find(fin, ["revenue"])
        ni  = find(fin, ["net income"], exclude=["loss", "minority"])
        oi  = find(fin, ["operating income"])
        ta  = find(bs,  ["total assets"])
        ca  = find(bs,  ["current assets"])
        cl  = find(bs,  ["current liabilities"])
        ocf = find(cf,  ["operating cash flow"])

        if rev is not None and len(rev) >= 2 and not pd.isna(rev.iloc[1]):
            f["revenue_growth"] = _safe(rev.iloc[0] - rev.iloc[1], abs(rev.iloc[1]))

        if ni is not None and ta is not None:
            f["roa"] = _safe(ni.iloc[0], ta.iloc[0])

        if oi is not None and rev is not None:
            f["op_margin"] = _safe(oi.iloc[0], rev.iloc[0])

        if ni is not None and ocf is not None and ta is not None:
            f["accruals"] = _safe(float(ni.iloc[0]) - float(ocf.iloc[0]), ta.iloc[0])

        if ca is not None and cl is not None:
            f["current_ratio"] = _safe(ca.iloc[0], cl.iloc[0])

        if ta is not None and len(ta) >= 2 and not pd.isna(ta.iloc[1]):
            f["asset_growth"] = _safe(ta.iloc[0] - ta.iloc[1], abs(ta.iloc[1]))

    except Exception:
        pass
    return f


# ── Macro snapshot ────────────────────────────────────────────────────────────

def _macro_features() -> dict:
    try:
        macro = build_macro_monthly()
        # Use the most recent non-null value per column (some series lag by 1–2 months)
        latest = {col: macro[col].dropna().iloc[-1] for col in macro.columns
                  if macro[col].notna().any()}
        return {k: _safe(v) for k, v in latest.items()}
    except Exception:
        return {}


# ── Public API ────────────────────────────────────────────────────────────────

def get_company_info(symbol: str) -> dict:
    """Name, sector, market cap, price, next earnings date."""
    result = {
        "name": symbol, "sector": "N/A",
        "market_cap": None, "current_price": None,
        "next_earnings": None, "currency": "USD",
    }
    try:
        t = yf.Ticker(symbol)
        info = t.info
        result["name"]          = info.get("longName") or info.get("shortName") or symbol
        result["sector"]        = info.get("sector", "N/A")
        result["market_cap"]    = info.get("marketCap")
        result["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
        result["currency"]      = info.get("currency", "USD")

        cal = t.calendar
        if cal is not None:
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date", [])
                if dates:
                    result["next_earnings"] = pd.Timestamp(dates[0])
            elif not cal.empty and "Earnings Date" in cal.index:
                ed = cal.loc["Earnings Date"]
                result["next_earnings"] = pd.Timestamp(ed.iloc[0]) if hasattr(ed, "iloc") else pd.Timestamp(ed)
    except Exception:
        pass
    return result


def fetch_live_features(symbol: str) -> tuple[dict, dict]:
    """
    Fetch all live features for a ticker.

    Returns:
        features : dict mapping each of the 27 FEATURE_COLS to a float (NaN if unavailable)
        debug    : metadata dict (coverage info)
    """
    t = yf.Ticker(symbol)

    raw = {}
    raw.update(_price_features(t))
    raw.update(_estimate_features(t))
    raw.update(_financial_features(t))
    raw.update(_macro_features())

    features = {col: raw.get(col, np.nan) for col in FEATURE_COLS}

    n_available = sum(1 for v in features.values() if not pd.isna(v))
    debug = {
        "available": n_available,
        "total": len(FEATURE_COLS),
        "missing": [k for k, v in features.items() if pd.isna(v)],
    }

    return features, debug


def build_feature_df(features: dict) -> pd.DataFrame:
    """Single-row DataFrame in the exact column order the model expects."""
    return pd.DataFrame([{col: features.get(col, np.nan) for col in FEATURE_COLS}])
