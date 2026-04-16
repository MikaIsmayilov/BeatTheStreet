"""
Macro Feature Engineering
==========================
Pulls macroeconomic data from FRED and engineers features that capture
the broad economic environment around each earnings announcement.

These features let the model detect economic shocks (oil price crashes,
credit market stress, rate spikes, recessions) that company-level data
misses entirely.

FRED series pulled:
  DCOILWTICO    — WTI crude oil price (daily)
  VIXCLS        — CBOE VIX volatility index (daily)
  GS10          — 10-year Treasury yield (monthly)
  BAMLH0A0HYM2  — ICE BofA HY credit spread (daily)
  A191RL1Q225SBEA — Real GDP growth QoQ % (quarterly)
  UNRATE        — Unemployment rate (monthly)

Features engineered (all joined on month prior to earnings announcement):
  oil_1m_ret     — WTI oil 1-month return
  oil_3m_ret     — WTI oil 3-month return
  vix_level      — VIX level (fear gauge)
  vix_1m_chg     — VIX 1-month change
  gs10_level     — 10yr yield level
  gs10_1m_chg    — 10yr yield 1-month change
  hy_spread      — HY credit spread level (risk appetite)
  hy_spread_chg  — HY credit spread 1-month change
  gdp_growth     — Real GDP QoQ growth rate
  unrate         — Unemployment rate
  unrate_chg     — Unemployment 1-month change

Usage (standalone — overwrites data/processed/macro_monthly.csv):
    python src/macro_features.py

Usage (from feature_engineering.py):
    from macro_features import build_macro_monthly, join_macro_to_earnings
"""

import os
import warnings
import numpy as np
import pandas as pd
import pandas_datareader.data as web

warnings.filterwarnings("ignore")

ROOT      = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR  = os.path.join(ROOT, "data", "processed")
MACRO_CSV = os.path.join(PROC_DIR, "macro_monthly.csv")

START_DATE = "2004-01-01"   # extra year of history for lagged calculations

FRED_SERIES = {
    "oil":        "DCOILWTICO",
    "vix":        "VIXCLS",
    "gs10":       "GS10",
    "hy_spread":  "BAMLH0A0HYM2",
    "gdp_growth": "A191RL1Q225SBEA",
    "unrate":     "UNRATE",
}


# ── Pull raw FRED data ────────────────────────────────────────────────────────

def pull_fred(start: str = START_DATE, end: str = None) -> pd.DataFrame:
    """
    Pull all FRED series and resample to month-end frequency.
    GDP is quarterly — forward-filled to monthly.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"Pulling FRED data ({start} → {end})...")
    frames = {}
    for name, series_id in FRED_SERIES.items():
        try:
            s = web.DataReader(series_id, "fred", start, end)[series_id]
            s = pd.to_numeric(s, errors="coerce")
            # Resample to month-end; GDP is quarterly so forward-fill gaps
            if name == "gdp_growth":
                frames[name] = s.resample("ME").last().ffill()
            else:
                frames[name] = s.resample("ME").mean()
            print(f"  ✓ {series_id}")
        except Exception as e:
            print(f"  ✗ {series_id}: {e}")

    df = pd.DataFrame(frames)
    df.index.name = "month"
    return df


# ── Engineer macro features ───────────────────────────────────────────────────

def engineer_macro_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute returns and changes from raw monthly FRED levels.
    All features are computed so that at month M, only data through
    month M is used (no lookahead).
    """
    out = pd.DataFrame(index=raw.index)

    # ── Oil ──────────────────────────────────────────────────────────────────
    if "oil" in raw:
        out["oil_1m_ret"]   = raw["oil"].pct_change(1)
        out["oil_3m_ret"]   = raw["oil"].pct_change(3)

    # ── VIX ──────────────────────────────────────────────────────────────────
    if "vix" in raw:
        out["vix_level"]    = raw["vix"]
        out["vix_1m_chg"]   = raw["vix"].diff(1)

    # ── 10yr Treasury ────────────────────────────────────────────────────────
    if "gs10" in raw:
        out["gs10_level"]   = raw["gs10"]
        out["gs10_1m_chg"]  = raw["gs10"].diff(1)

    # ── HY credit spread ─────────────────────────────────────────────────────
    if "hy_spread" in raw:
        out["hy_spread"]     = raw["hy_spread"]
        out["hy_spread_chg"] = raw["hy_spread"].diff(1)

    # ── GDP growth ───────────────────────────────────────────────────────────
    if "gdp_growth" in raw:
        out["gdp_growth"]   = raw["gdp_growth"]

    # ── Unemployment ─────────────────────────────────────────────────────────
    if "unrate" in raw:
        out["unrate"]       = raw["unrate"]
        out["unrate_chg"]   = raw["unrate"].diff(1)

    return out


# ── Main build function ───────────────────────────────────────────────────────

def build_macro_monthly(start: str = START_DATE, end: str = None) -> pd.DataFrame:
    """Pull FRED data, engineer features, return month-indexed DataFrame."""
    raw  = pull_fred(start, end)
    feat = engineer_macro_features(raw)
    print(f"  Macro feature matrix: {feat.shape[0]} months × {feat.shape[1]} features")
    print(f"  Date range: {feat.index.min().date()} → {feat.index.max().date()}")
    return feat


# ── Join macro features onto an earnings DataFrame ───────────────────────────

def join_macro_to_earnings(earnings_df: pd.DataFrame,
                           macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join macro features onto the earnings DataFrame.

    For each earnings row, we use macro features from the month PRIOR to
    the earnings announcement date (rdq) to avoid any lookahead.

    Args:
        earnings_df : DataFrame with an 'rdq' column (earnings announcement date)
        macro_df    : Month-indexed DataFrame from build_macro_monthly()

    Returns:
        earnings_df with macro columns added.
    """
    macro = macro_df.copy()
    macro.index = pd.to_datetime(macro.index)

    # Compute "month before rdq" as month-end timestamp
    df = earnings_df.copy()
    df["_macro_month"] = (
        pd.to_datetime(df["rdq"]).dt.to_period("M") - 1
    ).dt.to_timestamp("M")

    macro_reset = macro.reset_index().rename(columns={"month": "_macro_month"})
    macro_reset["_macro_month"] = pd.to_datetime(macro_reset["_macro_month"])

    merged = df.merge(macro_reset, on="_macro_month", how="left")
    merged = merged.drop(columns=["_macro_month"])

    macro_cols = macro_df.columns.tolist()
    filled = merged[macro_cols].notna().all(axis=1).sum()
    print(f"  Macro join: {filled:,} / {len(merged):,} rows matched "
          f"({filled/len(merged)*100:.1f}%)")

    return merged


# ── Standalone: save macro_monthly.csv ───────────────────────────────────────

def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    macro = build_macro_monthly()
    macro.to_csv(MACRO_CSV)
    print(f"\nSaved → data/processed/macro_monthly.csv")
    print(macro.tail(6).round(4).to_string())


if __name__ == "__main__":
    main()
