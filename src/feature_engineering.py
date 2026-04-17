"""
Phase 1 — Feature Engineering Script
======================================
Reads the four raw CSVs from data/raw/, merges them, engineers
predictive features, and writes data/processed/features.csv.

Run from the project root after completing the WRDS pull:
    python src/feature_engineering.py

Output columns in features.csv
────────────────────────────────
Identity:
  gvkey, permno, ticker, conm, rdq (earnings announcement date),
  datadate (fiscal quarter end), sector

Target variable:
  label  — 'beat' | 'meet' | 'miss'  (based on actual EPS vs. mean estimate)

I/B/E/S features:
  mean_est         – mean analyst EPS estimate
  med_est          – median analyst EPS estimate
  n_analysts       – number of analysts
  est_dispersion   – std dev of estimates / abs(mean estimate)  → analyst disagreement
  sue_lag1         – standardized unexpected earnings, prior quarter
  sue_lag2         – SUE two quarters ago

Compustat financial features:
  revenue_growth   – QoQ revenue growth
  gross_margin     – (saleq - cogsq) / saleq  (proxy using oiadpq)
  roa              – niq / atq  (return on assets)
  accruals         – (niq - operating_cf_proxy) / atq  (earnings quality signal)
  current_ratio    – actq / lctq
  asset_growth     – QoQ total asset growth

CRSP market features:
  mom_1m           – prior 1-month return
  mom_3m           – prior 3-month cumulative return
  mom_6m           – prior 6-month cumulative return
  vol_ratio        – trading volume relative to 6-month average

Estimate revision features (computed from I/B/E/S):
  est_revision_1m  – change in mean estimate over prior 30 days
  est_revision_2m  – change in mean estimate over prior 60 days
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from macro_features import build_macro_monthly, join_macro_to_earnings

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ── Surprise thresholds ──────────────────────────────────────────────────────
# Beat:  actual EPS >= estimate + BEAT_THRESH
# Miss:  actual EPS <= estimate - MISS_THRESH
# Meet:  everything in between
BEAT_THRESH = 0.02   # $0.02 per share
MISS_THRESH = 0.02


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading raw CSVs...")
    comp = pd.read_csv(f"{RAW_DIR}/compustat_quarterly.csv",
                       parse_dates=["datadate", "rdq"])
    ibes = pd.read_csv(f"{RAW_DIR}/ibes_summary.csv",
                       parse_dates=["statpers", "fpedats"])
    ccm  = pd.read_csv(f"{RAW_DIR}/ccm_links.csv",
                       parse_dates=["linkdt", "linkenddt"])
    crsp = pd.read_csv(f"{RAW_DIR}/crsp_monthly.csv",
                       parse_dates=["date"])
    print(f"  Compustat: {len(comp):,} rows | I/B/E/S: {len(ibes):,} rows | "
          f"CRSP: {len(crsp):,} rows | CCM: {len(ccm):,} links")
    return comp, ibes, ccm, crsp


def label_surprise(actual: pd.Series, estimate: pd.Series) -> pd.Series:
    """Assign beat / meet / miss label based on EPS surprise."""
    surprise = actual - estimate
    labels = np.where(surprise >= BEAT_THRESH, "beat",
              np.where(surprise <= -MISS_THRESH, "miss", "meet"))
    return pd.Series(labels, index=actual.index)


# ── Step 1: Compustat features ───────────────────────────────────────────────

def build_compustat_features(comp: pd.DataFrame) -> pd.DataFrame:
    print("\nEngineering Compustat features...")
    df = comp.sort_values(["gvkey", "datadate"]).copy()

    # QoQ revenue growth
    df["revenue_growth"] = df.groupby("gvkey")["saleq"].pct_change()

    # Return on assets
    df["roa"] = df["niq"] / df["atq"].replace(0, np.nan)

    # Accruals proxy: net income minus operating income after depreciation, scaled by assets
    # (positive accruals = earnings less backed by cash → negative signal)
    df["accruals"] = (df["niq"] - df["oiadpq"]) / df["atq"].replace(0, np.nan)

    # Current ratio (liquidity)
    df["current_ratio"] = df["actq"] / df["lctq"].replace(0, np.nan)

    # Asset growth QoQ
    df["asset_growth"] = df.groupby("gvkey")["atq"].pct_change()

    # Operating margin proxy
    df["op_margin"] = df["oiadpq"] / df["saleq"].replace(0, np.nan)

    keep = ["gvkey", "datadate", "rdq", "tic", "conm",
            "epspxq", "saleq", "atq",
            "revenue_growth", "roa", "accruals",
            "current_ratio", "asset_growth", "op_margin"]
    return df[keep].dropna(subset=["rdq"])


# ── Step 2: I/B/E/S features ────────────────────────────────────────────────

def build_ibes_features(ibes: pd.DataFrame) -> pd.DataFrame:
    """
    For each (ticker, fpedats) pair, select the most recent estimate
    snapshot before the earnings announcement — we use the last available
    statpers within 90 days before fpedats as our 'consensus' input.
    """
    print("Engineering I/B/E/S features...")
    df = ibes.sort_values(["ticker", "fpedats", "statpers"]).copy()

    # Analyst dispersion: std / |mean|  (NaN if only 1 analyst)
    df["est_dispersion"] = df["stdev"] / df["meanest"].abs().replace(0, np.nan)

    # Keep only snapshots within 90 days before the fiscal period end
    df["days_before_end"] = (df["fpedats"] - df["statpers"]).dt.days
    df = df[(df["days_before_end"] >= 0) & (df["days_before_end"] <= 90)]

    # Most recent snapshot per (ticker, fpedats)
    latest = (df.sort_values("statpers")
                .groupby(["ticker", "fpedats"])
                .last()
                .reset_index())

    # Estimate revision: change in mean estimate over prior 30 / 60 days
    # We compute this by comparing the latest snapshot to earlier ones
    df30 = (df[df["days_before_end"].between(25, 35)]
              .groupby(["ticker", "fpedats"])["meanest"]
              .mean()
              .reset_index()
              .rename(columns={"meanest": "est_30d_ago"}))

    df60 = (df[df["days_before_end"].between(55, 65)]
              .groupby(["ticker", "fpedats"])["meanest"]
              .mean()
              .reset_index()
              .rename(columns={"meanest": "est_60d_ago"}))

    latest = latest.merge(df30, on=["ticker", "fpedats"], how="left")
    latest = latest.merge(df60, on=["ticker", "fpedats"], how="left")
    latest["est_revision_1m"] = latest["meanest"] - latest["est_30d_ago"]
    latest["est_revision_2m"] = latest["meanest"] - latest["est_60d_ago"]

    # Surprise (actual - estimate) and label — only rows where actual is known
    latest_with_actual = latest[latest["actual"].notna()].copy()
    latest_with_actual["label"] = label_surprise(
        latest_with_actual["actual"], latest_with_actual["meanest"]
    )
    latest_with_actual["sue"] = (
        (latest_with_actual["actual"] - latest_with_actual["meanest"])
        / latest_with_actual["stdev"].replace(0, np.nan)
    )

    keep = ["ticker", "fpedats", "meanest", "medest", "numest",
            "est_dispersion", "actual", "label", "sue",
            "est_revision_1m", "est_revision_2m"]
    return latest_with_actual[keep]


# ── Step 3: CRSP momentum features ──────────────────────────────────────────

def build_crsp_features(crsp: pd.DataFrame) -> pd.DataFrame:
    print("Engineering CRSP momentum features...")
    df = crsp.sort_values(["permno", "date"]).copy()

    # Fill missing returns with 0 for rolling calculations
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

    # Compute cumulative returns over rolling windows
    df["ret_1m"] = df.groupby("permno")["ret"].shift(1)
    df["ret_3m"] = (df.groupby("permno")["ret"]
                      .transform(lambda x: (1 + x).rolling(3).apply(np.prod, raw=True) - 1)
                      .shift(1))
    df["ret_6m"] = (df.groupby("permno")["ret"]
                      .transform(lambda x: (1 + x).rolling(6).apply(np.prod, raw=True) - 1)
                      .shift(1))

    # Volume ratio: current month volume / 6-month avg volume
    df["vol_6m_avg"] = (df.groupby("permno")["vol"]
                          .transform(lambda x: x.rolling(6).mean())
                          .shift(1))
    df["vol_ratio"] = df["vol"] / df["vol_6m_avg"].replace(0, np.nan)

    keep = ["permno", "date", "ret_1m", "ret_3m", "ret_6m", "vol_ratio", "prc"]
    return df[keep].dropna(subset=["ret_1m"])


# ── Step 4: Merge everything ─────────────────────────────────────────────────

def merge_all(comp_feat: pd.DataFrame,
              ibes_feat: pd.DataFrame,
              ccm: pd.DataFrame,
              crsp_feat: pd.DataFrame) -> pd.DataFrame:
    print("\nMerging datasets...")

    # 4a. Link Compustat gvkey → CRSP permno via CCM
    #     Only keep valid links (linkenddt is null = still active, or after datadate)
    ccm = ccm.copy()
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31"))

    comp_ccm = comp_feat.merge(
        ccm[["gvkey", "permno", "linkdt", "linkenddt"]],
        on="gvkey", how="left"
    )
    comp_ccm = comp_ccm[
        (comp_ccm["datadate"] >= comp_ccm["linkdt"]) &
        (comp_ccm["datadate"] <= comp_ccm["linkenddt"])
    ].drop(columns=["linkdt", "linkenddt"])

    # 4b. Merge I/B/E/S onto Compustat via ticker + fiscal quarter-end date
    #     I/B/E/S fpedats ≈ Compustat datadate (within 15 days)
    comp_ccm["datadate_dt"] = pd.to_datetime(comp_ccm["datadate"])
    ibes_feat["fpedats_dt"]  = pd.to_datetime(ibes_feat["fpedats"])

    merged = pd.merge_asof(
        comp_ccm.sort_values("datadate_dt"),
        ibes_feat.sort_values("fpedats_dt").rename(columns={"ticker": "ibes_ticker"}),
        left_on="datadate_dt",
        right_on="fpedats_dt",
        left_by="tic",
        right_by="ibes_ticker",
        tolerance=pd.Timedelta("15 days"),
        direction="nearest"
    )

    print(f"  After Compustat-IBES merge: {len(merged):,} rows")

    # 4c. Merge CRSP features — align on month-end date closest to rdq
    #     We want the CRSP snapshot from the month before the earnings date
    merged["rdq_month"] = merged["rdq"].dt.to_period("M").dt.to_timestamp()

    crsp_feat["date_month"] = crsp_feat["date"].dt.to_period("M").dt.to_timestamp()

    merged = merged.merge(
        crsp_feat.rename(columns={"date_month": "rdq_month"}),
        on=["permno", "rdq_month"],
        how="left"
    )

    print(f"  After CRSP merge: {len(merged):,} rows")

    # 4d. Add lagged SUE (prior 1 and 2 quarters)
    merged = merged.sort_values(["gvkey", "datadate"])
    merged["sue_lag1"] = merged.groupby("gvkey")["sue"].shift(1)
    merged["sue_lag2"] = merged.groupby("gvkey")["sue"].shift(2)

    return merged


# ── Step 5: Final cleanup and save ───────────────────────────────────────────

FINAL_FEATURES = [
    # Identity
    "gvkey", "permno", "tic", "conm", "datadate", "rdq",
    # Target
    "label",
    # I/B/E/S
    "meanest", "numest", "est_dispersion",
    "est_revision_1m", "est_revision_2m",
    "sue_lag1", "sue_lag2",
    # Compustat
    "revenue_growth", "roa", "accruals",
    "current_ratio", "asset_growth", "op_margin",
    # CRSP
    "ret_1m", "ret_3m", "ret_6m", "vol_ratio", "prc",
    # Macro (FRED)
    "oil_1m_ret", "oil_3m_ret",
    "vix_level", "vix_1m_chg",
    "gs10_level", "gs10_1m_chg",
    "hy_spread", "hy_spread_chg",
    "gdp_growth",
    "unrate", "unrate_chg",
]


def save_features(df: pd.DataFrame) -> None:
    final = df[[c for c in FINAL_FEATURES if c in df.columns]].copy()
    final = final.dropna(subset=["label"])
    path  = os.path.join(PROC_DIR, "features.csv")
    final.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"\nSaved → data/processed/features.csv  ({size_mb:.1f} MB)")
    print(f"  Total samples: {len(final):,}")
    print(f"  Label distribution:\n{final['label'].value_counts(normalize=True).round(3)}")


def main():
    comp, ibes, ccm, crsp = load_raw()

    comp_feat  = build_compustat_features(comp)
    ibes_feat  = build_ibes_features(ibes)
    crsp_feat  = build_crsp_features(crsp)

    merged = merge_all(comp_feat, ibes_feat, ccm, crsp_feat)

    print("\nJoining macro features from FRED...")
    macro = build_macro_monthly()
    merged = join_macro_to_earnings(merged, macro)

    save_features(merged)

    print("\nNext step: run  python src/train_model.py")


if __name__ == "__main__":
    main()
