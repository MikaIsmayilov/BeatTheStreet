"""
Phase 1 — WRDS Data Pull Script
================================
Pulls four datasets from WRDS and saves them as CSVs to data/raw/.

Datasets:
  1. Compustat Quarterly Fundamentals  (comp.fundq)
  2. I/B/E/S Analyst EPS Estimates     (ibes.statsum_epsus)
  3. CCM Link Table                    (crsp.ccmxpf_linktable)
  4. CRSP Monthly Stock Returns        (crsp.msf)

Run from the project root:
    python src/wrds_pull.py

You will be prompted for your WRDS password on the first run.
Credentials are then stored in ~/.pgpass so you won't need to re-enter.

Requirements:
    pip install wrds pandas
"""

import os
import wrds
import pandas as pd

# ── Output folder ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Date range ───────────────────────────────────────────────────────────────
START_DATE = "2005-01-01"
END_DATE   = "2024-12-31"


def connect() -> wrds.Connection:
    """Open a WRDS connection.  Password is prompted once, then cached in ~/.pgpass."""
    print("Connecting to WRDS...")
    db = wrds.Connection(wrds_username="mikaismayilli")
    print("Connected.\n")
    return db


def pull_compustat(db: wrds.Connection) -> pd.DataFrame:
    """
    Compustat Fundamentals Quarterly (comp.fundq)
    ---------------------------------------------
    Key columns:
      gvkey   – Compustat firm identifier
      datadate – Fiscal quarter-end date
      rdq     – Earnings announcement date (critical for event-study alignment)
      tic     – Ticker symbol
      conm    – Company name
      epspxq  – EPS excluding extraordinary items (actual reported EPS)
      saleq   – Net sales / revenue
      niq     – Net income
      oiadpq  – Operating income after depreciation
      atq     – Total assets
      ltq     – Total liabilities
      ceqq    – Common equity
      actq    – Current assets
      lctq    – Current liabilities
    """
    print("Pulling Compustat Quarterly Fundamentals...")
    query = f"""
        SELECT
            gvkey, datadate, rdq, tic, conm, fic,
            epspxq, saleq, niq, oiadpq,
            atq, ltq, ceqq, actq, lctq,
            cshoq, prccq
        FROM comp.fundq
        WHERE fic = 'USA'
          AND datadate BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND indfmt  = 'INDL'
          AND datafmt = 'STD'
          AND popsrc  = 'D'
          AND consol  = 'C'
          AND epspxq IS NOT NULL
    """
    df = db.raw_sql(query, date_cols=["datadate", "rdq"])
    print(f"  → {len(df):,} rows, {df['gvkey'].nunique():,} unique firms")
    return df


def pull_ibes(db: wrds.Connection) -> pd.DataFrame:
    """
    I/B/E/S Summary Statistics — Quarterly EPS Estimates (ibes.statsum_epsus)
    --------------------------------------------------------------------------
    fpi = '6' means next fiscal quarter EPS forecast.

    Key columns:
      ticker  – I/B/E/S ticker (different from Compustat tic)
      statpers – Statistics period (date estimates were compiled)
      fpedats  – Fiscal period end date (the quarter being forecast)
      meanest  – Mean analyst EPS estimate
      medest   – Median analyst EPS estimate
      stdev    – Standard deviation of estimates (analyst dispersion)
      numest   – Number of analysts contributing estimates
      actual   – Actual reported EPS (filled in after announcement)
    """
    print("Pulling I/B/E/S Analyst EPS Estimates...")
    query = f"""
        SELECT
            ticker, statpers, fpedats,
            meanest, medest, stdev, numest, actual
        FROM ibes.statsum_epsus
        WHERE statpers BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND fpi      = '6'
          AND measure  = 'EPS'
          AND numest   >= 2
    """
    # fpi='6' = next fiscal quarter (not '1' which is next fiscal year)
    df = db.raw_sql(query, date_cols=["statpers", "fpedats"])
    print(f"  → {len(df):,} rows, {df['ticker'].nunique():,} unique tickers")
    return df


def pull_ccm(db: wrds.Connection) -> pd.DataFrame:
    """
    CRSP-Compustat Merged (CCM) Link Table (crsp.ccmxpf_linktable)
    ---------------------------------------------------------------
    Links gvkey (Compustat) to permno (CRSP) so we can merge the two datasets.

    linktype IN ('LU', 'LC') = primary links (use these, not secondary links)
    linkprim IN ('P', 'C')   = primary issue / primary class
    """
    print("Pulling CCM Link Table...")
    query = """
        SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
        FROM crsp.ccmxpf_linktable
        WHERE linktype  IN ('LU', 'LC')
          AND linkprim  IN ('P', 'C')
    """
    df = db.raw_sql(query, date_cols=["linkdt", "linkenddt"])
    print(f"  → {len(df):,} link records")
    return df


def pull_crsp(db: wrds.Connection, permnos: list) -> pd.DataFrame:
    """
    CRSP Monthly Stock File (crsp.msf)
    ------------------------------------
    Filtered to the permnos we actually need (firms in CCM) to keep
    the file size manageable.

    Key columns:
      permno  – CRSP firm identifier
      date    – Month-end date
      ret     – Monthly return (raw)
      vol     – Trading volume (shares)
      shrout  – Shares outstanding (thousands)
      prc     – Closing price (negative = average of bid/ask)
    """
    print("Pulling CRSP Monthly Returns (filtered to CCM firms)...")
    permno_list = ",".join(str(p) for p in permnos)
    query = f"""
        SELECT permno, date, ret, vol, shrout, prc
        FROM crsp.msf
        WHERE date    BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND permno  IN ({permno_list})
    """
    df = db.raw_sql(query, date_cols=["date"])
    print(f"  → {len(df):,} rows, {df['permno'].nunique():,} unique firms")
    return df


def save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  Saved → data/raw/{name}.csv  ({size_mb:.1f} MB)\n")


def main():
    db = connect()

    # 1. Compustat
    comp = pull_compustat(db)
    save(comp, "compustat_quarterly")

    # 2. I/B/E/S
    ibes = pull_ibes(db)
    save(ibes, "ibes_summary")

    # 3. CCM link table
    ccm = pull_ccm(db)
    save(ccm, "ccm_links")

    # 4. CRSP — filtered to permnos that appear in CCM
    permnos = ccm["permno"].dropna().astype(int).unique().tolist()
    crsp = pull_crsp(db, permnos)
    save(crsp, "crsp_monthly")

    db.close()
    print("All done! Files saved to data/raw/")
    print("Next step: run  python src/feature_engineering.py")


if __name__ == "__main__":
    main()
