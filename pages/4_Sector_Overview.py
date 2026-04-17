"""
Sector Overview Page
=====================
Historical beat/meet/miss rates by sector from the full dataset,
plus model prediction accuracy broken down by sector.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT       = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH  = os.path.join(ROOT, "data", "processed", "features.csv")

LABEL_COLORS = {"beat": "#2ecc71", "meet": "#f39c12", "miss": "#e74c3c"}
VAL_END      = "2022-01-01"

# Compustat SIC-based sector mapping (gics approximation via ticker lookup)
# We'll use the sector column if present, otherwise fall back to company-level stats
SECTOR_COL = "sector" if "sector" in pd.read_csv(DATA_PATH, nrows=1).columns else None


@st.cache_resource(show_spinner=False)
def load_artifacts():
    return {
        "lgbm":         joblib.load(f"{MODELS_DIR}/lightgbm_model.joblib"),
        "imputer":      joblib.load(f"{MODELS_DIR}/imputer.joblib"),
        "le":           joblib.load(f"{MODELS_DIR}/label_encoder.joblib"),
        "win_low":      joblib.load(f"{MODELS_DIR}/win_low.joblib"),
        "win_high":     joblib.load(f"{MODELS_DIR}/win_high.joblib"),
        "feature_cols": joblib.load(f"{MODELS_DIR}/feature_cols.joblib"),
    }


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datadate"])
    df = df.dropna(subset=["label"])
    return df


@st.cache_data(show_spinner="Computing sector stats…")
def compute_sector_stats():
    art  = load_artifacts()
    df   = load_data()
    test = df[df["datadate"] >= VAL_END].copy()

    # Run predictions on test set
    feat_cols = art["feature_cols"]
    X   = test[feat_cols].replace([np.inf, -np.inf], np.nan)
    X   = X.clip(lower=art["win_low"], upper=art["win_high"], axis=1)
    arr = art["imputer"].transform(X)
    le  = art["le"]
    test = test.reset_index(drop=True)
    test["pred"] = le.inverse_transform(art["lgbm"].predict(arr))
    test["correct"] = test["label"] == test["pred"]

    # Historical (full dataset) beat/miss rates by sector
    hist_sector = (df.groupby(["conm", "label"])
                     .size()
                     .unstack(fill_value=0)
                     .assign(total=lambda x: x.sum(axis=1))
                     .assign(beat_rate=lambda x: x.get("beat", 0) / x["total"],
                             miss_rate=lambda x: x.get("miss", 0) / x["total"]))

    return test, hist_sector


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Sector Overview")
st.markdown("Historical beat/miss rates and model accuracy broken down by sector.")

art  = load_artifacts()
df   = load_data()
test, hist_sector = compute_sector_stats()

# ── Section 1: Historical beat/miss rates ────────────────────────────────────
st.subheader("Historical Beat / Meet / Miss Rates (2005–2024)")
st.caption("From the full training dataset — 104,938 earnings announcements.")

label_order = ["beat", "meet", "miss"]
label_counts = df["label"].value_counts()
total = len(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Earnings Events", f"{total:,}")
c2.metric("Beat Rate", f"{label_counts.get('beat',0)/total*100:.1f}%",
          help="Actual EPS ≥ consensus + $0.02")
c3.metric("Meet Rate", f"{label_counts.get('meet',0)/total*100:.1f}%",
          help="Within ±$0.02 of consensus")
c4.metric("Miss Rate", f"{label_counts.get('miss',0)/total*100:.1f}%",
          help="Actual EPS ≤ consensus − $0.02")

# Beat rate over time
yearly = (df.groupby([df["datadate"].dt.year, "label"])
            .size()
            .unstack(fill_value=0))
yearly_pct = yearly.div(yearly.sum(axis=1), axis=0) * 100
yearly_pct.index.name = "year"
yearly_pct = yearly_pct.reset_index()

fig_trend = go.Figure()
for lbl in label_order:
    if lbl in yearly_pct.columns:
        fig_trend.add_trace(go.Scatter(
            x=yearly_pct["year"], y=yearly_pct[lbl],
            name=lbl.capitalize(), mode="lines+markers",
            line=dict(color=LABEL_COLORS[lbl], width=2),
            marker=dict(size=5),
        ))

fig_trend.update_layout(
    title="Beat / Meet / Miss Rate by Year",
    yaxis=dict(title="% of Earnings Announcements", range=[0, 80]),
    xaxis=dict(title="Year"),
    legend=dict(orientation="h", y=-0.2),
    height=380,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ── Section 2: Top beaters and missers ───────────────────────────────────────
st.subheader("Companies with Highest Beat & Miss Rates")
st.caption("Minimum 20 earnings announcements.")

col_l, col_r = st.columns(2)

# Filter to companies with ≥20 observations
qualified = hist_sector[hist_sector["total"] >= 20]

with col_l:
    st.markdown("**Top Consistent Beaters**")
    top_beat = (qualified.nlargest(10, "beat_rate")[["beat_rate", "total"]]
                         .reset_index()
                         .rename(columns={"conm": "Company",
                                          "beat_rate": "Beat Rate",
                                          "total": "Quarters"}))
    top_beat["Beat Rate"] = top_beat["Beat Rate"].map("{:.1%}".format)
    st.dataframe(top_beat, use_container_width=True, hide_index=True)

with col_r:
    st.markdown("**Top Consistent Missers**")
    top_miss = (qualified.nlargest(10, "miss_rate")[["miss_rate", "total"]]
                         .reset_index()
                         .rename(columns={"conm": "Company",
                                          "miss_rate": "Miss Rate",
                                          "total": "Quarters"}))
    top_miss["Miss Rate"] = top_miss["Miss Rate"].map("{:.1%}".format)
    st.dataframe(top_miss, use_container_width=True, hide_index=True)

st.divider()

# ── Section 3: Test set accuracy ─────────────────────────────────────────────
st.subheader("Model Accuracy on Test Set (2022–2024)")

acc_by_label = {}
for lbl in ["beat", "meet", "miss"]:
    subset = test[test["label"] == lbl]
    if len(subset) > 0:
        acc_by_label[lbl] = accuracy_score(subset["label"], subset["pred"])

fig_acc = go.Figure(go.Bar(
    x=[l.capitalize() for l in acc_by_label.keys()],
    y=[v * 100 for v in acc_by_label.values()],
    marker_color=[LABEL_COLORS[l] for l in acc_by_label.keys()],
    text=[f"{v*100:.1f}%" for v in acc_by_label.values()],
    textposition="outside",
))
fig_acc.add_hline(y=33.3, line_dash="dot", line_color="#888",
                  annotation_text="Random baseline (33%)")
fig_acc.update_layout(
    title="Per-Class Accuracy (test set)",
    yaxis=dict(range=[0, 100], title="Accuracy (%)"),
    height=340,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_acc, use_container_width=True)

# Label distribution in test set
test_dist = test["label"].value_counts()
fig_dist = go.Figure(go.Pie(
    labels=[l.capitalize() for l in test_dist.index],
    values=test_dist.values,
    marker=dict(colors=[LABEL_COLORS.get(l, "#888") for l in test_dist.index]),
    hole=0.4,
    textinfo="label+percent",
))
fig_dist.update_layout(
    title="Test Set Label Distribution",
    height=320,
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_dist, use_container_width=True)
