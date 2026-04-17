"""
Backtesting Page
=================
Historical model accuracy, confusion matrix, and performance over time.
All computed from the test set (2022–2024) stored in features.csv.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT       = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_PATH  = os.path.join(ROOT, "data", "processed", "features.csv")

LABEL_COLORS = {"beat": "#2ecc71", "meet": "#f39c12", "miss": "#e74c3c"}
VAL_END      = "2022-01-01"


@st.cache_resource(show_spinner=False)
def load_artifacts():
    return {
        "lgbm":    joblib.load(f"{MODELS_DIR}/lightgbm_model.joblib"),
        "lr":      joblib.load(f"{MODELS_DIR}/logistic_regression.joblib"),
        "imputer": joblib.load(f"{MODELS_DIR}/imputer.joblib"),
        "scaler":  joblib.load(f"{MODELS_DIR}/scaler.joblib"),
        "le":      joblib.load(f"{MODELS_DIR}/label_encoder.joblib"),
        "win_low": joblib.load(f"{MODELS_DIR}/win_low.joblib"),
        "win_high":joblib.load(f"{MODELS_DIR}/win_high.joblib"),
        "feature_cols": joblib.load(f"{MODELS_DIR}/feature_cols.joblib"),
    }


@st.cache_data(show_spinner="Running backtest on 2022–2024 data…")
def run_backtest():
    art = load_artifacts()
    df  = pd.read_csv(DATA_PATH, parse_dates=["datadate"])
    test = df[df["datadate"] >= VAL_END].dropna(subset=["label"]).copy()

    feat_cols = art["feature_cols"]
    X = test[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=art["win_low"], upper=art["win_high"], axis=1)
    X_imp    = art["imputer"].transform(X)
    X_scaled = art["scaler"].transform(X_imp)

    le = art["le"]
    y_true = le.transform(test["label"])

    y_lgbm = art["lgbm"].predict(X_imp)
    y_lr   = art["lr"].predict(X_scaled)

    # Per-quarter accuracy
    test = test.reset_index(drop=True)
    test["pred_lgbm"]  = le.inverse_transform(y_lgbm)
    test["pred_lr"]    = le.inverse_transform(y_lr)
    test["correct_lgbm"] = test["label"] == test["pred_lgbm"]
    test["quarter"]    = test["datadate"].dt.to_period("Q").astype(str)

    quarterly = (test.groupby("quarter")["correct_lgbm"]
                     .mean()
                     .reset_index()
                     .rename(columns={"correct_lgbm": "accuracy"}))

    return {
        "y_true": y_true, "y_lgbm": y_lgbm, "y_lr": y_lr,
        "le": le, "test": test, "quarterly": quarterly,
        "acc_lgbm": accuracy_score(y_true, y_lgbm),
        "acc_lr":   accuracy_score(y_true, y_lr),
    }


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("Model Backtesting")
st.markdown("Out-of-sample performance on the **2022–2024 test set** (never seen during training).")

bt = run_backtest()
le = bt["le"]

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("LightGBM Accuracy",       f"{bt['acc_lgbm']*100:.1f}%")
c2.metric("Logistic Regression",     f"{bt['acc_lr']*100:.1f}%")
c3.metric("Test Set Size",           f"{len(bt['test']):,}")
c4.metric("Test Period",             "2022 – 2024")

st.divider()

tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Accuracy Over Time", "Per-Class Metrics"])

# ── Tab 1: Confusion matrix ───────────────────────────────────────────────────
with tab1:
    cm = confusion_matrix(bt["y_true"], bt["y_lgbm"])
    labels = le.classes_

    # Normalize by row (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig_cm = go.Figure(go.Heatmap(
        z=cm_norm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"Actual: {l}" for l in labels],
        colorscale="Blues",
        text=[[f"{cm[i,j]:,}<br>({cm_norm[i,j]*100:.1f}%)"
               for j in range(len(labels))] for i in range(len(labels))],
        texttemplate="%{text}",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        showscale=False,
    ))
    fig_cm.update_layout(
        title="LightGBM Confusion Matrix (row-normalized)",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.caption("Diagonal = correct predictions. Off-diagonal = errors. "
               "Rows sum to 100% (recall per actual class).")

# ── Tab 2: Accuracy over time ─────────────────────────────────────────────────
with tab2:
    q = bt["quarterly"]

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=q["quarter"], y=q["accuracy"] * 100,
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(size=7),
        name="LightGBM",
    ))
    fig_ts.add_hline(y=bt["acc_lgbm"] * 100,
                     line_dash="dash", line_color="#888",
                     annotation_text=f"Overall {bt['acc_lgbm']*100:.1f}%",
                     annotation_position="right")
    fig_ts.add_hline(y=33.3, line_dash="dot", line_color="#e74c3c",
                     annotation_text="Random baseline (33%)",
                     annotation_position="right")

    fig_ts.update_layout(
        title="Quarterly Accuracy — Test Set (2022–2024)",
        yaxis=dict(title="Accuracy (%)", range=[0, 100]),
        xaxis=dict(title="Quarter"),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ── Tab 3: Per-class metrics ──────────────────────────────────────────────────
with tab3:
    report = classification_report(
        bt["y_true"], bt["y_lgbm"],
        target_names=le.classes_, output_dict=True
    )

    metrics_rows = []
    for cls in le.classes_:
        r = report[cls]
        metrics_rows.append({
            "Class":     cls.upper(),
            "Precision": f"{r['precision']:.3f}",
            "Recall":    f"{r['recall']:.3f}",
            "F1-Score":  f"{r['f1-score']:.3f}",
            "Support":   f"{int(r['support']):,}",
        })

    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Bar chart of F1 by class
    fig_f1 = go.Figure(go.Bar(
        x=[r["Class"] for r in metrics_rows],
        y=[float(r["F1-Score"]) for r in metrics_rows],
        marker_color=[LABEL_COLORS[c.lower()] for r in metrics_rows
                      for c in [r["Class"].lower()]],
        text=[r["F1-Score"] for r in metrics_rows],
        textposition="outside",
    ))
    fig_f1.update_layout(
        title="F1-Score by Class",
        yaxis=dict(range=[0, 1.1], title="F1-Score"),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_f1, use_container_width=True)

    st.markdown("""
    **Notes:**
    - **Beat** has the highest F1 — the most predictable outcome (companies tend to manage to consensus).
    - **Meet** is hardest — it's the boundary case between beat and miss.
    - Random baseline for a 3-class problem is 33%. The model runs at 60.9%.
    """)
