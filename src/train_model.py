"""
Phase 2 — Model Training Script
=================================
Reads data/processed/features.csv, trains two classifiers, evaluates them
on a time-based hold-out set, and saves artifacts to models/.

Time splits:
  Train:      2005 – 2019
  Validation: 2020 – 2021  (LightGBM early stopping)
  Test:       2022 – 2024  (final held-out evaluation)

Models trained:
  1. Logistic Regression  (baseline — with StandardScaler)
  2. LightGBM             (main model — with early stopping on val set)

Artifacts saved to models/:
  logistic_regression.joblib
  lightgbm_model.joblib
  imputer.joblib          — median imputer fitted on train set
  scaler.joblib           — StandardScaler fitted on train set (for LR)
  label_encoder.joblib    — maps beat/meet/miss <-> integers
  feature_cols.joblib     — ordered list of feature column names

Run from the project root:
    python src/train_model.py
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model  import LogisticRegression
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics       import classification_report, confusion_matrix, accuracy_score

import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "processed", "features.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Feature columns (16 predictors — revision cols excluded, 100% NaN) ───────
FEATURE_COLS = [
    # I/B/E/S
    "meanest", "numest", "est_dispersion",
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

# Time-based splits
TRAIN_END = "2020-01-01"   # train: 2005 – 2019
VAL_END   = "2022-01-01"   # val:   2020 – 2021
                            # test:  2022 – 2024

# Winsorize outliers at these percentiles before imputing
WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99


# ── Load & split ─────────────────────────────────────────────────────────────

def load_and_split():
    print("Loading features.csv...")
    df = pd.read_csv(DATA_PATH, parse_dates=["datadate"])
    df = df.dropna(subset=["label"])

    train = df[df["datadate"] <  TRAIN_END].copy()
    val   = df[(df["datadate"] >= TRAIN_END) & (df["datadate"] < VAL_END)].copy()
    test  = df[df["datadate"] >= VAL_END].copy()

    print(f"  Train: {len(train):,} rows  ({train['datadate'].min().date()} – "
          f"{train['datadate'].max().date()})")
    print(f"  Val:   {len(val):,} rows  ({val['datadate'].min().date()} – "
          f"{val['datadate'].max().date()})")
    print(f"  Test:  {len(test):,} rows  ({test['datadate'].min().date()} – "
          f"{test['datadate'].max().date()})")

    return (train[FEATURE_COLS], val[FEATURE_COLS], test[FEATURE_COLS],
            train["label"],      val["label"],      test["label"])


# ── Preprocessing ─────────────────────────────────────────────────────────────

def replace_inf(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan)


def winsorize(X: pd.DataFrame, low: pd.Series, high: pd.Series) -> pd.DataFrame:
    """Clip each column to [low, high] percentile bounds computed on train."""
    return X.clip(lower=low, upper=high, axis=1)


def fit_preprocessors(X_train_raw):
    """
    Fit on training data only:
      1. Compute winsorization bounds (1st / 99th percentile)
      2. Fit median imputer
      3. Fit StandardScaler
    Returns (win_low, win_high, imputer, scaler).
    """
    X = replace_inf(X_train_raw)

    win_low  = X.quantile(WINSOR_LOW)
    win_high = X.quantile(WINSOR_HIGH)

    X_win = winsorize(X, win_low, win_high)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_win)
    X_imp = imputer.transform(X_win)

    scaler = StandardScaler()
    scaler.fit(X_imp)

    return win_low, win_high, imputer, scaler


def preprocess(X_raw, win_low, win_high, imputer, scaler):
    X = replace_inf(X_raw)
    X = winsorize(X, win_low, win_high)
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    return X_imp, X_scaled   # raw-imputed for LightGBM, scaled for LR


# ── Model 1: Logistic Regression ─────────────────────────────────────────────

def train_logistic(X_train_scaled, y_enc_train):
    print("\nTraining Logistic Regression (scaled features)...")
    lr = LogisticRegression(
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        C=0.5,
        n_jobs=-1,
    )
    lr.fit(X_train_scaled, y_enc_train)
    return lr


# ── Model 2: LightGBM ────────────────────────────────────────────────────────

def train_lgbm(X_train_imp, y_train_enc, X_val_imp, y_val_enc, le):
    print("Training LightGBM (early stopping on validation set)...")
    n_classes = len(le.classes_)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=2000,        # large ceiling; early stopping will find the right number
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train_imp, y_train_enc,
        eval_set=[(X_val_imp, y_val_enc)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(name, model, X_imp, y_enc, le):
    print(f"\n{'─'*52}")
    print(f"  {name}")
    print(f"{'─'*52}")
    y_pred = model.predict(X_imp)
    acc = accuracy_score(y_enc, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_enc, y_pred, target_names=le.classes_, digits=3))
    cm = confusion_matrix(y_enc, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(cm_df.to_string())
    return acc


# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(model):
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    print("\n  LightGBM feature importance (top 10):")
    for feat, score in imp.head(10).items():
        bar = "█" * int(score / imp.max() * 20)
        print(f"    {feat:<20} {bar} {score:.0f}")


# ── Save artifacts ────────────────────────────────────────────────────────────

def save_artifacts(lr, lgbm, win_low, win_high, imputer, scaler, le):
    print("\nSaving artifacts to models/...")
    artifacts = {
        "logistic_regression.joblib": lr,
        "lightgbm_model.joblib":      lgbm,
        "win_low.joblib":             win_low,
        "win_high.joblib":            win_high,
        "imputer.joblib":             imputer,
        "scaler.joblib":              scaler,
        "label_encoder.joblib":       le,
        "feature_cols.joblib":        FEATURE_COLS,
    }
    for fname, obj in artifacts.items():
        joblib.dump(obj, os.path.join(MODEL_DIR, fname))
        print(f"  Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

    print("\nFitting preprocessors on training data...")
    win_low, win_high, imputer, scaler = fit_preprocessors(X_train)

    X_train_imp, X_train_sc = preprocess(X_train, win_low, win_high, imputer, scaler)
    X_val_imp,   X_val_sc   = preprocess(X_val,   win_low, win_high, imputer, scaler)
    X_test_imp,  X_test_sc  = preprocess(X_test,  win_low, win_high, imputer, scaler)

    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Train label distribution:\n{pd.Series(y_train).value_counts()}")

    lr   = train_logistic(X_train_sc, y_train_enc)
    lgbm = train_lgbm(X_train_imp, y_train_enc, X_val_imp, y_val_enc, le)

    print_feature_importance(lgbm)

    acc_lr   = evaluate("Logistic Regression (test set)", lr,   X_test_sc,  y_test_enc, le)
    acc_lgbm = evaluate("LightGBM (test set)",            lgbm, X_test_imp, y_test_enc, le)

    print(f"\n{'='*52}")
    print(f"  Final Results")
    print(f"{'='*52}")
    print(f"  Logistic Regression accuracy: {acc_lr:.4f}")
    print(f"  LightGBM accuracy:            {acc_lgbm:.4f}")
    print(f"  Best model: {'LightGBM' if acc_lgbm >= acc_lr else 'Logistic Regression'}")

    save_artifacts(lr, lgbm, win_low, win_high, imputer, scaler, le)
    print("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
