"""
spotify_analysis/src/train.py
───────────────────────────────
ML Classification Pipeline — predicts whether a track will chart.

Steps:
  1. Load train / val / test splits
  2. Fix class imbalance with SMOTE on training set only
  3. Train Random Forest + XGBoost
  4. Evaluate both on val set → pick best
  5. Final evaluation on held-out test set
  6. SHAP explainability analysis
  7. Save model, metrics, SHAP values → disk
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ── Load splits ───────────────────────────────────────────────────────────────

def load_splits() -> tuple:
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val   = pd.read_parquet(DATA_DIR / "val.parquet")
    test  = pd.read_parquet(DATA_DIR / "test.parquet")
    meta  = json.loads((DATA_DIR / "feature_meta.json").read_text())
    feature_cols = meta["feature_cols"]
    log.info(
        "Loaded splits — train:%d  val:%d  test:%d  features:%d",
        len(train), len(val), len(test), len(feature_cols),
    )
    return train, val, test, feature_cols


# ── Prepare X / y ─────────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].copy()
    y = df["charted"].values
    return X, y


# ── Impute remaining NaNs ─────────────────────────────────────────────────────

def fit_imputer(X_train: pd.DataFrame):
    imp = SimpleImputer(strategy="median")
    imp.fit(X_train)
    return imp


# ── SMOTE oversampling ────────────────────────────────────────────────────────

def apply_smote(X: np.ndarray, y: np.ndarray) -> tuple:
    log.info("Class distribution before SMOTE: %s", dict(zip(*np.unique(y, return_counts=True))))
    # k_neighbors capped at minority class size - 1
    minority = np.sum(y == 1)
    k = min(5, minority - 1)
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X, y)
    log.info("Class distribution after  SMOTE: %s", dict(zip(*np.unique(y_res, return_counts=True))))
    return X_res, y_res


# ── Models ────────────────────────────────────────────────────────────────────

def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators      = 300,
        max_depth         = 20,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = "sqrt",
        class_weight      = "balanced",   # extra guard on top of SMOTE
        n_jobs            = -1,
        random_state      = RANDOM_STATE,
    )


def build_xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators      = 400,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        scale_pos_weight  = scale_pos_weight,   # handles imbalance natively
        eval_metric       = "auc",
        use_label_encoder = False,
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbosity         = 0,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X: np.ndarray, y: np.ndarray, split_name: str = "val") -> dict:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "split":     split_name,
        "accuracy":  round(float(accuracy_score(y, y_pred)),         4),
        "auc_roc":   round(float(roc_auc_score(y, y_proba)),         4),
        "f1":        round(float(f1_score(y, y_pred)),               4),
        "precision": round(float(precision_score(y, y_pred)),        4),
        "recall":    round(float(recall_score(y, y_pred)),           4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    log.info(
        "[%s] acc=%.4f  auc=%.4f  f1=%.4f  prec=%.4f  rec=%.4f",
        split_name,
        metrics["accuracy"], metrics["auc_roc"],
        metrics["f1"], metrics["precision"], metrics["recall"],
    )
    return metrics


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def compute_shap(model, X: np.ndarray, feature_cols: list, model_name: str) -> dict:
    log.info("Computing SHAP values for %s (sample 2000 rows)…", model_name)
    sample_size = min(2000, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Normalise output across shap versions and model types:
    # - older shap + RF  -> list [class0_arr, class1_arr]
    # - newer shap       -> Explanation object or 3D array (n, features, classes)
    # - XGBoost binary   -> 2D array (n, features)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])          # class 1
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:                        # (n, features, classes)
            sv = sv[:, :, 1]
        elif sv.ndim == 1:
            sv = sv.reshape(1, -1)

    mean_abs_arr = np.abs(sv).mean(axis=0).tolist()
    mean_arr     = sv.mean(axis=0).tolist()

    importance = {
        feat: round(float(imp), 6)
        for feat, imp in sorted(
            zip(feature_cols, mean_abs_arr),
            key=lambda x: float(x[1]), reverse=True,
        )
    }

    mean_shap = {
        feat: round(float(imp), 6)
        for feat, imp in zip(feature_cols, mean_arr)
    }

    log.info("Top 5 SHAP features: %s", list(importance.items())[:5])
    return {
        "mean_abs_shap":  importance,
        "mean_shap":      mean_shap,
        "sample_size":    sample_size,
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(use_smote: bool = True, save: bool = True) -> dict:

    # 1. Load data
    train_df, val_df, test_df, feature_cols = load_splits()

    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val,   y_val   = prepare_xy(val_df,   feature_cols)
    X_test,  y_test  = prepare_xy(test_df,  feature_cols)

    # 2. Impute NaNs (fit on train only)
    log.info("Fitting imputer…")
    imputer  = fit_imputer(X_train)
    X_train  = imputer.transform(X_train)
    X_val    = imputer.transform(X_val)
    X_test   = imputer.transform(X_test)

    # 3. SMOTE on training set only
    if use_smote:
        X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    else:
        X_train_bal, y_train_bal = X_train, y_train

    # scale_pos_weight for XGBoost = n_negative / n_positive (on original train)
    neg  = np.sum(y_train == 0)
    pos  = np.sum(y_train == 1)
    spw  = round(neg / max(pos, 1), 2)
    log.info("XGBoost scale_pos_weight = %.2f", spw)

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    log.info("Training Random Forest…")
    rf = build_random_forest()
    rf.fit(X_train_bal, y_train_bal)
    rf_val_metrics  = evaluate(rf, X_val,  y_val,  "rf_val")
    rf_test_metrics = evaluate(rf, X_test, y_test, "rf_test")
    results["random_forest"] = {
        "val":  rf_val_metrics,
        "test": rf_test_metrics,
    }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    log.info("Training XGBoost…")
    xgb = build_xgboost(scale_pos_weight=spw)
    xgb.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_val_metrics  = evaluate(xgb, X_val,  y_val,  "xgb_val")
    xgb_test_metrics = evaluate(xgb, X_test, y_test, "xgb_test")
    results["xgboost"] = {
        "val":  xgb_val_metrics,
        "test": xgb_test_metrics,
    }

    # ── Pick best model by val AUC ────────────────────────────────────────────
    best_name  = max(["random_forest", "xgboost"],
                     key=lambda m: results[m]["val"]["auc_roc"])
    best_model = rf if best_name == "random_forest" else xgb
    log.info("Best model: %s  (val AUC %.4f)",
             best_name, results[best_name]["val"]["auc_roc"])

    # ── SHAP ──────────────────────────────────────────────────────────────────
    log.info("Running SHAP analysis on both models…")
    rf_shap  = compute_shap(rf,  X_val, feature_cols, "random_forest")
    xgb_shap = compute_shap(xgb, X_val, feature_cols, "xgboost")
    results["shap"] = {
        "random_forest": rf_shap,
        "xgboost":       xgb_shap,
    }

    # ── Feature importance from RF (Gini) ─────────────────────────────────────
    rf_gini = {
        feat: round(float(imp), 6)
        for feat, imp in sorted(
            zip(feature_cols, rf.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
    }
    results["rf_feature_importance_gini"] = rf_gini

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        joblib.dump(rf,      MODELS_DIR / "random_forest.joblib")
        joblib.dump(xgb,     MODELS_DIR / "xgboost.joblib")
        joblib.dump(imputer, MODELS_DIR / "imputer.joblib")

        results["best_model"]   = best_name
        results["feature_cols"] = feature_cols
        results["train_size"]   = len(train_df)
        results["smote_used"]   = use_smote

        (MODELS_DIR / "results.json").write_text(json.dumps(results, indent=2))
        log.info("Models + results saved → %s", MODELS_DIR)

    return results, best_model, imputer, feature_cols


# ── Pretty print summary ──────────────────────────────────────────────────────

def print_summary(results: dict):
    print("\n" + "═" * 60)
    print("  MODEL RESULTS SUMMARY")
    print("═" * 60)

    for model_name in ["random_forest", "xgboost"]:
        m = results[model_name]
        print(f"\n  {model_name.upper().replace('_', ' ')}")
        print(f"  {'Metric':<12} {'Val':>8} {'Test':>8}")
        print(f"  {'-'*30}")
        for metric in ["accuracy", "auc_roc", "f1", "precision", "recall"]:
            print(f"  {metric:<12} {m['val'][metric]:>8.4f} {m['test'][metric]:>8.4f}")

    best = results["best_model"]
    print(f"\n  ★  Best model: {best.upper().replace('_', ' ')}")
    print(f"     Val AUC:  {results[best]['val']['auc_roc']:.4f}")
    print(f"     Test AUC: {results[best]['test']['auc_roc']:.4f}")

    print(f"\n  Top 8 features (SHAP — {best}):")
    shap_importance = results["shap"][best]["mean_abs_shap"]
    for i, (feat, val) in enumerate(list(shap_importance.items())[:8]):
        bar = "█" * int(val / max(shap_importance.values()) * 20)
        print(f"    {i+1}. {feat:<25} {val:.4f}  {bar}")

    print("═" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    results, best_model, imputer, feature_cols = train(use_smote=True, save=True)
    print_summary(results)
