import os
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from features import build_features, FEATURE_COLUMNS

def train_and_evaluate_models():
    os.makedirs("models", exist_ok=True)
    
    data_path = "data/processed/labeled_disputes.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}. Run src/make_labels.py first.")
        
    df = pd.read_csv(data_path)
    
    # CRITICAL: Strict temporal sorting by dispute_date to prevent future leakage
    df["dispute_date"] = pd.to_datetime(df["dispute_date"])
    df = df.sort_values("dispute_date").reset_index(drop=True)
    
    # Extract features using src/features.py
    X = build_features(df)
    y = df["won_dispute"].values
    
    # 80/20 Time-based split
    split_idx = int(len(df) * 0.8)
    
    X_train, y_train = X.iloc[:split_idx], y[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]
    
    train_dates = (df["dispute_date"].iloc[0], df["dispute_date"].iloc[split_idx-1])
    test_dates = (df["dispute_date"].iloc[split_idx], df["dispute_date"].iloc[-1])
    
    print(f"Dataset Total: {len(df)} records")
    print(f"Train Period: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[1].strftime('%Y-%m-%d')} ({len(X_train)} rows)")
    print(f"Test Period:  {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[1].strftime('%Y-%m-%d')} ({len(X_test)} rows)")
    
    # 1. Train Baseline Model (Logistic Regression)
    scale_pos = float((y_train == 0).sum() / (y_train == 1).sum())
    baseline_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    baseline_model.fit(X_train, y_train)
    
    baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
    baseline_prauc = average_precision_score(y_test, baseline_probs)
    baseline_rocauc = roc_auc_score(y_test, baseline_probs)
    
    # 2. Train Winnability Classifier (XGBoost)
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_prauc = average_precision_score(y_test, xgb_probs)
    xgb_rocauc = roc_auc_score(y_test, xgb_probs)
    
    # Standard threshold 0.5 metrics comparison
    xgb_preds_05 = (xgb_probs >= 0.5).astype(int)
    base_preds_05 = (baseline_probs >= 0.5).astype(int)
    
    print("\n--- MODEL COMPARISON ON HELD-OUT TEST SET ---")
    print(f"Baseline (Logistic Regression) -> PR-AUC: {baseline_prauc:.4f} | ROC-AUC: {baseline_rocauc:.4f}")
    print(f"Winnability Engine (XGBoost)   -> PR-AUC: {xgb_prauc:.4f} | ROC-AUC: {xgb_rocauc:.4f}")
    print(f"PR-AUC Lift over Baseline: +{(xgb_prauc - baseline_prauc)*100:.2f}% points")
    
    # Save artifacts
    joblib.dump(xgb_model, "models/winnability_model.pkl")
    joblib.dump(baseline_model, "models/baseline_model.pkl")
    
    with open("models/feature_names.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
        
    summary = {
        "dataset_total": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "split_type": "time_based",
        "train_period": [d.strftime("%Y-%m-%d") for d in train_dates],
        "test_period": [d.strftime("%Y-%m-%d") for d in test_dates],
        "baseline_logistic_regression": {
            "pr_auc": round(float(baseline_prauc), 4),
            "roc_auc": round(float(baseline_rocauc), 4),
            "precision_05": round(float(precision_score(y_test, base_preds_05)), 4),
            "recall_05": round(float(recall_score(y_test, base_preds_05)), 4),
            "f1_05": round(float(f1_score(y_test, base_preds_05)), 4),
        },
        "xgboost_winnability_model": {
            "pr_auc": round(float(xgb_prauc), 4),
            "roc_auc": round(float(xgb_rocauc), 4),
            "precision_05": round(float(precision_score(y_test, xgb_preds_05)), 4),
            "recall_05": round(float(recall_score(y_test, xgb_preds_05)), 4),
            "f1_05": round(float(f1_score(y_test, xgb_preds_05)), 4),
        }
    }
    
    with open("models/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Successfully saved models and training summary to models/")

if __name__ == "__main__":
    train_and_evaluate_models()
