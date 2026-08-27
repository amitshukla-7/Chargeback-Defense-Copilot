import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from features import build_features

COST_FN = 250   # ₹250 loss when merchant forfeits a winnable chargeback
COST_FP = 15    # ₹15 loss in analyst ops time when contesting a losing chargeback

def evaluate_cost_optimal_threshold():
    os.makedirs("models", exist_ok=True)
    
    # Load model and data
    model = joblib.load("models/winnability_model.pkl")
    df = pd.read_csv("data/processed/labeled_disputes.csv")
    df["dispute_date"] = pd.to_datetime(df["dispute_date"])
    df = df.sort_values("dispute_date").reset_index(drop=True)
    
    X = build_features(df)
    y = df["won_dispute"].values
    
    # 80/20 Time-based split
    split_idx = int(len(df) * 0.8)
    X_test, y_test = X.iloc[split_idx:], y[split_idx:]
    
    probs = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.linspace(0.05, 0.95, 19)
    costs = []
    threshold_metrics = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        fn = int(((preds == 0) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        tp = int(((preds == 1) & (y_test == 1)).sum())
        tn = int(((preds == 0) & (y_test == 0)).sum())
        
        total_cost = fn * COST_FN + fp * COST_FP
        costs.append(total_cost)
        
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        threshold_metrics.append({
            "threshold": round(float(t), 2),
            "total_cost_inr": total_cost,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4)
        })
        
    best_idx = int(np.argmin(costs))
    best_t = float(thresholds[best_idx])
    min_cost = int(costs[best_idx])
    
    # 3-Way Baseline Financial Comparison
    # Baseline 1: Contest Everything (Threshold = 0.0) -> contest all 300 disputes
    all_contest_preds = np.ones_like(y_test)
    fn_all = int(((all_contest_preds == 0) & (y_test == 1)).sum()) # 0
    fp_all = int(((all_contest_preds == 1) & (y_test == 0)).sum()) # all 0-labels
    cost_contest_everything = fn_all * COST_FN + fp_all * COST_FP
    
    # Baseline 2: Contest Nothing (Threshold = 1.0) -> contest 0 disputes
    no_contest_preds = np.zeros_like(y_test)
    fn_none = int(((no_contest_preds == 0) & (y_test == 1)).sum()) # all 1-labels
    fp_none = int(((no_contest_preds == 1) & (y_test == 0)).sum()) # 0
    cost_contest_nothing = fn_none * COST_FN + fp_none * COST_FP
    
    savings_vs_everything = cost_contest_everything - min_cost
    savings_vs_nothing = cost_contest_nothing - min_cost
    
    print("\n=======================================================")
    print("       COST-AWARE THRESHOLD EVALUATION SUMMARY        ")
    print("=======================================================")
    print(f"Optimal Threshold (best_t): {best_t:.2f}")
    print(f"Minimum Expected Cost on Test Set: INR {min_cost:,}")
    print(f"Cost of 'Contest Everything' Baseline: INR {cost_contest_everything:,}")
    print(f"Cost of 'Contest Nothing' Baseline:    INR {cost_contest_nothing:,}")
    print(f"Savings vs Contest Everything:          INR {savings_vs_everything:,}")
    print(f"Savings vs Contest Nothing:             INR {savings_vs_nothing:,}")
    
    best_metrics = threshold_metrics[best_idx]
    print("\nMetrics at Optimal Threshold:")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1 Score:  {best_metrics['f1']:.4f}")
    print(f"Confusion Matrix (TP={best_metrics['tp']}, TN={best_metrics['tn']}, FP={best_metrics['fp']}, FN={best_metrics['fn']})")
    
    # Generate & Save Cost Curve Plot
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, costs, marker='o', linewidth=2.5, color='#1E88E5', label='Total Expected Risk Cost (INR)')
    plt.axvline(best_t, color='#D32F2F', linestyle='--', linewidth=2, label=f'Optimal Threshold ({best_t:.2f})')
    plt.scatter([best_t], [min_cost], color='#D32F2F', s=120, zorder=5)
    plt.title("Cost-Optimal Decision Threshold Tuning", fontsize=13, fontweight='bold')
    plt.xlabel("Contest Decision Threshold (Win Probability)", fontsize=11)
    plt.ylabel("Total Expected Risk Cost (INR)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_path = "models/cost_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Save JSON summary
    res = {
        "cost_fn_inr": COST_FN,
        "cost_fp_inr": COST_FP,
        "optimal_threshold": best_t,
        "test_set_min_cost": min_cost,
        "contest_everything_cost": cost_contest_everything,
        "contest_nothing_cost": cost_contest_nothing,
        "savings_vs_contest_everything": savings_vs_everything,
        "savings_vs_contest_nothing": savings_vs_nothing,
        "optimal_metrics": best_metrics,
        "threshold_curve": threshold_metrics
    }
    
    with open("models/evaluation_results.json", "w") as f:
        json.dump(res, f, indent=2)
        
    print(f"\nCost curve visualization saved to {plot_path}")
    print("Evaluation results saved to models/evaluation_results.json")

if __name__ == "__main__":
    evaluate_cost_optimal_threshold()
