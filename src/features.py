import pandas as pd
import numpy as np
from typing import List, Tuple

# Pre-defined list of feature columns to guarantee consistent column ordering across training & API inference
FEATURE_COLUMNS = [
    "days_to_dispute",
    "order_amount_inr",
    "customer_avg_order_val",
    "order_value_ratio",
    "avs_match",
    "cvv_match",
    "pod_confirmed",
    "address_match",
    "repeat_customer",
    "avs_cvv_both_match",
    "pod_and_address_match",
    "high_value_order",
    "is_first_time_buyer",
    "compelling_evidence_score",
    "reason_code_10.4_fraud",
    "reason_code_12.6_duplicate",
    "reason_code_13.1_not_received",
    "reason_code_13.3_not_as_described",
    "days_to_dispute_bucket_fast",
    "days_to_dispute_bucket_normal",
    "days_to_dispute_bucket_slow",
    "days_to_dispute_bucket_very_slow",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw dispute records into structured machine learning features.
    
    Feature Logic:
    1. days_to_dispute_bucket: Categorizes dispute filing window (fast: <=7 days, normal: 8-30, slow: 31-60, very_slow: >60)
    2. avs_cvv_both_match: Interaction feature for full checkout verification
    3. pod_and_address_match: Interaction feature for physical delivery verification
    4. high_value_order: Flag if order value > 1.5x customer historical average
    5. is_first_time_buyer: Reverse repeat customer flag
    6. compelling_evidence_score: Combined count of 5 key defense proof signals
    """
    processed = df.copy()
    
    # Bucket dispute filing latency
    processed["days_to_dispute_bucket"] = pd.cut(
        processed["days_to_dispute"],
        bins=[-1, 7, 30, 60, 999],
        labels=["fast", "normal", "slow", "very_slow"]
    )
    
    # Interaction & composite domain features
    processed["avs_cvv_both_match"] = (processed["avs_match"].astype(int) & processed["cvv_match"].astype(int)).astype(int)
    processed["pod_and_address_match"] = (processed["pod_confirmed"].astype(int) & processed["address_match"].astype(int)).astype(int)
    processed["high_value_order"] = (processed["order_value_ratio"] > 1.5).astype(int)
    processed["is_first_time_buyer"] = (processed["repeat_customer"] == 0).astype(int)
    
    processed["compelling_evidence_score"] = (
        processed["avs_match"].astype(int) +
        processed["cvv_match"].astype(int) +
        processed["pod_confirmed"].astype(int) +
        processed["address_match"].astype(int) +
        processed["repeat_customer"].astype(int)
    )
    
    # One-hot encoding for categorical variables
    processed = pd.get_dummies(processed, columns=["reason_code", "days_to_dispute_bucket"], dtype=int)
    
    # Ensure all expected columns exist (padding missing dummy categories with 0)
    for col in FEATURE_COLUMNS:
        if col not in processed.columns:
            processed[col] = 0
            
    # Return features strictly ordered by FEATURE_COLUMNS
    return processed[FEATURE_COLUMNS]

def build_single_order_features(order_dict: dict) -> pd.DataFrame:
    """
    Constructs a 1-row feature DataFrame from a single dictionary input for fast API inference.
    """
    df_single = pd.DataFrame([order_dict])
    return build_features(df_single)
