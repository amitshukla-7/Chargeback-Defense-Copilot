import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_chargeback_data(num_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    start_date = datetime(2024, 1, 1)
    
    records = []
    
    reason_codes = ["10.4_fraud", "13.1_not_received", "13.3_not_as_described", "12.6_duplicate"]
    reason_probs = [0.45, 0.30, 0.15, 0.10]
    
    for i in range(num_samples):
        dispute_id = f"DSP-2024-{1000 + i}"
        order_id = f"ORD-{800000 + i}"
        customer_id = f"CUST-{np.random.randint(10000, 99999)}"
        
        # Order date spaced over ~300 days
        days_offset = int((i / num_samples) * 280) + np.random.randint(0, 3)
        order_dt = start_date + timedelta(days=days_offset, hours=np.random.randint(8, 22))
        
        # Delivery date 2 to 6 days after order
        delivery_dt = order_dt + timedelta(days=int(np.random.randint(2, 7)))
        
        # Days to dispute: fraud early or late, not received ~10-30 days
        days_to_dispute = int(np.random.choice([
            np.random.randint(1, 8),
            np.random.randint(8, 30),
            np.random.randint(30, 75)
        ], p=[0.35, 0.45, 0.20]))
        
        dispute_dt = delivery_dt + timedelta(days=days_to_dispute)
        
        # Order values
        avg_ov = round(float(np.random.exponential(scale=2500) + 800), 2)
        ov_ratio = round(float(np.random.choice([
            np.random.uniform(0.7, 1.3),
            np.random.uniform(1.4, 3.5),
            np.random.uniform(0.3, 0.6)
        ], p=[0.70, 0.20, 0.10])), 2)
        
        order_amount_inr = round(avg_ov * ov_ratio, 2)
        
        # Verification features
        reason_code = np.random.choice(reason_codes, p=reason_probs)
        avs_match = int(np.random.binomial(1, 0.72))
        cvv_match = int(np.random.binomial(1, 0.85))
        pod_confirmed = int(np.random.binomial(1, 0.65))
        address_match = int(np.random.binomial(1, 0.78))
        repeat_customer = int(np.random.binomial(1, 0.38))
        
        # Calculate winning probability based on Visa/Mastercard compelling evidence rules
        logit = -1.2  # base intercept
        
        # Reason code impacts
        if reason_code == "10.4_fraud":
            if avs_match and cvv_match and pod_confirmed and address_match:
                logit += 2.8  # Strong Compelling Evidence 3.0
            elif avs_match and cvv_match:
                logit += 1.2
            else:
                logit -= 1.5
        elif reason_code == "13.1_not_received":
            if pod_confirmed:
                logit += 2.5
            else:
                logit -= 2.0
        elif reason_code == "13.3_not_as_described":
            if repeat_customer:
                logit += 0.8
            if days_to_dispute > 40:
                logit -= 1.0
        elif reason_code == "12.6_duplicate":
            logit += 0.5
            
        if repeat_customer:
            logit += 0.6
        if address_match:
            logit += 0.5
        if ov_ratio > 2.0:
            logit -= 0.4  # Anomalously high orders slightly harder to defend if disputed
            
        # Add realistic noise
        noise = np.random.normal(0, 0.4)
        prob_win = 1.0 / (1.0 + np.exp(-(logit + noise)))
        won_dispute = int(prob_win >= 0.5)
        
        records.append({
            "dispute_id": dispute_id,
            "order_id": order_id,
            "customer_id": customer_id,
            "order_date": order_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "delivery_date": delivery_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "dispute_date": dispute_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "days_to_dispute": days_to_dispute,
            "order_amount_inr": order_amount_inr,
            "customer_avg_order_val": avg_ov,
            "order_value_ratio": ov_ratio,
            "reason_code": reason_code,
            "avs_match": avs_match,
            "cvv_match": cvv_match,
            "pod_confirmed": pod_confirmed,
            "address_match": address_match,
            "repeat_customer": repeat_customer,
            "win_probability_ground_truth": round(prob_win, 4),
            "won_dispute": won_dispute
        })
        
    df = pd.DataFrame(records)
    # Sort strictly by dispute_date for proper time series split
    df = df.sort_values("dispute_date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    df = generate_synthetic_chargeback_data(num_samples=1500)
    
    raw_path = "data/raw/chargeback_orders_raw.csv"
    processed_path = "data/processed/labeled_disputes.csv"
    
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)
    
    print(f"Generated {len(df)} records.")
    print(f"Raw dataset saved to: {raw_path}")
    print(f"Processed labeled dataset saved to: {processed_path}")
    print(f"Overall win rate: {df['won_dispute'].mean():.2%}")
