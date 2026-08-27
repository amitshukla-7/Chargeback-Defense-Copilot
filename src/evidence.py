import pandas as pd
from typing import Dict, Any

def compile_evidence(order_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compiles structured facts for a specific order dispute from transaction records.
    
    GUARANTEE OF DEFENSE-ONLY ACCURACY:
    This evidence dictionary constitutes the strict, closed-world universe of facts
    that the LLM is permitted to reference when drafting dispute representations.
    """
    matches = df.loc[df["order_id"] == order_id]
    if len(matches) == 0:
        raise ValueError(f"Order ID '{order_id}' not found in dispute database.")
        
    row = matches.iloc[0]
    
    evidence = {
        "order_id": str(row["order_id"]),
        "dispute_id": str(row.get("dispute_id", f"DSP-{row['order_id']}")),
        "customer_id": str(row.get("customer_id", "CUST-UNKNOWN")),
        "order_date": str(row.get("order_date", "N/A")),
        "delivery_date": str(row.get("delivery_date", "N/A")),
        "dispute_date": str(row.get("dispute_date", "N/A")),
        "order_amount_inr": float(row["order_amount_inr"]),
        "reason_code": str(row["reason_code"]),
        "avs_match": bool(row["avs_match"]),
        "cvv_match": bool(row["cvv_match"]),
        "pod_confirmed": bool(row["pod_confirmed"]),
        "delivery_address_matches_billing": bool(row["address_match"]),
        "days_between_delivery_and_dispute": int(row["days_to_dispute"]),
        "repeat_customer": bool(row["repeat_customer"]),
        "order_value_ratio": float(row["order_value_ratio"])
    }
    
    return evidence
