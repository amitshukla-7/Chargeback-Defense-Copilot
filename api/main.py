import os
import sys
import json
import joblib
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from features import build_single_order_features, FEATURE_COLUMNS
from evidence import compile_evidence
from letter_generator import draft_response
from guardrails import verify_traceability

app = FastAPI(
    title="Chargeback Defense Copilot API",
    description="Razorpay Buildathon — AI Risk Manager Track decision engine & evidence generator.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached model & data
MODEL = None
DATA_DF = None
OPTIMAL_THRESHOLD = 0.15
EVALUATION_RESULTS = {}

@app.on_event("startup")
def load_artifacts():
    global MODEL, DATA_DF, OPTIMAL_THRESHOLD, EVALUATION_RESULTS
    
    model_path = os.path.join("models", "winnability_model.pkl")
    data_path = os.path.join("data", "processed", "labeled_disputes.csv")
    eval_path = os.path.join("models", "evaluation_results.json")
    
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}")
        
    if os.path.exists(data_path):
        DATA_DF = pd.read_csv(data_path)
    else:
        print(f"Warning: Data not found at {data_path}")
        
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            EVALUATION_RESULTS = json.load(f)
            OPTIMAL_THRESHOLD = EVALUATION_RESULTS.get("optimal_threshold", 0.15)

class DisputeRequest(BaseModel):
    order_id: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Chargeback Defense Copilot API",
        "optimal_contest_threshold": OPTIMAL_THRESHOLD,
        "model_loaded": MODEL is not None,
        "total_disputes_indexed": len(DATA_DF) if DATA_DF is not None else 0
    }

@app.get("/disputes")
def list_disputes(limit: int = 50, reason_code: Optional[str] = None):
    if DATA_DF is None:
        raise HTTPException(status_code=500, detail="Dispute dataset not loaded.")
        
    df_filtered = DATA_DF.copy()
    if reason_code:
        df_filtered = df_filtered[df_filtered["reason_code"] == reason_code]
        
    disputes = df_filtered.head(limit).to_dict(orient="records")
    return {"total": len(df_filtered), "returned": len(disputes), "disputes": disputes}

@app.get("/model-summary")
def get_model_summary():
    eval_path = os.path.join("models", "evaluation_results.json")
    train_path = os.path.join("models", "training_summary.json")
    
    summary = {}
    if os.path.exists(train_path):
        with open(train_path, "r") as f:
            summary["training_summary"] = json.load(f)
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            summary["evaluation_results"] = json.load(f)
            
    return summary

@app.post("/assess-dispute")
def assess_dispute(req: DisputeRequest):
    if MODEL is None or DATA_DF is None:
        raise HTTPException(status_code=500, detail="Model or dispute dataset not initialized.")
        
    matches = DATA_DF[DATA_DF["order_id"] == req.order_id]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"Order ID '{req.order_id}' not found.")
        
    order_row = matches.iloc[0].to_dict()
    features_df = build_single_order_features(order_row)
    
    # Model prediction
    prob = float(MODEL.predict_proba(features_df)[0][1])
    recommend_contest = bool(prob >= OPTIMAL_THRESHOLD)
    
    evidence = compile_evidence(req.order_id, DATA_DF)
    
    result = {
        "order_id": req.order_id,
        "dispute_id": evidence["dispute_id"],
        "reason_code": evidence["reason_code"],
        "order_amount_inr": evidence["order_amount_inr"],
        "win_probability": round(prob, 4),
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "recommend_contest": recommend_contest,
        "evidence": evidence
    }
    
    if recommend_contest:
        letter_obj = draft_response(evidence)
        audit_res = verify_traceability(letter_obj["raw"], evidence)
        
        result["letter"] = letter_obj["raw"]
        result["generator_mode"] = letter_obj["generator_mode"]
        result["traceability_audit"] = audit_res
    else:
        result["letter"] = None
        result["reason"] = f"Win probability ({prob:.1%}) is below cost-optimal contest threshold ({OPTIMAL_THRESHOLD:.1%}). Contesting expected to incur net loss."
        
    return result
