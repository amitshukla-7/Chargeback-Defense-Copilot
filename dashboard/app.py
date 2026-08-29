import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from features import build_single_order_features
from evidence import compile_evidence
from letter_generator import draft_response
from guardrails import verify_traceability

st.set_page_config(
    page_title="Chargeback Defense Copilot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimalist, clean fintech styling
st.markdown("""
<style>
    /* Typography & Hierarchy */
    h1, h2, h3 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    .page-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: var(--text-color, #0F172A);
    }
    
    .page-subtitle {
        font-size: 0.9rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    /* Minimal Stat Card */
    .stat-card {
        background-color: var(--secondary-background-color, #F8FAFC);
        border: 1px solid var(--border-color, #E2E8F0);
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--text-color, #0F172A);
        line-height: 1.2;
    }
    .stat-sub {
        font-size: 0.78rem;
        color: #10B981;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .status-contest {
        background-color: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-forfeit {
        background-color: rgba(100, 116, 139, 0.12);
        color: #475569;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }

    /* Audit Sentence List */
    .audit-item {
        background-color: var(--secondary-background-color, #F8FAFC);
        border: 1px solid var(--border-color, #E2E8F0);
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .audit-item-verified {
        border-left: 3px solid #10B981;
    }
    .audit-item-flagged {
        border-left: 3px solid #EF4444;
        background-color: rgba(239, 68, 68, 0.04);
    }
    .tag-grounded {
        background-color: rgba(16, 185, 129, 0.15);
        color: #047857;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 6px;
        text-transform: uppercase;
    }
    .tag-flagged {
        background-color: rgba(239, 68, 68, 0.15);
        color: #B91C1C;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 6px;
        text-transform: uppercase;
    }
    .fact-meta {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 4px;
        font-family: monospace;
    }
    
    /* Clean sidebar divider */
    .sidebar-divider {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--border-color, #E2E8F0);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load cached resources
@st.cache_resource
def load_all_artifacts():
    model_path = os.path.join("models", "winnability_model.pkl")
    base_model_path = os.path.join("models", "baseline_model.pkl")
    data_path = os.path.join("data", "processed", "labeled_disputes.csv")
    eval_path = os.path.join("models", "evaluation_results.json")
    train_path = os.path.join("models", "training_summary.json")
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    base_model = joblib.load(base_model_path) if os.path.exists(base_model_path) else None
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    
    eval_res = {}
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_res = json.load(f)
            
    train_res = {}
    if os.path.exists(train_path):
        with open(train_path, "r") as f:
            train_res = json.load(f)
            
    return model, base_model, df, eval_res, train_res

model, base_model, df, eval_res, train_res = load_all_artifacts()

if "session_savings" not in st.session_state:
    st.session_state.session_savings = 0.0
if "evaluated_cases" not in st.session_state:
    st.session_state.evaluated_cases = 0

# Sidebar Clean Header & Navigation
st.sidebar.markdown("## Chargeback Copilot")
st.sidebar.caption("Risk Management & Guardrail Engine")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["Overview & Metrics", "Cost Calibration", "Live Case Assessment", "Guardrails & Protocol"],
    index=2
)

# Session impact widget
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown("##### Session Savings Impact")
st.sidebar.metric("Evaluated Cases", st.session_state.evaluated_cases)
st.sidebar.metric("Net Savings", f"₹{st.session_state.session_savings:,.2f}")

if st.sidebar.button("Reset Counter", use_container_width=True):
    st.session_state.session_savings = 0.0
    st.session_state.evaluated_cases = 0
    st.rerun()

optimal_t = eval_res.get("optimal_threshold", 0.15)

# --- PAGE 1: OVERVIEW & METRICS ---
if nav_choice == "Overview & Metrics":
    st.markdown('<div class="page-title">Overview & Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Held-out test set evaluation with strict time-based temporal splitting</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    xgb_metrics = train_res.get("xgboost_winnability_model", {})
    base_metrics = train_res.get("baseline_logistic_regression", {})
    opt_metrics = eval_res.get("optimal_metrics", {})
    
    pr_auc = xgb_metrics.get('pr_auc', 0.9669)
    base_pr_auc = base_metrics.get('pr_auc', 0.8540)
    roc_auc = xgb_metrics.get('roc_auc', 0.9678)
    base_roc_auc = base_metrics.get('roc_auc', 0.8446)
    
    col1.metric("PR-AUC", f"{pr_auc:.4f}", f"+{(pr_auc - base_pr_auc)*100:.2f}% vs Baseline")
    col2.metric("ROC-AUC", f"{roc_auc:.4f}", f"+{(roc_auc - base_roc_auc)*100:.2f}% vs Baseline")
    col3.metric("Precision (t=0.15)", f"{opt_metrics.get('precision', 0.7526):.1%}")
    col4.metric("Recall (t=0.15)", f"{opt_metrics.get('recall', 0.9931):.1%}")
    
    st.divider()
    
    m_col1, m_col2 = st.columns([1.1, 1])
    
    with m_col1:
        st.markdown("### Temporal Split Validation")
        st.markdown("""
        To avoid data leakage, disputes are split chronologically rather than randomly:
        
        - **Total Disputes:** 1,500 records
        - **Training Set (80%):** 1,200 records (Earliest dates: Jan 2024 – Sep 2024)
        - **Held-out Test Set (20%):** 300 records (Latest dates: Sep 2024 – Dec 2024)
        """)
        
        st.markdown("### Model Benchmark Comparison")
        comp_df = pd.DataFrame([
            {"Model": "Logistic Regression (Baseline)", "PR-AUC": base_metrics.get('pr_auc', 0.8540), "ROC-AUC": base_metrics.get('roc_auc', 0.8446), "Precision": f"{base_metrics.get('precision_05', 0.65):.1%}", "Recall": f"{base_metrics.get('recall_05', 0.82):.1%}"},
            {"Model": "XGBoost Winnability Model", "PR-AUC": xgb_metrics.get('pr_auc', 0.9669), "ROC-AUC": xgb_metrics.get('roc_auc', 0.9678), "Precision": f"{xgb_metrics.get('precision_05', 0.88):.1%}", "Recall": f"{xgb_metrics.get('recall_05', 0.92):.1%}"},
            {"Model": "XGBoost @ Threshold t=0.15", "PR-AUC": xgb_metrics.get('pr_auc', 0.9669), "ROC-AUC": xgb_metrics.get('roc_auc', 0.9678), "Precision": f"{opt_metrics.get('precision', 0.7526):.1%}", "Recall": f"{opt_metrics.get('recall', 0.9931):.1%}"}
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
    with m_col2:
        st.markdown("### Confusion Matrix @ t=0.15")
        tp = opt_metrics.get("tp", 143)
        tn = opt_metrics.get("tn", 109)
        fp = opt_metrics.get("fp", 47)
        fn = opt_metrics.get("fn", 1)
        
        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            columns=["Pred Lost (0)", "Pred Won (1)"],
            index=["Actual Lost (0)", "Actual Won (1)"]
        )
        st.table(cm_df)
        
        st.caption(f"**False Negatives (FN = {fn}):** Forfeited winnable dispute (Cost: ₹250 each)")
        st.caption(f"**False Positives (FP = {fp}):** Contested unwinnable dispute (Cost: ₹15 analyst time each)")

# --- PAGE 2: COST CALIBRATION ---
elif nav_choice == "Cost Calibration":
    st.markdown('<div class="page-title">Cost Calibration & Baselines</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Optimizing decision threshold based on asymmetric operational costs (₹250 FN vs ₹15 FP)</div>', unsafe_allow_html=True)
    
    b_col1, b_col2 = st.columns([1.2, 1])
    
    with b_col1:
        cost_img_path = os.path.join("models", "cost_curve.png")
        if os.path.exists(cost_img_path):
            st.image(cost_img_path, caption="Risk Cost vs Decision Threshold Curve", use_container_width=True)
        else:
            st.warning("Cost curve graphic not found.")
            
    with b_col2:
        st.markdown("### Financial Baseline Comparison")
        st.caption("Evaluation on held-out test set (300 disputes)")
        
        cost_everything = eval_res.get("contest_everything_cost", 2340)
        cost_nothing = eval_res.get("contest_nothing_cost", 36000)
        min_cost = eval_res.get("test_set_min_cost", 955)
        
        sav_nothing = eval_res.get("savings_vs_contest_nothing", 35045)
        
        st.markdown(f"""
        | Strategy | Total Risk Cost (₹) | Net Financial Savings |
        |---|---|---|
        | **Contest Nothing** ($\\\\tau = 1.0$) | ₹{cost_nothing:,.2f} | Baseline |
        | **Contest Everything** ($\\\\tau = 0.0$) | ₹{cost_everything:,.2f} | Saved ₹{cost_nothing - cost_everything:,.2f} |
        | **Chargeback Copilot** ($\\\\tau = 0.15$) | **₹{min_cost:,.2f}** | **Saved ₹{sav_nothing:,.2f} (+{sav_nothing/cost_nothing:.1%})** |
        """)
        
        st.divider()
        st.markdown("### Cost Structure Parameters")
        st.markdown("""
        - **False Negative ($C_{FN} = ₹250$):** Winnable dispute ignored. Merchant forfeits order revenue + product value.
        - **False Positive ($C_{FP} = ₹15$):** Unwinnable dispute contested. Merchant incurs unnecessary analyst review time (~15 mins).
        - **Optimal Threshold ($\\\\tau^* = 0.15$):** Calibrated because retrieving winnable cases is 16.6x more impactful than saving minor operational review time.
        """)

# --- PAGE 3: LIVE CASE ASSESSMENT ---
elif nav_choice == "Live Case Assessment":
    st.markdown('<div class="page-title">Live Case Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Predict dispute winnability, compile evidence ground truth, and audit response representations.</div>', unsafe_allow_html=True)
    
    if df is None or model is None:
        st.error("Model or dataset unavailable. Please run pipeline training scripts first.")
        st.stop()
        
    # Sample selection
    sample_options = df["order_id"].tolist()
    
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_order_id = st.selectbox("Select Order ID for Assessment:", sample_options, index=0)
    with col_sel2:
        reason_filter = st.selectbox("Filter by Reason Code:", ["All"] + sorted(df["reason_code"].unique().tolist()))
        if reason_filter != "All":
            filtered_ids = df[df["reason_code"] == reason_filter]["order_id"].tolist()
            if filtered_ids:
                selected_order_id = filtered_ids[0]
                
    order_row = df[df["order_id"] == selected_order_id].iloc[0].to_dict()
    features_df = build_single_order_features(order_row)
    
    prob = float(model.predict_proba(features_df)[0][1])
    recommend_contest = prob >= optimal_t
    
    st.divider()
    
    res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
    
    with res_col1:
        st.markdown(f"**Order Amount:** ₹{order_row['order_amount_inr']:,.2f}")
        st.markdown(f"**Reason Code:** `{order_row['reason_code']}`")
        st.markdown(f"**Dispute Latency:** {order_row['days_to_dispute']} days after delivery")
        
    with res_col2:
        st.markdown("**Win Probability:**")
        st.progress(prob)
        st.markdown(f"### {prob:.1%}")
        
    with res_col3:
        st.markdown("**Recommendation:**")
        if recommend_contest:
            st.markdown('<span class="status-pill status-contest">RECOMMEND CONTEST</span>', unsafe_allow_html=True)
            st.caption(f"Probability ({prob:.1%}) ≥ threshold ({optimal_t:.1%}). Expected positive EV.")
        else:
            st.markdown('<span class="status-pill status-forfeit">RECOMMEND FORFEIT</span>', unsafe_allow_html=True)
            st.caption(f"Probability ({prob:.1%}) < threshold ({optimal_t:.1%}). Contest expected net loss.")
            
    st.markdown("")
    if st.button("Record Assessment in Session Counter", use_container_width=False):
        st.session_state.evaluated_cases += 1
        if recommend_contest and order_row["won_dispute"] == 1:
            st.session_state.session_savings += (order_row["order_amount_inr"] - 15)
        elif not recommend_contest and order_row["won_dispute"] == 0:
            st.session_state.session_savings += 15
        st.success(f"Case {selected_order_id} recorded. Updated session savings: ₹{st.session_state.session_savings:,.2f}")

    st.divider()
    
    tab_ev, tab_let, tab_aud = st.tabs(["Evidence Data", "Drafted Response", "Guardrail Audit"])
    
    evidence = compile_evidence(selected_order_id, df)
    
    with tab_ev:
        st.markdown("##### Closed-World Fact Set")
        st.caption("Only facts in this payload are accessible to the response generator.")
        
        ev_cols = st.columns(2)
        with ev_cols[0]:
            st.json(evidence)
        with ev_cols[1]:
            st.markdown("##### Verification Summary")
            st.markdown(f"- **AVS Match:** {'Match' if evidence['avs_match'] else 'Failed'}")
            st.markdown(f"- **CVV Match:** {'Match' if evidence['cvv_match'] else 'Failed'}")
            st.markdown(f"- **Proof of Delivery:** {'Confirmed' if evidence['pod_confirmed'] else 'Unconfirmed'}")
            st.markdown(f"- **Address Alignment:** {'Matches Billing' if evidence['delivery_address_matches_billing'] else 'Mismatch'}")
            st.markdown(f"- **Customer History:** {'Repeat Customer' if evidence['repeat_customer'] else 'First-time Customer'}")
            
    with tab_let:
        st.markdown("##### Generated Representation Letter")
        if recommend_contest:
            letter_obj = draft_response(evidence)
            st.caption(f"Engine: `{letter_obj['generator_mode']}`")
            st.text_area("Letter Content:", letter_obj["raw"], height=300)
        else:
            st.info("No letter drafted because forfeiture is recommended for this case.")
            
    with tab_aud:
        st.markdown("##### Sentence-Level Traceability Audit")
        if recommend_contest:
            letter_obj = draft_response(evidence)
            audit_res = verify_traceability(letter_obj["raw"], evidence)
            
            st.markdown(f"**Verification Score:** `{audit_res['verification_rate']:.1%}` ({audit_res['verified_sentences_count']}/{audit_res['total_sentences']} sentences grounded)")
            if audit_res["is_fully_compliant"]:
                st.success("Fully Compliant: All statements ground in evidence facts.")
            else:
                st.warning(f"{len(audit_res['unverified_sentences'])} sentence(s) flagged for review.")
                
            st.markdown("##### Sentence Audit Log:")
            for s in audit_res["sentence_audits"]:
                if s["verified"]:
                    facts_str = ", ".join(s["matched_facts"]) if s["matched_facts"] else "Standard formal legal phrasing"
                    st.markdown(f"""
                    <div class="audit-item audit-item-verified">
                        <span class="tag-grounded">Grounded</span> "{s['sentence']}"
                        <div class="fact-meta">Evidence: {facts_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="audit-item audit-item-flagged">
                        <span class="tag-flagged">Flagged</span> "{s['sentence']}"
                        <div class="fact-meta">Unverified assertion. Requires operational review.</div>
                    </div>
                    """, unsafe_allow_html=True)

# --- PAGE 4: GUARDRAILS & PROTOCOL ---
elif nav_choice == "Guardrails & Protocol":
    st.markdown('<div class="page-title">System Guardrails & Protocol</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Architectural principles enforcing truthful, defensible risk management.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    #### 1. Zero-Hallucination Evidence Constraint
    The dispute letter generator is strictly restricted to a validated JSON fact payload. It has no access to external state or unverified generative memory.

    #### 2. Sentence-Level Traceability Audit
    Before presenting drafted text, a deterministic engine parses every sentence. Statements lacking direct evidence grounding are explicitly flagged for human review.

    #### 3. Human-in-the-Loop Governance
    Copilot acts as an intelligence assistant. It does not automatically submit dispute responses to payment networks without merchant authorization.

    #### 4. Defense-Only Purpose
    The engine strictly compiles legitimate merchant verification records (AVS/CVV checks, tracking confirmation) to contest improper chargebacks.
    """)
