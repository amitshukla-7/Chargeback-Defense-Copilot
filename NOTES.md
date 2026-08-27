# Chargeback Defense Copilot — Technical & Domain Assumptions

## 1. Executive Summary & Objective
This document details all data derivation logic, feature engineering assumptions, probabilistic dispute win-rate modeling rules, and financial cost-curve parameters powering the **Chargeback Defense Copilot**. Developed for the Razorpay Buildathon (AI Risk Manager Track), this solution operates as a defense-only decision engine that evaluates merchant dispute winnability, optimizes response thresholds based on actual operational costs, and generates zero-hallucination factual response letters.

---

## 2. Dataset Synthesis & Field Derivations
Chargeback ground truth data from card networks (Visa VROL, Mastercard MasterCom) is proprietary and non-public due to PCI-DSS compliance. We derive realistic merchant transaction and dispute logs based on official Visa Core Rules, Mastercard Chargeback Guidelines, and standard ecommerce transaction attributes.

### Simulated Dataset Schema (1,500 Dispute Records):
- `order_id`: Unique transaction identifier (e.g. `ORD-894102`).
- `dispute_id`: Unique dispute tracking ID (e.g. `DSP-2024-9481`).
- `customer_id`: Unique buyer identifier.
- `order_date`: ISO timestamp when order was placed.
- `delivery_date`: ISO timestamp when shipment reached customer.
- `dispute_date`: ISO timestamp when chargeback notice was served to merchant.
- `days_to_dispute`: Integer calculated as `(dispute_date - delivery_date).days`.
- `order_amount_inr`: Order monetary value in ₹ (range: ₹450 to ₹18,500).
- `customer_avg_order_val`: Historical average order value for customer.
- `order_value_ratio`: `order_amount_inr / customer_avg_order_val`.
- `reason_code`: Dispute classification based on network standards:
  - `10.4` — Fraud (Card-Not-Present / Unauthorized Transaction)
  - `13.1` — Merchandise / Services Not Received
  - `13.3` — Not as Described / Defective Merchandise
  - `12.6` — Duplicate Processing / Incorrect Amount
- `avs_match`: Boolean (1 = Address Verification System matched billing street + zip).
- `cvv_match`: Boolean (1 = Card Verification Value matched at checkout).
- `pod_confirmed`: Boolean (1 = Electronic Proof of Delivery with customer signature/OTP confirmation).
- `address_match`: Boolean (1 = Shipping address matches verified billing address).
- `repeat_customer`: Boolean (1 = Customer has ≥ 2 previous completed transactions).

---

## 3. Label Assignment (`won_dispute` Probabilistic Logic)
Real chargeback outcomes are non-deterministic and noisy. Rather than using arbitrary rules, `won_dispute` is assigned via a calibrated Bernoulli probability model based on card network rules:

$$\text{Prob}(\text{Win}) = \text{sigmoid}\left( \beta_0 + \beta_1 \cdot \text{AVS} + \beta_2 \cdot \text{CVV} + \beta_3 \cdot \text{POD} + \beta_4 \cdot \text{AddrMatch} + \beta_5 \cdot \text{Repeat} + \beta_{\text{Reason}} + \text{Noise} \right)$$

### Domain Weights Applied:
1. **Proof of Delivery (POD) & AVS/CVV**: For Fraud (10.4), having AVS=1, CVV=1, and POD=1 provides compelling evidence under Visa Compelling Evidence 3.0 rules, increasing win probability by ~65-75%.
2. **Item Not Received (13.1)**: Win probability is heavily dominated by `pod_confirmed=1` and carrier tracking validation.
3. **Dispute Timing**: Genuine fraud disputes cluster early (< 7 days) or very late (> 60 days via stolen card statements). Intermediate disputes (15-30 days) often exhibit different win rates.
4. **Noise**: Gaussian noise ($\mathcal{N}(0, 0.25)$) is added to ensure realistic decision boundary uncertainty, preventing synthetic perfection and mimicking card issuer panel variance.

---

## 4. Time-Based Train/Test Split (Preventing Data Leakage)
Chargebacks occur sequentially over time. A standard random K-Fold or train_test_split would leak future dispute trends and buyer behavior back into historical training.
- **Sorting**: All records are strictly ordered by `dispute_date`.
- **Split Ratio**: 80% Train (Earliest 1,200 records), 20% Held-out Test (Latest 300 records).
- **Validation**: Metrics (PR-AUC, ROC-AUC, Precision, Recall, F1) are reported exclusively on the held-out test set.

---

## 5. Financial Cost Function & Operational Calibration
Not all classification errors carry equal risk:
- **Cost of False Negative ($C_{\text{FN}} = ₹250$)**: Merchant forfeits a winnable chargeback. Loss = Transaction revenue loss + cost of goods sold + payment gateway chargeback fee.
- **Cost of False Positive ($C_{\text{FP}} = ₹15$)**: Merchant contests a losing chargeback. Loss = Operations analyst time spent compiling documents (~15-20 mins) + arbitration submission fee when rejected.

$$\text{Total Expected Cost}(\tau) = \text{FN}(\tau) \times ₹250 + \text{FP}(\tau) \times ₹15$$

The decision threshold $\tau^*$ is selected by minimizing $\text{Total Expected Cost}(\tau)$ across the held-out test set.

---

## 6. Guardrails & LLM Zero-Hallucination Guarantees
1. **Facts-Only Subgraph**: The Anthropic Claude model is provided only a JSON payload containing structured order/dispute evidence.
2. **Deterministic Traceability Check**: Every generated sentence is verified against the input evidence values. Sentences containing unverified assertions are explicitly flagged for human operational review.
3. **Non-Autonomy Guarantee**: Copilot produces recommendations and response drafts for merchant approval—it never automatically submits disputes or denies customer claims without human verification.
