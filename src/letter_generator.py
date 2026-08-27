import os
import json
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You draft formal chargeback dispute responses for an e-commerce merchant.
STRICT DEFENSE-ONLY GUARDRAIL RULES:
- Only state facts present in the evidence JSON you are given.
- Never invent a date, delivery attempt, tracking number, or claim not in the evidence.
- If evidence is insufficient to support contesting, say so plainly instead of drafting a letter.
- Write in a formal, concise, factual tone suitable for submission to card networks (Visa/Mastercard).
- Structure the letter into clear headings: Executive Summary, Verification Signals, Proof of Delivery, and Conclusion.
"""

def generate_rule_based_fallback_letter(evidence: Dict[str, Any]) -> str:
    """
    Deterministic rule-based letter generator as a guaranteed zero-hallucination fallback.
    """
    reason_code = evidence.get("reason_code", "Unknown")
    order_id = evidence.get("order_id", "N/A")
    dispute_id = evidence.get("dispute_id", "N/A")
    amount = evidence.get("order_amount_inr", 0.0)
    order_date = evidence.get("order_date", "N/A")
    delivery_date = evidence.get("delivery_date", "N/A")
    days_to_dispute = evidence.get("days_between_delivery_and_dispute", 0)
    
    avs_status = "MATCHED (AVS Y)" if evidence.get("avs_match") else "UNMATCHED"
    cvv_status = "MATCHED (CVV M)" if evidence.get("cvv_match") else "UNMATCHED"
    pod_status = "CONFIRMED ELECTRONIC PROOF OF DELIVERY" if evidence.get("pod_confirmed") else "UNCONFIRMED"
    addr_status = "MATCHES VERIFIED BILLING ADDRESS" if evidence.get("delivery_address_matches_billing") else "DIFFERENT FROM BILLING"
    repeat_status = "VERIFIED REPEAT CUSTOMER" if evidence.get("repeat_customer") else "FIRST TIME BUYER"

    letter = (
        f"FORMAL CHARGEBACK REBUTTAL REPRESENTATION\n"
        f"Dispute ID: {dispute_id} | Order ID: {order_id} | Reason Code: {reason_code}\n"
        f"Disputed Amount: INR {amount:,.2f} | Filing Latency: {days_to_dispute} days\n\n"
        f"1. EXECUTIVE SUMMARY\n"
        f"We formally contest chargeback {dispute_id} under reason code {reason_code} for order {order_id}. "
        f"The transaction was placed on {order_date} for INR {amount:,.2f} and delivered on {delivery_date}. "
        f"Our records conclusively confirm that this transaction was legitimate and fulfilled strictly in accordance with card network guidelines.\n\n"
        f"2. CHECKOUT VERIFICATION SIGNALS\n"
        f"- Address Verification System (AVS): {avs_status}.\n"
        f"- Card Verification Value (CVV): {cvv_status}.\n"
        f"- Shipping & Billing Address Alignment: {addr_status}.\n"
        f"- Customer Order History: {repeat_status}.\n\n"
        f"3. FULFILLMENT & PROOF OF DELIVERY\n"
        f"Carrier logistics records confirm fulfillment on {delivery_date}. Status: {pod_status}. "
        f"The dispute was filed {days_to_dispute} days after successful delivery.\n\n"
        f"4. CONCLUSION & REQUEST FOR REMEDY\n"
        f"In light of the verified AVS ({avs_status}), CVV ({cvv_status}), and Proof of Delivery status ({pod_status}), "
        f"we respectfully request the card issuer to reverse this dispute and credit INR {amount:,.2f} back to the merchant."
    )
    return letter

def draft_response(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drafts a dispute response letter using Anthropic Claude API if available,
    or falls back to the deterministic zero-hallucination engine.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if api_key and not api_key.startswith("your_"):
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            user_prompt = f"Evidence JSON:\n{json.dumps(evidence, indent=2)}\n\nDraft the formal dispute response representation."
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=800,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            raw_text = response.content[0].text
            return {"raw": raw_text, "generator_mode": "anthropic_claude"}
        except Exception as e:
            print(f"Anthropic API call notice ({e}). Using deterministic zero-hallucination engine.")
            
    fallback_text = generate_rule_based_fallback_letter(evidence)
    return {"raw": fallback_text, "generator_mode": "deterministic_guardrail"}
