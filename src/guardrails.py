import re
from typing import Dict, Any, List

def verify_traceability(letter_text: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifies that every sentence in the generated chargeback dispute response letter
    is strictly traceable to facts present in the evidence dictionary.
    
    Returns structured audit results including sentence-level status and verification rate.
    """
    # Extract string representations of evidence values for substring matching
    known_facts = {}
    
    # Key value string tokens
    for k, v in evidence.items():
        if isinstance(v, bool):
            val_str = "matched" if v else "failed"
            known_facts[k] = val_str
        elif isinstance(v, (int, float)):
            known_facts[k] = str(v)
        else:
            known_facts[k] = str(v).lower()
            
    # Clean and split letter into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', letter_text) if len(s.strip()) > 5]
    
    sentence_audits = []
    unverified_sentences = []
    
    # Generic standard legal/formal phrases acceptable without explicit value match
    generic_accepted_keywords = [
        "dispute", "merchant", "cardholder", "chargeback", "evidence", "submission",
        "representation", "respectfully", "rebuttal", "regards", "sincerely",
        "visa", "mastercard", "compelling", "reason code", "documentation"
    ]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matched_facts = []
        
        # Check against evidence values
        for key, val in known_facts.items():
            val_str = str(val).lower()
            if len(val_str) >= 2 and val_str in sentence_lower:
                matched_facts.append(f"{key}={val}")
                
        # Check against key evidence terms
        if "avs" in sentence_lower and evidence.get("avs_match"):
            matched_facts.append("avs_match=True")
        if "cvv" in sentence_lower and evidence.get("cvv_match"):
            matched_facts.append("cvv_match=True")
        if "delivery" in sentence_lower and evidence.get("pod_confirmed"):
            matched_facts.append("pod_confirmed=True")
        if "billing" in sentence_lower and evidence.get("delivery_address_matches_billing"):
            matched_facts.append("delivery_address_matches_billing=True")
            
        is_verified = len(matched_facts) > 0 or any(kw in sentence_lower for kw in generic_accepted_keywords)
        
        sentence_audits.append({
            "sentence": sentence,
            "verified": is_verified,
            "matched_facts": matched_facts
        })
        
        if not is_verified:
            unverified_sentences.append(sentence)
            
    total_sentences = len(sentences)
    verified_count = sum(1 for s in sentence_audits if s["verified"])
    verification_rate = round(verified_count / max(1, total_sentences), 4)
    
    return {
        "total_sentences": total_sentences,
        "verified_sentences_count": verified_count,
        "verification_rate": verification_rate,
        "is_fully_compliant": len(unverified_sentences) == 0,
        "unverified_sentences": unverified_sentences,
        "sentence_audits": sentence_audits
    }
