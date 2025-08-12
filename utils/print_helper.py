from datetime import datetime
from typing import Dict, Any, List, Optional
# ---------------------------
# Pretty-print helpers
# ---------------------------

def hr(title: str) -> None:
    print(f"\n{'=' * 8} {title.upper()} {'=' * 8}")

def kv(key: str, value: Any) -> None:
    print(f"- {key}: {value}")

def yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"

def pretty_duplicate(result: Dict[str, Any]) -> None:
    hr("Duplicate Check")
    kv("Duplicate", yes_no(result.get("is_duplicate", False)))
    if result.get("is_duplicate"):
        kv("Reason", result.get("reason", "Possible duplicate detected"))
    kv("Customer ID", result.get("customer_id"))
    kv("Transaction ID", result.get("transaction_id"))
    ts = result.get("checked_at")
    if ts:
        kv("Checked At (utc)", datetime.utcfromtimestamp(ts).isoformat() + "Z")
    kv("Next Step", result.get("recommendation"))

def pretty_fraud(analysis: Dict[str, Any], amount: float) -> None:
    hr("Fraud Classification")
    kv("Is Fraud", yes_no(analysis.get("is_fraud", False)))
    kv("Amount", amount)

def pretty_similarity(data: Dict[str, Any]) -> None:
    hr("Similarity Detection")
    sigs: List[str] = data.get("query_signatures", [])
    if sigs:
        kv("Generated Signatures", ", ".join(sigs))
    sims: List[Dict[str, Any]] = data.get("similar_cases", []) or []
    if not sims:
        print("No similar cases found.")
        return
    print("Top Similar Cases:")
    # We assume each result may include a similarity score under 'score'
    # and may contain 'transaction_id' or an '_id'. We handle both.
    for i, doc in enumerate(sims, 1):
        tid = doc.get("transaction_id") or doc.get("_id")
        score = doc.get("score")
        line = f"  {i}. id={tid}"
        if score is not None:
            line += f" | score={round(float(score), 4)}"
        print(line)
        # If stored signatures are present in similar doc, show a short preview
        if isinstance(doc.get("signatures"), list) and doc["signatures"]:
            preview = ", ".join(doc["signatures"][:3])
            print(f"     signatures: {preview}{' ...' if len(doc['signatures']) > 3 else ''}")

def pretty_reflection(text: Optional[str]) -> None:
    hr("Agent Reflection")
    if text:
        print(text.strip())
    else:
        print("No reflection available.")

def pretty_recommendation(text: Optional[str]) -> None:
    hr("Action Recommendation")
    if text:
        print(text.strip())
    else:
        print("No recommendation available.")

def pretty_storage(status: Optional[str], txn_id: str) -> None:
    hr("Storage")
    kv("Transaction ID", txn_id)
    kv("Status", status or "unknown")

def pretty_final(state: "WorkflowState") -> None:  # type: ignore
    hr("Final Summary")
    kv("Transaction ID", state.transaction.transaction_id)
    kv("Duplicate", yes_no(state.duplicate_check.get("is_duplicate") if state.duplicate_check else False))
    kv("Fraud", yes_no(state.fraud_analysis.get("is_fraud") if state.fraud_analysis else False))
    kv("Similar Cases", len((state.similarity_data or {}).get("similar_cases", [])))
    kv("Recommendation", state.recommendation or "N/A")
    if state.errors:
        print("\nErrors:")
        for e in state.errors:
            print(f"  - {e}")
