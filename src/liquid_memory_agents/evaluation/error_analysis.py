from __future__ import annotations

from typing import Any


CATEGORIES = (
    "correction_failure", "stale_fact_interference", "slot_collision",
    "temporal_reasoning", "entity_confusion", "number_or_date",
    "multi_session", "decoder_hallucination", "reader_failure",
    "retrieval_miss", "seed_sensitivity",
)


def categorize(row: dict[str, Any]) -> list[str]:
    if row.get("correct"):
        return []
    categories = []
    qtype = str(row.get("question_type", "")).casefold()
    diagnostics = row.get("diagnostics") or {}
    if "correction" in qtype: categories.append("correction_failure")
    if diagnostics.get("stale_value_in_prediction"): categories.append("stale_fact_interference")
    attention = diagnostics.get("attention", [])
    if attention and max(attention) > 0.95: categories.append("slot_collision")
    if "temporal" in qtype: categories.append("temporal_reasoning")
    if diagnostics.get("wrong_entity"): categories.append("entity_confusion")
    if diagnostics.get("number_or_date_error"): categories.append("number_or_date")
    if diagnostics.get("sessions", 1) > 1: categories.append("multi_session")
    if not row.get("evidence_supports_prediction", True): categories.append("decoder_hallucination")
    if row.get("evidence_present") and not row.get("correct"): categories.append("reader_failure")
    if row.get("agent") == "rag" and not row.get("gold_session_retrieved", True):
        categories.append("retrieval_miss")
    return categories or ["decoder_hallucination"]


def error_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": row.get("example_id"),
        "agent": row.get("agent"),
        "categories": categorize(row),
        "evidence": row.get("evidence", ""),
        "query": row.get("query", ""),
        "expected": row.get("reference", ""),
        "generated": row.get("prediction", ""),
        "decoded_liquid_representation": row.get("decoded_liquid_representation", ""),
        "state_slot_diagnostics": row.get("diagnostics", {}),
        "likely_mechanism": row.get("likely_mechanism", "requires reviewer annotation"),
    }
