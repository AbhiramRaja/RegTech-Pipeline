"""
src/graph/state.py

ComplianceState TypedDict — exactly as specified in architecture.md §5.
Do NOT add, remove, or rename fields without explicit user approval.
"""

from typing import TypedDict, Literal, Optional
from datetime import datetime


class ComplianceState(TypedDict):
    document_id: str
    source_path: str
    extracted_clauses: list[dict]          # [{clause_id, text, page, section}]
    retrieved_context: list[dict]          # regulatory chunks pulled for comparison
    verification_status: Literal["pending", "verified", "flagged", "escalated"]
    confidence_score: float
    flagged_issues: list[dict]             # [{clause_id, issue, evidence_chunk_ids}]
    retry_count: int
    guardrail_passed: bool
    audit_log: list[dict]                  # append-only, never overwritten
    last_updated: str


def initial_state(document_id: str, source_path: str) -> ComplianceState:
    """Factory for a fresh ComplianceState with all required defaults."""
    return ComplianceState(
        document_id=document_id,
        source_path=source_path,
        extracted_clauses=[],
        retrieved_context=[],
        verification_status="pending",
        confidence_score=0.0,
        flagged_issues=[],
        retry_count=0,
        guardrail_passed=False,
        audit_log=[],
        last_updated=datetime.utcnow().isoformat(),
    )
