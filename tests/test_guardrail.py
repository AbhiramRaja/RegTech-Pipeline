"""
tests/test_guardrail.py

Guardrail node tests — verifies that malformed/hallucinated LLM output is caught,
retried, and escalated correctly per architecture.md §5.
"""

import pytest
from unittest.mock import patch

from src.graph.nodes import guardrail_node
from src.graph.supervisor import route_from_supervisor
from src.llm.provider import reset_provider


class TestGuardrailValidation:
    """Test that the guardrail catches various forms of bad LLM output."""

    def test_guardrail_passes_on_valid_output(self, sample_state):
        """Valid chunk IDs, valid confidence, valid clause IDs → guardrail passes."""
        state = {
            **sample_state,
            "verification_status": "flagged",
            "confidence_score": 0.88,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "CET1 ratio too low",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }
        result = guardrail_node(state)
        assert result["guardrail_passed"] is True
        assert result["retry_count"] == 0  # no increment on pass

    def test_guardrail_fails_on_hallucinated_chunk_id(self, sample_state):
        """evidence_chunk_ids containing a non-existent chunk_id → guardrail fails."""
        state = {
            **sample_state,
            "verification_status": "flagged",
            "confidence_score": 0.80,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "Some issue",
                    "evidence_chunk_ids": ["HALLUCINATED-CHUNK-XYZ"],  # not in retrieved_context
                }
            ],
        }
        result = guardrail_node(state)
        assert result["guardrail_passed"] is False
        assert result["retry_count"] == 1  # incremented

    def test_guardrail_fails_on_hallucinated_clause_id(self, sample_state):
        """Flagged issue referencing a clause_id not in extracted_clauses → fails."""
        state = {
            **sample_state,
            "verification_status": "flagged",
            "confidence_score": 0.80,
            "flagged_issues": [
                {
                    "clause_id": "NONEXISTENT-CLAUSE-999",  # hallucinated
                    "issue": "Some issue",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }
        result = guardrail_node(state)
        assert result["guardrail_passed"] is False
        assert result["retry_count"] == 1

    def test_guardrail_fails_on_invalid_confidence(self, sample_state):
        """Confidence score outside [0.0, 1.0] → fails."""
        state = {**sample_state, "confidence_score": 1.5, "flagged_issues": []}
        result = guardrail_node(state)
        assert result["guardrail_passed"] is False
        assert result["retry_count"] == 1

    def test_guardrail_fails_on_negative_confidence(self, sample_state):
        state = {**sample_state, "confidence_score": -0.1, "flagged_issues": []}
        result = guardrail_node(state)
        assert result["guardrail_passed"] is False

    def test_guardrail_passes_on_no_issues(self, sample_state):
        """No flagged issues + valid confidence → guardrail passes (verified path)."""
        state = {**sample_state, "confidence_score": 0.95, "flagged_issues": []}
        result = guardrail_node(state)
        assert result["guardrail_passed"] is True

    def test_guardrail_appends_to_audit_log(self, sample_state):
        """Guardrail must always append an audit entry."""
        result = guardrail_node(sample_state)
        assert any(e["node_name"] == "guardrail_node" for e in result["audit_log"])

    def test_guardrail_increments_retry_count_on_each_failure(self, sample_state):
        """Each guardrail failure increments retry_count by exactly 1."""
        state = {
            **sample_state,
            "retry_count": 1,
            "confidence_score": 0.80,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "bad",
                    "evidence_chunk_ids": ["HALLUCINATED"],
                }
            ],
        }
        result = guardrail_node(state)
        assert result["retry_count"] == 2  # was 1, now 2


class TestGuardrailRetryRouting:
    """Test supervisor routing after guardrail failure."""

    def test_routes_to_retry_when_below_max(self, sample_state):
        """After guardrail fail with retry_count < MAX_RETRIES → re-verify."""
        state = {
            **sample_state,
            "guardrail_passed": False,
            "retry_count": 1,        # below max_retries=2
            "verification_status": "flagged",
        }
        route = route_from_supervisor(state)
        assert route == "verification_agent_node"

    def test_routes_to_escalation_when_at_max(self, sample_state):
        """After guardrail fail with retry_count >= MAX_RETRIES → human review."""
        state = {
            **sample_state,
            "guardrail_passed": False,
            "retry_count": 2,        # at max_retries=2
            "verification_status": "flagged",
        }
        route = route_from_supervisor(state)
        assert route == "human_review_node"

    def test_routes_to_human_review_when_escalated(self, sample_state):
        """verification_status == escalated → always human_review."""
        state = {**sample_state, "verification_status": "escalated"}
        route = route_from_supervisor(state)
        assert route == "human_review_node"

    def test_routes_to_human_review_when_guardrail_passed(self, sample_state):
        """guardrail_passed=True → human_review."""
        state = {**sample_state, "guardrail_passed": True}
        route = route_from_supervisor(state)
        assert route == "human_review_node"

    def test_routes_to_verification_when_pending(self, sample_state):
        """Fresh document (pending) → verification_agent_node."""
        state = {**sample_state, "verification_status": "pending", "guardrail_passed": False}
        route = route_from_supervisor(state)
        assert route == "verification_agent_node"

    def test_routes_to_cross_ref_after_verified(self, sample_state):
        """After verification passes (status=verified, no cross-ref yet) → cross_reference."""
        state = {
            **sample_state,
            "verification_status": "verified",
            "guardrail_passed": False,
            "audit_log": [{"node_name": "verification_agent_node", "timestamp": "t1"}],
        }
        route = route_from_supervisor(state)
        assert route == "cross_reference_agent_node"

    def test_routes_to_guardrail_after_cross_ref(self, sample_state):
        """After cross_ref ran → guardrail."""
        state = {
            **sample_state,
            "verification_status": "verified",
            "guardrail_passed": False,
            "audit_log": [
                {"node_name": "verification_agent_node", "timestamp": "t1"},
                {"node_name": "cross_reference_agent_node", "timestamp": "t2"},
            ],
        }
        route = route_from_supervisor(state)
        assert route == "guardrail_node"


class TestFullRetryFlow:
    """Integration: simulate the retry loop in isolation."""

    def test_guardrail_fail_retry_then_escalate(self, sample_state):
        """
        Simulate: guardrail fails twice → supervisor escalates.
        Tests the full retry cycle end-to-end without LLM calls.
        """
        from config import settings

        state = {
            **sample_state,
            "verification_status": "flagged",
            "guardrail_passed": False,
            "retry_count": 0,
            "confidence_score": 0.80,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "bad",
                    "evidence_chunk_ids": ["HALLUCINATED"],
                }
            ],
        }

        # First guardrail failure
        state = guardrail_node(state)
        assert state["guardrail_passed"] is False
        assert state["retry_count"] == 1
        route = route_from_supervisor(state)
        assert route == "verification_agent_node"  # retry

        # Second guardrail failure
        state = guardrail_node(state)
        assert state["retry_count"] == 2
        route = route_from_supervisor(state)
        assert route == "human_review_node"  # escalate — max retries reached
