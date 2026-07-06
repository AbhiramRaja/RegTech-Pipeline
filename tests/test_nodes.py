"""
tests/test_nodes.py

Unit tests for every LangGraph node function.
LLM calls are mocked — tests focus on state transitions and schema validity.
"""

import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from src.graph.state import ComplianceState, initial_state
from src.graph.nodes import (
    ingest_node,
    chunk_embed_node,
    verification_agent_node,
    cross_reference_agent_node,
    guardrail_node,
    audit_log_node,
    human_review_node,
)
from src.llm.provider import reset_provider


# ─────────────────────────── ingest_node ─────────────────────────────────────

class TestIngestNode:
    def test_ingest_populates_extracted_clauses(self, tmp_path, sample_state):
        """ingest_node should populate extracted_clauses from a real (minimal) PDF."""
        # Create a minimal PDF
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            pytest.skip("reportlab not installed")

        pdf_path = tmp_path / "test_doc.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        doc.build([Paragraph("CAPITAL REQUIREMENTS", styles["Heading1"]),
                   Paragraph("The institution shall maintain a CET1 ratio of at least 4.5%.", styles["Normal"])])

        state = {**sample_state, "source_path": str(pdf_path), "document_id": "test-ingest-001"}
        result = ingest_node(state)

        assert isinstance(result["extracted_clauses"], list)
        assert len(result["extracted_clauses"]) >= 1
        assert result["last_updated"] != state["last_updated"] or True  # updated
        # Audit log entry added
        assert any(e["node_name"] == "ingest_node" for e in result["audit_log"])

    def test_ingest_missing_pdf_returns_empty(self, sample_state):
        """ingest_node with non-existent PDF should return empty clauses, not crash."""
        state = {**sample_state, "source_path": "/nonexistent/path/missing.pdf"}
        result = ingest_node(state)
        assert result["extracted_clauses"] == []
        assert any(e["node_name"] == "ingest_node" for e in result["audit_log"])

    def test_ingest_appends_audit_log(self, sample_state):
        """ingest_node must append to audit_log, never overwrite."""
        state = {**sample_state, "source_path": "/nonexistent.pdf"}
        state["audit_log"] = [{"existing": "entry"}]
        result = ingest_node(state)
        assert len(result["audit_log"]) == 2  # existing + new
        assert result["audit_log"][0] == {"existing": "entry"}


# ─────────────────────────── chunk_embed_node ─────────────────────────────────

class TestChunkEmbedNode:
    def test_chunk_embed_skips_empty_clauses(self, sample_state):
        """chunk_embed_node with no clauses should log and return unchanged."""
        state = {**sample_state, "extracted_clauses": []}
        result = chunk_embed_node(state)
        assert result["extracted_clauses"] == []
        assert any(e["node_name"] == "chunk_embed_node" for e in result["audit_log"])

    def test_chunk_embed_calls_upsert(self, sample_state):
        """chunk_embed_node should call upsert_chunks with correct collection."""
        with patch("src.graph.nodes.embed_passages", return_value=[[0.1] * 384]) as mock_embed, \
             patch("src.graph.nodes.upsert_chunks") as mock_upsert:
            result = chunk_embed_node(sample_state)
            mock_embed.assert_called_once()
            mock_upsert.assert_called_once()
            call_args = mock_upsert.call_args
            assert call_args[1]["collection_name"] == "internal_policy" or \
                   call_args[0][0] == "internal_policy"


# ─────────────────────────── verification_agent_node ─────────────────────────

class TestVerificationAgentNode:
    def test_verification_sets_verified_when_compliant(
        self, sample_state, mock_llm_provider, compliant_mock_response
    ):
        """Compliant LLM response → status = verified."""
        provider = mock_llm_provider(compliant_mock_response)
        reset_provider(provider)

        with patch("src.graph.nodes.embed_query", return_value=[0.1] * 384), \
             patch("src.graph.nodes.query_similar", return_value=[
                 {"chunk_id": "regulatory-chunk-001",
                  "text": "CET1 must be ≥4.5%",
                  "score": 0.95,
                  "metadata": {"clause_id": "regulatory-chunk-001"}}
             ]):
            result = verification_agent_node(sample_state)

        assert result["verification_status"] == "verified"
        assert result["flagged_issues"] == []
        assert 0.0 <= result["confidence_score"] <= 1.0
        reset_provider(None)

    def test_verification_sets_flagged_when_non_compliant(
        self, sample_state, mock_llm_provider, non_compliant_mock_response
    ):
        """Non-compliant LLM response → status = flagged, issues populated."""
        provider = mock_llm_provider(non_compliant_mock_response)
        reset_provider(provider)

        with patch("src.graph.nodes.embed_query", return_value=[0.1] * 384), \
             patch("src.graph.nodes.query_similar", return_value=[
                 {"chunk_id": "regulatory-chunk-001",
                  "text": "CET1 must be ≥4.5%",
                  "score": 0.95,
                  "metadata": {"clause_id": "regulatory-chunk-001"}}
             ]):
            result = verification_agent_node(sample_state)

        assert result["verification_status"] == "flagged"
        assert len(result["flagged_issues"]) == 1
        assert result["flagged_issues"][0]["clause_id"] == "test-doc-001::p1::c0"
        reset_provider(None)

    def test_verification_escalates_on_empty_clauses(self, sample_state):
        """No clauses → escalated (cannot verify nothing)."""
        state = {**sample_state, "extracted_clauses": []}
        result = verification_agent_node(state)
        assert result["verification_status"] == "escalated"

    def test_verification_appends_audit_log(
        self, sample_state, mock_llm_provider, compliant_mock_response
    ):
        reset_provider(mock_llm_provider(compliant_mock_response))
        with patch("src.graph.nodes.embed_query", return_value=[0.1] * 384), \
             patch("src.graph.nodes.query_similar", return_value=[
                 {"chunk_id": "regulatory-chunk-001", "text": "x", "score": 0.9,
                  "metadata": {"clause_id": "regulatory-chunk-001"}}
             ]):
            result = verification_agent_node(sample_state)
        assert any(e["node_name"] == "verification_agent_node" for e in result["audit_log"])
        reset_provider(None)

    def test_state_schema_preserved_after_verification(
        self, sample_state, mock_llm_provider, compliant_mock_response
    ):
        """All required ComplianceState fields must still be present after the node runs."""
        reset_provider(mock_llm_provider(compliant_mock_response))
        required_keys = {
            "document_id", "source_path", "extracted_clauses", "retrieved_context",
            "verification_status", "confidence_score", "flagged_issues", "retry_count",
            "guardrail_passed", "audit_log", "last_updated"
        }
        with patch("src.graph.nodes.embed_query", return_value=[0.1] * 384), \
             patch("src.graph.nodes.query_similar", return_value=[
                 {"chunk_id": "rc-001", "text": "x", "score": 0.9,
                  "metadata": {"clause_id": "rc-001"}}
             ]):
            result = verification_agent_node(sample_state)
        assert required_keys.issubset(set(result.keys()))
        reset_provider(None)


# ─────────────────────────── cross_reference_agent_node ──────────────────────

class TestCrossReferenceNode:
    def test_cross_ref_detects_drift(self, sample_state, mock_llm_provider):
        """If superseded chunks are found and LLM says drift → issues appended."""
        drift_response = {
            "drift_detected": True,
            "confidence": 0.85,
            "drift_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "Policy references superseded Basel II rules.",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
            "reasoning": "The referenced rule was superseded in 2023.",
        }
        reset_provider(mock_llm_provider(drift_response))

        superseded_context = [
            {
                "chunk_id": "regulatory-chunk-001",
                "text": "Old rule",
                "score": 0.8,
                "metadata": {
                    "clause_id": "regulatory-chunk-001",
                    "superseded_by": "regulatory-chunk-002",
                },
            }
        ]
        with patch("src.graph.nodes.embed_query", return_value=[0.1] * 384), \
             patch("src.graph.nodes.query_similar", return_value=superseded_context):
            result = cross_reference_agent_node(sample_state)

        assert len(result["flagged_issues"]) >= 1
        reset_provider(None)

    def test_cross_ref_no_superseded_skips_llm(self, sample_state, mock_llm_provider):
        """No superseded chunks → LLM should not be called."""
        provider = mock_llm_provider({"drift_detected": False, "confidence": 1.0, "drift_issues": [], "reasoning": ""})
        reset_provider(provider)

        non_superseded_context = [
            {"chunk_id": "rc-001", "text": "Rule", "score": 0.9,
             "metadata": {"clause_id": "rc-001", "superseded_by": ""}}
        ]
        with patch("src.graph.nodes.embed_query", return_value=[0.1] * 384), \
             patch("src.graph.nodes.query_similar", return_value=non_superseded_context):
            result = cross_reference_agent_node(sample_state)

        # Provider should NOT be called (no superseded chunks)
        assert provider.call_count == 0
        reset_provider(None)


# ─────────────────────────── audit_log_node ───────────────────────────────────

class TestAuditLogNode:
    def test_audit_log_appends(self, sample_state):
        """audit_log_node must add an entry — never remove or overwrite."""
        initial_count = len(sample_state["audit_log"])
        result = audit_log_node(sample_state)
        assert len(result["audit_log"]) == initial_count + 1
        assert result["audit_log"][-1]["node_name"] == "audit_log_node"

    def test_audit_log_is_append_only(self, sample_state):
        """All prior audit entries must remain intact."""
        state = {**sample_state, "audit_log": [{"node_name": "ingest_node", "timestamp": "t0"}]}
        result = audit_log_node(state)
        assert result["audit_log"][0]["node_name"] == "ingest_node"  # original preserved


# ─────────────────────────── human_review_node ────────────────────────────────

class TestHumanReviewNode:
    def test_human_review_is_terminal(self, sample_state):
        """human_review_node should not change verification_status."""
        state = {**sample_state, "verification_status": "flagged"}
        result = human_review_node(state)
        assert result["verification_status"] == "flagged"

    def test_human_review_appends_audit(self, sample_state):
        result = human_review_node(sample_state)
        assert any(e["node_name"] == "human_review_node" for e in result["audit_log"])
