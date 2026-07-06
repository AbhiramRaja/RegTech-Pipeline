"""
tests/test_audit_traceback.py

Audit trace-back tests — for every flagged issue, assert the full chain from
flag → audit_log entries → source chunk IDs resolves with no gaps.
"""

import json
from datetime import datetime

import pytest

from src.audit.writer import (
    append_audit_entry,
    persist_flagged_issues,
    get_trace,
    get_all_flagged_issues,
    update_issue_status,
)
from src.audit.models import init_db
from src.graph.state import initial_state


class TestAuditTrail:
    """Audit log append-only and full trace-back tests."""

    def test_append_audit_entry_creates_row(self, sample_state):
        """append_audit_entry should create a row in audit_log table."""
        append_audit_entry(sample_state, "ingest_node", "input_x", "output_x")
        trace = get_trace(sample_state["document_id"])
        assert len(trace) == 1
        assert trace[0]["node_name"] == "ingest_node"

    def test_multiple_nodes_append_in_order(self, sample_state):
        """Entries from multiple nodes must appear in insertion order."""
        nodes = ["ingest_node", "chunk_embed_node", "verification_agent_node",
                 "guardrail_node", "audit_log_node"]
        for node in nodes:
            append_audit_entry(sample_state, node, f"in_{node}", f"out_{node}")

        trace = get_trace(sample_state["document_id"])
        assert len(trace) == len(nodes)
        for i, entry in enumerate(trace):
            assert entry["node_name"] == nodes[i]

    def test_audit_entries_are_never_modified(self, sample_state):
        """Once written, audit entries must not change — append-only invariant."""
        append_audit_entry(sample_state, "ingest_node", "original_input", "original_output")
        # Write another entry — first must remain unchanged
        append_audit_entry(sample_state, "chunk_embed_node", "second_input", "second_output")

        trace = get_trace(sample_state["document_id"])
        assert trace[0]["input_summary"] == "original_input"
        assert trace[0]["output_summary"] == "original_output"
        assert trace[1]["node_name"] == "chunk_embed_node"

    def test_state_snapshot_persisted(self, sample_state):
        """State snapshot JSON must be recoverable from audit_log."""
        append_audit_entry(sample_state, "verification_agent_node", "in", "out")
        trace = get_trace(sample_state["document_id"])
        snapshot = trace[0]["state_snapshot"]
        assert snapshot["document_id"] == sample_state["document_id"]
        assert "extracted_clauses" in snapshot
        assert "verification_status" in snapshot

    def test_state_snapshot_excludes_audit_log_recursion(self, sample_state):
        """State snapshot must not include audit_log to avoid circular JSON."""
        append_audit_entry(sample_state, "guardrail_node", "in", "out")
        trace = get_trace(sample_state["document_id"])
        snapshot = trace[0]["state_snapshot"]
        # audit_log should be excluded from the snapshot (per writer.py design)
        assert "audit_log" not in snapshot


class TestFlaggedIssueTraceback:
    """End-to-end trace: flagged issue → audit_log → source chunk IDs."""

    def test_persist_flagged_issues(self, sample_state):
        """persist_flagged_issues should write issues to flagged_issues table."""
        state = {
            **sample_state,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "CET1 ratio below minimum",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }
        persist_flagged_issues(state)
        issues = get_all_flagged_issues()
        assert len(issues) == 1
        assert issues[0]["document_id"] == "test-doc-001"
        assert issues[0]["clause_id"] == "test-doc-001::p1::c0"
        assert issues[0]["status"] == "open"

    def test_traceback_resolves_to_source_chunk_ids(self, sample_state):
        """
        Given a flagged issue, the full audit chain must trace back to
        the evidence chunk IDs — no gaps in the chain.
        """
        # Simulate the pipeline: ingest → verification (with flag) → guardrail → human_review
        flagged_state = {
            **sample_state,
            "verification_status": "flagged",
            "confidence_score": 0.82,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "CET1 ratio 3.5% below Basel III minimum 4.5%",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }

        for node in ["ingest_node", "verification_agent_node", "guardrail_node", "human_review_node"]:
            append_audit_entry(flagged_state, node, f"input_{node}", f"output_{node}")

        persist_flagged_issues(flagged_state)

        # Full traceback
        trace = get_trace("test-doc-001")
        assert len(trace) == 4

        # Trace for the specific clause
        clause_trace = get_trace("test-doc-001", clause_id="test-doc-001::p1::c0")
        assert len(clause_trace) >= 1

        # Verify source chunk IDs are recoverable from the trace
        all_chunk_ids: set[str] = set()
        for entry in clause_trace:
            flagged = entry["state_snapshot"].get("flagged_issues", [])
            for issue in flagged:
                if issue.get("clause_id") == "test-doc-001::p1::c0":
                    all_chunk_ids.update(issue.get("evidence_chunk_ids", []))

        assert "regulatory-chunk-001" in all_chunk_ids, \
            "Evidence chunk ID must be recoverable from the audit trace"

    def test_persist_is_idempotent(self, sample_state):
        """Calling persist_flagged_issues twice must not create duplicates."""
        state = {
            **sample_state,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "Duplicate test",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }
        persist_flagged_issues(state)
        persist_flagged_issues(state)  # second call must be a no-op
        issues = get_all_flagged_issues()
        assert len(issues) == 1  # not 2


class TestAnalystSignOff:
    """Test analyst review workflow."""

    def test_update_status_confirmed(self, sample_state):
        """Analyst can mark issue as confirmed."""
        state = {
            **sample_state,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "Issue",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }
        persist_flagged_issues(state)
        issues = get_all_flagged_issues()
        issue_id = issues[0]["id"]

        success = update_issue_status(issue_id, "confirmed", "analyst@bank.com")
        assert success is True

        updated = get_all_flagged_issues()
        assert updated[0]["status"] == "confirmed"
        assert updated[0]["reviewed_by"] == "analyst@bank.com"

    def test_update_status_dismissed(self, sample_state):
        """Analyst can dismiss an issue."""
        state = {
            **sample_state,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "False positive",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }
        persist_flagged_issues(state)
        issues = get_all_flagged_issues()
        success = update_issue_status(issues[0]["id"], "dismissed", "senior_analyst")
        assert success is True

    def test_update_status_invalid_raises(self, sample_state):
        """Invalid status value must raise ValueError."""
        state = {
            **sample_state,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "x",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }
        persist_flagged_issues(state)
        issues = get_all_flagged_issues()
        with pytest.raises(ValueError, match="Invalid status"):
            update_issue_status(issues[0]["id"], "auto_approve", "rogue_bot")

    def test_update_nonexistent_issue_returns_false(self):
        """Updating a non-existent issue ID must return False."""
        result = update_issue_status(99999, "confirmed", "analyst")
        assert result is False

    def test_audit_log_not_touched_by_analyst_update(self, sample_state):
        """Analyst sign-off must never modify audit_log rows."""
        append_audit_entry(sample_state, "ingest_node", "in", "out")
        state = {
            **sample_state,
            "flagged_issues": [
                {
                    "clause_id": "test-doc-001::p1::c0",
                    "issue": "x",
                    "evidence_chunk_ids": ["regulatory-chunk-001"],
                }
            ],
        }
        persist_flagged_issues(state)
        issues = get_all_flagged_issues()
        update_issue_status(issues[0]["id"], "confirmed", "analyst")

        # Audit log must be unchanged
        trace = get_trace("test-doc-001")
        assert len(trace) == 1
        assert trace[0]["node_name"] == "ingest_node"
