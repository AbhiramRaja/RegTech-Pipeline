"""
src/audit/writer.py

Audit trail writer — append-only writes to SQLite.

Design principles:
  - append_audit_entry() NEVER updates existing rows.
  - get_trace() reconstructs the full chain of state transitions for any flag.
  - persist_flagged_issues() writes flagged issues for analyst review.
  - update_issue_status() is the only write that modifies existing data
    (analyst sign-off). It touches only the flagged_issues table.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from src.audit.models import AuditLog, FlaggedIssue, init_db, get_session
from src.graph.state import ComplianceState

logger = logging.getLogger(__name__)

# Ensure tables exist on first import
try:
    init_db()
except Exception as exc:
    logger.warning("Could not init audit DB on import: %s", exc)


def append_audit_entry(
    state: ComplianceState,
    node_name: str,
    input_summary: str = "",
    output_summary: str = "",
) -> None:
    """
    Append a single audit entry for the current node execution.
    append-only: never updates existing rows.
    """
    try:
        session = get_session()
        # Serialize full state snapshot — exclude audit_log from snapshot to avoid recursion
        snapshot = {k: v for k, v in state.items() if k != "audit_log"}
        entry = AuditLog(
            document_id=state["document_id"],
            node_name=node_name,
            timestamp=datetime.utcnow().isoformat(),
            input_summary=input_summary[:1000],
            output_summary=output_summary[:1000],
            confidence_score=float(state.get("confidence_score", 0.0)),
            state_snapshot_json=json.dumps(snapshot, default=str),
        )
        session.add(entry)
        session.commit()
        session.close()
    except Exception as exc:
        logger.error("Failed to append audit entry for node '%s': %s", node_name, exc)


def persist_flagged_issues(state: ComplianceState) -> None:
    """
    Write all flagged issues from state to the flagged_issues table.
    Called after guardrail passes — idempotent on clause_id + document_id.
    """
    flagged = state.get("flagged_issues", [])
    if not flagged:
        return

    try:
        session = get_session()
        for issue in flagged:
            # Check for existing entry to avoid duplicates
            existing = (
                session.query(FlaggedIssue)
                .filter_by(
                    document_id=state["document_id"],
                    clause_id=issue.get("clause_id", ""),
                )
                .first()
            )
            if existing:
                continue  # Already persisted

            record = FlaggedIssue(
                document_id=state["document_id"],
                clause_id=issue.get("clause_id", ""),
                issue_description=issue.get("issue", ""),
                evidence_chunk_ids=json.dumps(issue.get("evidence_chunk_ids", [])),
                status="open",
            )
            session.add(record)
        session.commit()
        session.close()
        logger.info("Persisted %d flagged issues for doc '%s'", len(flagged), state["document_id"])
    except Exception as exc:
        logger.error("Failed to persist flagged issues: %s", exc)


def update_issue_status(
    issue_id: int,
    new_status: str,
    reviewed_by: str,
) -> bool:
    """
    Analyst sign-off: update a flagged issue's status.
    Only modifies flagged_issues — never touches audit_log.

    Args:
        issue_id:    The flagged_issues.id to update.
        new_status:  One of: reviewed, confirmed, dismissed.
        reviewed_by: Analyst identifier (name or email).

    Returns:
        True if updated, False if not found.
    """
    valid_statuses = {"reviewed", "confirmed", "dismissed"}
    if new_status not in valid_statuses:
        raise ValueError(f"Invalid status '{new_status}'. Must be one of {valid_statuses}")

    try:
        session = get_session()
        issue = session.query(FlaggedIssue).filter_by(id=issue_id).first()
        if not issue:
            session.close()
            return False
        issue.status = new_status
        issue.reviewed_by = reviewed_by
        issue.reviewed_at = datetime.utcnow().isoformat()
        session.commit()
        session.close()
        logger.info("Issue %d marked as '%s' by '%s'", issue_id, new_status, reviewed_by)
        return True
    except Exception as exc:
        logger.error("Failed to update issue %d status: %s", issue_id, exc)
        return False


def get_trace(document_id: str, clause_id: Optional[str] = None) -> list[dict]:
    """
    Reconstruct the full chain of state transitions for a document (or specific clause).

    Returns:
        Ordered list of audit log entries with their state snapshots.
        Can be used to trace any flagged issue back to its source chunk IDs.
    """
    try:
        session = get_session()
        query = session.query(AuditLog).filter_by(document_id=document_id).order_by(AuditLog.id)
        entries = query.all()
        session.close()

        result = []
        for entry in entries:
            snapshot = {}
            try:
                snapshot = json.loads(entry.state_snapshot_json)
            except Exception:
                pass
            result.append(
                {
                    "id": entry.id,
                    "node_name": entry.node_name,
                    "timestamp": entry.timestamp,
                    "input_summary": entry.input_summary,
                    "output_summary": entry.output_summary,
                    "confidence_score": entry.confidence_score,
                    "state_snapshot": snapshot,
                }
            )

        if clause_id:
            # Filter to entries where this clause appears in flagged_issues
            result = [
                e for e in result
                if any(
                    fi.get("clause_id") == clause_id
                    for fi in e["state_snapshot"].get("flagged_issues", [])
                )
            ]

        return result
    except Exception as exc:
        logger.error("get_trace failed for doc '%s': %s", document_id, exc)
        return []


def get_all_flagged_issues(status_filter: Optional[str] = None) -> list[dict]:
    """
    Return all flagged issues (optionally filtered by status) for the dashboard.
    """
    try:
        session = get_session()
        query = session.query(FlaggedIssue)
        if status_filter:
            query = query.filter_by(status=status_filter)
        issues = query.all()
        session.close()
        return [
            {
                "id": i.id,
                "document_id": i.document_id,
                "clause_id": i.clause_id,
                "issue_description": i.issue_description,
                "evidence_chunk_ids": json.loads(i.evidence_chunk_ids or "[]"),
                "status": i.status,
                "reviewed_by": i.reviewed_by,
                "reviewed_at": i.reviewed_at,
            }
            for i in issues
        ]
    except Exception as exc:
        logger.error("get_all_flagged_issues failed: %s", exc)
        return []
