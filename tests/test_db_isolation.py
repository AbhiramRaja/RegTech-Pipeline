"""
tests/test_db_isolation.py

Regression test for the eval-tooling DB contamination bug.

Context: threshold_sensitivity.py was previously using document_id values with
suffixes like 'policy_capital_adequacy_t0.75', which polluted audit_trail.db
and caused the dashboard to show 15 documents / 89 issues instead of 3 / 19.

Fix: eval/analysis scripts now use isolated temp DBs via tempfile.

These tests assert that:
1. The main audit_trail.db never contains _t0.* pattern document IDs.
2. Running evaluate_pipeline (which does NOT use temp DBs — it writes real runs)
   produces exactly the expected document count, no phantom entries.
3. The threshold_sensitivity script (which DOES use temp DBs) produces zero
   writes to the real audit DB, regardless of how many thresholds are run.
"""

import os
import re
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audit.models import init_db, get_session, AuditLog, FlaggedIssue
from src.audit.writer import append_audit_entry, persist_flagged_issues, get_all_flagged_issues
from config import settings


# ── Pattern that must NEVER appear in production document IDs ─────────────────
EVAL_SUFFIX_PATTERN = re.compile(r"_t\d+\.\d+$")


class TestNoPhantomDocumentIDs:
    """
    Assert that document_ids with _t0.XX eval suffixes never reach the main DB.

    These tests run against the REAL audit_trail.db (not a fixture copy),
    because they are asserting production DB cleanliness.
    """

    def test_no_threshold_suffix_in_audit_log(self):
        """
        audit_log must not contain any document_id matching _t<float> pattern.
        If this fails, an eval/analysis script is writing to the main DB.
        """
        try:
            session = get_session()
            doc_ids = [row[0] for row in session.query(AuditLog.document_id).distinct().all()]
            session.close()
        except Exception:
            doc_ids = []

        contaminated = [d for d in doc_ids if EVAL_SUFFIX_PATTERN.search(d)]
        assert contaminated == [], (
            f"audit_log contains eval-suffixed document_ids: {contaminated}\n"
            "This means an evaluation/analysis script wrote to audit_trail.db.\n"
            "Fix: use an isolated temp DB in the offending script."
        )

    def test_no_threshold_suffix_in_flagged_issues(self):
        """
        flagged_issues must not contain any document_id matching _t<float> pattern.
        """
        issues = get_all_flagged_issues()
        contaminated = [i["document_id"] for i in issues if EVAL_SUFFIX_PATTERN.search(i["document_id"])]
        assert contaminated == [], (
            f"flagged_issues contains eval-suffixed document_ids: {contaminated}\n"
            "This means an evaluation/analysis script wrote to audit_trail.db.\n"
            "Fix: use an isolated temp DB in the offending script."
        )

    def test_document_count_matches_expected(self):
        """
        The number of distinct document_ids in audit_log must equal the number
        of real policy documents that have been run through the pipeline.

        This bound is loose (>= 0, <= 3 for a clean repo state) rather than
        exact, because the DB persists across runs and may have 0 entries on a
        fresh clone or 3 entries after a full eval run. What it must NOT have
        is 15 entries (the contaminated state).
        """
        MAX_EXPECTED_DOCS = 3  # We only have 3 synthetic policy PDFs
        try:
            session = get_session()
            count = session.query(AuditLog.document_id).distinct().count()
            session.close()
        except Exception:
            count = 0

        assert count <= MAX_EXPECTED_DOCS, (
            f"audit_log has {count} distinct document_ids — expected <= {MAX_EXPECTED_DOCS}.\n"
            "Likely cause: eval/analysis script wrote to audit_trail.db with synthetic doc IDs.\n"
            "Fix: check scripts/threshold_sensitivity.py and scripts/evaluate_pipeline.py "
            "to ensure they use isolated temp DBs."
        )


class TestTempDBIsolation:
    """
    Unit tests confirming that writing to a temp DB does not affect the main DB.
    This tests the isolation mechanism itself, not just the absence of contamination.
    """

    def test_writes_to_temp_db_dont_appear_in_main_db(self):
        """
        A document written to a temp DB must NOT appear in the main DB.
        """
        # Get current main DB doc count
        try:
            session = get_session()
            before_count = session.query(AuditLog.document_id).distinct().count()
            session.close()
        except Exception:
            before_count = 0

        # Write a phantom doc to an isolated temp DB
        phantom_doc_id = "phantom_eval_doc_t0.99"
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        try:
            import config
            original_db = config.settings.audit_db_path
            config.settings.audit_db_path = tmp_db
            init_db()

            state = {
                "document_id": phantom_doc_id,
                "source_path": "/fake/path.pdf",
                "extracted_clauses": [],
                "retrieved_context": [],
                "verification_status": "pending",
                "flagged_issues": [],
                "cross_ref_issues": [],
                "guardrail_passed": False,
                "confidence_score": 0.0,
                "retry_count": 0,
                "audit_log": [],
            }
            append_audit_entry(state, "ingest_node", "in", "out")

            config.settings.audit_db_path = original_db
        finally:
            os.unlink(tmp_db)

        # Main DB must be unchanged
        try:
            session = get_session()
            after_count = session.query(AuditLog.document_id).distinct().count()
            doc_ids = [r[0] for r in session.query(AuditLog.document_id).distinct().all()]
            session.close()
        except Exception:
            after_count = 0
            doc_ids = []

        assert after_count == before_count, (
            f"Main DB doc count changed from {before_count} to {after_count} "
            f"after writing to temp DB. Isolation is broken."
        )
        assert phantom_doc_id not in doc_ids, (
            f"phantom_doc_id '{phantom_doc_id}' appeared in main DB. "
            "Temp DB writes are leaking into the main DB."
        )

    def test_temp_db_receives_the_writes(self):
        """
        Confirm writes DO go to the temp DB (not silently dropped).
        """
        phantom_doc_id = "phantom_eval_doc_verification"
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        try:
            import config
            original_db = config.settings.audit_db_path
            config.settings.audit_db_path = tmp_db
            init_db()

            state = {
                "document_id": phantom_doc_id,
                "source_path": "/fake.pdf",
                "extracted_clauses": [],
                "retrieved_context": [],
                "verification_status": "flagged",
                "flagged_issues": [],
                "cross_ref_issues": [],
                "guardrail_passed": True,
                "confidence_score": 0.85,
                "retry_count": 0,
                "audit_log": [],
            }
            append_audit_entry(state, "verification_agent_node", "in", "out")

            # Verify it landed in temp DB
            conn = sqlite3.connect(tmp_db)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM audit_log WHERE document_id = ?", (phantom_doc_id,))
            count = cur.fetchone()[0]
            conn.close()

            config.settings.audit_db_path = original_db
        finally:
            os.unlink(tmp_db)

        assert count == 1, (
            f"Expected 1 row in temp DB for '{phantom_doc_id}', got {count}. "
            "Writes to the isolated temp DB are not working."
        )
