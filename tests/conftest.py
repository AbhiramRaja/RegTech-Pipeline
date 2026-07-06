"""
tests/conftest.py — Shared pytest fixtures.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

# ── Fixture: temp SQLite DB ────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    """
    Override AUDIT_DB_PATH so every test gets its own isolated SQLite file.
    Also resets the ChromaDB client and LLM provider singletons.
    """
    db_path = str(tmp_path / "test_audit.db")
    monkeypatch.setenv("AUDIT_DB_PATH", db_path)

    # Patch config.settings directly
    import config
    monkeypatch.setattr(config.settings, "audit_db_path", db_path)

    # Reset audit module singletons
    import src.audit.models as models_mod
    models_mod.init_db()

    yield db_path


@pytest.fixture(autouse=True)
def use_temp_chroma(tmp_path, monkeypatch):
    """Each test gets its own isolated ChromaDB directory."""
    chroma_dir = str(tmp_path / "chroma")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", chroma_dir)

    import config
    monkeypatch.setattr(config.settings, "chroma_persist_dir", chroma_dir)

    # Reset the ChromaDB singleton
    import src.vectorstore.chroma_client as cc
    cc._client = None

    yield chroma_dir

    cc._client = None


@pytest.fixture
def reset_llm_provider():
    """Reset the LLM provider singleton after each test."""
    from src.llm import provider as prov_mod
    yield
    prov_mod.reset_provider(None)


# ── Sample state factory ───────────────────────────────────────────────────────
@pytest.fixture
def sample_state():
    """Return a minimal valid ComplianceState for testing."""
    from src.graph.state import ComplianceState

    return ComplianceState(
        document_id="test-doc-001",
        source_path="/tmp/test.pdf",
        extracted_clauses=[
            {
                "clause_id": "test-doc-001::p1::c0",
                "text": "The institution shall maintain a CET1 ratio of no less than 4.5%.",
                "page": 1,
                "section": "Capital Requirements",
            }
        ],
        retrieved_context=[
            {
                "chunk_id": "regulatory-chunk-001",
                "text": "CET1 capital must be at least 4.5% of RWA.",
                "score": 0.95,
                "metadata": {
                    "doc_id": "basel3",
                    "clause_id": "regulatory-chunk-001",
                    "effective_date": "2023-01-01",
                    "superseded_by": "",
                    "source": "basel3_capital_requirements.pdf",
                },
            }
        ],
        verification_status="pending",
        confidence_score=0.0,
        flagged_issues=[],
        retry_count=0,
        guardrail_passed=False,
        audit_log=[],
        last_updated=datetime.utcnow().isoformat(),
    )


@pytest.fixture
def mock_llm_provider():
    """Return a mock LLM provider that returns valid compliant JSON."""
    from src.llm.provider import LLMProvider

    class MockProvider(LLMProvider):
        def __init__(self, response: dict):
            self.response = response
            self.call_count = 0

        def generate(self, prompt, system_prompt=None):
            self.call_count += 1
            return self.response

    return MockProvider


@pytest.fixture
def compliant_mock_response():
    return {
        "compliant": True,
        "confidence": 0.92,
        "issues": [],
        "reasoning": "The clause meets the minimum CET1 requirement of 4.5%.",
    }


@pytest.fixture
def non_compliant_mock_response():
    return {
        "compliant": False,
        "confidence": 0.88,
        "issues": [
            {
                "clause_id": "test-doc-001::p1::c0",
                "issue": "CET1 ratio of 3.5% is below the Basel III minimum of 4.5%.",
                "evidence_chunk_ids": ["regulatory-chunk-001"],
            }
        ],
        "reasoning": "The clause specifies 3.5% CET1 which violates the 4.5% minimum.",
    }


@pytest.fixture
def malformed_mock_response():
    """Simulates an LLM returning a response with a hallucinated chunk ID."""
    return {
        "compliant": False,
        "confidence": 0.80,
        "issues": [
            {
                "clause_id": "test-doc-001::p1::c0",
                "issue": "Non-compliance detected.",
                "evidence_chunk_ids": ["HALLUCINATED-CHUNK-XYZ"],  # Not in retrieved_context
            }
        ],
        "reasoning": "Some issue.",
    }
