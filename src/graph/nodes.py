"""
src/graph/nodes.py

All 8 LangGraph node functions for the compliance pipeline.

Node responsibilities per architecture.md §6:
  1. ingest_node              — load PDF, extract text/tables → state
  2. chunk_embed_node         — split, embed, upsert to Chroma
  3. supervisor_node          — inspect status/retry_count → route (in supervisor.py)
  4. verification_agent_node  — top-k retrieval + Groq LLM → structured JSON
  5. cross_reference_agent_node — current vs superseded chunk comparison
  6. guardrail_node           — validate chunk IDs, schema, confidence
  7. audit_log_node           — append state snapshot to SQLite (runs after EVERY node)
  8. human_review_node        — terminal: surface to Streamlit dashboard

Architecture rule: conditional branching lives ONLY in supervisor.py.
Every other node has a single deterministic next step.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from config import settings
from src.graph.state import ComplianceState
from src.ingestion.pdf_parser import extract_clauses
from src.embeddings.embedder import embed_passages, embed_query
from src.vectorstore.chroma_client import upsert_chunks, query_similar
from src.llm.provider import get_provider, LLMProviderError
from src.audit.writer import append_audit_entry

logger = logging.getLogger(__name__)

# ── Prompt templates ───────────────────────────────────────────────────────────

VERIFICATION_SYSTEM_PROMPT = """You are a financial regulatory compliance expert.
Analyse whether the provided internal policy clause complies with the retrieved regulatory rules.
Respond ONLY with valid JSON matching this exact schema:
{
  "compliant": true or false,
  "confidence": <float 0.0–1.0>,
  "issues": [
    {
      "clause_id": "<the internal clause_id>",
      "issue": "<concise description of the non-compliance>",
      "evidence_chunk_ids": ["<regulatory chunk_id1>", ...]
    }
  ],
  "reasoning": "<brief explanation>"
}
If the clause is fully compliant, return an empty list for "issues".
CRITICAL: evidence_chunk_ids must only contain chunk_ids from the provided regulatory context."""

CROSS_REF_SYSTEM_PROMPT = """You are a regulatory compliance expert specializing in version drift.
Review whether the internal policy clause references any superseded regulatory rule.
Respond ONLY with valid JSON:
{
  "drift_detected": true or false,
  "confidence": <float 0.0–1.0>,
  "drift_issues": [
    {
      "clause_id": "<internal clause_id>",
      "issue": "<description of the version drift>",
      "evidence_chunk_ids": ["<chunk_id of the superseded rule>"]
    }
  ],
  "reasoning": "<brief explanation>"
}
CRITICAL: evidence_chunk_ids must only contain chunk_ids from the provided context."""


def _now() -> str:
    return datetime.utcnow().isoformat()


def _audit_append(state: ComplianceState, node_name: str, input_summary: str, output_summary: str) -> ComplianceState:
    """Append an entry to audit_log and write to SQLite. Returns updated state."""
    entry = {
        "timestamp": _now(),
        "node_name": node_name,
        "input_summary": input_summary,
        "output_summary": output_summary,
        "confidence": state.get("confidence_score", 0.0),
    }
    new_log = list(state["audit_log"]) + [entry]
    updated = {**state, "audit_log": new_log, "last_updated": _now()}
    # Persist to SQLite
    try:
        append_audit_entry(updated, node_name, input_summary, output_summary)
    except Exception as exc:
        logger.warning("Failed to write audit entry for node '%s': %s", node_name, exc)
    return updated


# ── Node 1: ingest_node ────────────────────────────────────────────────────────

def ingest_node(state: ComplianceState) -> ComplianceState:
    """
    Load a PDF and extract raw clauses into state.
    Writes raw extraction count to audit_log.
    """
    source_path = state["source_path"]
    doc_id = state["document_id"]

    clauses = extract_clauses(source_path, doc_id=doc_id)

    logger.info("[ingest_node] Extracted %d clauses from '%s'", len(clauses), source_path)
    state = {
        **state,
        "extracted_clauses": clauses,
        "last_updated": _now(),
    }
    return _audit_append(
        state,
        node_name="ingest_node",
        input_summary=f"source_path={source_path}",
        output_summary=f"extracted {len(clauses)} clauses",
    )


# ── Node 2: chunk_embed_node ───────────────────────────────────────────────────

def chunk_embed_node(state: ComplianceState) -> ComplianceState:
    """
    Embed extracted internal policy clauses and upsert to ChromaDB internal_policy collection.
    """
    clauses = state["extracted_clauses"]
    doc_id = state["document_id"]

    if not clauses:
        logger.warning("[chunk_embed_node] No clauses to embed for doc '%s'", doc_id)
        return _audit_append(
            state,
            node_name="chunk_embed_node",
            input_summary=f"doc_id={doc_id}",
            output_summary="0 clauses embedded (empty doc)",
        )

    texts = [c["text"] for c in clauses]
    embeddings = embed_passages(texts)

    # Enrich clauses with doc_id for metadata
    enriched = [{**c, "doc_id": doc_id} for c in clauses]

    upsert_chunks(
        collection_name="internal_policy",
        chunks=enriched,
        embeddings=embeddings,
    )

    logger.info("[chunk_embed_node] Embedded and upserted %d clauses for doc '%s'", len(clauses), doc_id)
    return _audit_append(
        state,
        node_name="chunk_embed_node",
        input_summary=f"doc_id={doc_id}, {len(clauses)} clauses",
        output_summary=f"upserted {len(clauses)} chunks to internal_policy",
    )


# ── Node 4: verification_agent_node ───────────────────────────────────────────

def verification_agent_node(state: ComplianceState) -> ComplianceState:
    """
    For each internal clause, retrieve top-k regulatory chunks and call Groq LLM
    to verify compliance. Produces structured JSON with issues and evidence chunk IDs.
    """
    clauses = state["extracted_clauses"]
    doc_id = state["document_id"]

    if not clauses:
        return _audit_append(
            {**state, "verification_status": "escalated", "last_updated": _now()},
            node_name="verification_agent_node",
            input_summary=f"doc_id={doc_id}",
            output_summary="no clauses to verify — escalated",
        )

    provider = get_provider()
    all_retrieved_context: list[dict] = []
    all_flagged_issues: list[dict] = []
    confidence_scores: list[float] = []

    for clause in clauses:
        clause_text = clause["text"]
        clause_id = clause["clause_id"]

        # Retrieve top-5 relevant regulatory chunks
        q_emb = embed_query(clause_text)
        retrieved = query_similar("regulatory_corpus", q_emb, top_k=5)
        all_retrieved_context.extend(retrieved)

        if not retrieved:
            logger.warning("[verification_agent] No regulatory context found for clause '%s'", clause_id)
            continue

        # Build prompt
        context_text = "\n\n".join(
            f"[chunk_id: {r['chunk_id']}]\n{r['text']}" for r in retrieved
        )
        prompt = (
            f"Internal policy clause (clause_id: {clause_id}):\n{clause_text}\n\n"
            f"Regulatory context:\n{context_text}\n\n"
            "Analyse compliance and respond in JSON."
        )

        try:
            result = provider.generate(prompt, system_prompt=VERIFICATION_SYSTEM_PROMPT)
        except LLMProviderError as exc:
            logger.error("[verification_agent] LLM error for clause '%s': %s", clause_id, exc)
            # Mark as needing retry
            return _audit_append(
                {**state, "verification_status": "flagged", "last_updated": _now()},
                node_name="verification_agent_node",
                input_summary=f"clause_id={clause_id}",
                output_summary=f"LLM error: {exc}",
            )

        confidence_scores.append(float(result.get("confidence", 0.0)))
        issues = result.get("issues", [])
        all_flagged_issues.extend(issues)

    # Deduplicate retrieved_context by chunk_id
    seen_ids: set = set()
    deduped_context: list[dict] = []
    for ctx in all_retrieved_context:
        if ctx["chunk_id"] not in seen_ids:
            seen_ids.add(ctx["chunk_id"])
            deduped_context.append(ctx)

    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    new_status = "flagged" if all_flagged_issues else "verified"

    state = {
        **state,
        "retrieved_context": deduped_context,
        "flagged_issues": all_flagged_issues,
        "confidence_score": avg_confidence,
        "verification_status": new_status,
        "last_updated": _now(),
    }
    return _audit_append(
        state,
        node_name="verification_agent_node",
        input_summary=f"doc_id={doc_id}, {len(clauses)} clauses",
        output_summary=(
            f"status={new_status}, confidence={avg_confidence:.2f}, "
            f"issues={len(all_flagged_issues)}"
        ),
    )


# ── Node 5: cross_reference_agent_node ────────────────────────────────────────

def cross_reference_agent_node(state: ComplianceState) -> ComplianceState:
    """
    Compare internal policy clauses against current vs superseded regulatory chunks.
    Flags drift where policy cites outdated rule versions.
    """
    clauses = state["extracted_clauses"]
    doc_id = state["document_id"]
    provider = get_provider()
    drift_issues: list[dict] = []

    for clause in clauses:
        clause_text = clause["text"]
        clause_id = clause["clause_id"]

        # Retrieve regulatory chunks (including superseded ones)
        q_emb = embed_query(clause_text)
        all_chunks = query_similar("regulatory_corpus", q_emb, top_k=10)

        # Identify any superseded chunks in the results
        superseded = [c for c in all_chunks if c["metadata"].get("superseded_by", "")]
        if not superseded:
            continue

        context_text = "\n\n".join(
            f"[chunk_id: {r['chunk_id']} | superseded_by: {r['metadata'].get('superseded_by', 'N/A')}]\n{r['text']}"
            for r in all_chunks[:8]
        )
        prompt = (
            f"Internal policy clause (clause_id: {clause_id}):\n{clause_text}\n\n"
            f"Regulatory context (some may be superseded):\n{context_text}\n\n"
            "Identify any version drift and respond in JSON."
        )

        try:
            result = provider.generate(prompt, system_prompt=CROSS_REF_SYSTEM_PROMPT)
        except LLMProviderError as exc:
            logger.error("[cross_ref_agent] LLM error for clause '%s': %s", clause_id, exc)
            continue

        if result.get("drift_detected", False):
            drift_issues.extend(result.get("drift_issues", []))

    # Merge drift issues with existing flagged issues
    existing = list(state.get("flagged_issues", []))
    merged = existing + drift_issues

    new_status = state["verification_status"]
    if drift_issues and new_status == "verified":
        new_status = "flagged"

    state = {
        **state,
        "flagged_issues": merged,
        "verification_status": new_status,
        "last_updated": _now(),
    }
    return _audit_append(
        state,
        node_name="cross_reference_agent_node",
        input_summary=f"doc_id={doc_id}, {len(clauses)} clauses",
        output_summary=f"drift_issues={len(drift_issues)}, total_issues={len(merged)}",
    )


# ── Node 6: guardrail_node ────────────────────────────────────────────────────

def guardrail_node(state: ComplianceState) -> ComplianceState:
    """
    Validate LLM output before it reaches the audit log and human reviewer:
      (a) Every flagged issue's evidence_chunk_ids must exist in retrieved_context.
      (b) confidence_score must be a valid float in [0.0, 1.0].
      (c) No hallucinated clause IDs (must appear in extracted_clauses).

    Passes  → set guardrail_passed=True.
    Fails   → set guardrail_passed=False, increment retry_count.
              Supervisor will then route to retry or escalate.
    """
    flagged_issues = state.get("flagged_issues", [])
    retrieved_context = state.get("retrieved_context", [])
    extracted_clauses = state.get("extracted_clauses", [])
    confidence_score = state.get("confidence_score", 0.0)
    retry_count = state.get("retry_count", 0)

    valid_chunk_ids = {ctx["chunk_id"] for ctx in retrieved_context}
    valid_clause_ids = {c["clause_id"] for c in extracted_clauses}

    validation_errors: list[str] = []

    # (b) Confidence score validity
    if not isinstance(confidence_score, (int, float)) or not (0.0 <= float(confidence_score) <= 1.0):
        validation_errors.append(f"Invalid confidence_score: {confidence_score!r}")

    # (a) & (c) Per-issue validation
    for issue in flagged_issues:
        clause_id = issue.get("clause_id", "")
        evidence_ids = issue.get("evidence_chunk_ids", [])

        if clause_id and clause_id not in valid_clause_ids:
            validation_errors.append(f"Hallucinated clause_id: {clause_id!r}")

        for cid in evidence_ids:
            if cid not in valid_chunk_ids:
                validation_errors.append(f"evidence_chunk_id not in retrieved_context: {cid!r}")

    if validation_errors:
        logger.warning("[guardrail_node] %d validation error(s): %s", len(validation_errors), validation_errors)
        state = {
            **state,
            "guardrail_passed": False,
            "retry_count": retry_count + 1,
            "last_updated": _now(),
        }
        return _audit_append(
            state,
            node_name="guardrail_node",
            input_summary=f"issues={len(flagged_issues)}, confidence={confidence_score}",
            output_summary=f"FAILED — {len(validation_errors)} errors: {validation_errors[:3]}",
        )

    logger.info("[guardrail_node] All validations passed.")
    state = {
        **state,
        "guardrail_passed": True,
        "last_updated": _now(),
    }
    return _audit_append(
        state,
        node_name="guardrail_node",
        input_summary=f"issues={len(flagged_issues)}, confidence={confidence_score}",
        output_summary="PASSED",
    )


# ── Node 7: audit_log_node ────────────────────────────────────────────────────

def audit_log_node(state: ComplianceState) -> ComplianceState:
    """
    Append the current full state snapshot to the SQLite audit table.
    This node runs after EVERY other node — not just at the end.
    audit_log is append-only; this node never removes or edits prior entries.
    """
    # The _audit_append calls inside each node already write to SQLite.
    # This dedicated node provides an explicit checkpoint snapshot.
    return _audit_append(
        state,
        node_name="audit_log_node",
        input_summary=f"status={state['verification_status']}, retry_count={state['retry_count']}",
        output_summary=f"snapshot logged — {len(state['audit_log'])} total entries",
    )


# ── Node 8: human_review_node ─────────────────────────────────────────────────

def human_review_node(state: ComplianceState) -> ComplianceState:
    """
    Terminal node. Surfaces flagged/escalated items to the Streamlit dashboard.
    Does not auto-apply any action — human analyst must sign off.
    """
    logger.info(
        "[human_review_node] Document '%s' ready for human review. "
        "Status: %s, Issues: %d",
        state["document_id"],
        state["verification_status"],
        len(state.get("flagged_issues", [])),
    )
    return _audit_append(
        state,
        node_name="human_review_node",
        input_summary=f"doc_id={state['document_id']}, status={state['verification_status']}",
        output_summary=f"awaiting analyst review — {len(state.get('flagged_issues', []))} issues",
    )
