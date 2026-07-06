"""
src/graph/build_graph.py

LangGraph graph assembly.

Wires all nodes and conditional edges per architecture.md §4 & §6.
Conditional branching lives ONLY in supervisor_node (via route_from_supervisor).

Graph flow:
  START
    → ingest_node
    → chunk_embed_node
    → supervisor_node  ←─────────────────────────────┐
        │ (conditional edge via route_from_supervisor) │
        ├── verification_agent_node → supervisor_node ─┤
        ├── cross_reference_agent_node → supervisor_node ┤
        ├── guardrail_node → supervisor_node ──────────┤
        └── human_review_node → END
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.graph.state import ComplianceState
from src.graph.nodes import (
    ingest_node,
    chunk_embed_node,
    verification_agent_node,
    cross_reference_agent_node,
    guardrail_node,
    audit_log_node,
    human_review_node,
)
from src.graph.supervisor import supervisor_node, route_from_supervisor

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Returns:
        A compiled LangGraph graph ready to invoke with a ComplianceState.
    """
    graph = StateGraph(ComplianceState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("ingest_node", ingest_node)
    graph.add_node("chunk_embed_node", chunk_embed_node)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("verification_agent_node", verification_agent_node)
    graph.add_node("cross_reference_agent_node", cross_reference_agent_node)
    graph.add_node("guardrail_node", guardrail_node)
    graph.add_node("audit_log_node", audit_log_node)
    graph.add_node("human_review_node", human_review_node)

    # ── Deterministic edges ────────────────────────────────────────────────────
    # Entry point
    graph.add_edge(START, "ingest_node")
    graph.add_edge("ingest_node", "chunk_embed_node")
    graph.add_edge("chunk_embed_node", "supervisor_node")

    # After each agent node → back to supervisor for re-routing
    graph.add_edge("verification_agent_node", "supervisor_node")
    graph.add_edge("cross_reference_agent_node", "supervisor_node")
    graph.add_edge("guardrail_node", "supervisor_node")

    # Terminal: human_review_node → END
    graph.add_edge("human_review_node", END)

    # ── Conditional edge (ONLY in supervisor) ──────────────────────────────────
    graph.add_conditional_edges(
        "supervisor_node",
        route_from_supervisor,
        {
            "verification_agent_node": "verification_agent_node",
            "cross_reference_agent_node": "cross_reference_agent_node",
            "guardrail_node": "guardrail_node",
            "human_review_node": "human_review_node",
        },
    )

    compiled = graph.compile()
    logger.info("Compliance graph compiled successfully.")
    return compiled


# Singleton compiled graph
_graph = None


def get_graph():
    """Return the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(document_id: str, source_path: str) -> ComplianceState:
    """
    Convenience function: run the full compliance pipeline on one document.

    Args:
        document_id: Unique identifier for this document.
        source_path: Absolute or relative path to the PDF.

    Returns:
        Final ComplianceState after the graph reaches END.
    """
    from src.graph.state import initial_state
    from src.audit.writer import persist_flagged_issues

    graph = get_graph()
    state = initial_state(document_id=document_id, source_path=source_path)

    logger.info("Starting pipeline for document '%s' at '%s'", document_id, source_path)
    final_state = graph.invoke(state)

    # Persist flagged issues to the dedicated table (for dashboard queries)
    if final_state.get("flagged_issues"):
        persist_flagged_issues(final_state)

    logger.info(
        "Pipeline complete for '%s'. Status: %s, Issues: %d, Audit entries: %d",
        document_id,
        final_state.get("verification_status"),
        len(final_state.get("flagged_issues", [])),
        len(final_state.get("audit_log", [])),
    )
    return final_state
