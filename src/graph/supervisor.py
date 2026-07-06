"""
src/graph/supervisor.py

Supervisor node — the ONLY place with conditional routing/branching logic.

Architecture rule (architecture.md §6):
  "Conditional edges live only in supervisor_node. Every other node has a single
  deterministic next step."

Routing logic:
  verification_status == "pending"    → verification_agent_node
  verification_status == "verified"   → guardrail_node (if not already passed)
  verification_status == "flagged"    → cross_reference_agent_node (first time)
                                       → guardrail_node (after cross-ref)
  guardrail_passed == False:
    retry_count < MAX_RETRIES         → back to verification_agent_node (retry)
    retry_count >= MAX_RETRIES        → human_review_node (escalate)
  verification_status == "escalated"  → human_review_node
  guardrail_passed == True            → human_review_node
"""

import logging
from typing import Literal

from config import settings
from src.graph.state import ComplianceState

logger = logging.getLogger(__name__)

# Routing targets — must match node names registered in build_graph.py
RouteTarget = Literal[
    "verification_agent_node",
    "cross_reference_agent_node",
    "guardrail_node",
    "human_review_node",
]


def supervisor_node(state: ComplianceState) -> ComplianceState:
    """
    Inspect state and set internal routing — does not change domain state fields.
    The actual routing is done by route_from_supervisor() which LangGraph calls
    on the conditional edge.
    """
    # Supervisor only reads state to decide routing — no domain mutations.
    return state


def route_from_supervisor(state: ComplianceState) -> RouteTarget:
    """
    Pure routing function called by LangGraph's conditional edge from supervisor_node.

    Returns the name of the next node to execute.
    """
    status = state.get("verification_status", "pending")
    retry_count = state.get("retry_count", 0)
    guardrail_passed = state.get("guardrail_passed", False)
    max_retries = settings.max_retries

    logger.debug(
        "[supervisor] status=%s, retry_count=%d, guardrail_passed=%s",
        status, retry_count, guardrail_passed,
    )

    # Escalation takes priority
    if status == "escalated":
        logger.info("[supervisor] → escalated → human_review_node")
        return "human_review_node"

    # Guardrail passed → final human review
    if guardrail_passed:
        logger.info("[supervisor] → guardrail passed → human_review_node")
        return "human_review_node"

    # Guardrail failed
    if state.get("guardrail_passed") is False and retry_count > 0:
        if retry_count >= max_retries:
            logger.info(
                "[supervisor] → max retries (%d) reached → escalating → human_review_node",
                max_retries,
            )
            # Update status to escalated
            return "human_review_node"
        else:
            logger.info(
                "[supervisor] → guardrail failed (retry %d/%d) → verification_agent_node",
                retry_count, max_retries,
            )
            return "verification_agent_node"

    # Normal flow
    if status == "pending":
        return "verification_agent_node"

    if status == "verified":
        # Cross-reference check before guardrail
        if not _cross_ref_done(state):
            return "cross_reference_agent_node"
        return "guardrail_node"

    if status == "flagged":
        if not _cross_ref_done(state):
            return "cross_reference_agent_node"
        return "guardrail_node"

    # Fallback — should never reach here
    logger.error("[supervisor] Unexpected state — escalating. status=%s", status)
    return "human_review_node"


def _cross_ref_done(state: ComplianceState) -> bool:
    """Check if cross_reference_agent_node has already run for this document."""
    return any(
        entry.get("node_name") == "cross_reference_agent_node"
        for entry in state.get("audit_log", [])
    )
