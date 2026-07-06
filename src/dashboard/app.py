"""
src/dashboard/app.py

Streamlit analyst review dashboard.

Features:
  - Overview: stats on flagged/verified/escalated documents
  - Flagged issues table with filtering by status
  - Full audit trail drill-down for any issue
  - Analyst sign-off (confirm / dismiss issues)
  - Run pipeline on new documents from the UI
"""

import json
import sys
from pathlib import Path

# Ensure src/ is importable when running via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime

from src.audit.writer import (
    get_all_flagged_issues,
    get_trace,
    update_issue_status,
    persist_flagged_issues,
)
from src.audit.models import init_db, get_session, AuditLog

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RegTech Compliance Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure DB exists
init_db()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: #f1f5f9;
}
section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

/* Status badges */
.badge-open      { background:#dc2626; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-confirmed { background:#16a34a; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-dismissed { background:#6b7280; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-reviewed  { background:#d97706; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #60a5fa;
    margin: 16px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #334155;
}

/* Audit trail entry */
.audit-entry {
    background: #0f172a;
    border-left: 3px solid #3b82f6;
    padding: 10px 14px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
}

/* Violation highlight */
.violation-card {
    background: linear-gradient(135deg, #1e0a0a, #2d1515);
    border: 1px solid #ef4444;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def get_stats():
    """Return aggregate stats from the audit log."""
    try:
        session = get_session()
        total_docs = session.query(AuditLog.document_id).distinct().count()
        session.close()
    except Exception:
        total_docs = 0

    issues = get_all_flagged_issues()
    open_count = sum(1 for i in issues if i["status"] == "open")
    confirmed_count = sum(1 for i in issues if i["status"] == "confirmed")
    dismissed_count = sum(1 for i in issues if i["status"] == "dismissed")
    reviewed_count = sum(1 for i in issues if i["status"] == "reviewed")
    return {
        "total_docs": total_docs,
        "total_issues": len(issues),
        "open": open_count,
        "confirmed": confirmed_count,
        "dismissed": dismissed_count,
        "reviewed": reviewed_count,
    }


def status_badge(status: str) -> str:
    return f'<span class="badge-{status}">{status.upper()}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 RegTech Pipeline")
    st.markdown("**Compliance & Audit Dashboard**")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🚩 Flagged Issues", "🔍 Audit Trail", "▶️ Run Pipeline"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Quick Stats**")
    stats = get_stats()
    st.metric("Documents Processed", stats["total_docs"])
    st.metric("Open Issues", stats["open"], delta=None)
    st.divider()
    st.caption("v1.0 | Architecture v2")


# ── Page: Overview ─────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Compliance Pipeline Overview")
    st.caption(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📁 Documents", stats["total_docs"])
    col2.metric("🚩 Open Issues", stats["open"])
    col3.metric("✅ Confirmed", stats["confirmed"])
    col4.metric("🔕 Dismissed", stats["dismissed"])
    col5.metric("👁️ Reviewed", stats["reviewed"])

    st.divider()
    st.subheader("Issue Status Breakdown")

    all_issues = get_all_flagged_issues()
    if all_issues:
        df = pd.DataFrame(all_issues)
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        st.bar_chart(status_counts.set_index("Status"))

        st.subheader("All Issues (Summary)")
        display_df = df[["id", "document_id", "clause_id", "status", "reviewed_by", "reviewed_at"]].copy()
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No flagged issues yet. Run the pipeline to analyse documents.")

    st.divider()
    st.subheader("Recent Audit Activity")
    try:
        session = get_session()
        recent = session.query(AuditLog).order_by(AuditLog.id.desc()).limit(20).all()
        session.close()
        if recent:
            for entry in recent:
                col_a, col_b, col_c = st.columns([2, 2, 4])
                col_a.caption(entry.timestamp[:19])
                col_b.caption(f"`{entry.node_name}`")
                col_c.caption(entry.output_summary or "—")
        else:
            st.info("No audit activity yet.")
    except Exception as e:
        st.warning(f"Could not load audit activity: {e}")


# ── Page: Flagged Issues ───────────────────────────────────────────────────────
elif page == "🚩 Flagged Issues":
    st.title("🚩 Flagged Compliance Issues")

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        status_filter = st.selectbox(
            "Filter by status",
            ["All", "open", "confirmed", "dismissed", "reviewed"],
        )
    with col_f2:
        doc_filter = st.text_input("Filter by document ID", placeholder="e.g. policy_capital_adequacy")

    issues = get_all_flagged_issues(
        status_filter=None if status_filter == "All" else status_filter
    )
    if doc_filter:
        issues = [i for i in issues if doc_filter.lower() in i["document_id"].lower()]

    st.caption(f"Showing {len(issues)} issue(s)")

    if not issues:
        st.info("No issues match the current filter.")
    else:
        for issue in issues:
            with st.expander(
                f"[{issue['status'].upper()}] {issue['document_id']} — {issue['clause_id']}",
                expanded=(issue["status"] == "open"),
            ):
                st.markdown(
                    f'<div class="violation-card">'
                    f'<strong>Issue:</strong> {issue["issue_description"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                col_i1, col_i2 = st.columns(2)
                col_i1.write(f"**Document:** `{issue['document_id']}`")
                col_i2.write(f"**Clause:** `{issue['clause_id']}`")

                st.write(f"**Evidence Chunk IDs:** `{', '.join(issue['evidence_chunk_ids']) or 'none'}`")

                if issue["reviewed_by"]:
                    st.write(f"**Reviewed by:** {issue['reviewed_by']} @ {issue['reviewed_at'][:19] if issue['reviewed_at'] else '—'}")

                # Analyst sign-off
                if issue["status"] == "open":
                    st.markdown("---")
                    st.markdown("**Analyst Sign-Off**")
                    analyst_name = st.text_input(
                        "Your name / ID", key=f"analyst_{issue['id']}", placeholder="analyst@bank.com"
                    )
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Confirm", key=f"confirm_{issue['id']}"):
                        if analyst_name:
                            update_issue_status(issue["id"], "confirmed", analyst_name)
                            st.success("Marked as confirmed.")
                            st.rerun()
                        else:
                            st.warning("Enter your name before signing off.")
                    if c2.button("🔕 Dismiss", key=f"dismiss_{issue['id']}"):
                        if analyst_name:
                            update_issue_status(issue["id"], "dismissed", analyst_name)
                            st.success("Marked as dismissed.")
                            st.rerun()
                        else:
                            st.warning("Enter your name before signing off.")


# ── Page: Audit Trail ─────────────────────────────────────────────────────────
elif page == "🔍 Audit Trail":
    st.title("🔍 Audit Trail Explorer")
    st.caption(
        "Drill down into the full state-transition chain for any document. "
        "Every node writes an entry — nothing is ever overwritten or deleted."
    )

    doc_id = st.text_input(
        "Document ID to trace",
        placeholder="e.g. policy_capital_adequacy",
    )
    clause_id_filter = st.text_input(
        "Clause ID (optional — narrows to entries involving this clause)",
        placeholder="e.g. policy_capital_adequacy::p2::c0",
    )

    if st.button("🔍 Load Audit Chain") and doc_id:
        trace = get_trace(doc_id, clause_id=clause_id_filter or None)
        if not trace:
            st.warning(f"No audit entries found for document: `{doc_id}`")
        else:
            st.success(f"Found {len(trace)} audit entries for `{doc_id}`")
            for i, entry in enumerate(trace, 1):
                with st.expander(
                    f"Step {i}: `{entry['node_name']}` — {entry['timestamp'][:19]}",
                    expanded=(i == 1),
                ):
                    col_t1, col_t2 = st.columns(2)
                    col_t1.write(f"**Input:** {entry['input_summary']}")
                    col_t2.write(f"**Output:** {entry['output_summary']}")
                    col_t1.write(f"**Confidence:** {entry['confidence_score']:.3f}")

                    snapshot = entry["state_snapshot"]
                    if snapshot:
                        st.markdown("**State Snapshot**")
                        tab_s, tab_f, tab_r = st.tabs(
                            ["Verification Status", "Flagged Issues", "Retrieved Context"]
                        )
                        with tab_s:
                            st.json({
                                "verification_status": snapshot.get("verification_status"),
                                "confidence_score": snapshot.get("confidence_score"),
                                "retry_count": snapshot.get("retry_count"),
                                "guardrail_passed": snapshot.get("guardrail_passed"),
                            })
                        with tab_f:
                            flagged = snapshot.get("flagged_issues", [])
                            if flagged:
                                st.json(flagged)
                            else:
                                st.caption("No flagged issues at this step.")
                        with tab_r:
                            context = snapshot.get("retrieved_context", [])
                            if context:
                                for ctx in context[:5]:
                                    st.markdown(
                                        f"**`{ctx.get('chunk_id', '?')}`** "
                                        f"(score: {ctx.get('score', 0):.3f}): "
                                        f"{ctx.get('text', '')[:200]}..."
                                    )
                            else:
                                st.caption("No retrieved context at this step.")


# ── Page: Run Pipeline ─────────────────────────────────────────────────────────
elif page == "▶️ Run Pipeline":
    st.title("▶️ Run Compliance Pipeline")
    st.caption(
        "Select an internal policy document and run the full compliance pipeline. "
        "Results appear in Flagged Issues after completion."
    )

    from config import settings

    policy_dir = Path(settings.raw_internal_policy_dir)
    regulatory_dir = Path(settings.raw_regulatory_dir)

    # Check regulatory corpus
    reg_pdfs = list(regulatory_dir.glob("*.pdf")) if regulatory_dir.exists() else []
    if not reg_pdfs:
        st.error(
            "⚠️ No regulatory PDFs found. Run `python scripts/download_regulatory_docs.py` first."
        )
    else:
        st.success(f"✓ Regulatory corpus: {len(reg_pdfs)} documents loaded")

    policy_pdfs = list(policy_dir.glob("*.pdf")) if policy_dir.exists() else []
    if not policy_pdfs:
        st.warning(
            "No internal policy PDFs found. Run `python scripts/generate_synthetic_policies.py` first."
        )
    else:
        selected_pdf = st.selectbox(
            "Select policy document to verify",
            options=policy_pdfs,
            format_func=lambda p: p.name,
        )
        doc_id = st.text_input(
            "Document ID (auto-generated from filename)",
            value=selected_pdf.stem if selected_pdf else "",
        )

        st.divider()
        st.warning(
            "⚠️ **Pipeline will make Groq API calls.** "
            "This may take 1–3 minutes depending on document size."
        )

        if st.button("🚀 Run Pipeline", type="primary"):
            if not reg_pdfs:
                st.error("Cannot run: no regulatory corpus.")
            else:
                with st.spinner("Running compliance pipeline..."):
                    # First: ensure regulatory corpus is indexed
                    try:
                        from src.vectorstore.chroma_client import collection_count
                        reg_count = collection_count("regulatory_corpus")

                        if reg_count == 0:
                            st.info("📥 Indexing regulatory corpus (first run)...")
                            from src.ingestion.pdf_parser import extract_clauses
                            from src.embeddings.embedder import embed_passages
                            from src.vectorstore.chroma_client import upsert_chunks

                            for pdf in reg_pdfs:
                                clauses = extract_clauses(pdf, doc_id=pdf.stem)
                                if clauses:
                                    enriched = [
                                        {**c, "doc_id": pdf.stem, "effective_date": "2024-01-01",
                                         "superseded_by": "", "source": pdf.name}
                                        for c in clauses
                                    ]
                                    embeddings = embed_passages([c["text"] for c in enriched])
                                    upsert_chunks("regulatory_corpus", enriched, embeddings)

                        from src.graph.build_graph import run_pipeline
                        final_state = run_pipeline(
                            document_id=doc_id,
                            source_path=str(selected_pdf),
                        )

                        st.success("✅ Pipeline complete!")
                        col_r1, col_r2, col_r3 = st.columns(3)
                        col_r1.metric("Status", final_state["verification_status"])
                        col_r2.metric("Issues Found", len(final_state.get("flagged_issues", [])))
                        col_r3.metric("Confidence", f"{final_state.get('confidence_score', 0):.2f}")

                        if final_state.get("flagged_issues"):
                            st.warning("🚩 Issues found — see Flagged Issues tab for review.")
                            st.json(final_state["flagged_issues"])
                        else:
                            st.success("✅ No compliance issues detected.")

                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
                        st.exception(e)
