# Automated Multi-Agent Financial Compliance & Audit Pipeline

A production-grade multi-agent system that ingests regulatory documents, verifies internal policy against the current regulatory corpus, flags anomalies and drift, and produces a fully traceable audit trail for every decision. A human analyst reviews and signs off on every flag before any action is taken.

**Design principle:** the system assists, it never auto-decides. Every output is traceable back to its exact source text.

---

## Architecture

See [`architecture.md`](../architecture.md) for the full architecture (v2 — refined). The pipeline is built on:

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (state machine with conditional edges) |
| LLM | LangChain + Groq API (Llama 3.3-70b) |
| Embeddings | sentence-transformers `BAAI/bge-small-en-v1.5` (local, free) |
| Vector store | ChromaDB (local persistent, 2 collections) |
| Audit trail | SQLite via SQLAlchemy |
| PDF parsing | PyMuPDF |
| Dashboard | Streamlit |
| Config | Pydantic Settings + `.env` |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A Groq API key ([console.groq.com](https://console.groq.com))

### 2. Clone and set up environment

```bash
git clone https://github.com/AbhiramRaja/RegTech-Pipeline.git
cd RegTech-Pipeline

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=gsk_...
```

### 4. Prepare sample documents

```bash
# Download real regulatory PDFs (SEC, FINRA, Basel III)
python scripts/download_regulatory_docs.py

# Generate synthetic internal policy PDFs with deliberate violations
python scripts/generate_synthetic_policies.py
```

### 5. Verify setup

```bash
python -c "from config import settings; print('✓ Config OK:', settings.llm_model_name)"
```

---

## Running the Pipeline

### Via command line

```python
from src.graph.build_graph import run_pipeline

final_state = run_pipeline(
    document_id="policy_capital_adequacy",
    source_path="data/raw_internal_policy/policy_capital_adequacy.pdf",
)
print("Status:", final_state["verification_status"])
print("Issues:", len(final_state["flagged_issues"]))
```

### Via Streamlit dashboard

```bash
streamlit run src/dashboard/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

The dashboard provides:
- **Overview** — aggregate stats across all processed documents
- **Flagged Issues** — analyst review queue with confirm/dismiss sign-off
- **Audit Trail** — full state-transition drill-down for any document/clause
- **Run Pipeline** — process new documents directly from the UI

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Test suite covers:
- **Unit tests per node** (`test_nodes.py`) — mocked LLM, state transition and schema validity assertions
- **Guardrail tests** (`test_guardrail.py`) — malformed output caught, retry routing, escalation
- **Audit trace-back tests** (`test_audit_traceback.py`) — full chain resolution, append-only invariant, analyst sign-off

All tests use isolated temp SQLite databases and ChromaDB directories — no shared state between tests.

---

## Configuration (`.env` variables)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `LLM_MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage directory |
| `AUDIT_DB_PATH` | `./audit_trail.db` | SQLite database path |
| `CONFIDENCE_THRESHOLD` | `0.75` | Minimum confidence to pass guardrail |
| `MAX_RETRIES` | `2` | Max LLM retries before human escalation |

---

## Repository Structure

```
compliance-pipeline/
├── .env.example               # Template — copy to .env and fill in
├── config.py                  # Pydantic settings: all thresholds, keys, paths
├── requirements.txt
├── pyproject.toml             # pytest config
├── data/
│   ├── raw_regulatory/        # Downloaded regulatory PDFs (Basel III, FINRA, SEC)
│   └── raw_internal_policy/   # Generated synthetic policy PDFs
├── scripts/
│   ├── download_regulatory_docs.py    # Downloads public regulatory PDFs
│   └── generate_synthetic_policies.py # Generates test policy PDFs with violations
├── src/
│   ├── ingestion/pdf_parser.py        # PyMuPDF section-aware extractor
│   ├── embeddings/embedder.py         # sentence-transformers wrapper
│   ├── vectorstore/chroma_client.py   # ChromaDB (2 collections + metadata)
│   ├── graph/
│   │   ├── state.py                   # ComplianceState TypedDict
│   │   ├── nodes.py                   # All 8 node functions
│   │   ├── supervisor.py              # Routing logic (ONLY conditional branching here)
│   │   └── build_graph.py             # LangGraph assembly + run_pipeline()
│   ├── llm/provider.py               # Groq adapter (swappable interface)
│   ├── audit/
│   │   ├── models.py                  # SQLAlchemy ORM (audit_log + flagged_issues)
│   │   └── writer.py                  # Append-only writes + trace-back
│   └── dashboard/app.py              # Streamlit review UI
└── tests/
    ├── conftest.py                    # Shared fixtures (isolated DB, Chroma, mock LLM)
    ├── test_nodes.py                  # Unit tests per node
    ├── test_guardrail.py              # Guardrail + retry + escalation tests
    └── test_audit_traceback.py        # Full trace-back + analyst sign-off tests
```

---

## Pipeline Flow

```
PDF Input
  → ingest_node (extract clauses)
  → chunk_embed_node (embed + upsert to Chroma)
  → supervisor_node (route based on state)
      ↓
  verification_agent_node (top-k retrieval + Groq LLM)
      → supervisor_node
  cross_reference_agent_node (superseded rule detection)
      → supervisor_node
  guardrail_node (validate chunk IDs, confidence, schema)
      → supervisor_node (retry if failed, escalate at max retries)
  human_review_node (terminal — Streamlit dashboard)
```

Every node writes a state snapshot to the SQLite audit trail. The full chain from any flagged issue back to its exact source regulatory chunk is always recoverable.

---

## What This Project Does NOT Do

Per architecture.md §12 (explicit non-goals):

- **Not a production system** — SQLite and local ChromaDB are fine for portfolio scale
- **Not doing real regulatory monitoring** — sample/public documents only
- **Not claiming legal compliance determinations** — the system flags for human review, full stop. No decision is auto-applied.

---

## Security Notes

- API keys are loaded only from `.env` (which is in `.gitignore`) — never hardcoded
- Audit log stores chunk IDs and summaries; full raw sensitive text is not stored in plaintext beyond what's needed for traceability
- No agent output is ever auto-applied — `human_review_node` is a hard terminal gate
- Local embeddings model avoids sending internal policy text to third-party APIs