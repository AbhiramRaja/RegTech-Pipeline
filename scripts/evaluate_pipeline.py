"""
scripts/evaluate_pipeline.py

Run the compliance pipeline on all synthetic policy PDFs and measure
accuracy against the known ground-truth violations.

Known violations (ground truth):
  policy_capital_adequacy    : 2 violations (CET1 3.5%, Tier1 5.0%)
  policy_business_continuity : 1 violation  (BCP testing biennial)
  policy_risk_management     : 3 violations (VaR 95%, stress quarterly, no CVA)

Total known violations: 6
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

from config import settings
from src.vectorstore.chroma_client import collection_count, upsert_chunks, reset_collection
from src.ingestion.pdf_parser import extract_clauses
from src.embeddings.embedder import embed_passages
from src.graph.build_graph import run_pipeline

# ── Ground truth ───────────────────────────────────────────────────────────────
GROUND_TRUTH = {
    "policy_capital_adequacy": {
        "violations": [
            {"keyword": "3.5", "description": "CET1 ratio 3.5% below Basel III minimum of 4.5%"},
            {"keyword": "5.0",  "description": "Tier 1 ratio 5.0% below Basel III minimum of 6.0%"},
        ]
    },
    "policy_business_continuity": {
        "violations": [
            {"keyword": "biennial", "description": "BCP testing biennial — FINRA Rule 4370 requires annual"},
        ]
    },
    "policy_risk_management": {
        "violations": [
            {"keyword": "95", "description": "VaR at 95% CI — Basel III requires 99% for internal models"},
            {"keyword": "quarterly", "description": "Stress testing quarterly — Basel III requires monthly"},
            {"keyword": "CVA",  "description": "No CVA capital charge applied to counterparty credit risk"},
        ]
    },
}

POLICY_DIR = Path(settings.raw_internal_policy_dir)
REG_DIR = Path(settings.raw_regulatory_dir)


def index_regulatory_corpus():
    """Index regulatory PDFs into Chroma if not already indexed."""
    count = collection_count("regulatory_corpus")
    if count > 0:
        print(f"  Regulatory corpus already indexed: {count} chunks")
        return

    print("  Indexing regulatory corpus...")
    for pdf in sorted(REG_DIR.glob("*.pdf")):
        clauses = extract_clauses(pdf, doc_id=pdf.stem)
        if clauses:
            enriched = [
                {**c, "doc_id": pdf.stem, "effective_date": "2024-01-01",
                 "superseded_by": "", "source": pdf.name}
                for c in clauses
            ]
            embeddings = embed_passages([c["text"] for c in enriched])
            upsert_chunks("regulatory_corpus", enriched, embeddings)
            print(f"    Indexed {len(clauses)} chunks from {pdf.name}")


def score_document(doc_id: str, final_state: dict, ground_truth: dict) -> dict:
    """Check how many ground-truth violations were caught."""
    detected_issues = final_state.get("flagged_issues", [])
    all_issue_text = " ".join(
        (issue.get("issue", "") + " " + issue.get("clause_id", "")).lower()
        for issue in detected_issues
    )
    # Also check the raw extracted clauses that were flagged
    flagged_clause_ids = {issue.get("clause_id", "") for issue in detected_issues}
    extracted = final_state.get("extracted_clauses", [])
    flagged_clause_texts = " ".join(
        c["text"].lower() for c in extracted if c["clause_id"] in flagged_clause_ids
    )
    combined_text = all_issue_text + " " + flagged_clause_texts

    true_positives = []
    false_negatives = []
    for v in ground_truth["violations"]:
        if v["keyword"].lower() in combined_text:
            true_positives.append(v)
        else:
            false_negatives.append(v)

    total_gt = len(ground_truth["violations"])
    tp = len(true_positives)
    fn = len(false_negatives)
    fp = max(0, len(detected_issues) - tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / total_gt if total_gt > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "doc_id": doc_id,
        "ground_truth_violations": total_gt,
        "detected_issues": len(detected_issues),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confidence_score": final_state.get("confidence_score", 0.0),
        "verification_status": final_state.get("verification_status", ""),
        "audit_log_entries": len(final_state.get("audit_log", [])),
        "caught": [v["description"] for v in true_positives],
        "missed": [v["description"] for v in false_negatives],
        "guardrail_passed": final_state.get("guardrail_passed", False),
        "retry_count": final_state.get("retry_count", 0),
    }


def main():
    print("=" * 65)
    print("  RegTech Compliance Pipeline — Evaluation")
    print("=" * 65)

    # Index regulatory corpus
    print("\n[1/4] Regulatory corpus")
    index_regulatory_corpus()

    results = []
    policy_pdfs = sorted(POLICY_DIR.glob("*.pdf"))

    if not policy_pdfs:
        print("No policy PDFs found. Run generate_synthetic_policies.py first.")
        sys.exit(1)

    print(f"\n[2/4] Running pipeline on {len(policy_pdfs)} document(s)...\n")

    for pdf in policy_pdfs:
        doc_id = pdf.stem
        if doc_id not in GROUND_TRUTH:
            print(f"  Skipping {doc_id} (no ground truth)")
            continue

        print(f"  ▶ {doc_id}...")
        try:
            final_state = run_pipeline(document_id=doc_id, source_path=str(pdf))
            scored = score_document(doc_id, final_state, GROUND_TRUTH[doc_id])
            results.append(scored)
            print(f"    Status: {scored['verification_status']} | "
                  f"Issues detected: {scored['detected_issues']} | "
                  f"TP: {scored['true_positives']}/{scored['ground_truth_violations']} | "
                  f"Confidence: {scored['confidence_score']:.2f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    if not results:
        print("No results to report.")
        return

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    total_gt   = sum(r["ground_truth_violations"] for r in results)
    total_tp   = sum(r["true_positives"] for r in results)
    total_fp   = sum(r["false_positives"] for r in results)
    total_fn   = sum(r["false_negatives"] for r in results)
    avg_conf   = sum(r["confidence_score"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_prec   = sum(r["precision"] for r in results) / len(results)
    avg_f1     = sum(r["f1"] for r in results) / len(results)

    # ── Per-document report ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Per-Document Results")
    print("=" * 65)

    for r in results:
        print(f"\n📄 {r['doc_id']}")
        print(f"   Status          : {r['verification_status']}")
        print(f"   Confidence      : {r['confidence_score']:.3f}")
        print(f"   Guardrail passed: {r['guardrail_passed']}")
        print(f"   Retry count     : {r['retry_count']}")
        print(f"   Audit entries   : {r['audit_log_entries']}")
        print(f"   Ground truth    : {r['ground_truth_violations']} violations")
        print(f"   Detected issues : {r['detected_issues']}")
        print(f"   Precision       : {r['precision']:.1%}")
        print(f"   Recall          : {r['recall']:.1%}")
        print(f"   F1 Score        : {r['f1']:.1%}")
        if r["caught"]:
            print("   ✅ Caught:")
            for c in r["caught"]:
                print(f"      - {c}")
        if r["missed"]:
            print("   ❌ Missed:")
            for m in r["missed"]:
                print(f"      - {m}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Overall Pipeline Accuracy")
    print("=" * 65)
    print(f"  Documents evaluated : {len(results)}")
    print(f"  Known violations    : {total_gt}")
    print(f"  True positives (TP) : {total_tp}")
    print(f"  False positives (FP): {total_fp}")
    print(f"  False negatives (FN): {total_fn}")
    print(f"  Avg Precision       : {avg_prec:.1%}")
    print(f"  Avg Recall          : {avg_recall:.1%}")
    print(f"  Avg F1 Score        : {avg_f1:.1%}")
    print(f"  Avg Confidence      : {avg_conf:.3f}")
    print("=" * 65)

    # Save as JSON for reference
    out = {
        "summary": {
            "documents_evaluated": len(results),
            "total_known_violations": total_gt,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "avg_precision": round(avg_prec, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1": round(avg_f1, 4),
            "avg_confidence": round(avg_conf, 4),
        },
        "per_document": results,
    }
    out_path = Path("evaluation_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
