"""
scripts/threshold_sensitivity.py

Run the pipeline at multiple confidence thresholds and plot recall vs precision.
This is the threshold sensitivity analysis — the interview story for precision.

Usage:
    python scripts/threshold_sensitivity.py

Outputs:
    - Console table of precision/recall/F1 at each threshold
    - threshold_sensitivity.json with full results
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

# ── We reuse the evaluation harness ──────────────────────────────────────────
from scripts.evaluate_pipeline import GROUND_TRUTH, score_document, index_regulatory_corpus
from config import settings
from src.graph.build_graph import run_pipeline
from src.audit.models import init_db
from src.vectorstore.chroma_client import collection_count

POLICY_DIR = Path(settings.raw_internal_policy_dir)
THRESHOLDS = [0.75, 0.80, 0.85, 0.90]


def run_at_threshold(threshold: float) -> dict:
    """Run all 3 policy docs at a given confidence threshold."""
    # Temporarily patch settings
    import config
    original = config.settings.confidence_threshold
    config.settings.confidence_threshold = threshold

    results = []
    for pdf in sorted(POLICY_DIR.glob("*.pdf")):
        doc_id = pdf.stem
        if doc_id not in GROUND_TRUTH:
            continue
        try:
            final_state = run_pipeline(document_id=f"{doc_id}_t{threshold}", source_path=str(pdf))
            # Override guardrail check with threshold
            scored = score_document(doc_id, final_state, GROUND_TRUTH[doc_id])
            results.append(scored)
        except Exception as e:
            print(f"    ERROR at threshold {threshold} for {doc_id}: {e}")

    config.settings.confidence_threshold = original

    if not results:
        return {"threshold": threshold, "precision": 0, "recall": 0, "f1": 0,
                "tp": 0, "fp": 0, "fn": 0}

    total_tp = sum(r["true_positives"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_prec   = sum(r["precision"] for r in results) / len(results)
    avg_f1     = sum(r["f1"] for r in results) / len(results)

    return {
        "threshold": threshold,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "avg_precision": round(avg_prec, 4),
        "avg_recall": round(avg_recall, 4),
        "avg_f1": round(avg_f1, 4),
        "per_doc": results,
    }


def main():
    print("=" * 65)
    print("  Threshold Sensitivity Analysis")
    print("=" * 65)

    print("\n[1] Regulatory corpus")
    index_regulatory_corpus()

    all_results = []
    print(f"\n[2] Running at thresholds: {THRESHOLDS}\n")
    print(f"  {'Threshold':>10} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
    print("  " + "-" * 62)

    for t in THRESHOLDS:
        print(f"  Threshold {t}...", end="", flush=True)
        result = run_at_threshold(t)
        all_results.append(result)
        print(f"\r  {t:>10.2f} | {result['avg_recall']:>7.1%} | {result['avg_precision']:>9.1%} | "
              f"{result['avg_f1']:>7.1%} | {result['true_positives']:>4} | "
              f"{result['false_positives']:>4} | {result['false_negatives']:>4}")

    print("\n" + "=" * 65)

    # Find the recommendation: highest threshold where recall is still 100%
    perfect_recall = [r for r in all_results if r["avg_recall"] >= 1.0]
    if perfect_recall:
        best = max(perfect_recall, key=lambda r: r["avg_precision"])
        print(f"\n  RECOMMENDATION: threshold={best['threshold']}")
        print(f"  At this threshold: recall={best['avg_recall']:.1%}, "
              f"precision={best['avg_precision']:.1%}, F1={best['avg_f1']:.1%}")
        print(f"  Precision improved from {all_results[0]['avg_precision']:.1%} "
              f"(at 0.75) to {best['avg_precision']:.1%} while maintaining 100% recall.")
    else:
        drop = next((r for r in all_results if r["avg_recall"] < 1.0), None)
        if drop:
            print(f"\n  NOTE: Recall dropped below 100% at threshold={drop['threshold']}")
            print(f"  This means model confidence isn't perfectly calibrated to correctness.")

    # Save results
    out_path = Path("threshold_sensitivity.json")
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Results saved to: {out_path}")

    return all_results


if __name__ == "__main__":
    main()
