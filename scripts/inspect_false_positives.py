"""
scripts/inspect_false_positives.py

Read flagged issues from the audit DB and print the full list,
separating known true-positive violations from the extra flags.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audit.writer import get_all_flagged_issues

KNOWN_VIOLATIONS = {
    "3.5": "CET1 ratio",
    "5.0": "Tier 1 ratio",
    "biennial": "BCP testing frequency",
    "95%": "VaR confidence interval",
    "95": "VaR confidence interval",
    "quarterly": "Stress testing frequency",
    "CVA": "CVA capital charge",
}

def classify(issue_desc: str) -> str:
    low = issue_desc.lower()
    for kw, label in KNOWN_VIOLATIONS.items():
        if kw.lower() in low:
            return f"✅ TRUE POSITIVE ({label})"
    return "⚠️  EXTRA FLAG"

def main():
    issues = get_all_flagged_issues()
    if not issues:
        print("No flagged issues in DB. Run evaluate_pipeline.py first.")
        return

    by_doc: dict = {}
    for i in issues:
        by_doc.setdefault(i["document_id"], []).append(i)

    print("=" * 70)
    print("  Flagged Issue Breakdown — True Positives vs Extra Flags")
    print("=" * 70)

    for doc_id, doc_issues in sorted(by_doc.items()):
        print(f"\n📄 {doc_id}  ({len(doc_issues)} flags)")
        print("-" * 60)
        for idx, issue in enumerate(doc_issues, 1):
            tag = classify(issue["issue_description"])
            print(f"  [{idx}] {tag}")
            print(f"       Clause  : {issue['clause_id']}")
            # Wrap long description
            desc = issue["issue_description"]
            print(f"       Issue   : {desc[:120]}")
            if len(desc) > 120:
                print(f"                 {desc[120:240]}")
            evidence = issue["evidence_chunk_ids"]
            print(f"       Evidence: {evidence[:3]}{'...' if len(evidence)>3 else ''}")
            print()

    total = len(issues)
    tps = sum(1 for i in issues if classify(i["issue_description"]).startswith("✅"))
    fps = total - tps
    print("=" * 70)
    print(f"  Total flags : {total}")
    print(f"  True positives  : {tps}")
    print(f"  Extra flags     : {fps}")
    print()

    # Categorize the extra flags
    extra = [i for i in issues if not classify(i["issue_description"]).startswith("✅")]
    if extra:
        print("  Extra flags detail:")
        for e in extra:
            desc = e["issue_description"][:150]
            print(f"    • [{e['document_id']}] {desc}")
    print("=" * 70)

if __name__ == "__main__":
    main()
