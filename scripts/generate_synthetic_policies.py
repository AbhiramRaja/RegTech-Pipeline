"""
scripts/generate_synthetic_policies.py

Generate synthetic internal policy PDFs with deliberate non-compliance clauses.
These are used for development and testing — the violations provide signal for
the compliance pipeline's flagging logic.

Run from the repo root:
    python scripts/generate_synthetic_policies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
except ImportError:
    print("reportlab not installed. Run: pip install reportlab")
    sys.exit(1)

from config import settings

OUTPUT_DIR = Path(settings.raw_internal_policy_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_pdf(filename: str, title: str, sections: list[dict]) -> Path:
    """
    Build a simple policy PDF.
    sections: list of {heading: str, clauses: list[str]}
    """
    path = OUTPUT_DIR / filename
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
        leftMargin=1.25 * inch,
        rightMargin=1.25 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"], fontSize=16, spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "Heading2", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8
    )

    story = [
        Paragraph(title, title_style),
        HRFlowable(width="100%", thickness=1, color=colors.grey),
        Spacer(1, 0.2 * inch),
    ]

    for section in sections:
        story.append(Paragraph(section["heading"], heading_style))
        for i, clause in enumerate(section["clauses"], 1):
            story.append(Paragraph(f"{i}. {clause}", body_style))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    print(f"  ✓ Generated: {path}")
    return path


def generate_capital_adequacy_policy():
    """Mostly compliant — 2 deliberate violations."""
    build_pdf(
        "policy_capital_adequacy.pdf",
        "INTERNAL POLICY: Capital Adequacy & Risk-Weighted Assets",
        sections=[
            {
                "heading": "1. Purpose and Scope",
                "clauses": [
                    "This policy establishes the minimum capital requirements for the institution "
                    "in accordance with applicable regulatory frameworks.",
                    "This policy applies to all business lines and subsidiaries that are subject "
                    "to consolidated capital reporting obligations.",
                ],
            },
            {
                "heading": "2. Minimum Capital Ratios",
                "clauses": [
                    # VIOLATION 1: Basel III requires CET1 >= 4.5%, policy says 3.5%
                    "The institution shall maintain a Common Equity Tier 1 (CET1) capital ratio "
                    "of no less than 3.5% of risk-weighted assets at all times. "
                    "[NOTE: Internal target — subject to board approval.]",
                    # VIOLATION 2: Basel III requires Tier 1 >= 6%, policy says 5%
                    "The Tier 1 capital ratio shall be maintained at a minimum of 5.0% of "
                    "total risk-weighted assets, as calculated on a quarterly basis.",
                    # Compliant
                    "The Total Capital ratio (Tier 1 + Tier 2) shall be maintained at a minimum "
                    "of 8.0% of risk-weighted assets, consistent with Basel III requirements.",
                ],
            },
            {
                "heading": "3. Capital Conservation Buffer",
                "clauses": [
                    "The institution shall maintain a capital conservation buffer of 2.5% "
                    "above the minimum CET1 ratio, as required under Basel III framework.",
                    "Breach of the conservation buffer triggers automatic restrictions on "
                    "discretionary distributions including dividends and bonus payments.",
                ],
            },
            {
                "heading": "4. Reporting and Monitoring",
                "clauses": [
                    "Capital adequacy ratios shall be calculated and reported to the Risk Committee "
                    "on a monthly basis and to regulators on a quarterly basis.",
                    "Any projected breach of minimum ratios must be escalated to the Chief Risk "
                    "Officer within 24 hours of identification.",
                ],
            },
        ],
    )


def generate_business_continuity_policy():
    """Mostly compliant — 1 deliberate violation."""
    build_pdf(
        "policy_business_continuity.pdf",
        "INTERNAL POLICY: Business Continuity Planning (BCP)",
        sections=[
            {
                "heading": "1. Purpose",
                "clauses": [
                    "This policy ensures the institution maintains operational resilience "
                    "and continuity of critical business functions during disruptive events.",
                ],
            },
            {
                "heading": "2. BCP Testing Requirements",
                "clauses": [
                    # VIOLATION: FINRA Rule 4370 requires annual BCP testing; policy says every 2 years
                    "Business continuity plans shall be tested and reviewed on a biennial basis "
                    "(every two years) or following any significant organizational change.",
                    # Compliant
                    "All BCP tests must be documented with results, gaps identified, and "
                    "remediation timelines approved by senior management.",
                ],
            },
            {
                "heading": "3. Recovery Time Objectives",
                "clauses": [
                    "Critical trading systems must achieve a Recovery Time Objective (RTO) "
                    "of no more than 4 hours following a disruptive event.",
                    "Non-critical systems must achieve an RTO of no more than 24 hours.",
                    "Recovery Point Objectives (RPO) for all critical data shall not exceed 1 hour.",
                ],
            },
            {
                "heading": "4. Emergency Contact and Notification",
                "clauses": [
                    "In the event of a significant business disruption, the institution shall "
                    "notify FINRA within the time frame required under Rule 4370.",
                    "An updated emergency contact list must be maintained and reviewed quarterly.",
                ],
            },
        ],
    )


def generate_risk_management_policy():
    """Mostly compliant — 3 deliberate violations."""
    build_pdf(
        "policy_risk_management.pdf",
        "INTERNAL POLICY: Market Risk Management Framework",
        sections=[
            {
                "heading": "1. Scope",
                "clauses": [
                    "This policy governs the identification, measurement, monitoring, and "
                    "control of market risk across all trading and banking book positions.",
                ],
            },
            {
                "heading": "2. Value-at-Risk Limits",
                "clauses": [
                    # VIOLATION 1: Using 95% confidence VaR; Basel III requires 99% for internal models
                    "The institution shall calculate daily Value-at-Risk (VaR) using a "
                    "95% confidence interval and a 10-day holding period.",
                    # VIOLATION 2: Stress testing quarterly; Basel requires at least monthly
                    "Stress testing of the trading portfolio shall be conducted on a quarterly basis "
                    "to assess potential losses under adverse market conditions.",
                    # Compliant
                    "VaR models shall be backtested daily against actual P&L outcomes to "
                    "validate model accuracy.",
                ],
            },
            {
                "heading": "3. Counterparty Credit Risk",
                "clauses": [
                    # VIOLATION 3: Not recognizing CVA capital charge requirement
                    "Counterparty credit risk exposure shall be measured using Current Exposure "
                    "Method (CEM) without application of a Credit Valuation Adjustment (CVA) "
                    "capital charge.",
                    # Compliant
                    "Netting agreements shall be documented and legally reviewed annually to "
                    "confirm enforceability.",
                ],
            },
            {
                "heading": "4. Risk Limit Breach Escalation",
                "clauses": [
                    "Any breach of established risk limits must be reported to the Market Risk "
                    "Committee within one business day.",
                    "Repeated breaches require a formal remediation plan approved by the CRO "
                    "and Board Risk Committee.",
                ],
            },
        ],
    )


def generate_all():
    print("Generating synthetic internal policy PDFs...")
    generate_capital_adequacy_policy()
    generate_business_continuity_policy()
    generate_risk_management_policy()
    print(f"\n✓ All policy PDFs written to: {OUTPUT_DIR}")
    print("\nViolations embedded:")
    print("  capital_adequacy:     CET1 ratio 3.5% (should be ≥4.5%), Tier1 5.0% (should be ≥6.0%)")
    print("  business_continuity:  BCP testing biennial (should be annual per FINRA 4370)")
    print("  risk_management:      VaR 95% CI (should be 99%), stress testing quarterly (should be monthly), no CVA capital charge")


if __name__ == "__main__":
    generate_all()
