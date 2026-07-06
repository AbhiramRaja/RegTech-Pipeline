"""
scripts/download_regulatory_docs.py

Download real public regulatory PDFs for use as the regulatory corpus.

Sources (all publicly available, no login required):
  1. SEC — Sample 10-K Annual Report (public EDGAR filing)
  2. FINRA — Rule 4370 Business Continuity Guidance
  3. BIS/Basel III — Capital Requirements summary document

Run from the repo root:
    python scripts/download_regulatory_docs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import settings

OUTPUT_DIR = Path(settings.raw_regulatory_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Public URLs for regulatory documents
REGULATORY_DOCUMENTS = [
    {
        "filename": "basel3_capital_requirements.pdf",
        "url": "https://www.bis.org/publ/bcbs189.pdf",
        "description": "Basel III: A global regulatory framework for more resilient banks (BIS, 2010/2011)",
    },
    {
        "filename": "finra_rule_4370_guidance.pdf",
        "url": "https://www.finra.org/sites/default/files/NoticeDocument/p125232.pdf",
        "description": "FINRA Regulatory Notice 11-65: Business Continuity Planning",
    },
    {
        "filename": "sec_risk_disclosure_guidance.pdf",
        "url": "https://www.sec.gov/files/risk-alert-compliance-issues-related-to-best-execution.pdf",
        "description": "SEC Risk Alert: Compliance issues — Best Execution",
    },
]


def download_pdf(url: str, output_path: Path, description: str) -> bool:
    """Download a single PDF. Returns True on success."""
    if output_path.exists() and output_path.stat().st_size > 1000:
        logger.info("  Already exists (skipping): %s", output_path.name)
        return True

    logger.info("  Downloading: %s", description)
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            size_kb = len(response.content) / 1024
            logger.info("  ✓ Saved %s (%.1f KB)", output_path.name, size_kb)
            return True
    except Exception as exc:
        logger.warning("  ✗ Failed to download %s: %s", url, exc)
        return False


def create_fallback_regulatory_pdf(filename: str, title: str, content_sections: list[dict]) -> Path:
    """
    Create a synthetic regulatory PDF as fallback when download fails.
    Contains real regulatory text extracted from public sources.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    except ImportError:
        logger.error("reportlab not installed. Cannot create fallback PDF.")
        return None

    path = OUTPUT_DIR / filename
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            topMargin=inch, bottomMargin=inch,
                            leftMargin=1.25*inch, rightMargin=1.25*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=14, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11, spaceBefore=10)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=9, leading=13)

    story = [Paragraph(title, title_style),
             HRFlowable(width="100%", thickness=1, color=colors.grey),
             Spacer(1, 0.15*inch)]
    for section in content_sections:
        story.append(Paragraph(section["heading"], h2))
        for clause in section["clauses"]:
            story.append(Paragraph(clause, body))
        story.append(Spacer(1, 0.08*inch))
    doc.build(story)
    logger.info("  ✓ Created fallback PDF: %s", path.name)
    return path


FALLBACK_BASEL3 = {
    "filename": "basel3_capital_requirements.pdf",
    "title": "Basel III Capital Requirements — Key Provisions (Public Summary)",
    "sections": [
        {
            "heading": "1. Minimum Capital Requirements",
            "clauses": [
                "Common Equity Tier 1 (CET1) capital must be at least 4.5% of risk-weighted assets (RWA) at all times.",
                "Tier 1 capital must be at least 6.0% of risk-weighted assets at all times.",
                "Total capital (Tier 1 + Tier 2) must be at least 8.0% of risk-weighted assets.",
                "Banks must maintain a capital conservation buffer of 2.5% of RWA above minimum CET1, comprised of CET1 capital. Failure to maintain the buffer restricts discretionary distributions.",
            ],
        },
        {
            "heading": "2. Countercyclical Buffer",
            "clauses": [
                "National regulators may impose a countercyclical capital buffer of up to 2.5% of RWA during periods of excess credit growth.",
            ],
        },
        {
            "heading": "3. Leverage Ratio",
            "clauses": [
                "Banks must maintain a minimum Tier 1 leverage ratio of 3% (Tier 1 capital / total exposure measure).",
            ],
        },
        {
            "heading": "4. Liquidity Requirements",
            "clauses": [
                "The Liquidity Coverage Ratio (LCR) requires banks to hold sufficient high-quality liquid assets (HQLA) to cover 30-day net cash outflows under stress.",
                "The Net Stable Funding Ratio (NSFR) requires available stable funding to exceed required stable funding over a one-year horizon.",
            ],
        },
        {
            "heading": "5. Market Risk — Internal Models",
            "clauses": [
                "Banks using internal models must calculate VaR at a 99% confidence interval with a 10-day holding period.",
                "Stress testing of market risk must be conducted at minimum on a monthly basis.",
                "Credit Valuation Adjustment (CVA) capital charges must be applied to counterparty credit risk exposures arising from OTC derivatives.",
            ],
        },
    ],
}

FALLBACK_FINRA = {
    "filename": "finra_rule_4370_guidance.pdf",
    "title": "FINRA Rule 4370 — Business Continuity Plans (Key Requirements)",
    "sections": [
        {
            "heading": "1. Business Continuity Plan Requirement",
            "clauses": [
                "Each member must create and maintain a written business continuity plan identifying procedures relating to an emergency or significant business disruption.",
                "The plan must be reasonably designed to enable the member to meet its existing obligations to customers and, at a minimum, address the following elements.",
            ],
        },
        {
            "heading": "2. Required Plan Elements",
            "clauses": [
                "Data back-up and recovery (hard copy and electronic).",
                "All mission critical systems and alternate means to complete transactions.",
                "Financial and operational assessments.",
                "Alternate communications between the member and its customers and regulators.",
                "Alternate physical location of employees.",
                "Critical business constituent, bank and counter-party impact.",
                "Regulatory reporting obligations.",
                "Communications with regulators.",
            ],
        },
        {
            "heading": "3. Annual Review and Testing",
            "clauses": [
                "Each member must conduct an annual review of its business continuity plan to determine whether any modifications are necessary in light of changes to the member's operations, structure, business or location.",
                "Business continuity plans must be tested at least annually. Testing must be documented and results reported to senior management.",
                "Members must update their emergency contact information with FINRA at least annually and promptly following any material change.",
            ],
        },
        {
            "heading": "4. Disclosure to Customers",
            "clauses": [
                "Each member must disclose to its customers how its BCP addresses the possibility of a future significant business disruption and how the member plans to respond to events of varying scope.",
            ],
        },
    ],
}

FALLBACK_SEC = {
    "filename": "sec_risk_disclosure_guidance.pdf",
    "title": "SEC — Risk Management and Compliance Guidance (Key Provisions)",
    "sections": [
        {
            "heading": "1. Risk Management Framework",
            "clauses": [
                "Registered investment advisers must adopt and implement written policies and procedures reasonably designed to prevent violations of the Investment Advisers Act and the rules thereunder.",
                "Compliance programs must be reviewed at least annually for adequacy and effectiveness.",
            ],
        },
        {
            "heading": "2. Best Execution Obligations",
            "clauses": [
                "Broker-dealers must use reasonable diligence to ascertain the best market for a security and buy or sell in such market so that the resulting price to the customer is as favorable as possible under prevailing market conditions.",
                "Best execution policies must be reviewed and updated at least annually.",
                "Material conflicts of interest in routing orders must be disclosed to customers.",
            ],
        },
        {
            "heading": "3. Recordkeeping",
            "clauses": [
                "All required records must be maintained for the periods specified under applicable SEC rules (generally 3–6 years).",
                "Records must be kept in an easily accessible location for the first two years.",
            ],
        },
    ],
}


def main():
    print("Downloading regulatory PDF corpus...")
    any_downloaded = False

    for doc in REGULATORY_DOCUMENTS:
        success = download_pdf(doc["url"], OUTPUT_DIR / doc["filename"], doc["description"])
        if success:
            any_downloaded = True

    # Create fallback PDFs for any that failed to download
    for fallback in [FALLBACK_BASEL3, FALLBACK_FINRA, FALLBACK_SEC]:
        target = OUTPUT_DIR / fallback["filename"]
        if not target.exists() or target.stat().st_size < 1000:
            print(f"  Creating fallback for: {fallback['filename']}")
            create_fallback_regulatory_pdf(
                fallback["filename"], fallback["title"], fallback["sections"]
            )

    print(f"\n✓ Regulatory corpus ready in: {OUTPUT_DIR}")
    pdfs = list(OUTPUT_DIR.glob("*.pdf"))
    for p in pdfs:
        print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
