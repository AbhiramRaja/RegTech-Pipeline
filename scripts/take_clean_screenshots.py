"""
scripts/take_clean_screenshots.py

Captures pixel-perfect, borderless, clean screenshots of the Streamlit dashboard
using Playwright Headless Chromium.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT_DIR = Path("docs/screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # 1920x1080 high-res viewport, dark color scheme
        page = browser.new_page(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2,
            color_scheme="dark"
        )
        
        # 1. Overview Page
        print("Capturing Overview...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(2)
        page.screenshot(path=str(OUT_DIR / "overview.png"))
        
        # 2. Flagged Issues Page
        print("Capturing Flagged Issues...")
        page.get_by_text("🚩 Flagged Issues").click()
        time.sleep(2)
        page.screenshot(path=str(OUT_DIR / "flagged_issues.png"))
        
        # 3. Audit Trail Page
        print("Capturing Audit Trail...")
        page.get_by_text("🔍 Audit Trail").click()
        time.sleep(1)
        # Fill in policy_capital_adequacy and load
        page.get_by_placeholder("e.g. policy_capital_adequacy").fill("policy_capital_adequacy")
        page.get_by_role("button", name="🔍 Load Audit Chain").click()
        time.sleep(2)
        page.screenshot(path=str(OUT_DIR / "audit_trail.png"))
        
        browser.close()
        print("✓ All screenshots captured cleanly without any browser borders or glows!")

if __name__ == "__main__":
    main()
