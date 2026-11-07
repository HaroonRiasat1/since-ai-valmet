"""Render every diagram in docs/diagrams/*.html to docs/media/diagram_<name>.png (2x, for crisp README images).

    uv run --no-project --with playwright python docs/diagrams/render.py
Uses Microsoft Edge through Playwright (channel="msedge"); change CHANNEL for Chrome.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "media"
CHANNEL = "msedge"

with sync_playwright() as p:
    browser = p.chromium.launch(channel=CHANNEL)
    page = browser.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=2)
    for html in sorted(HERE.glob("*.html")):
        page.goto(html.as_uri())
        page.wait_for_timeout(300)
        out = MEDIA / f"diagram_{html.stem}.png"
        page.locator("#d").screenshot(path=str(out), omit_background=True)
        print("wrote", out.relative_to(HERE.parent.parent))
    browser.close()
