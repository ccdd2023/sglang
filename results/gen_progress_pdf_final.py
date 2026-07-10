#!/usr/bin/env python3
"""Generate PDF for the final triple-falsification deck."""
from pathlib import Path
from playwright.sync_api import sync_playwright

HTML = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/CODE_AWARE_LOSSY_KV_PROGRESS_FINAL.html").resolve()
PDF = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/CODE_AWARE_LOSSY_KV_PROGRESS_FINAL.pdf")

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    context = browser.new_context(viewport={"width": 1280, "height": 720})
    page = context.new_page()
    page.goto(f"file://{HTML}", wait_until="networkidle")
    page.emulate_media(media="print")
    page.pdf(
        path=str(PDF),
        width="1280px",
        height="720px",
        print_background=True,
        margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
    )
    browser.close()

print(f"Wrote {PDF} ({PDF.stat().st_size // 1024} KB)")
