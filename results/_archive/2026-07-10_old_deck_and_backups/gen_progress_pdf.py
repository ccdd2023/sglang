#!/usr/bin/env python3
"""Regenerate CODE_AWARE_LOSSY_KV_PROGRESS.pdf from the HTML via Playwright.

Uses Chromium headless + the deck's @media print CSS at @page { size: 1280px 1600px }.
Same pipeline as the original (no gen_*.py existed in repo at this commit).
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

HTML = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/CODE_AWARE_LOSSY_KV_PROGRESS.html").resolve()
PDF = Path("/home/gfy/CodeMAS_Project/sglang-kvflow/results/CODE_AWARE_LOSSY_KV_PROGRESS.pdf")

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    context = browser.new_context(viewport={"width": 1280, "height": 1600})
    page = context.new_page()
    page.goto(f"file://{HTML}", wait_until="networkidle")
    # Use the deck's @media print CSS
    page.emulate_media(media="print")
    page.pdf(
        path=str(PDF),
        width="1280px",
        height="1600px",
        print_background=True,
        margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
    )
    browser.close()

print(f"Wrote {PDF} ({PDF.stat().st_size // 1024} KB)")