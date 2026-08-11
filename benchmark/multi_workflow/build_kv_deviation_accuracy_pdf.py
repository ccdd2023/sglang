#!/usr/bin/env python3
"""Build the advisor-facing KV-deviation audit as A4 HTML and PDF."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

from markdown_it import MarkdownIt
from pypdf import PdfReader
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "docs/kvflow/KV_DEVIATION_ACCURACY_DECOUPLING_AUDIT_20260806.md"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slugify(value: str, index: int) -> str:
    stem = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value).strip("-").lower()
    return f"{stem or 'section'}-{index}"


def render_markdown(source: str) -> tuple[str, list[tuple[str, str]]]:
    parser = MarkdownIt(
        "commonmark",
        {"html": True, "typographer": True, "linkify": False},
    ).enable("table")
    tokens = parser.parse(source)
    toc: list[tuple[str, str]] = []
    section_index = 0
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.tag != "h2":
            continue
        title = tokens[index + 1].content
        slug = slugify(title, section_index)
        section_index += 1
        token.attrSet("id", slug)
        toc.append((title, slug))
    body = parser.renderer.render(tokens, parser.options, {})
    body = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>',
        lambda match: (
            f'<figure><img src="{match.group(1)}" alt="{match.group(2)}">'
            f"<figcaption>{html.escape(match.group(2))}</figcaption></figure>"
        ),
        body,
    )
    return body, toc


def build_html(source_path: Path) -> str:
    body, toc = render_markdown(source_path.read_text(encoding="utf-8"))
    toc_html = "".join(
        f'<li><a href="#{slug}">{html.escape(title)}</a></li>'
        for title, slug in toc
    )
    css = r"""
    :root {
      --ink: #182536;
      --muted: #5b6878;
      --navy: #153456;
      --blue: #2b6595;
      --cyan: #168aa1;
      --green: #26775d;
      --amber: #a76716;
      --red: #a44543;
      --line: #ccd6df;
      --wash: #f2f6f9;
    }
    * { box-sizing: border-box; }
    html { background: white; }
    body {
      margin: 0;
      color: var(--ink);
      background: white;
      font-family: "FandolSong", "Noto Serif CJK SC", "Source Han Serif SC", serif;
      font-size: 10.3pt;
      line-height: 1.56;
    }
    .cover {
      min-height: 250mm;
      page-break-after: always;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 24mm 17mm 15mm;
      color: white;
      background: linear-gradient(145deg, #102740 0%, #174a70 63%, #147b83 100%);
    }
    .cover .eyebrow {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 9pt;
      letter-spacing: .13em;
      color: #a9dce5;
    }
    .cover h1 {
      margin: 15mm 0 7mm;
      max-width: 160mm;
      color: white;
      font-family: "FandolHei", "Noto Sans CJK SC", sans-serif;
      font-size: 31pt;
      font-weight: 600;
      line-height: 1.18;
      border: 0;
    }
    .cover .subtitle {
      max-width: 155mm;
      margin: 0;
      color: #d9ebf1;
      font-family: "FandolHei", "Noto Sans CJK SC", sans-serif;
      font-size: 15pt;
      line-height: 1.5;
    }
    .cover-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 5mm;
      margin-top: 18mm;
    }
    .cover-grid div {
      padding-top: 3mm;
      border-top: 1.2pt solid #7dc6d2;
    }
    .cover-grid strong {
      display: block;
      color: white;
      font-family: "DejaVu Sans", "FandolHei", sans-serif;
      font-size: 15pt;
    }
    .cover-grid span { color: #d2e6ed; font-size: 8.5pt; }
    .cover-meta {
      display: flex;
      justify-content: space-between;
      padding-top: 4mm;
      border-top: .6pt solid rgba(255,255,255,.45);
      color: #d5e5eb;
      font-size: 8.5pt;
    }
    main { width: 100%; }
    article > h1:first-child { display: none; }
    .toc {
      margin: 0 0 10mm;
      padding: 7mm 8mm;
      border-left: 3pt solid var(--cyan);
      background: var(--wash);
      break-inside: avoid;
    }
    .toc h2 {
      margin: 0 0 3mm;
      padding: 0;
      border: 0;
      font-size: 15pt;
    }
    .toc ol {
      columns: 2;
      column-gap: 10mm;
      margin: 0;
      padding-left: 5mm;
    }
    .toc li { margin: 1.2mm 0; break-inside: avoid; }
    .toc a { color: var(--ink); text-decoration: none; }
    h1, h2, h3 {
      font-family: "FandolHei", "Noto Sans CJK SC", sans-serif;
      color: var(--navy);
      font-weight: 600;
    }
    h2 {
      margin: 10mm 0 4mm;
      padding-top: 2.5mm;
      border-top: 1.4pt solid var(--navy);
      font-size: 18pt;
      line-height: 1.3;
      break-after: avoid;
    }
    h3 {
      margin: 7mm 0 3mm;
      color: var(--blue);
      font-size: 13.2pt;
      line-height: 1.35;
      break-after: avoid;
    }
    p { margin: 2.2mm 0 3.4mm; orphans: 3; widows: 3; }
    strong { color: #164a72; }
    a { color: var(--blue); text-decoration: none; }
    ul, ol { margin: 2mm 0 4mm; padding-left: 6mm; }
    li { margin: 1mm 0; orphans: 2; widows: 2; }
    blockquote {
      margin: 5mm 0;
      padding: 4mm 5mm;
      border-left: 3pt solid var(--cyan);
      background: #edf7f8;
      color: #1e4356;
      font-size: 11pt;
      break-inside: avoid;
    }
    blockquote p { margin: 0; }
    code {
      padding: .25mm .9mm;
      border-radius: 1mm;
      color: #7d3150;
      background: #edf2f5;
      font-family: "DejaVu Sans Mono", monospace;
      font-size: 8.4pt;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 4mm 0;
      padding: 4mm;
      color: #e1edf3;
      background: #132b43;
      font-size: 8.2pt;
      line-height: 1.42;
      white-space: pre-wrap;
      break-inside: avoid;
    }
    pre code { padding: 0; color: inherit; background: transparent; }
    table {
      width: 100%;
      margin: 4mm 0 6mm;
      border-collapse: collapse;
      table-layout: auto;
      font-family: "FandolSong", serif;
      font-size: 7.7pt;
      line-height: 1.38;
      break-inside: avoid;
    }
    thead { display: table-header-group; }
    tr { break-inside: avoid; }
    th {
      padding: 2.2mm 2mm;
      color: white;
      background: var(--navy);
      text-align: left;
      vertical-align: top;
    }
    td {
      padding: 2mm;
      border-bottom: .45pt solid var(--line);
      vertical-align: top;
    }
    tbody tr:nth-child(even) td { background: #f5f8fa; }
    figure {
      margin: 5mm 0 7mm;
      text-align: center;
      break-inside: avoid;
    }
    figure img {
      display: block;
      max-width: 100%;
      max-height: 171mm;
      margin: 0 auto;
      object-fit: contain;
    }
    figcaption {
      margin-top: 2mm;
      color: var(--muted);
      font-family: "FandolHei", sans-serif;
      font-size: 8pt;
      line-height: 1.35;
    }
    hr { border: 0; border-top: .5pt solid var(--line); margin: 7mm 0; }
    @page { size: A4; margin: 14mm 14mm 17mm; }
    @media print {
      body { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
      .cover { min-height: 250mm; }
      h2, h3 { break-after: avoid-page; }
      p, li { orphans: 3; widows: 3; }
    }
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KV deviation 与 Coding Lossy Reuse 任务精度审计</title>
  <style>{css}</style>
</head>
<body>
  <section class="cover">
    <div>
      <div class="eyebrow">IMPACTKV · CODING-AWARE LOSSY KV REUSE · EVIDENCE AUDIT</div>
      <h1>KV deviation 能否代表 Coding Lossy Reuse 的任务精度？</h1>
      <p class="subtitle">从表示接近、固定预算 selector 到官方 execution accuracy 的证据审计</p>
      <div class="cover-grid">
        <div><strong>23.00% → 9.28%</strong><span>stale fraction 显著下降</span></div>
        <div><strong>11/50 → 10/50</strong><span>DS-1000 官方通过未提高</span></div>
        <div><strong>AUROC 0.276</strong><span>residual V-mass 未校准为失败风险</span></div>
      </div>
    </div>
    <div class="cover-meta">
      <span>阶段研究备忘录</span><span>Frozen evidence · 2026-08-06</span>
    </div>
  </section>
  <main>
    <nav class="toc"><h2>目录</h2><ol>{toc_html}</ol></nav>
    <article>{body}</article>
  </main>
</body>
</html>
"""


def render_pdf(html_path: Path, pdf_path: Path) -> dict[str, object]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        image_failures = page.eval_on_selector_all(
            "img",
            "elements => elements.filter(img => !img.complete || img.naturalWidth === 0).map(img => img.src)",
        )
        page.emulate_media(media="print")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            display_header_footer=True,
            header_template=(
                '<div style="width:100%;font-size:7px;color:#6c7885;padding:0 14mm;'
                'font-family:Arial,sans-serif">ImpactKV · KV deviation accuracy audit</div>'
            ),
            footer_template=(
                '<div style="width:100%;font-size:7px;color:#6c7885;padding:0 14mm;'
                'font-family:Arial,sans-serif;text-align:right">'
                '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
            ),
        )
        browser.close()

    reader = PdfReader(str(pdf_path))
    blank_pages: list[int] = []
    for page_number, pdf_page in enumerate(reader.pages, start=1):
        if len((pdf_page.extract_text() or "").strip()) < 8:
            blank_pages.append(page_number)
    return {
        "pages": len(reader.pages),
        "blank_pages": blank_pages,
        "image_failures": image_failures,
        "passed": not blank_pages and not image_failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--pdf", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    html_path = (args.html or source.with_name(f"{source.stem}_A4.html")).resolve()
    pdf_path = (args.pdf or source.with_name(f"{source.stem}_A4.pdf")).resolve()
    qa_path = pdf_path.with_name(f"{pdf_path.stem}_QA.json")

    html_path.write_text(build_html(source), encoding="utf-8")
    html_path.chmod(0o644)
    qa = render_pdf(html_path, pdf_path)
    pdf_path.chmod(0o644)
    result = {
        **qa,
        "source": str(source),
        "html": str(html_path),
        "pdf": str(pdf_path),
        "source_sha256": sha256(source),
        "html_sha256": sha256(html_path),
        "pdf_sha256": sha256(pdf_path),
        "pdf_bytes": pdf_path.stat().st_size,
    }
    qa_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    qa_path.chmod(0o644)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not qa["passed"]:
        raise SystemExit("PDF validation failed")


if __name__ == "__main__":
    main()
