#!/usr/bin/env python3
"""Generate an HTML report visualizing KVCOMM anchor-level matching for the 6-case lossy suite."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MASCODER_SRC = PROJECT_ROOT / "MAScoder" / "src"
SGLANG_PYTHON = PROJECT_ROOT / "sglang-kvflow" / "python"
for entry in (str(MASCODER_SRC), str(SGLANG_PYTHON)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from mascoder.code_anchor import (
    CodeAnchor,
    build_code_anchors,
    compute_anchor_signature,
    compute_syntax_fingerprint,
    serialize_code_anchor_spans,
)
from sglang.srt.mem_cache.anchor_match import (
    AnchorMatchResult,
    AnchorMetadata,
    build_anchor_metadata,
    match_request_to_candidate,
    _span_similarity,
)

BASE_CODE = """from typing import List

def count_up_to(n: int) -> List[int]:
    result = []
    for value in range(2, n):
        is_prime = True
        for factor in range(2, int(value ** 0.5) + 1):
            if value % factor == 0:
                is_prime = False
                break
        if is_prime:
            result.append(value)
    return result
"""

CASES = [
    {"case_id": "exact_same", "desc": "完全相同的代码", "code": BASE_CODE},
    {
        "case_id": "rename_variables",
        "desc": "变量重命名: result→primes, value→candidate",
        "code": BASE_CODE.replace("result", "primes").replace("value", "candidate"),
    },
    {
        "case_id": "comment_only",
        "desc": "仅在顶部增加注释",
        "code": "# Count primes below n\n" + BASE_CODE,
    },
    {
        "case_id": "add_helper",
        "desc": "增加一个 helper 函数",
        "code": "def _identity(x):\n    return x\n\n" + BASE_CODE,
    },
    {
        "case_id": "structure_rewrite",
        "desc": "将内部逻辑重构为嵌套函数 + 列表推导式",
        "code": """from typing import List

def count_up_to(n: int) -> List[int]:
    def is_prime(value: int) -> bool:
        if value < 2:
            return False
        for factor in range(2, int(value ** 0.5) + 1):
            if value % factor == 0:
                return False
        return True

    return [value for value in range(2, n) if is_prime(value)]
""",
    },
    {
        "case_id": "different_function",
        "desc": "完全不同的函数",
        "code": """def reverse_words(text: str) -> str:
    return " ".join(reversed(text.split()))
""",
    },
]

WARMUP_ANCHORS = build_code_anchors(BASE_CODE, language="python")
WARMUP_SPANS = serialize_code_anchor_spans(WARMUP_ANCHORS)
WARMUP_SIGNATURE = compute_anchor_signature(WARMUP_ANCHORS)
WARMUP_FINGERPRINT = compute_syntax_fingerprint(WARMUP_ANCHORS)

ANCHOR_COLORS = {
    "function": "#e6a817",
    "for": "#4a90d9",
    "if": "#6ebd6e",
    "class": "#d94a90",
    "while": "#d97a4a",
    "try": "#9b59b6",
}
ANCHOR_DEFAULT_COLOR = "#888888"
CONFIDENCE_COLORS = {
    "exact_code_content_signature": "#2e7d32",
    "exact_anchor_signature": "#2e7d32",
    "span_overlap_high": "#1565c0",
    "span_overlap_medium": "#e65100",
    "span_overlap_low": "#b71c1c",
    "no_anchor_overlap": "#616161",
}


def highlight_code(text: str, anchors: list[CodeAnchor], label: str) -> str:
    """Build an HTML snippet showing the code with highlighted anchor regions."""
    lines = text.splitlines()
    anchor_by_line: dict[int, str] = {}
    for a in anchors:
        color = ANCHOR_COLORS.get(a.anchor_type, ANCHOR_DEFAULT_COLOR)
        for ln in range(a.start_line, a.end_line + 1):
            anchor_by_line[ln] = color

    buf = [f'<div class="code-block"><div class="code-label">{label}</div><pre>']
    for i, line in enumerate(lines, start=1):
        color = anchor_by_line.get(i, "")
        if color:
            buf.append(
                f'<span style="display:block;border-left:4px solid {color};padding-left:8px;background:{color}10">'
            )
        else:
            buf.append('<span style="display:block;padding-left:12px">')
        buf.append(f"{i:>3} {_escape_html(line)}</span>")
    buf.append("</pre></div>")
    return "\n".join(buf)


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_anchor_table(
    warmup_anchors: list[CodeAnchor],
    candidate_anchors: list[CodeAnchor],
) -> str:
    """Build an HTML table showing per-anchor matching details."""
    rows = []
    for wa in warmup_anchors:
        best_overlap = 0.0
        best_ca: CodeAnchor | None = None
        best_sig_match = False
        for ca in candidate_anchors:
            if ca.anchor_type != wa.anchor_type:
                continue
            overlap = _span_ratio(wa, ca)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ca = ca
                best_sig_match = wa.signature == ca.signature

        wa_color = ANCHOR_COLORS.get(wa.anchor_type, ANCHOR_DEFAULT_COLOR)
        if best_ca is None:
            status = "❌ 无同类型锚点"
            style = "color:#b71c1c"
        elif best_sig_match:
            status = "✓ 签名匹配"
            style = f"color:{CONFIDENCE_COLORS['exact_anchor_signature']}"
        elif best_overlap >= 0.8:
            status = f"≈ 高重叠 ({best_overlap:.0%})"
            style = f"color:{CONFIDENCE_COLORS['span_overlap_high']}"
        elif best_overlap >= 0.5:
            status = f"≈ 中重叠 ({best_overlap:.0%})"
            style = f"color:{CONFIDENCE_COLORS['span_overlap_medium']}"
        elif best_overlap >= 0.3:
            status = f"≈ 低重叠 ({best_overlap:.0%})"
            style = f"color:{CONFIDENCE_COLORS['span_overlap_low']}"
        else:
            status = "✗ 无重叠"
            style = "color:#616161"

        rows.append(
            f'<tr>'
            f'<td style="border-left:4px solid {wa_color};padding-left:8px">{wa.anchor_type}</td>'
            f'<td>{wa.name}</td>'
            f'<td><code>{wa.signature[:8]}</code></td>'
            f'<td>L{wa.start_line}-L{wa.end_line}</td>'
            f'<td>{best_ca.name if best_ca else "—"}</td>'
            f'<td><code>{best_ca.signature[:8] if best_ca else "—"}</code></td>'
            f'<td>{f"L{best_ca.start_line}-L{best_ca.end_line}" if best_ca else "—"}</td>'
            f'<td style="{style}">{status}</td>'
            f"</tr>"
        )

    # also show extra candidate anchors not matched to any warmup
    matched_ca_names = set()
    for wa in warmup_anchors:
        for ca in candidate_anchors:
            if ca.anchor_type == wa.anchor_type and _span_ratio(wa, ca) > 0:
                matched_ca_names.add(ca.name)
    for ca in candidate_anchors:
        if ca.name not in matched_ca_names:
            rows.append(
                f'<tr style="opacity:0.5">'
                f"<td>—</td>"
                f"<td>—</td>"
                f"<td>—</td>"
                f"<td>—</td>"
                f'<td>{ca.name}</td>'
                f'<td><code>{ca.signature[:8]}</code></td>'
                f"<td>L{ca.start_line}-L{ca.end_line}</td>"
                f'<td style="color:#888">新增锚点</td>'
                f"</tr>"
            )

    header = (
        "<thead><tr>"
        "<th>类型</th><th>warmup 名称</th><th>warmup 签名</th><th>warmup 行</th>"
        "<th>候选名称</th><th>候选签名</th><th>候选行</th><th>匹配</th>"
        "</tr></thead>"
    )
    return f'<table class="anchor-table">{header}<tbody>{"".join(rows)}</tbody></table>'


def _span_ratio(a: CodeAnchor, b: CodeAnchor) -> float:
    """line-level overlap ratio matching the server-side logic."""
    if not a.start_line or not a.end_line or not b.start_line or not b.end_line:
        return 0.0
    inter = max(0, min(a.end_line, b.end_line) - max(a.start_line, b.start_line) + 1)
    if inter <= 0:
        return 0.0
    left_len = max(1, a.end_line - a.start_line + 1)
    right_len = max(1, b.end_line - b.start_line + 1)
    return inter / max(left_len, right_len)


def build_case_html(case: dict) -> str:
    cid = case["case_id"]
    desc = case["desc"]
    code = case["code"]

    cand_anchors = build_code_anchors(code, language="python")
    cand_spans = serialize_code_anchor_spans(cand_anchors)
    cand_signature = compute_anchor_signature(cand_anchors)
    cand_fingerprint = compute_syntax_fingerprint(cand_anchors)

    request_meta = build_anchor_metadata(
        code_anchor_signature=cand_signature,
        code_anchor_spans=cand_spans,
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
        template_task_family="code_generation",
        template_workflow_signature="agents=planner,implementer,reviewer",
        template_structural_fingerprint="loop_for",
    )
    candidate_meta = build_anchor_metadata(
        code_anchor_signature=WARMUP_SIGNATURE,
        code_anchor_spans=WARMUP_SPANS,
        reuse_mode="lossy",
        lossy_alignment_method="kvcomm",
        template_task_family="code_generation",
        template_workflow_signature="agents=planner,implementer,reviewer",
        template_structural_fingerprint="loop_for",
    )
    result = match_request_to_candidate(request_meta, candidate_meta)

    verdict_color = "#2e7d32" if result.reuse_allowed else "#b71c1c"
    reason = result.match_reason or result.rejected_reason or "unknown"
    confidence_pct = f"{result.reuse_confidence:.0%}"

    return f"""
    <div class="case">
        <div class="case-header" style="border-left: 6px solid {verdict_color}">
            <span class="case-id">{cid}</span>
            <span class="case-desc">{desc}</span>
            <span class="case-verdict" style="color:{verdict_color}">
                {'✅ 复用允许' if result.reuse_allowed else '❌ 复用拒绝'}
                &nbsp;|&nbsp; 原因: {reason} &nbsp;|&nbsp; 置信度: {confidence_pct}
            </span>
        </div>
        <div class="case-meta">
            <span>warmup 签名: <code>{WARMUP_SIGNATURE}</code></span>
            <span>候选签名: <code>{cand_signature}</code></span>
            <span>结构指纹: {WARMUP_FINGERPRINT} → {cand_fingerprint}</span>
        </div>
        <div class="side-by-side">
            {highlight_code(BASE_CODE, WARMUP_ANCHORS, "warmup (BASE_CODE)")}
            {highlight_code(code, cand_anchors, f"candidate ({cid})")}
        </div>
        <div class="legend">
            {_color_legend_html()}
        </div>
        <h4>锚点级别匹配明细</h4>
        {build_anchor_table(WARMUP_ANCHORS, cand_anchors)}
    </div>
    """


def _color_legend_html() -> str:
    items = []
    for atype, color in sorted(ANCHOR_COLORS.items()):
        items.append(
            f'<span style="display:inline-block;width:14px;height:14px;background:{color};border-radius:3px;margin-right:4px"></span>'
            f"{atype}&nbsp;&nbsp;"
        )
    return (
        '<div style="font-size:13px;color:#666;margin-top:4px">锚点颜色: '
        + "".join(items)
        + "</div>"
    )


def build_html() -> str:
    case_html = "\n".join(build_case_html(c) for c in CASES)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>KVCOMM Anchor-Level Matching Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
    background: #fafafa;
    color: #222;
}}
h1 {{ font-size: 24px; margin-bottom: 4px; }}
.summary {{ color: #666; margin-bottom: 32px; }}
.case {{
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    margin-bottom: 32px;
    padding: 20px;
}}
.case-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 14px;
    margin: -20px -20px 12px -20px;
    background: #f5f5f5;
    border-radius: 8px 8px 0 0;
}}
.case-id {{ font-weight: 700; font-size: 18px; font-family: monospace; }}
.case-desc {{ color: #555; }}
.case-verdict {{ margin-left: auto; font-weight: 600; font-size: 14px; }}
.case-meta {{
    font-size: 12px;
    color: #888;
    display: flex;
    gap: 20px;
    margin-bottom: 12px;
}}
.side-by-side {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 8px;
}}
.code-block pre {{
    margin: 0;
    padding: 0;
    font-size: 13px;
    line-height: 1.5;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    background: #f8f8f8;
    border-radius: 4px;
    padding: 8px 0;
    overflow-x: auto;
}}
.code-label {{
    font-weight: 600;
    font-size: 12px;
    color: #888;
    margin-bottom: 2px;
    padding-left: 4px;
}}
.anchor-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 8px;
}}
.anchor-table th {{
    background: #f0f0f0;
    padding: 6px 10px;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #ddd;
}}
.anchor-table td {{
    padding: 6px 10px;
    border-bottom: 1px solid #eee;
}}
.anchor-table code {{
    background: #f0f0f0;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 12px;
}}
h4 {{ margin: 16px 0 4px; font-size: 14px; color: #555; }}
</style>
</head>
<body>
<h1>KVCOMM Anchor-Level Matching Report</h1>
<div class="summary">
    warmup (BASE_CODE): {len(WARMUP_ANCHORS)} 个锚点, 签名 <code>{WARMUP_SIGNATURE}</code><br>
    lossy_alignment_method=kvcomm, template_task_family=code_generation
</div>
{case_html}
</body>
</html>"""


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "kvcomm_anchor_report.html"
    out.write_text(build_html(), encoding="utf-8")
    print(f"Report written to {out}")
