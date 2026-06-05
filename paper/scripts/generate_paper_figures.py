#!/usr/bin/env python3
"""Generate paper tables and vector PDF figures from existing experiment results.

This script intentionally uses only the Python standard library so the paper
package can be regenerated on a fresh machine without numpy/matplotlib.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
TAB = PAPER / "tables"
DATA = PAPER / "data_manifest.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def optional_csv(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: list[float], pct_value: float) -> float:
    if not xs:
        return 0.0
    vals = sorted(xs)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * pct_value))))
    return vals[idx]


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}\\%"


def boolish(s: str) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def tex_escape(s: object) -> str:
    text = str(s)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


class Pdf:
    def __init__(self, width: int = 420, height: int = 300):
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def rgb(self, r: float, g: float, b: float) -> None:
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg {r:.3f} {g:.3f} {b:.3f} RG")

    def text(self, x: float, y: float, text: str, size: int = 9, bold: bool = False) -> None:
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        font = "/F2" if bold else "/F1"
        self.ops.append(f"BT {font} {size} Tf {x:.1f} {y:.1f} Td ({safe}) Tj ET")

    def line(self, x1: float, y1: float, x2: float, y2: float, width: float = 0.8) -> None:
        self.ops.append(f"{width:.2f} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S")

    def rect(self, x: float, y: float, w: float, h: float, fill: bool = True) -> None:
        op = "f" if fill else "S"
        self.ops.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re {op}")

    def save(self, path: Path) -> None:
        stream = "\n".join(self.ops).encode("latin-1", "replace")
        objs: list[bytes] = []
        objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
            f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode()
        objs.append(page)
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, obj in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode()
        out += (
            f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        ).encode()
        path.write_bytes(out)


COLORS = [
    (0.22, 0.45, 0.70),
    (0.20, 0.62, 0.45),
    (0.90, 0.55, 0.20),
    (0.65, 0.35, 0.70),
]


def grouped_bar_pdf(
    path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    ymax: float | None = None,
    yfmt=lambda v: f"{v:.0f}",
) -> None:
    pdf = Pdf()
    pdf.rgb(0, 0, 0)
    pdf.text(18, 278, title, 12, bold=True)
    left, bottom, width, height = 50, 55, 345, 200
    ymax = ymax or max(max(vals) for _, vals in series) * 1.15 or 1
    tick_count = 4
    pdf.rgb(0.82, 0.84, 0.86)
    for i in range(tick_count + 1):
        y = bottom + height * i / tick_count
        pdf.line(left, y, left + width, y, 0.35)
        pdf.rgb(0.25, 0.25, 0.25)
        pdf.text(8, y - 3, yfmt(ymax * i / tick_count), 7)
        pdf.rgb(0.82, 0.84, 0.86)
    pdf.rgb(0.1, 0.1, 0.1)
    pdf.line(left, bottom, left, bottom + height, 0.8)
    pdf.line(left, bottom, left + width, bottom, 0.8)

    ncat = len(categories)
    nser = len(series)
    group_w = width / max(ncat, 1)
    bar_w = min(22, group_w / (nser + 1.3))
    for si, (label, vals) in enumerate(series):
        r, g, b = COLORS[si % len(COLORS)]
        pdf.rgb(r, g, b)
        for ci, val in enumerate(vals):
            x = left + ci * group_w + (group_w - nser * bar_w) / 2 + si * bar_w
            h = 0 if ymax == 0 else height * val / ymax
            pdf.rect(x, bottom, bar_w * 0.85, h, True)
            pdf.rgb(0.1, 0.1, 0.1)
            pdf.text(x - 2, bottom + h + 4, yfmt(val), 7)
            pdf.rgb(r, g, b)
    pdf.rgb(0.1, 0.1, 0.1)
    for ci, cat in enumerate(categories):
        x = left + ci * group_w + 3
        pdf.text(x, 36, cat[:22], 7)
    lx = 54
    for si, (label, _) in enumerate(series):
        r, g, b = COLORS[si % len(COLORS)]
        pdf.rgb(r, g, b)
        pdf.rect(lx, 260, 10, 8, True)
        pdf.rgb(0.1, 0.1, 0.1)
        pdf.text(lx + 14, 260, label, 8)
        lx += 105
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(path)


def simple_note_pdf(path: Path, title: str, lines: list[str]) -> None:
    pdf = Pdf()
    pdf.rgb(0, 0, 0)
    pdf.text(18, 278, title, 12, bold=True)
    y = 245
    for line in lines:
        pdf.text(24, y, line, 10)
        y -= 22
    pdf.save(path)


def placeholder_pdf(path: Path, title: str) -> None:
    simple_note_pdf(
        path,
        title,
        [
            "Run benchmark/multi_workflow/bench_kvcomm_ttft_stress.py",
            "to populate this camera-ready figure.",
        ],
    )


def table_safety(gate_rows: list[dict[str, str]]) -> None:
    by_policy = defaultdict(lambda: {"false_accept": 0, "false_reject": 0, "allow": 0, "total": 0})
    for r in gate_rows:
        p = r["policy"]
        by_policy[p]["total"] += 1
        by_policy[p]["false_accept"] += int(boolish(r["false_accept"]))
        by_policy[p]["false_reject"] += int(boolish(r["false_reject"]))
        by_policy[p]["allow"] += int(boolish(r["reuse_allowed"]))
    order = ["full_kvcomm", "ast_only", "span_overlap_only", "content_only", "token_text_exact", "no_gate"]
    display = {
        "full_kvcomm": "exact_content_gate",
        "ast_only": "ast_only",
        "span_overlap_only": "span_overlap_only",
        "content_only": "content_only",
        "token_text_exact": "token_text_exact",
        "no_gate": "no_gate",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Gate safety ablation. The exact-content policy permits reuse only on identical code text.}",
        "\\label{tab:safety}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Policy & Allows & False accepts & False rejects \\\\",
        "\\midrule",
    ]
    for p in order:
        s = by_policy[p]
        lines.append(f"{tex_escape(display[p])} & {s['allow']} & {s['false_accept']} & {s['false_reject']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "table_safety.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_nearmatch_safety(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    by_policy = defaultdict(lambda: {"pairs": 0, "allow": 0, "false_accept": 0, "false_reject": 0})
    for r in rows:
        p = r["policy"]
        by_policy[p]["pairs"] += 1
        by_policy[p]["allow"] += int(boolish(r["reuse_allowed"]))
        by_policy[p]["false_accept"] += int(boolish(r["false_accept"]))
        by_policy[p]["false_reject"] += int(boolish(r["false_reject"]))
    order = ["exact_content_gate", "ast_only", "span_overlap_only", "path_function_name", "content_signature", "token_text_exact", "no_gate"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Near-match safety expansion from 500 repo-level negative pairs plus exact controls.}",
        "\\label{tab:nearmatch-safety}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Policy & Allows & False accepts & False rejects \\\\",
        "\\midrule",
    ]
    for p in order:
        s = by_policy[p]
        lines.append(f"{tex_escape(p)} & {s['allow']} & {s['false_accept']} & {s['false_reject']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "table_nearmatch_safety.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return by_policy


def table_passrate(pass_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_mode = defaultdict(list)
    for r in pass_rows:
        by_mode[r["mode"]].append(r)
    summary = {}
    for mode, rows in by_mode.items():
        summary[mode] = {
            "cases": len(rows),
            "diff": sum(boolish(r["diff_extracted"]) for r in rows),
            "synth": sum(boolish(r["synthesis_ok"]) for r in rows),
            "apply": sum(boolish(r["apply_clean"]) for r in rows),
            "pass": sum(boolish(r["pass1"]) for r in rows),
            "cached": mean([float(r["cached_tokens"]) for r in rows]),
            "latency": mean([float(r["elapsed_ms"]) for r in rows]),
            "exact_hits": sum(1 for r in rows if r.get("match_reason") == "exact_code_content_signature"),
        }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{30-case lossless KV vs. exact-content segment reuse pass@1 comparison.}",
        "\\label{tab:passrate}",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & Lossless KV & Exact-content reuse \\\\",
        "\\midrule",
    ]
    lossless, lossy = summary["lossless"], summary["lossy"]
    metrics = [
        ("Diff extracted", f"{int(lossless['diff'])}/{int(lossless['cases'])}", f"{int(lossy['diff'])}/{int(lossy['cases'])}"),
        ("Clean apply", f"{int(lossless['apply'])}/{int(lossless['cases'])}", f"{int(lossy['apply'])}/{int(lossy['cases'])}"),
        ("pass@1", f"{int(lossless['pass'])}/{int(lossless['cases'])}", f"{int(lossy['pass'])}/{int(lossy['cases'])}"),
        ("Avg cached tokens", f"{lossless['cached']:.1f}", f"{lossy['cached']:.1f}"),
        ("Avg latency (ms)", f"{lossless['latency']:.1f}", f"{lossy['latency']:.1f}"),
        ("Exact-content hits", "--", f"{int(lossy['exact_hits'])}/{int(lossy['cases'])}"),
    ]
    for name, a, b in metrics:
        lines.append(f"{name} & {a} & {b} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "table_passrate.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def table_prefetch(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    summary = {}
    for mode, rs in by_mode.items():
        summary[mode] = {
            "cases": len(rs),
            "latency": mean([float(r["elapsed_ms"]) for r in rs]),
            "latency_p50": percentile([float(r["elapsed_ms"]) for r in rs], 0.50),
            "latency_p90": percentile([float(r["elapsed_ms"]) for r in rs], 0.90),
            "cached": mean([float(r["cached_tokens"]) for r in rs]),
            "hints": mean([float(r["codebase_prefetch_hint_count"]) for r in rs]),
            "hits": sum(1 for r in rs if r.get("lossy_match_reason") == "exact_code_content_signature"),
        }
    labels = [
        ("baseline_prefix_cache_only", "Baseline prefix cache"),
        ("kvflow_prefix_only", "KVFlow baseline"),
        ("kvflow_prefix_plus_codebase_prefetch", "Baseline + code hints"),
        ("kvcomm_lossy_plus_codebase_prefetch", "Exact reuse + code hints"),
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Realistic E2E serving check on {int(summary[labels[0][0]]['cases'])} cases. See Table~\\ref{{tab:ttft-stress}} for the primary prefill-dominated TTFT speedup claim.}}",
        "\\label{tab:prefetch}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Mode & Avg lat (ms) & P50 (ms) & P90 (ms) & Avg cached tok. & Avg hints & Exact hits \\\\",
        "\\midrule",
    ]
    for key, label in labels:
        s = summary[key]
        lines.append(f"{label} & {s['latency']:.1f} & {s['latency_p50']:.1f} & {s['latency_p90']:.1f} & {s['cached']:.1f} & {s['hints']:.1f} & {int(s['hits'])}/{int(s['cases'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (TAB / "table_prefetch.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def table_ttft_stress(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    if not rows:
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{KVCOMM-style TTFT stress experiment plan. Results are generated by \\texttt{bench\\_kvcomm\\_ttft\\_stress.py}.}",
            "\\label{tab:ttft-stress}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Mode & Max chars & P50 TTFT (ms) & Speedup & Exact hits \\\\",
            "\\midrule",
            "\\multicolumn{5}{c}{Pending local stress run} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
        (TAB / "table_ttft_stress.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        placeholder_pdf(FIG / "fig_ttft_speedup_by_length.pdf", "TTFT speedup by length pending")
        placeholder_pdf(FIG / "fig_agent_scaling_speedup.pdf", "Agent scaling speedup pending")
        return {}

    e6 = [r for r in rows if r.get("experiment") == "ttft_stress" and str(r.get("max_tokens")) == "1"]
    e7 = [r for r in rows if r.get("experiment") == "agent_scaling_workflow"]

    def group(items, keys):
        out = defaultdict(list)
        for item in items:
            out[tuple(item[k] for k in keys)].append(item)
        return out

    e6_groups = group(e6, ["mode", "max_file_chars"])
    prefix_by_len = {
        length: percentile([float(r["ttft_ms"]) for r in rs], 0.50)
        for (mode, length), rs in e6_groups.items()
        if mode == "prefix_cache_only"
    }
    summary = {}
    preferred_modes = [
        ("prefix_cache_only", "Prefix only"),
        ("exact_reuse_no_hints", "Exact reuse"),
        ("exact_reuse_plus_code_hints", "Exact reuse + hints"),
    ]
    lengths = sorted({r["max_file_chars"] for r in e6}, key=lambda x: int(x))
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{KVCOMM-style long-code TTFT stress results. Speedup is relative to prefix-only at the same length bucket.}",
        "\\label{tab:ttft-stress}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Mode & Max chars & Cases & P50 TTFT (ms) & Speedup & Exact hits \\\\",
        "\\midrule",
    ]
    for mode, label in preferred_modes:
        for length in lengths:
            rs = e6_groups.get((mode, length), [])
            if not rs:
                continue
            ttfts = [float(r["ttft_ms"]) for r in rs]
            p50 = percentile(ttfts, 0.50)
            base = prefix_by_len.get(length, p50)
            speedup = base / p50 if p50 else 0.0
            hits = sum(1 for r in rs if boolish(r.get("exact_hit", "")))
            lines.append(f"{label} & {length} & {len(rs)} & {p50:.1f} & {speedup:.2f}$\\times$ & {hits}/{len(rs)} \\\\")
            summary[f"ttft_{mode}_{length}"] = {
                "cases": len(rs),
                "p50_ttft": p50,
                "p90_ttft": percentile(ttfts, 0.90),
                "speedup_vs_prefix": speedup,
                "exact_hits": hits,
            }
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (TAB / "table_ttft_stress.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    exact_speedups = []
    for length in lengths:
        rs = e6_groups.get(("exact_reuse_plus_code_hints", length), [])
        if not rs:
            exact_speedups.append(0.0)
            continue
        p50 = percentile([float(r["ttft_ms"]) for r in rs], 0.50)
        exact_speedups.append(prefix_by_len.get(length, p50) / p50 if p50 else 0.0)
    grouped_bar_pdf(
        FIG / "fig_ttft_speedup_by_length.pdf",
        "TTFT speedup by code length",
        [f"{int(x)//1000}k" for x in lengths],
        [("exact+hints vs prefix", exact_speedups)],
        ymax=max(exact_speedups + [1.0]) * 1.15,
        yfmt=lambda v: f"{v:.2f}x",
    )

    e7_groups = group(e7, ["mode", "agent_count", "segment_count", "max_file_chars"])
    agent_counts = sorted({r["agent_count"] for r in e7 if r.get("segment_count") == "3"}, key=lambda x: int(x))
    agent_speedups = []
    for count in agent_counts:
        candidates = [k for k in e7_groups if k[1] == count and k[2] == "3"]
        if not candidates:
            agent_speedups.append(0.0)
            continue
        length = max((k[3] for k in candidates), key=lambda x: int(x))
        pfx = e7_groups.get(("prefix_cache_only", count, "3", length), [])
        exact = e7_groups.get(("exact_reuse_plus_code_hints", count, "3", length), [])
        pfx_p50 = percentile([float(r["ttft_ms"]) for r in pfx], 0.50)
        exact_p50 = percentile([float(r["ttft_ms"]) for r in exact], 0.50)
        agent_speedups.append(pfx_p50 / exact_p50 if exact_p50 else 0.0)
    if agent_counts:
        grouped_bar_pdf(
            FIG / "fig_agent_scaling_speedup.pdf",
            "Cumulative workflow TTFT speedup",
            [f"{c} agents" for c in agent_counts],
            [("exact+hints vs prefix", agent_speedups)],
            ymax=max(agent_speedups + [1.0]) * 1.15,
            yfmt=lambda v: f"{v:.2f}x",
        )
    else:
        placeholder_pdf(FIG / "fig_agent_scaling_speedup.pdf", "Agent scaling speedup pending")
    return summary


def table_ablation(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    ablation_rows = [r for r in rows if r.get("experiment") == "ablation"]
    if not ablation_rows:
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{KVCOMM performance ablation. Results are generated by \\texttt{bench\\_kvcomm\\_ttft\\_stress.py}.}",
            "\\label{tab:ablation}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            "Mode & P50 TTFT (ms) & Speedup & Cached gain & Exact hits & Output F1 \\\\",
            "\\midrule",
            "\\multicolumn{6}{c}{Pending ablation run} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
        (TAB / "table_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {}

    by_mode = defaultdict(list)
    for r in ablation_rows:
        by_mode[r["mode"]].append(r)

    prefix_rows = by_mode.get("ablation_prefix_only", [])
    baseline_p50 = percentile([float(r["ttft_ms"]) for r in prefix_rows], 0.50) if prefix_rows else 1.0
    baseline_cached = mean([float(r["cached_tokens"]) for r in prefix_rows]) if prefix_rows else 0.0

    mode_labels = [
        ("ablation_prefix_only", "Prefix only"),
        ("ablation_hints_no_exact", "Hints only"),
        ("ablation_exact_no_hints", "Exact gate (no hints)"),
        ("ablation_exact_gate_rope", "Exact gate + RoPE $\\delta$"),
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{KVCOMM performance ablation. Speedup is relative to prefix-only; cached-token gain is increase over prefix-only.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Mode & P50 TTFT (ms) & Speedup & Cached gain & Exact hits & Output F1 \\\\",
        "\\midrule",
    ]
    summary = {}
    for mode, label in mode_labels:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        p50 = percentile([float(r["ttft_ms"]) for r in rs], 0.50)
        speedup = baseline_p50 / p50 if p50 else 0.0
        avg_cached = mean([float(r["cached_tokens"]) for r in rs])
        cached_gain = avg_cached - baseline_cached
        hits = sum(1 for r in rs if boolish(r.get("exact_hit", "")))
        f1_vals = [float(r["output_token_f1_vs_baseline"]) for r in rs
                   if str(r.get("output_token_f1_vs_baseline", "")) not in {"", "None"}]
        f1 = mean(f1_vals) if f1_vals else 0.0
        lines.append(f"{label} & {p50:.1f} & {speedup:.2f}$\\times$ & {cached_gain:+.0f} & {hits}/{len(rs)} & {f1:.3f} \\\\")
        summary[mode] = {
            "p50_ttft": p50,
            "speedup": speedup,
            "cached_gain": cached_gain,
            "exact_hits": hits,
            "f1": f1,
        }
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "table_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def table_template_segments(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Template segment ablation. More exposed code-base segments and downstream agents increase exact reuse opportunity.}",
        "\\label{tab:template-segments}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Workflow & Segments & Exact hits & Est. cached tok. \\\\",
        "\\midrule",
    ]
    summary = {}
    for r in rows:
        workflow = "P$\\rightarrow$I" if r["workflow"] == "planner_implementer" else "P$\\rightarrow$I$\\rightarrow$D"
        lines.append(f"{workflow} & {r['segment_count']} & {r['exact_hits']} & {r['estimated_cached_tokens']} \\\\")
        summary[f"{r['workflow']}_s{r['segment_count']}"] = {
            "exact_hits": float(r["exact_hits"]),
            "estimated_cached_tokens": float(r["estimated_cached_tokens"]),
        }
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "table_template_segments.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def table_scalability(manifest_30, manifest_100, manifest_500, gate_500_rows) -> dict[str, float]:
    rows = []
    for label, manifest in [("30-case", manifest_30), ("100-case", manifest_100), ("500-case", manifest_500)]:
        fs = manifest["file_stats"]
        rows.append((label, manifest["case_count"], len(manifest["repo_distribution"]), fs["total_files"], fs["total_lines"], fs["total_chars"]))
    approx_tokens = sum(int(r["approx_tokens"]) for r in gate_500_rows)
    unique_sigs = len({r["content_signature"] for r in gate_500_rows})
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Repo-level dataset scale.}",
        "\\label{tab:scalability}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & Cases & Repos & Files & Source lines \\\\",
        "\\midrule",
    ]
    for label, cases, repos, files, total_lines, _ in rows:
        lines.append(f"{label} & {cases} & {repos} & {files} & {total_lines:,} \\\\")
    lines += [
        "\\midrule",
        f"500-case reusable tokens & \\multicolumn{{4}}{{r}}{{{approx_tokens:,}}} \\\\",
        f"500-case unique signatures & \\multicolumn{{4}}{{r}}{{{unique_sigs:,}}} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (TAB / "table_scalability.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"approx_tokens": approx_tokens, "unique_sigs": unique_sigs}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    gate = read_csv(ROOT / "results/kvcomm_ablation_package/gate_safety_ablation.csv")
    nearmatch = optional_csv(ROOT / "results/kvcomm_ablation_package/gate_nearmatch_500.csv")
    rope = read_csv(ROOT / "results/kvcomm_ablation_package/rope_delta_ablation.csv")
    logits = read_csv(ROOT / "results/kvcomm_ablation_package/logit_alignment_ablation.csv")
    passrate_candidates = [
        ROOT / "results/swe_generated_patch_kvcomm/qwen2_5_32b_gptq_json_30/passrate_table.csv",
        ROOT / "results/swe_generated_patch_kvcomm/qwen2_5_32b_gptq_json_30_smallctx/passrate_table.csv",
        ROOT / "results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/passrate_table.csv",
    ]
    passrate_path = next((p for p in passrate_candidates if p.exists()), passrate_candidates[-1])
    prefetch_path = ROOT / "results/coding_kvflow_prefetch/qwen2_5_7b_100/prefetch_table.csv"
    if not prefetch_path.exists():
        prefetch_path = ROOT / "results/coding_kvflow_prefetch/qwen2_5_7b_30/prefetch_table.csv"
    ttft_stress_path = ROOT / "results/kvcomm_ttft_stress/qwen2_5_7b/ttft_stress_table.csv"
    passrate = read_csv(passrate_path)
    prefetch = read_csv(prefetch_path)
    ttft_stress = optional_csv(ttft_stress_path)
    template_segments = optional_csv(ROOT / "results/template_codebase_segments/template_segment_ablation.csv")
    gate500 = read_csv(ROOT / "results/repo_level_datasets/500_gate_anchor_stats.csv")
    manifest30 = read_json(ROOT / "results/repo_level_datasets/manifest_30.json")
    manifest100 = read_json(ROOT / "results/repo_level_datasets/manifest_100.json")
    manifest500 = read_json(ROOT / "results/repo_level_datasets/manifest_500.json")

    table_safety(gate)
    nearmatch_summary = table_nearmatch_safety(nearmatch)
    pass_summary = table_passrate(passrate)
    prefetch_summary = table_prefetch(prefetch)
    ttft_stress_summary = table_ttft_stress(ttft_stress)
    ablation_summary = table_ablation(ttft_stress)
    template_summary = table_template_segments(template_segments)
    scale_summary = table_scalability(manifest30, manifest100, manifest500, gate500)

    # Safety chart.
    by_policy = Counter()
    for r in gate:
        by_policy[r["policy"]] += int(boolish(r["false_accept"]))
    policies = ["full_kvcomm", "ast_only", "span_overlap_only", "content_only", "token_text_exact", "no_gate"]
    grouped_bar_pdf(
        FIG / "fig_gate_false_accepts.pdf",
        "Gate safety: false accepts",
        [p.replace("_", " ") for p in policies],
        [("false accepts", [by_policy[p] for p in policies])],
        ymax=max(by_policy.values()) + 1,
    )

    if nearmatch_summary:
        policies_near = ["exact_content_gate", "ast_only", "span_overlap_only", "path_function_name", "content_signature", "token_text_exact", "no_gate"]
        grouped_bar_pdf(
            FIG / "fig_gate_nearmatch_false_accepts.pdf",
            "Near-match safety: false accepts",
            [p.replace("_", " ") for p in policies_near],
            [("false accepts", [nearmatch_summary[p]["false_accept"] for p in policies_near])],
            ymax=max(nearmatch_summary[p]["false_accept"] for p in policies_near) + 50,
        )

    # RoPE/logit chart.
    correct_k = mean([float(r["k_cosine"]) for r in rope if r["variant"] == "correct_delta"])
    wrong_large_k = mean([float(r["k_cosine"]) for r in rope if r["variant"] == "wrong_delta" and abs(int(r["delta_error"])) >= 16])
    no_rot_k = mean([float(r["k_cosine"]) for r in rope if r["variant"] == "no_rotation"])
    top1 = mean([1.0 if boolish(r["top1_agree"]) else 0.0 for r in logits])
    top5 = mean([1.0 if boolish(r["top5_agree"]) else 0.0 for r in logits])
    kl = mean([float(r["kl_b_to_a"]) for r in logits])
    grouped_bar_pdf(
        FIG / "fig_rope_and_logits.pdf",
        f"RoPE/logit alignment (mean KL={kl:.4f})",
        ["correct K", "no-rot K", "wrong K", "top-1", "top-5"],
        [("agreement/cosine", [correct_k, no_rot_k, wrong_large_k, top1, top5])],
        ymax=1.05,
        yfmt=lambda v: f"{v:.2f}",
    )

    # H12 mode chart.
    mode_order = [
        ("baseline_prefix_cache_only", "baseline"),
        ("kvflow_prefix_only", "KVFlow base"),
        ("kvflow_prefix_plus_codebase_prefetch", "+ code hints"),
        ("kvcomm_lossy_plus_codebase_prefetch", "exact reuse"),
    ]
    grouped_bar_pdf(
        FIG / "fig_prefetch_modes.pdf",
        "Coding-aware codebase hint modes",
        [label for _, label in mode_order],
        [
            ("cached tokens", [prefetch_summary[k]["cached"] for k, _ in mode_order]),
            ("latency ms", [prefetch_summary[k]["latency"] for k, _ in mode_order]),
        ],
        yfmt=lambda v: f"{v:.0f}",
    )

    if template_segments:
        grouped_bar_pdf(
            FIG / "fig_template_segment_ablation.pdf",
            "Template segment reuse opportunity",
            [f"{r['workflow'].replace('planner_implementer_debugger', 'P-I-D').replace('planner_implementer', 'P-I')} s{r['segment_count']}" for r in template_segments],
            [("exact hits", [float(r["exact_hits"]) for r in template_segments])],
            yfmt=lambda v: f"{v:.0f}",
        )

    # H10 passrate chart.
    grouped_bar_pdf(
        FIG / "fig_passrate_main.pdf",
        "Lossless KV vs exact-content reuse",
        ["pass@1", "clean apply", "cached tok/1k", "latency s"],
        [
            (
                "lossless",
                [
                    pass_summary["lossless"]["pass"],
                    pass_summary["lossless"]["apply"],
                    pass_summary["lossless"]["cached"] / 1000.0,
                    pass_summary["lossless"]["latency"] / 1000.0,
                ],
            ),
            (
                "exact reuse",
                [
                    pass_summary["lossy"]["pass"],
                    pass_summary["lossy"]["apply"],
                    pass_summary["lossy"]["cached"] / 1000.0,
                    pass_summary["lossy"]["latency"] / 1000.0,
                ],
            ),
        ],
        yfmt=lambda v: f"{v:.1f}",
    )

    # Scalability chart: top 6 repo distribution plus token note.
    repo_counts = Counter(manifest500["repo_distribution"])
    # Counter(dict) creates key counts incorrectly; use sorted items instead.
    top = sorted(manifest500["repo_distribution"].items(), key=lambda kv: kv[1], reverse=True)[:6]
    grouped_bar_pdf(
        FIG / "fig_scalability.pdf",
        f"500-case scale: {scale_summary['approx_tokens']:,} reusable tokens",
        [repo.split("/")[-1] for repo, _ in top],
        [("cases", [count for _, count in top])],
        yfmt=lambda v: f"{v:.0f}",
    )

    manifest = {
        "generated_by": "paper/scripts/generate_paper_figures.py",
        "sources": {
            "gate_safety": "results/kvcomm_ablation_package/gate_safety_ablation.csv",
            "gate_nearmatch": "results/kvcomm_ablation_package/gate_nearmatch_500.csv" if nearmatch else "",
            "rope_delta": "results/kvcomm_ablation_package/rope_delta_ablation.csv",
            "logit_alignment": "results/kvcomm_ablation_package/logit_alignment_ablation.csv",
            "passrate_primary": str(passrate_path.relative_to(ROOT)),
            "prefetch_primary": str(prefetch_path.relative_to(ROOT)),
            "ttft_stress": str(ttft_stress_path.relative_to(ROOT)) if ttft_stress else "",
            "ablation": str(ttft_stress_path.relative_to(ROOT)) if ablation_summary else "",
            "template_segments": "results/template_codebase_segments/template_segment_ablation.csv" if template_segments else "",
            "manifest_30": "results/repo_level_datasets/manifest_30.json",
            "manifest_100": "results/repo_level_datasets/manifest_100.json",
            "manifest_500": "results/repo_level_datasets/manifest_500.json",
            "gate_500": "results/repo_level_datasets/500_gate_anchor_stats.csv",
        },
        "summaries": {
            "passrate": pass_summary,
            "prefetch": prefetch_summary,
            "ttft_stress": ttft_stress_summary,
            "ablation": ablation_summary,
            "template_segments": template_summary,
            "nearmatch_safety": nearmatch_summary,
            "scale": scale_summary,
            "rope_logit": {
                "correct_k_cosine": correct_k,
                "no_rotation_k_cosine": no_rot_k,
                "wrong_large_delta_k_cosine": wrong_large_k,
                "mean_kl": kl,
                "top1_agreement": top1,
                "top5_agreement": top5,
            },
        },
    }
    DATA.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ----- Section 7 additions: anchor-distance experiment figures -----
    # Pull the per-axis aggregated d_norm from the same-code-different-
    # context experiment and emit three bar charts plus a sample of
    # the predicted_distance_table.json as a paper table.
    ctx = read_json(ROOT / "results/same_code_context_variation/data/context_distance_7b.json")
    table_json = read_json(ROOT / "results/same_code_context_variation/data/predicted_distance_table.json")
    if ctx.get("per_segment"):
        per_seg = ctx["per_segment"]
        by_pos: dict = defaultdict(list)
        by_sys: dict = defaultdict(list)
        by_sur: dict = defaultdict(list)
        for s in per_seg:
            for k, v in s["by_position_offset"].items():
                by_pos[k].append(v["mean"])
            for k, v in s["by_system_prompt_class"].items():
                by_sys[k].append(v["mean"])
            for k, v in s["by_surrounding_code_class"].items():
                by_sur[k].append(v["mean"])
        pos_keys = sorted(by_pos, key=lambda k: int(k))
        sys_keys = sorted(by_sys)
        sur_keys = sorted(by_sur)
        grouped_bar_pdf(
            FIG / "fig_anchor_distance_by_position_offset.pdf",
            "d_norm vs position offset (same code, 7B, last-4 layers)",
            pos_keys,
            [("mean d_norm", [sum(by_pos[k]) / len(by_pos[k]) for k in pos_keys])],
            yfmt=lambda v: f"{v:.2f}",
        )
        grouped_bar_pdf(
            FIG / "fig_anchor_distance_by_system_prompt.pdf",
            "d_norm vs system prompt class",
            sys_keys,
            [("mean d_norm", [sum(by_sys[k]) / len(by_sys[k]) for k in sys_keys])],
            yfmt=lambda v: f"{v:.2f}",
        )
        grouped_bar_pdf(
            FIG / "fig_anchor_distance_by_surrounding_code.pdf",
            "d_norm vs surrounding code wrap",
            sur_keys,
            [("mean d_norm", [sum(by_sur[k]) / len(by_sur[k]) for k in sur_keys])],
            yfmt=lambda v: f"{v:.2f}",
        )

    if table_json.get("cells"):
        # Sample the 4D lookup table as a paper-ready LaTeX table: pick
        # the (50-200, 50-100) row across all 4 sys_cls x 4 surr_cls = 16
        # cells. This is the most interesting "high offset" range.
        target_lb = "50-200"
        target_pos = "50-100"
        sampled = [
            c for c in table_json["cells"]
            if c["length_bin"] == target_lb and c["position_offset"] == target_pos
        ]
        # Sort by (sys_cls, surr_cls) for readability
        sampled.sort(key=lambda c: (c["system_prompt_class"], c["surrounding_code_class"]))
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Predicted KV distance (\(d_{\text{norm}}\)) for the (50--200 tokens, 50--100 offset) bucket of the \texttt{context\_aware\_confidence} lookup table. "
            r"Rows are system prompt classes, columns are surrounding code wraps. "
            r"Values closer to 0 mean the K/V is more reusable; the multiplier drops the 0.95 base confidence proportionally.}"
            r"\label{tab:predicted-distance-50-100}",
            r"\begin{tabular}{l|rrrr}",
            r"\toprule",
            r" & none & class\_wrap & try\_wrap & imports\_wrap \\",
            r"\midrule",
        ]
        sys_order = ["planner", "coder", "reviewer", "tester"]
        for sys_cls in sys_order:
            row_cells = [c for c in sampled if c["system_prompt_class"] == sys_cls]
            row_cells.sort(key=lambda c: c["surrounding_code_class"])
            cells = [f"{c['predicted_d_norm_mean']:.2f}" for c in row_cells]
            lines.append(f"{sys_cls} & " + " & ".join(cells) + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        (TAB / "tab_predicted_distance_50_100.tex").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    # ----- Section 7 additions: cross-model transferability (Study 1) -----
    # Read all per-model tables and produce a 2-row comparison table
    # summarising d_norm at the canonical (0, planner, none) cell and
    # the 50-100 offset cell, plus the global max. Embedded in the
    # paper as tab_cross_model_summary.tex.
    xmodel_dir = ROOT / "results" / "lookup_table_transferability" / "data"
    xmodel_files = sorted(xmodel_dir.glob("predicted_distance_table_*.json"))

    def _slug_to_pretty(slug: str) -> str:
        """e.g. 'qwen-qwen2.5-coder-3b-instruct' -> 'Qwen2.5-Coder-3B-Instruct'."""
        body = slug.split("qwen-", 1)[-1]
        # Capitalise only the first letter of each token (so 3B stays 3B)
        parts = []
        for tok in body.split("-"):
            if not tok:
                continue
            if tok[0].isdigit():
                parts.append(tok.upper())
            else:
                parts.append(tok[0].upper() + tok[1:])
        return "-".join(parts)

    if xmodel_files:
        xrows = []
        for path in xmodel_files:
            slug = path.stem.replace("predicted_distance_table_", "")
            try:
                t = read_json(path)
            except Exception:
                continue
            cells = t.get("cells", [])
            if not cells:
                continue
            # canonical = 0, planner, none (find the cell with lowest d_norm)
            canonical = [c for c in cells if c["position_offset"] == "0"
                         and c["system_prompt_class"] == "planner"
                         and c["surrounding_code_class"] == "none"]
            hi = [c for c in cells if c["position_offset"] == "50-100"
                  and c["system_prompt_class"] == "tester"
                  and c["surrounding_code_class"] == "imports_wrap"]
            if not canonical or not hi:
                continue
            xrows.append({
                "model": _slug_to_pretty(slug),
                "n_cells": len(cells),
                "canonical": f"{canonical[0]['predicted_d_norm_mean']:.3f}",
                "high_offset": f"{hi[0]['predicted_d_norm_mean']:.3f}",
                "global_max": f"{t.get('global', {}).get('predicted_d_norm_max_observed', 0):.3f}",
            })
        if xrows:
            # Also emit a per-model d_norm vs position_offset table
            xlines = [
                r"\begin{table}[t]",
                r"\centering",
                r"\small",
                r"\caption{Cross-model transferability of the \texttt{predicted\_distance\_table}. "
                r"\emph{canonical} is the (0 offset, planner, none) cell; "
                r"\emph{high\_offset} is the (50--100 offset, tester, imports\_wrap) cell. "
                r"Smaller values mean the K/V is more reusable.}"
                r"\label{tab:cross-model-summary}",
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                r"model & canonical & high offset & global max \\",
                r"\midrule",
            ]
            for r in xrows:
                xlines.append(
                    f"{tex_escape(r['model'])} & {r['canonical']} & {r['high_offset']} & {r['global_max']} \\\\"
                )
            xlines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
            (TAB / "tab_cross_model_summary.tex").write_text(
                "\n".join(xlines), encoding="utf-8"
            )

    # ----- Section 7 additions: real-trace reuse (Study 2) ----------------
    # Read swe_bench_aggregate.json (from real_trace_reuse) and emit a
    # per-agent-pair hit rate table + a tokens-saved table.
    real_trace = ROOT / "results" / "real_trace_reuse" / "data" / "swe_bench_aggregate.json"
    if real_trace.exists():
        try:
            agg = read_json(real_trace)
        except Exception:
            agg = {}
        if agg:
            pairs = agg.get("agent_pair_stats", {})
            if pairs:
                plines = [
                    r"\begin{table}[t]",
                    r"\centering",
                    r"\small",
                    r"\caption{SWE-bench Verified replay: KVCOMM hit rate by agent pair. "
                    r"\emph{n} is the number of cross-agent request pairs; "
                    r"\emph{matched} is how many had the same content signature.}"
                    r"\label{tab:real-trace-hit-rate}",
                    r"\begin{tabular}{lrrr}",
                    r"\toprule",
                    r"agent pair & n & matched & hit rate \\",
                    r"\midrule",
                ]
                for pair, s in pairs.items():
                    plines.append(
                        f"{tex_escape(pair)} & {s['n_pairs']} & {s['n_matched_content']} & {s['hit_rate']:.1%} \\\\"
                    )
                plines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
                (TAB / "tab_real_trace_reuse_stats.tex").write_text(
                    "\n".join(plines), encoding="utf-8"
                )

    print(f"Wrote figures to {FIG}")
    print(f"Wrote tables to {TAB}")
    print(f"Wrote manifest to {DATA}")


if __name__ == "__main__":
    main()
