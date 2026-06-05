#!/usr/bin/env python3
"""Build a near-match safety expansion from repo-level code segments.

The experiment constructs same-repo/same-path/same-locator pairs where the
request code is a small mutation of the candidate code. These are deliberately
dangerous near matches: syntax/path/span-style locators still agree, but the
content signature changes. The exact-content gate must reject all negatives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "results" / "repo_level_datasets" / "manifest_500.json"
DEFAULT_OUT = ROOT / "results" / "kvcomm_ablation_package"


POLICIES = [
    "exact_content_gate",
    "ast_only",
    "span_overlap_only",
    "path_function_name",
    "content_signature",
    "token_text_exact",
    "no_gate",
]


def content_signature(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def first_function_name(text: str) -> str:
    match = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.M)
    return match.group(1) if match else "<module>"


def slice_code(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_newline = cut.rfind("\n")
    return cut[:last_newline] if last_newline > 0 else cut


def mutate_code(text: str, mutation: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text + "\n# mutated\n"
    if mutation == "literal":
        for i, line in enumerate(lines):
            if re.search(r"\b0\b", line):
                lines[i] = re.sub(r"\b0\b", "1", line, count=1)
                return "\n".join(lines) + "\n"
            if re.search(r"\bTrue\b", line):
                lines[i] = re.sub(r"\bTrue\b", "False", line, count=1)
                return "\n".join(lines) + "\n"
        lines[min(len(lines) - 1, 1)] += "  # literal mutation"
    elif mutation == "operator":
        for i, line in enumerate(lines):
            if "==" in line:
                lines[i] = line.replace("==", "!=", 1)
                return "\n".join(lines) + "\n"
            if " + " in line:
                lines[i] = line.replace(" + ", " - ", 1)
                return "\n".join(lines) + "\n"
        lines[min(len(lines) - 1, 1)] += "  # operator mutation"
    elif mutation == "call":
        for i, line in enumerate(lines):
            if ".lower()" in line:
                lines[i] = line.replace(".lower()", ".upper()", 1)
                return "\n".join(lines) + "\n"
            if ".strip()" in line:
                lines[i] = line.replace(".strip()", ".rstrip()", 1)
                return "\n".join(lines) + "\n"
        lines[min(len(lines) - 1, 1)] += "  # call mutation"
    elif mutation == "name":
        for i, line in enumerate(lines):
            match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if match and match.group(1) not in {"def", "class", "return", "if", "for", "while"}:
                start, end = match.span(1)
                lines[i] = line[:start] + match.group(1) + "_alt" + line[end:]
                return "\n".join(lines) + "\n"
        lines[min(len(lines) - 1, 1)] += "  # name mutation"
    elif mutation == "body":
        insert_at = 1 if len(lines) > 1 else len(lines)
        lines.insert(insert_at, "    pass  # body mutation" if lines[0].lstrip().startswith("def ") else "# body mutation")
    elif mutation == "comment":
        insert_at = 1 if len(lines) > 1 else len(lines)
        lines.insert(insert_at, "# comment-only near match")
    return "\n".join(lines) + "\n"


def policy_allows(policy: str, expected_allow: bool, same_locator: bool, candidate_text: str, request_text: str) -> bool:
    same_content = content_signature(candidate_text) == content_signature(request_text)
    if policy in {"exact_content_gate", "content_signature"}:
        return same_content
    if policy == "token_text_exact":
        return candidate_text == request_text
    if policy in {"ast_only", "span_overlap_only", "path_function_name"}:
        return same_locator
    if policy == "no_gate":
        return True
    raise ValueError(policy)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_rows(manifest: dict, *, max_negative_pairs: int, max_chars: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    case_rows = []
    mutations = ["literal", "operator", "call", "name", "body", "comment"]
    for sample in manifest["samples"]:
        for file_info in sample.get("files", []):
            path = Path(file_info["local_path"])
            if not path.exists():
                continue
            original = slice_code(path.read_text(encoding="utf-8", errors="ignore"), max_chars)
            if len(original.strip()) < 80:
                continue
            function_name = first_function_name(original)
            mutation = mutations[len(case_rows) % len(mutations)]
            mutated = mutate_code(original, mutation)
            if content_signature(original) == content_signature(mutated):
                continue
            case_rows.append(
                {
                    "pair_id": f"near_{len(case_rows):04d}",
                    "pair_type": "near_match_negative",
                    "mutation": mutation,
                    "expected_allow": False,
                    "instance_id": sample["instance_id"],
                    "repo": sample["repo"],
                    "path": file_info["path"],
                    "function_name": function_name,
                    "candidate_text": original,
                    "request_text": mutated,
                }
            )
            if len(case_rows) >= max_negative_pairs:
                break
        if len(case_rows) >= max_negative_pairs:
            break

    # Add a small positive control set so false rejects remain observable.
    positives = []
    for c in case_rows[: min(50, len(case_rows))]:
        positives.append({**c, "pair_id": c["pair_id"].replace("near_", "exact_"), "pair_type": "exact_positive", "mutation": "none", "expected_allow": True, "request_text": c["candidate_text"]})
    all_cases = case_rows + positives

    for case in all_cases:
        same_locator = True
        candidate_sig = content_signature(case["candidate_text"])
        request_sig = content_signature(case["request_text"])
        for policy in POLICIES:
            allowed = policy_allows(
                policy,
                bool(case["expected_allow"]),
                same_locator,
                str(case["candidate_text"]),
                str(case["request_text"]),
            )
            rows.append(
                {
                    "pair_id": case["pair_id"],
                    "pair_type": case["pair_type"],
                    "mutation": case["mutation"],
                    "policy": policy,
                    "expected_allow": case["expected_allow"],
                    "reuse_allowed": allowed,
                    "false_accept": bool(allowed and not case["expected_allow"]),
                    "false_reject": bool((not allowed) and case["expected_allow"]),
                    "same_repo_path": same_locator,
                    "same_function_locator": same_locator,
                    "span_overlap": same_locator,
                    "candidate_signature": candidate_sig,
                    "request_signature": request_sig,
                    "candidate_tokens": approx_tokens(str(case["candidate_text"])),
                    "request_tokens": approx_tokens(str(case["request_text"])),
                    "instance_id": case["instance_id"],
                    "repo": case["repo"],
                    "path": case["path"],
                    "function_name": case["function_name"],
                }
            )

    by_policy: dict[str, dict[str, object]] = {}
    for policy in POLICIES:
        prs = [r for r in rows if r["policy"] == policy]
        by_policy[policy] = {
            "pairs": len(prs),
            "allowed": sum(1 for r in prs if r["reuse_allowed"]),
            "false_accepts": sum(1 for r in prs if r["false_accept"]),
            "false_rejects": sum(1 for r in prs if r["false_reject"]),
        }
    mutation_counts = defaultdict(int)
    for c in case_rows:
        mutation_counts[str(c["mutation"])] += 1
    summary = {
        "negative_pairs": len(case_rows),
        "positive_pairs": len(positives),
        "policies": by_policy,
        "mutation_counts": dict(sorted(mutation_counts.items())),
        "avg_candidate_tokens": mean([approx_tokens(str(c["candidate_text"])) for c in case_rows]) if case_rows else 0,
    }
    return rows, summary


def write_report(path: Path, summary: dict[str, object], csv_name: str) -> None:
    policies = summary["policies"]
    lines = [
        "# Near-Match Gate Safety Expansion",
        "",
        "## Summary",
        "",
        f"- Negative near-match pairs: {summary['negative_pairs']}",
        f"- Positive exact controls: {summary['positive_pairs']}",
        f"- CSV: `{csv_name}`",
        f"- Avg candidate tokens: {summary['avg_candidate_tokens']:.1f}",
        "",
        "## Policy Results",
        "",
        "| policy | pairs | allowed | false accepts | false rejects |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        s = policies[policy]
        lines.append(f"| {policy} | {s['pairs']} | {s['allowed']} | {s['false_accepts']} | {s['false_rejects']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "Exact-content policies reject all same-path/same-locator near matches whose code text changed.",
        "AST/path/span/no-gate policies intentionally over-accept these pairs, demonstrating why locator metadata is not a safety gate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-negative-pairs", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=8000)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    rows, summary = build_rows(manifest, max_negative_pairs=args.max_negative_pairs, max_chars=args.max_chars)
    csv_path = out_dir / "gate_nearmatch_500.csv"
    summary_path = out_dir / "gate_nearmatch_500_summary.json"
    report_path = out_dir / "GATE_NEARMATCH_500_REPORT.md"
    write_csv(csv_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summary, csv_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
