#!/usr/bin/env python3
"""Lightweight code-graph bundle census for code-specific KV reuse precision.

This script is intentionally static and dependency-light. It uses Python's
standard-library `ast` module to build a conservative repository-local symbol
index, then derives reusable bundles around SWE-bench patch targets:

* ast_function_only
* call_neighborhood_1hop
* reverse_callers_1hop
* import_dependency_bundle
* test_target_bundle

The output is a derived research artifact for lossy-reuse precision studies.
Token counts are recorded only as scope covariates; the primary goal is to
prepare code-graph bundles for KV-distance and output-drift experiments.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import math
import json
import os
import re
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
BASE = ROOT / "results" / "code_graph_kv_reuse"
DATA = BASE / "data"
FIG = BASE / "figures"
REPO_ROOT = ROOT / "results" / "repo_level_datasets"
LOCAL_ENV_REPOS = ROOT / "results" / "swebench_local_envs" / "repos"
DEFAULT_INSTANCES = REPO_ROOT / "swe_verified_100_instances.json"

PY_EXT = ".py"
BUNDLE_TYPES = (
    "ast_function_only",
    "call_neighborhood_1hop",
    "reverse_callers_1hop",
    "import_dependency_bundle",
    "test_target_bundle",
)

PRECISION_HYPOTHESES = {
    "ast_function_only": (
        "baseline exact target span; useful when the patch-local symbol is "
        "self-contained"
    ),
    "call_neighborhood_1hop": (
        "expected to improve lossy precision when target behavior depends on "
        "local helpers or dispatch utilities"
    ),
    "reverse_callers_1hop": (
        "diagnostic high-context variant; may increase semantic alignment but "
        "also raises drift risk from caller/test-specific context"
    ),
    "import_dependency_bundle": (
        "expected to preserve stable API/import context with limited semantic "
        "contamination"
    ),
    "test_target_bundle": (
        "task-aligned variant for SWE-style repair; useful for output-drift "
        "and pass@1 non-degradation checks"
    ),
}

PRECISION_PRIORITY = {
    "import_dependency_bundle": "high",
    "call_neighborhood_1hop": "high",
    "ast_function_only": "medium",
    "test_target_bundle": "medium",
    "reverse_callers_1hop": "diagnostic",
}

ROLE_SYSTEM_PROMPTS = {
    "planner": "You are AgentTemplateKV Planner. Identify reusable code evidence without writing a patch.",
    "coder": "You are AgentTemplateKV Coder. Use the provided code evidence to draft the minimal repair.",
    "reviewer": "You are AgentTemplateKV Reviewer. Check whether the code evidence supports the proposed repair.",
}


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text))


def sha1_short(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def parse_patch_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        candidate = parts[3]
        if candidate.startswith("b/"):
            candidate = candidate[2:]
        if candidate.endswith(PY_EXT) and candidate not in files:
            files.append(candidate)
    return files


def changed_new_lines_for_file(patch_text: str, rel_path: str) -> set[int]:
    """Return new-file line numbers touched by a unified diff."""
    touched: set[int] = set()
    in_file = False
    new_line = 0
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            in_file = f" b/{rel_path}" in line or line.endswith(f" {rel_path}")
            new_line = 0
            continue
        if not in_file:
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            new_line = int(match.group(1)) if match else 0
            continue
        if not new_line:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            touched.add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            new_line += 1
    return touched


@dataclass(frozen=True)
class SymbolRef:
    symbol_id: str
    file: str
    qualname: str
    name: str
    start_line: int
    end_line: int
    kind: str
    text: str
    token_count: int
    content_signature: str
    calls: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()


@dataclass
class FileInfo:
    rel_path: str
    text: str
    imports_text: str
    imports: list[str]
    symbols: list[SymbolRef] = field(default_factory=list)


@dataclass
class BundleRecord:
    instance_id: str
    repo: str
    target_file: str
    target_symbol: str
    bundle_type: str
    symbol_count: int
    file_count: int
    token_count: int
    content_signature: str
    exact_signature_hit_rate: float
    overlap_with_file_prefix: float
    expansion_ratio: float
    includes_imports: bool
    includes_tests: bool
    precision_priority: str
    precision_hypothesis: str
    reason: str
    files: list[str]
    symbols: list[str]


class RepoIndex:
    def __init__(self, repo_dir: Path, max_files: int = 300):
        self.repo_dir = repo_dir
        self.max_files = max_files
        self.files: dict[str, FileInfo] = {}
        self.symbols: dict[str, SymbolRef] = {}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.callers: dict[str, set[str]] = defaultdict(set)

    def build(self, priority_files: Iterable[str]) -> None:
        candidates = self._candidate_files(priority_files)
        for rel in candidates:
            info = self._parse_file(rel)
            if info is None:
                continue
            self.files[rel] = info
            for sym in info.symbols:
                self.symbols[sym.symbol_id] = sym
                self.by_name[sym.name].append(sym.symbol_id)
                self.by_name[sym.qualname.split(".")[-1]].append(sym.symbol_id)
        self._build_reverse_edges()

    def _candidate_files(self, priority_files: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for rel in priority_files:
            p = self.repo_dir / rel
            if p.exists() and p.suffix == PY_EXT and rel not in seen:
                seen.add(rel)
                out.append(rel)
        skip_dirs = {
            ".git",
            ".github",
            ".pytest_cache",
            ".tox",
            ".eggs",
            "__pycache__",
            "build",
            "dist",
            "docs",
            "doc",
            "node_modules",
            "venv",
            ".venv",
        }
        for dirpath, dirnames, filenames in os.walk(self.repo_dir):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.endswith(".egg-info")]
            for name in filenames:
                if not name.endswith(PY_EXT):
                    continue
                p = Path(dirpath) / name
                rel = safe_rel(p, self.repo_dir)
                if rel in seen:
                    continue
                seen.add(rel)
                out.append(rel)
                if len(out) >= self.max_files:
                    return out
        return out

    def _parse_file(self, rel: str) -> FileInfo | None:
        path = self.repo_dir / rel
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return None
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        lines = text.splitlines()
        imports: list[str] = []
        import_lines: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.extend(lines[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)])
                imports.extend(import_names(node))
        info = FileInfo(rel_path=rel, text=text, imports_text="\n".join(import_lines), imports=imports)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                continue
            qualname = qualified_name(node, parents)
            src = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            calls = sorted(call_names(node))
            sid = f"{rel}::{qualname}:{node.lineno}-{node.end_lineno}"
            info.symbols.append(
                SymbolRef(
                    symbol_id=sid,
                    file=rel,
                    qualname=qualname,
                    name=node.name,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    kind=type(node).__name__,
                    text=src,
                    token_count=approx_tokens(src),
                    content_signature=sha1_short(src),
                    calls=tuple(calls),
                    imports=tuple(imports),
                )
            )
        return info

    def _build_reverse_edges(self) -> None:
        for sid, sym in self.symbols.items():
            for call in sym.calls:
                for callee_id in self.resolve_call(sym.file, call):
                    if callee_id != sid:
                        self.callers[callee_id].add(sid)

    def resolve_call(self, current_file: str, call: str) -> list[str]:
        short = call.split(".")[-1]
        candidates = self.by_name.get(call, []) + self.by_name.get(short, [])
        if not candidates:
            return []
        same_file = [sid for sid in candidates if self.symbols[sid].file == current_file]
        return same_file or candidates[:5]

    def import_dependency_files(self, rel: str) -> list[str]:
        info = self.files.get(rel)
        if not info:
            return []
        out: list[str] = []
        for name in info.imports:
            mod_path = name.replace(".", "/") + ".py"
            init_path = name.replace(".", "/") + "/__init__.py"
            for candidate in (mod_path, init_path):
                if candidate in self.files and candidate not in out:
                    out.append(candidate)
        return out[:5]


def import_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        prefix = "." * node.level + (node.module or "")
        return [prefix.strip(".")] if prefix.strip(".") else []
    return []


def call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = expr_name(child.func)
        if name:
            names.add(name)
    return names


def expr_name(expr: ast.AST) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        base = expr_name(expr.value)
        return f"{base}.{expr.attr}" if base else expr.attr
    return ""


def qualified_name(target: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names = [getattr(target, "name", "")]
    cur = parents.get(target)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(cur.name)
        cur = parents.get(cur)
    return ".".join(reversed([n for n in names if n]))


def select_target_symbols(index: RepoIndex, rel_path: str, touched_lines: set[int]) -> list[SymbolRef]:
    symbols = index.files.get(rel_path, FileInfo(rel_path, "", "", [])).symbols
    if not symbols:
        return []
    if not touched_lines:
        return symbols[:1]
    hits = [
        sym
        for sym in symbols
        if any(sym.start_line <= line <= sym.end_line for line in touched_lines)
    ]
    if hits:
        return sorted(hits, key=lambda s: (s.end_line - s.start_line, s.start_line))[:3]
    nearest = sorted(
        symbols,
        key=lambda s: min(abs(s.start_line - line) for line in touched_lines),
    )
    return nearest[:1]


def file_prefix_text(index: RepoIndex, rel_path: str, max_lines: int = 160) -> str:
    info = index.files.get(rel_path)
    if not info:
        return ""
    return "\n".join(info.text.splitlines()[:max_lines])


def make_bundle(index: RepoIndex, target: SymbolRef, bundle_type: str, test_files: list[str]) -> tuple[str, list[SymbolRef], list[str], str]:
    symbols = [target]
    extra_texts: list[str] = []
    reason = "target function/method/class only"
    if bundle_type == "call_neighborhood_1hop":
        callee_ids: list[str] = []
        for call in target.calls:
            callee_ids.extend(index.resolve_call(target.file, call))
        for sid in sorted(set(callee_ids)):
            if sid in index.symbols and sid != target.symbol_id:
                symbols.append(index.symbols[sid])
        reason = "target plus statically resolved direct callees"
    elif bundle_type == "reverse_callers_1hop":
        for sid in sorted(index.callers.get(target.symbol_id, set())):
            if sid in index.symbols:
                symbols.append(index.symbols[sid])
        reason = "target plus same-repo callers that reference it"
    elif bundle_type == "import_dependency_bundle":
        info = index.files.get(target.file)
        if info and info.imports_text:
            extra_texts.append(info.imports_text)
        for dep in index.import_dependency_files(target.file):
            dep_info = index.files.get(dep)
            if dep_info and dep_info.imports_text:
                extra_texts.append(f"# imports from {dep}\n{dep_info.imports_text}")
        reason = "target plus local import context and resolved import-file front matter"
    elif bundle_type == "test_target_bundle":
        for tf in test_files[:3]:
            info = index.files.get(tf)
            if not info:
                continue
            test_syms = [s for s in info.symbols if s.name.startswith("test")]
            for sym in test_syms[:3]:
                symbols.append(sym)
        reason = "target plus changed/FAIL_TO_PASS test functions"

    # Preserve deterministic order and avoid duplicate symbols.
    deduped: list[SymbolRef] = []
    seen: set[str] = set()
    for sym in symbols:
        if sym.symbol_id not in seen:
            seen.add(sym.symbol_id)
            deduped.append(sym)
    text = "\n\n".join([s.text for s in deduped] + extra_texts)
    return text, deduped, extra_texts, reason


def build_precision_prompt(
    *,
    bundle_text: str,
    bundle_type: str,
    role: str,
    instance_id: str,
    target_file: str,
    target_symbol: str,
) -> dict:
    user_prompt = (
        f"## SWE Instance\n{instance_id}\n\n"
        f"## Target\nfile: {target_file}\nsymbol: {target_symbol}\n"
        f"bundle_type: {bundle_type}\n\n"
        "## Code Graph Bundle\n"
        "```python\n"
        f"{bundle_text.rstrip()}\n"
        "```\n\n"
        "## Task\n"
        "Reason about whether this exact code bundle is sufficient evidence for a lossy KV reuse decision. "
        "Do not repeat the bundle. Return a compact JSON object with keys: relevant_symbols, missing_context, reuse_risk."
    )
    return {
        "agent_role": role,
        "system_prompt": ROLE_SYSTEM_PROMPTS[role],
        "user_prompt": user_prompt,
    }


def analyze(instances_path: Path, limit: int, max_files_per_repo: int) -> dict:
    instances = json.loads(instances_path.read_text(encoding="utf-8"))[:limit]
    records: list[BundleRecord] = []
    precision_manifest: list[dict] = []
    case_summaries: list[dict] = []
    skipped: Counter[str] = Counter()

    for inst in instances:
        instance_id = inst["instance_id"]
        repo_dir = resolve_repo_dir(instance_id)
        if not repo_dir.exists():
            skipped["missing_repo_dir"] += 1
            continue
        patch_files = parse_patch_files(inst.get("patch", ""))
        test_files = parse_patch_files(inst.get("test_patch", ""))
        py_targets = [p for p in patch_files if p.endswith(PY_EXT)]
        if not py_targets:
            skipped["no_python_patch_target"] += 1
            continue
        priority = list(dict.fromkeys(py_targets + test_files))
        index = RepoIndex(repo_dir, max_files=max_files_per_repo)
        index.build(priority)
        if not index.files:
            skipped["parse_failed"] += 1
            continue
        case_target_count = 0
        for rel in py_targets[:3]:
            touched = changed_new_lines_for_file(inst.get("patch", ""), rel)
            targets = select_target_symbols(index, rel, touched)
            for target in targets:
                prefix_toks = max(1, approx_tokens(file_prefix_text(index, rel)))
                base_toks = max(1, target.token_count)
                for bundle_type in BUNDLE_TYPES:
                    text, symbols, extra_texts, reason = make_bundle(index, target, bundle_type, test_files)
                    if not text.strip():
                        continue
                    files = sorted({s.file for s in symbols})
                    token_count = approx_tokens(text)
                    signature = sha1_short(text)
                    records.append(
                        BundleRecord(
                            instance_id=instance_id,
                            repo=inst.get("repo", ""),
                            target_file=rel,
                            target_symbol=target.qualname,
                            bundle_type=bundle_type,
                            symbol_count=len(symbols),
                            file_count=len(files),
                            token_count=token_count,
                            content_signature=signature,
                            exact_signature_hit_rate=1.0,
                            overlap_with_file_prefix=min(1.0, token_count / prefix_toks),
                            expansion_ratio=token_count / base_toks,
                            includes_imports=bool(extra_texts),
                            includes_tests=any(s.file in test_files for s in symbols),
                            precision_priority=PRECISION_PRIORITY[bundle_type],
                            precision_hypothesis=PRECISION_HYPOTHESES[bundle_type],
                            reason=reason,
                            files=files,
                            symbols=[s.qualname for s in symbols],
                        )
                    )
                    for role in ROLE_SYSTEM_PROMPTS:
                        prompt = build_precision_prompt(
                            bundle_text=text,
                            bundle_type=bundle_type,
                            role=role,
                            instance_id=instance_id,
                            target_file=rel,
                            target_symbol=target.qualname,
                        )
                        precision_manifest.append(
                            {
                                "variant_id": f"{instance_id}::{target.symbol_id}::{bundle_type}::{role}",
                                "instance_id": instance_id,
                                "repo": inst.get("repo", ""),
                                "target_file": rel,
                                "target_symbol": target.qualname,
                                "bundle_type": bundle_type,
                                "content_signature": signature,
                                "token_count": token_count,
                                "precision_priority": PRECISION_PRIORITY[bundle_type],
                                "precision_hypothesis": PRECISION_HYPOTHESES[bundle_type],
                                "files": files,
                                "symbols": [s.qualname for s in symbols],
                                "bundle_text": text,
                                **prompt,
                            }
                        )
                case_target_count += 1
        case_summaries.append(
            {
                "instance_id": instance_id,
                "repo": inst.get("repo", ""),
                "patch_files": py_targets,
                "test_files": test_files,
                "targets": case_target_count,
                "indexed_files": len(index.files),
                "indexed_symbols": len(index.symbols),
            }
        )

    summary = summarize_records(records, case_summaries, skipped)
    return {
        "schema_version": "code_graph_bundle_census.v1",
        "instances_path": str(instances_path),
        "limit": limit,
        "bundle_types": list(BUNDLE_TYPES),
        "summary": summary,
        "cases": case_summaries,
        "records": [asdict(r) for r in records],
        "precision_manifest_rows": len(precision_manifest),
        "precision_manifest": precision_manifest,
        "skipped": dict(skipped),
    }


def resolve_repo_dir(instance_id: str) -> Path:
    for root in (LOCAL_ENV_REPOS, REPO_ROOT):
        candidate = root / instance_id
        if candidate.exists():
            return candidate
    return LOCAL_ENV_REPOS / instance_id


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize_records(records: list[BundleRecord], cases: list[dict], skipped: Counter[str]) -> dict:
    by_type: dict[str, list[BundleRecord]] = defaultdict(list)
    for row in records:
        by_type[row.bundle_type].append(row)
    bundle_summary = {}
    for bundle_type in BUNDLE_TYPES:
        rows = by_type[bundle_type]
        toks = [r.token_count for r in rows]
        expansions = [r.expansion_ratio for r in rows]
        bundle_summary[bundle_type] = {
            "n": len(rows),
            "mean_tokens": mean(toks),
            "p50_tokens": percentile(toks, 0.5),
            "p90_tokens": percentile(toks, 0.9),
            "mean_expansion_ratio": mean(expansions),
            "p90_expansion_ratio": percentile(expansions, 0.9),
            "mean_file_count": mean([r.file_count for r in rows]),
            "mean_symbol_count": mean([r.symbol_count for r in rows]),
            "imports_rate": mean([1.0 if r.includes_imports else 0.0 for r in rows]),
            "tests_rate": mean([1.0 if r.includes_tests else 0.0 for r in rows]),
            "exact_signature_hit_rate": mean([r.exact_signature_hit_rate for r in rows]),
            "mean_overlap_with_file_prefix": mean([r.overlap_with_file_prefix for r in rows]),
        }
    return {
        "cases_analyzed": len(cases),
        "targets_analyzed": sum(c["targets"] for c in cases),
        "records": len(records),
        "skipped": dict(skipped),
        "by_bundle_type": bundle_summary,
    }


def write_outputs(payload: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    census_payload = dict(payload)
    precision_manifest = census_payload.pop("precision_manifest", [])
    (DATA / "code_graph_bundle_census.json").write_text(
        json.dumps(census_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (DATA / "code_graph_precision_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in precision_manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (DATA / "code_graph_bundle_table.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instance_id",
            "repo",
            "target_file",
            "target_symbol",
            "bundle_type",
            "symbol_count",
            "file_count",
            "token_count",
            "content_signature",
            "exact_signature_hit_rate",
            "overlap_with_file_prefix",
            "expansion_ratio",
            "includes_imports",
            "includes_tests",
            "precision_priority",
            "precision_hypothesis",
            "reason",
            "files",
            "symbols",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["records"]:
            out = dict(row)
            out["files"] = ";".join(row["files"])
            out["symbols"] = ";".join(row["symbols"])
            writer.writerow(out)
    draw_figures(payload)
    write_report(payload)


def draw_figures(payload: dict) -> None:
    by_type = payload["summary"]["by_bundle_type"]
    labels = list(BUNDLE_TYPES)
    mean_tokens = [by_type[k]["mean_tokens"] for k in labels]
    p90_tokens = [by_type[k]["p90_tokens"] for k in labels]
    expansion = [by_type[k]["mean_expansion_ratio"] for k in labels]
    symbol_count = [by_type[k]["mean_symbol_count"] for k in labels]

    rates = [
        by_type[k]["imports_rate"] if "import" in k else by_type[k]["tests_rate"] if "test" in k else by_type[k]["mean_overlap_with_file_prefix"]
        for k in labels
    ]
    draw_grouped_bar_png(
        FIG / "fig_code_graph_bundle_scope.png",
        "Code-graph bundle scope covariate",
        labels,
        [("mean tokens", mean_tokens, (54, 104, 141)), ("p90 tokens", p90_tokens, (216, 132, 50))],
    )
    draw_scatter_png(
        FIG / "fig_code_graph_precision_design_space.png",
        "Lossy-reuse precision design space",
        "Expansion ratio vs target",
        "Mean scope tokens",
        labels,
        expansion,
        mean_tokens,
        symbol_count,
    )
    draw_grouped_bar_png(
        FIG / "fig_code_graph_bundle_diagnostics.png",
        "Code-graph precision diagnostic rates",
        labels,
        [("rate / overlap", rates, (91, 126, 170))],
        ymax=1.05,
    )


class Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.px = bytearray(bg * (width * height))

    def _idx(self, x: int, y: int) -> int:
        return (y * self.width + x) * 3

    def point(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = self._idx(x, y)
            self.px[i : i + 3] = bytes(color)

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        for yy in range(max(0, y), min(self.height, y + h)):
            start = self._idx(max(0, x), yy)
            end = self._idx(min(self.width, x + w), yy)
            self.px[start:end] = bytes(color) * max(0, min(self.width, x + w) - max(0, x))

    def line(self, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            self.point(x1, y1, color)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def circle(self, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.point(x, y, color)

    def save_png(self, path: Path) -> None:
        rows = []
        stride = self.width * 3
        for y in range(self.height):
            rows.append(b"\x00" + bytes(self.px[y * stride : (y + 1) * stride]))
        raw = b"".join(rows)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(raw, 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "a": ["010", "101", "111", "101", "101"],
    "b": ["110", "101", "110", "101", "110"],
    "c": ["011", "100", "100", "100", "011"],
    "d": ["110", "101", "101", "101", "110"],
    "e": ["111", "100", "110", "100", "111"],
    "f": ["111", "100", "110", "100", "100"],
    "g": ["011", "100", "101", "101", "011"],
    "h": ["101", "101", "111", "101", "101"],
    "i": ["111", "010", "010", "010", "111"],
    "j": ["001", "001", "001", "101", "010"],
    "k": ["101", "101", "110", "101", "101"],
    "l": ["100", "100", "100", "100", "111"],
    "m": ["101", "111", "111", "101", "101"],
    "n": ["110", "101", "101", "101", "101"],
    "o": ["010", "101", "101", "101", "010"],
    "p": ["110", "101", "110", "100", "100"],
    "q": ["010", "101", "101", "111", "011"],
    "r": ["110", "101", "110", "101", "101"],
    "s": ["011", "100", "010", "001", "110"],
    "t": ["111", "010", "010", "010", "010"],
    "u": ["101", "101", "101", "101", "111"],
    "v": ["101", "101", "101", "101", "010"],
    "w": ["101", "101", "111", "111", "101"],
    "x": ["101", "101", "010", "101", "101"],
    "y": ["101", "101", "010", "010", "010"],
    "z": ["111", "001", "010", "100", "111"],
    "_": ["000", "000", "000", "000", "111"],
    "-": ["000", "000", "111", "000", "000"],
    ".": ["000", "000", "000", "000", "010"],
    "/": ["001", "001", "010", "100", "100"],
    ":": ["000", "010", "000", "010", "000"],
    " ": ["000", "000", "000", "000", "000"],
}


def draw_text(canvas: Canvas, x: int, y: int, text: str, color: tuple[int, int, int] = (30, 30, 30), scale: int = 2) -> None:
    cursor = x
    for ch in text.lower():
        glyph = FONT.get(ch, FONT.get(" ", []))
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    canvas.rect(cursor + gx * scale, y + gy * scale, scale, scale, color)
        cursor += 4 * scale


def draw_grouped_bar_png(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    ymax: float | None = None,
) -> None:
    c = Canvas(980, 520)
    left, top, right, bottom = 80, 80, 940, 420
    ymax = ymax or max([max(vals or [0]) for _, vals, _ in series] + [1]) * 1.15
    draw_text(c, 80, 25, title[:42], scale=3)
    for i in range(6):
        y = bottom - int((bottom - top) * i / 5)
        c.line(left, y, right, y, (220, 225, 230))
        draw_text(c, 18, y - 7, f"{ymax * i / 5:.0f}", scale=2)
    c.line(left, top, left, bottom, (60, 60, 60))
    c.line(left, bottom, right, bottom, (60, 60, 60))
    group_w = (right - left) / len(labels)
    bar_w = max(10, int(group_w / (len(series) + 2)))
    for gi, label in enumerate(labels):
        center = int(left + group_w * gi + group_w / 2)
        for si, (_, vals, color) in enumerate(series):
            h = int((bottom - top) * vals[gi] / ymax)
            x = center - int((len(series) * bar_w) / 2) + si * bar_w
            c.rect(x, bottom - h, bar_w - 2, h, color)
        draw_text(c, int(left + group_w * gi + 4), bottom + 18, label.replace("_", " ")[:14], scale=1)
    lx = 690
    for name, _, color in series:
        c.rect(lx, 42, 18, 12, color)
        draw_text(c, lx + 26, 40, name[:18], scale=2)
        lx += 150
    c.save_png(path)


def draw_scatter_png(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: list[str],
    xs: list[float],
    ys: list[float],
    sizes: list[float],
) -> None:
    c = Canvas(920, 520)
    left, top, right, bottom = 90, 70, 860, 420
    xmin, xmax = min(xs or [0]), max(xs or [1])
    ymin, ymax = min(ys or [0]), max(ys or [1])
    if math.isclose(xmin, xmax):
        xmax = xmin + 1
    if math.isclose(ymin, ymax):
        ymax = ymin + 1
    xpad, ypad = (xmax - xmin) * 0.12, (ymax - ymin) * 0.12
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = max(0, ymin - ypad), ymax + ypad
    draw_text(c, 82, 24, title[:38], scale=3)
    for i in range(6):
        y = bottom - int((bottom - top) * i / 5)
        c.line(left, y, right, y, (225, 228, 232))
        draw_text(c, 20, y - 7, f"{ymin + (ymax-ymin)*i/5:.0f}", scale=2)
    c.line(left, top, left, bottom, (60, 60, 60))
    c.line(left, bottom, right, bottom, (60, 60, 60))
    draw_text(c, 330, 470, xlabel[:28], scale=2)
    draw_text(c, 10, 46, ylabel[:22], scale=2)
    for label, x, y, size in zip(labels, xs, ys, sizes):
        px = left + int((right - left) * (x - xmin) / (xmax - xmin))
        py = bottom - int((bottom - top) * (y - ymin) / (ymax - ymin))
        c.circle(px, py, max(5, min(16, int(4 + size * 2))), (77, 127, 95))
        draw_text(c, min(px + 10, right - 150), max(top, py - 8), label.replace("_", " ")[:18], scale=1)
    c.save_png(path)


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def write_report(payload: dict) -> None:
    summary = payload["summary"]
    by_type = summary["by_bundle_type"]
    rows = []
    for key in BUNDLE_TYPES:
        row = by_type[key]
        rows.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | `{}` |".format(
                key,
                row["n"],
                fmt(row["mean_tokens"]),
                fmt(row["p90_tokens"]),
                fmt(row["mean_expansion_ratio"], 2),
                fmt(row["mean_symbol_count"], 2),
                fmt(row["exact_signature_hit_rate"], 2),
                PRECISION_PRIORITY[key],
            )
        )
    examples = []
    seen_examples: set[str] = set()
    for row in payload["records"]:
        key = row["bundle_type"]
        if key in seen_examples:
            continue
        seen_examples.add(key)
        examples.append(
            "| `{}` | `{}` | `{}` | `{}` | {} | `{}` |".format(
                row["bundle_type"],
                row["instance_id"],
                row["target_file"],
                row["target_symbol"],
                row["token_count"],
                "; ".join(row["symbols"][:4]),
            )
        )
        if len(examples) >= len(BUNDLE_TYPES):
            break
    md = f"""# Code Graph-Aware Lossy Reuse Precision Study

> 自动生成：`results/code_graph_kv_reuse/code_graph_bundle_analyzer.py`

## 1. 这项实验回答什么

已有 AST 粒度实验回答的是“代码块切多大更稳定”。本实验进一步问：在真实 SWE-style 代码修改中，调用、导入和测试触达关系能否帮助选择 **lossy reuse 精度更高** 的 exact code bundle。

这里的 token 数不是优化目标，只作为 scope covariate 记录。我们后续会靠调度和预取处理执行成本；本贡献主要关心：哪类 code graph bundle 在跨 agent、跨 prompt 位置时 KV 更稳定，输出漂移更小，pass@1 损失更可控。

换句话说，AST 是 span boundary，code graph 是 precision-oriented bundle selection signal，安全 gate 仍然是 exact normalized content signature。

## 2. 工具链

- 解析器：Python 标准库 `ast`，用于函数、方法、类、import 和 call expression 抽取。
- 图构建：轻量静态 call graph。`ast.Call` 的 `Name`/`Attribute` 会解析到同文件优先、再 repo-local 同名 symbol。
- Import resolver：将 `a.b` 映射到 repo 内 `a/b.py` 或 `a/b/__init__.py`，只保留可静态定位的本地依赖。
- Test bundle：从 `test_patch` 和 FAIL_TO_PASS 对应的 Python 测试文件中抽取 `test*` 函数。
- 可选重工具：PyCG、CodeQL、Jedi 目前只作为后续 robustness，不进入第一版必需依赖。

参考关系：PyCG 说明 Python 静态 call graph 的可行性；CodeQL 的 data-flow/call graph 说明 AST 之外的程序关系可以作为代码理解对象；Tree-sitter/Jedi 可在后续扩展到多语言或更强 definition/reference resolution。本实验第一版刻意不用这些重依赖，以保证单机可复现。

## 3. 数据设置

- Manifest：`{payload["instances_path"]}`
- 分析 case：{summary["cases_analyzed"]}
- 目标 symbol：{summary["targets_analyzed"]}
- 派生 bundle 记录：{summary["records"]}
- Precision manifest 行数：{payload["precision_manifest_rows"]}，即每个 bundle 生成 planner/coder/reviewer 三角色 prompt，用于 paired KV distance 和 output-drift 实验。
- Bundle 类型：`ast_function_only`、`call_neighborhood_1hop`、`reverse_callers_1hop`、`import_dependency_bundle`、`test_target_bundle`

## 4. Bundle 定义

- `ast_function_only`：patch 命中的最小函数/方法/类 span。
- `call_neighborhood_1hop`：target 加上静态解析到的直接 callee。
- `reverse_callers_1hop`：target 加上 repo 内直接 caller。
- `import_dependency_bundle`：target 加上本文件 import front matter 和可定位本地 import 文件的 import front matter。
- `test_target_bundle`：target 加上变更测试文件中的 `test*` 函数。

## 5. Precision-first 静态结果

下表中的 token/symbol 不是“越小越好”的结论，而是后续解释精度差异时的控制变量。真正要比较的是下一阶段的 cross-role KV distance、output F1/drift 和 paired pass@1 non-degradation。

| bundle | n | mean scope tokens | p90 scope tokens | scope expansion | mean symbols | exact signature hit | precision priority |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## 6. Bundle 示例

| bundle | case | target file | target symbol | scope tokens | included symbols |
|---|---|---|---|---:|---|
{chr(10).join(examples)}

![Bundle scope covariate](/home/gfy/CodeMAS_Project/sglang-kvflow/results/code_graph_kv_reuse/figures/fig_code_graph_bundle_scope.png)

![Precision design space](/home/gfy/CodeMAS_Project/sglang-kvflow/results/code_graph_kv_reuse/figures/fig_code_graph_precision_design_space.png)

![Bundle diagnostics](/home/gfy/CodeMAS_Project/sglang-kvflow/results/code_graph_kv_reuse/figures/fig_code_graph_bundle_diagnostics.png)

## 7. 论文可用解释

Code graph-aware bundling is not a safety mechanism. It decides which exact code spans should be compared, retained, or prefetched together for lossy reuse. The actual reuse gate remains the normalized content signature and token-level exact match. This distinction lets AgentTemplateKV use code structure to improve precision-oriented candidate selection while preserving exact-content safety.

在论文中可以把它写成一个 design validation：当 function/method 缺少局部上下文时，调用邻域、导入依赖和测试触达关系提供了比盲目扩展到 file prefix 更精细的 lossy-reuse precision 策略。file prefix 可以继续作为高复用稳定前缀候选，但不是本节关注点。

## 8. 下一步实验接口

本报告已经输出 `data/code_graph_bundle_table.csv`，其中每行都有 `bundle_type`、`target_file`、`target_symbol`、`token_count`、`content_signature`、`precision_priority`、`files` 和 `symbols`。同时输出 `data/code_graph_precision_manifest.jsonl`，其中包含 planner/coder/reviewer 三角色 prompt 和完整 bundle 文本。

当前完成的是 P-G0 precision scaffold。这里的 scope tokens、bundle expansion 和 exact signature 只能证明“这些 code graph bundle 可以被定义和追踪”，不能直接证明 KV distance、TTFT 或 accuracy。P-G1/P-G2/P-G3 必须继续用同一 manifest 跑 paired KV distance、output drift 和 pass@1 non-degradation，才能进入精度结果表。

建议优先顺序：先比较 `ast_function_only`、`import_dependency_bundle`、`call_neighborhood_1hop` 三类；`test_target_bundle` 用于 SWE-style output drift；`reverse_callers_1hop` 只作为诊断上界，不作为默认策略。

## 9. 边界

- Python-only；动态 dispatch、monkey patch、反射调用不会被完整解析。
- 当前 call graph 是 conservative locator，不保证语义完整性。
- `exact_signature_hit_rate=1.0` 表示派生 bundle 自身有 exact signature，并不表示线上 cache 一定 device-hit。
"""
    (BASE / "CODE_GRAPH_KV_REUSE_REPORT.md").write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-files-per-repo", type=int, default=80)
    args = parser.parse_args()
    payload = analyze(args.instances, args.limit, args.max_files_per_repo)
    write_outputs(payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
