#!/usr/bin/env python3
"""Offline signal labeler for multi-signal HKVD extension (2026-07-11).

Builds on the triple-falsification finding (Direction A/B/HKVD-by-node-kind):
interface/body was the ONLY AST-axis we tested. This script labels each pool
chunk's tokens with **8 distinct code-structure signals** so the HKVD driver
can measure per-signal KV deviation:

  1. first_use vs reuse             (Name/Attribute first occurrence)
  2. def vs ref                     (Name in Store vs Load context)
  3. control_flow vs data_flow      (AST node type buckets)
  4. import-graph distance          (per-file module_globals BFS)
  5. cyclomatic complexity bucket   (per-enclosing-function: low/med/high)
  6. type complexity                (reuse ChunkSpan.type_complexity bucket)
  7. linter risky                   (ruff subprocess per file, graceful skip)
  8. rare vs common identifier      (corpus-wide Name.id frequency)

Output: <out-dir>/signal_labels_per_chunk.jsonl
        one record per chunk; signals = dict of {signal_name_toks: [...token indices within chunk...]}

Reuses:
  - compute_dataflow_budget.py: collect_module_globals, collect_chunk_uses,
    collect_chunk_local_defs, line_byte_offsets, node_byte_range, _BUILTINS
  - compute_nodekind_budget.py: token_count_to_byte (used to verify, but we
    return TOKEN INDICES directly so this isn't strictly needed)
  - ast_chunker.py: ASTChunker for type_complexity

Does NOT touch sglang runtime. Reads pool manifest + source files.
"""
from __future__ import annotations
import argparse
import ast
import bisect
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
sys.path.insert(0, str(ROOT / "python/sglang/srt/mem_cache"))
sys.path.insert(0, str(ROOT / "results"))

# Reuse utilities from compute_dataflow_budget.py
from compute_dataflow_budget import (  # noqa: E402
    _BUILTINS, collect_module_globals, collect_chunk_local_defs,
    line_byte_offsets, node_byte_range,
)
from ast_chunker import ASTChunker  # noqa: E402


CHUNKER = ASTChunker()
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Type-complexity threshold (R40: 0-10 score). Picked 3 to filter "trivially
# typed" code (single param, no generic) out of "interesting" (≥3 means
# multiple typed params + return type annotation).
TYPE_COMPLEXITY_THRESHOLD = 3

# Cyclomatic complexity buckets (CC = 1 + branches in function body)
CC_LOW_MAX = 3      # CC ≤ 3 → low
CC_MED_MAX = 7      # CC 4-7 → med; CC ≥ 8 → high

# Rare-identifier threshold (corpus frequency below this is "rare")
RARE_ID_FREQ_MAX = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--no-ruff", action="store_true",
                   help="skip ruff-based linter risky signal (auto-detected if ruff absent)")
    return p.parse_args()


def load_tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def tokenize_with_offsets(text: str, tok) -> list[tuple[int, int]]:
    """Return list of (char_start, char_end) per token (mirrors radix_cache
    Encoding-unwrap fix)."""
    out = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    om = out["offset_mapping"] if isinstance(out, dict) else out
    first = om[0] if om else None
    if first is not None and hasattr(first, "offsets"):
        offsets = list(first.offsets)  # unwrap Encoding
    elif first is not None:
        offsets = list(first)
    else:
        offsets = []
    return [(s, e) for s, e in offsets if s is not None and e is not None and s >= 0 and e >= 0]


def bytes_to_toks(bs: int, be: int, offsets: list[tuple[int, int]]) -> list[int]:
    """Token indices whose char span overlaps [bs, be) within chunk_text."""
    if bs < 0 or be <= bs:
        return []
    return [i for i, (s, e) in enumerate(offsets) if s < be and e > bs]


# ---------- Signal extractors (each operates on chunk_text + parsed tree) ----

def extract_first_use_reuse(tree: ast.AST, lbo: list[int],
                            offsets: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """first_use = first occurrence in this chunk; reuse = later occurrences."""
    seen: set[str] = set()
    first_use_ranges: list[tuple[int, int]] = []
    reuse_ranges: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _BUILTINS:
                continue
            bs, be = node_byte_range(node, lbo)
            if bs < 0:
                continue
            (first_use_ranges if node.id not in seen else reuse_ranges).append((bs, be))
            seen.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id not in _BUILTINS:
                bs, be = node_byte_range(node, lbo)
                if bs < 0:
                    continue
                (first_use_ranges if base.id not in seen else reuse_ranges).append((bs, be))
                seen.add(base.id)

    first_toks = sorted({t for r in first_use_ranges for t in bytes_to_toks(*r, offsets)})
    reuse_toks = sorted({t for r in reuse_ranges for t in bytes_to_toks(*r, offsets)})
    return first_toks, reuse_toks


def extract_def_ref(tree: ast.AST, lbo: list[int],
                    offsets: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """def = Name in Store ctx (LHS of assign, for-target, arg); ref = Name in Load ctx."""
    def_ranges: list[tuple[int, int]] = []
    ref_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                bs, be = node_byte_range(node, lbo)
                if bs >= 0:
                    def_ranges.append((bs, be))
            elif isinstance(node.ctx, ast.Load):
                if node.id in _BUILTINS:
                    continue
                bs, be = node_byte_range(node, lbo)
                if bs >= 0:
                    ref_ranges.append((bs, be))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id not in _BUILTINS:
                bs, be = node_byte_range(node, lbo)
                if bs >= 0:
                    ref_ranges.append((bs, be))
    def_toks = sorted({t for r in def_ranges for t in bytes_to_toks(*r, offsets)})
    ref_toks = sorted({t for r in ref_ranges for t in bytes_to_toks(*r, offsets)})
    return def_toks, ref_toks


CONTROL_TYPES = (ast.If, ast.For, ast.While, ast.With, ast.Try,
                 ast.Return, ast.Raise, ast.Yield, ast.Assert,
                 ast.Break, ast.Continue, ast.Pass)
DATA_TYPES = (ast.BinOp, ast.Compare, ast.Constant, ast.Call,
              ast.UnaryOp, ast.BoolOp, ast.JoinedStr, ast.Dict, ast.Set,
              ast.List, ast.Tuple, ast.Subscript, ast.Starred)


def extract_control_data(tree: ast.AST, lbo: list[int],
                         offsets: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """control_flow = control AST nodes; data_flow = data AST nodes (no overlap)."""
    control_ranges: list[tuple[int, int]] = []
    data_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, CONTROL_TYPES):
            bs, be = node_byte_range(node, lbo)
            if bs >= 0:
                control_ranges.append((bs, be))
        elif isinstance(node, DATA_TYPES):
            bs, be = node_byte_range(node, lbo)
            if bs >= 0:
                data_ranges.append((bs, be))
    # Disjoint: drop data ranges that fully lie inside a control range
    control_set = [(bs, be) for bs, be in control_ranges]
    filtered_data = []
    for bs, be in data_ranges:
        inside = any(cbs <= bs and be <= cbe for cbs, cbe in control_set)
        if not inside:
            filtered_data.append((bs, be))
    ctl_toks = sorted({t for r in control_ranges for t in bytes_to_toks(*r, offsets)})
    data_toks = sorted({t for r in filtered_data for t in bytes_to_toks(*r, offsets)})
    return ctl_toks, data_toks


def cyclomatic_complexity(node: ast.AST) -> int:
    """Standard CC = 1 + (number of branching constructs)."""
    cc = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                              ast.IfExp, ast.Match, ast.Assert)):
            cc += 1
        elif isinstance(child, ast.BoolOp):
            cc += len(child.values) - 1
    return cc


def extract_cyclomatic(tree: ast.AST, lbo: list[int],
                       offsets: list[tuple[int, int]]) -> dict[str, list[int]]:
    """For each token, the CC bucket of its enclosing function/class. Buckets:
    low (CC ≤ 3), med (CC 4-7), high (CC ≥ 8)."""
    token_bucket: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            cc = cyclomatic_complexity(node)
            if cc <= CC_LOW_MAX:
                bucket = "low"
            elif cc <= CC_MED_MAX:
                bucket = "med"
            else:
                bucket = "high"
            bs, be = node_byte_range(node, lbo)
            if bs < 0:
                continue
            for t in bytes_to_toks(bs, be, offsets):
                token_bucket[t] = bucket
    return {
        "cyc_low_toks": sorted(t for t, b in token_bucket.items() if b == "low"),
        "cyc_med_toks": sorted(t for t, b in token_bucket.items() if b == "med"),
        "cyc_high_toks": sorted(t for t, b in token_bucket.items() if b == "high"),
    }


def extract_type_complexity(chunk_text: str, chunk_rec: dict,
                           offsets: list[tuple[int, int]]) -> list[int]:
    """Reuse ASTChunker to get type_complexity; mark interface tokens as type_complex
    if complexity ≥ threshold. Empty list if untyped (pandas 0.x typical)."""
    try:
        spans = CHUNKER.chunk_text(chunk_text)
    except Exception:
        return []
    match = next((s for s in spans
                  if s.name == chunk_rec["name"] and s.anchor_type == chunk_rec["anchor_type"]),
                 None)
    if match is None or match.type_complexity < TYPE_COMPLEXITY_THRESHOLD:
        return []
    # Mark the chunk's interface tokens (signature + leading docstring)
    end = match.interface_end_byte or match.signature_end_byte or 0
    if end <= 0:
        return []
    return sorted(bytes_to_toks(0, end, offsets))


def run_ruff_on_file(repo_root: Path, relpath: str) -> set[int]:
    """Return set of byte offsets inside file that ruff flags. Empty if ruff
    unavailable or file missing."""
    fpath = repo_root / relpath
    if not fpath.exists():
        return set()
    try:
        proc = subprocess.run(
            ["ruff", "check", "--output-format=json", "--no-fix", str(fpath)],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return set()
    if proc.returncode not in (0, 1):  # 0 = clean, 1 = issues found
        return set()
    try:
        issues = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return set()
    flagged: set[int] = set()
    for iss in issues:
        loc = iss.get("location", {})
        row = loc.get("row", 0)
        col = loc.get("column", 0)
        if row > 0 and col >= 0:
            flagged.add(row * 100000 + col)
    return flagged


def extract_linter_risky(chunk_text: str, repo_root: Path, relpath: str,
                         lbo: list[int], offsets: list[tuple[int, int]]) -> list[int]:
    """Map ruff-flagged byte positions to token indices within chunk_text."""
    flagged = run_ruff_on_file(repo_root, relpath)
    if not flagged:
        return []
    # Map file-level flagged positions to chunk-text positions
    # For simplicity, map by line: flagged line numbers → chunk-text line
    flagged_lines = {pos // 100000 for pos in flagged}
    risky_toks: set[int] = set()
    for line_idx in flagged_lines:
        if line_idx - 1 >= len(lbo):
            continue
        bs = lbo[line_idx - 1]
        be = lbo[line_idx] if line_idx < len(lbo) else bs + 200
        risky_toks.update(bytes_to_toks(bs, be, offsets))
    return sorted(risky_toks)


def extract_import_graph_dist(chunk_text: str, chunk_tree: ast.AST,
                              module_globals: set[str], local_defs: set[str],
                              lbo: list[int],
                              offsets: list[tuple[int, int]]) -> dict[str, list[int]]:
    """Bucket tokens by import-graph distance of the symbol they reference.

    dist 0 = local-only (name defined in this chunk or builtin)
    dist 1 = cross-chunk reference (name defined elsewhere in same file)
    dist 2+ = external / unresolvable

    For pandas pool, most refs fall into dist 1 (cross-chunk module_globals).
    """
    dist_ranges: dict[str, list[tuple[int, int]]] = {
        "import_dist_0_toks": [],
        "import_dist_1_toks": [],
        "import_dist_2plus_toks": [],
    }
    for node in ast.walk(chunk_tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _BUILTINS or node.id in local_defs:
                bucket = "import_dist_0_toks"
            elif node.id in module_globals:
                bucket = "import_dist_1_toks"
            else:
                bucket = "import_dist_2plus_toks"
            bs, be = node_byte_range(node, lbo)
            if bs >= 0:
                dist_ranges[bucket].append((bs, be))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id not in _BUILTINS:
                if base.id in local_defs:
                    bucket = "import_dist_0_toks"
                elif base.id in module_globals:
                    bucket = "import_dist_1_toks"
                else:
                    bucket = "import_dist_2plus_toks"
                bs, be = node_byte_range(node, lbo)
                if bs >= 0:
                    dist_ranges[bucket].append((bs, be))
    return {k: sorted({t for r in v for t in bytes_to_toks(*r, offsets)})
            for k, v in dist_ranges.items()}


def extract_rare_common(chunk_tree: ast.AST, lbo: list[int],
                        offsets: list[tuple[int, int]],
                        corpus_freq: Counter) -> dict[str, list[int]]:
    """rare = Name.id appearing < RARE_ID_FREQ_MAX times across full corpus.
    common = ≥ RARE_ID_FREQ_MAX."""
    rare_ranges: list[tuple[int, int]] = []
    common_ranges: list[tuple[int, int]] = []
    for node in ast.walk(chunk_tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _BUILTINS:
                continue
            bs, be = node_byte_range(node, lbo)
            if bs < 0:
                continue
            if corpus_freq.get(node.id, 0) < RARE_ID_FREQ_MAX:
                rare_ranges.append((bs, be))
            else:
                common_ranges.append((bs, be))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id not in _BUILTINS:
                bs, be = node_byte_range(node, lbo)
                if bs < 0:
                    continue
                if corpus_freq.get(base.id, 0) < RARE_ID_FREQ_MAX:
                    rare_ranges.append((bs, be))
                else:
                    common_ranges.append((bs, be))
    return {
        "rare_id_toks": sorted({t for r in rare_ranges for t in bytes_to_toks(*r, offsets)}),
        "common_id_toks": sorted({t for r in common_ranges for t in bytes_to_toks(*r, offsets)}),
    }


# ---------- Corpus-level passes (run once over all chunks) -------------------

def build_corpus_passes(recs: list[dict], repo_root: Path) -> tuple[
    dict[str, set[str]],  # module_globals per slot_id (file-level)
    Counter,               # corpus-wide Name.id frequency
]:
    """Two passes over the corpus:
    Pass 1: parse each source file once, collect module_globals per file.
    Pass 2: walk every chunk, count Name.id occurrences.

    Returns (per-file module_globals, corpus name frequency).
    """
    # Group by source file
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_file[r["slot_id"]].append(r)

    module_globals_per_file: dict[str, set[str]] = {}
    file_text_cache: dict[str, str] = {}

    for slot_id, recs_in_file in by_file.items():
        relpath = slot_id.split(":", 1)[1]
        fpath = repo_root / relpath
        if not fpath.exists():
            continue
        if slot_id not in file_text_cache:
            file_text_cache[slot_id] = fpath.read_text()
            try:
                mod_tree = ast.parse(file_text_cache[slot_id])
                module_globals_per_file[slot_id] = collect_module_globals(mod_tree)
            except SyntaxError:
                module_globals_per_file[slot_id] = set()

    # Corpus-wide name frequency
    name_freq: Counter = Counter()
    for slot_id, ftext in file_text_cache.items():
        try:
            tree = ast.parse(ftext)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in _BUILTINS:
                    name_freq[node.id] += 1
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                base = node
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name) and base.id not in _BUILTINS:
                    name_freq[base.id] += 1

    return module_globals_per_file, name_freq


# ---------- Per-chunk main loop ----------------------------------------------

def main():
    args = parse_args()
    print(f"[labels] loading tokenizer {args.model} ...", flush=True)
    tok = load_tokenizer(args.model)

    use_ruff = not args.no_ruff and shutil.which("ruff") is not None
    if not use_ruff:
        print(f"[labels] ruff not available; skipping linter-risky signal",
              flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    print(f"[labels] {len(recs)} chunks in manifest", flush=True)

    print(f"[labels] building corpus-level passes (module_globals + name_freq) ...",
          flush=True)
    module_globals_per_file, corpus_freq = build_corpus_passes(recs, args.repo_root)
    print(f"[labels] {len(module_globals_per_file)} files indexed, "
          f"{len(corpus_freq)} unique names, "
          f"{sum(1 for v in corpus_freq.values() if v < RARE_ID_FREQ_MAX)} rare "
          f"(<{RARE_ID_FREQ_MAX} occ)", flush=True)

    # Group by file (reuse same logic as compute_dataflow_budget)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_file[r["slot_id"]].append(r)

    out_records: list[dict[str, Any]] = []
    n = 0
    for slot_id, recs_in_file in by_file.items():
        relpath = slot_id.split(":", 1)[1]
        fpath = args.repo_root / relpath
        if not fpath.exists():
            for r in recs_in_file:
                out_records.append({
                    "slot_id": slot_id, "name": r["name"],
                    "anchor_type": r["anchor_type"], "n_tokens": r["n_tokens"],
                    "signals": {},
                })
            continue
        ftext = fpath.read_text()
        module_globals = module_globals_per_file.get(slot_id, set())

        for r in recs_in_file:
            n += 1
            if n % 20 == 0:
                print(f"  ... {n}/{len(recs)}", flush=True)
            ct = ftext[r["byte_start"]:r["byte_end"]]
            rec: dict[str, Any] = {
                "slot_id": slot_id, "name": r["name"],
                "anchor_type": r["anchor_type"], "n_tokens": r["n_tokens"],
            }
            if not ct.strip():
                rec["signals"] = {}
                out_records.append(rec)
                continue
            try:
                tree = ast.parse(ct)
            except SyntaxError:
                rec["signals"] = {}
                out_records.append(rec)
                continue

            offsets = tokenize_with_offsets(ct, tok)
            lbo = line_byte_offsets(ct)
            local_defs = collect_chunk_local_defs(tree)
            signals: dict[str, list[int]] = {}

            # Signal 1: first_use vs reuse
            fu, ru = extract_first_use_reuse(tree, lbo, offsets)
            signals["first_use_toks"] = fu
            signals["reuse_toks"] = ru

            # Signal 2: def vs ref
            df, rf = extract_def_ref(tree, lbo, offsets)
            signals["def_toks"] = df
            signals["ref_toks"] = rf

            # Signal 3: control_flow vs data_flow
            cf, daf = extract_control_data(tree, lbo, offsets)
            signals["control_flow_toks"] = cf
            signals["data_flow_toks"] = daf

            # Signal 4: import-graph distance
            igd = extract_import_graph_dist(ct, tree, module_globals, local_defs, lbo, offsets)
            signals.update(igd)

            # Signal 5: cyclomatic complexity bucket
            cyc = extract_cyclomatic(tree, lbo, offsets)
            signals.update(cyc)

            # Signal 6: type complexity (reuse ChunkSpan)
            signals["type_complexity_toks"] = extract_type_complexity(ct, r, offsets)

            # Signal 7: linter risky (ruff) — graceful skip
            if use_ruff:
                signals["linter_risky_toks"] = extract_linter_risky(
                    ct, args.repo_root, relpath, lbo, offsets,
                )
            else:
                signals["linter_risky_toks"] = []

            # Signal 8: rare vs common identifier
            rare_common = extract_rare_common(tree, lbo, offsets, corpus_freq)
            signals.update(rare_common)

            rec["signals"] = signals
            out_records.append(rec)

    out_path = args.out_dir / "signal_labels_per_chunk.jsonl"
    out_path.write_text("\n".join(json.dumps(r) for r in out_records) + "\n")
    print(f"\n[labels] wrote {out_path}  ({len(out_records)} chunks)")

    # Quick summary: per-signal fire rate + mean token count
    signal_keys = [
        "first_use_toks", "reuse_toks",
        "def_toks", "ref_toks",
        "control_flow_toks", "data_flow_toks",
        "import_dist_0_toks", "import_dist_1_toks", "import_dist_2plus_toks",
        "cyc_low_toks", "cyc_med_toks", "cyc_high_toks",
        "type_complexity_toks",
        "linter_risky_toks",
        "rare_id_toks", "common_id_toks",
    ]
    print("\n=== signal fire rates + mean token counts ===")
    print(f"{'signal':<24} {'fire_rate':>10} {'mean_toks':>10} {'max_toks':>10}")
    for k in signal_keys:
        hits = [r["signals"].get(k, []) for r in out_records if "signals" in r]
        non_empty = [v for v in hits if v]
        if not hits:
            continue
        fr = len(non_empty) / len(hits)
        mean_n = sum(len(v) for v in non_empty) / len(non_empty) if non_empty else 0
        max_n = max((len(v) for v in non_empty), default=0)
        print(f"{k:<24} {fr*100:>9.1f}% {mean_n:>10.1f} {max_n:>10}")

    summary_path = args.out_dir / "signal_labels_summary.json"
    summary = {
        "n_chunks": len(out_records),
        "n_files_indexed": len(module_globals_per_file),
        "n_unique_names": len(corpus_freq),
        "n_rare_names": sum(1 for v in corpus_freq.values() if v < RARE_ID_FREQ_MAX),
        "ruff_used": use_ruff,
        "per_signal": {
            k: {
                "fire_rate": (sum(1 for r in out_records
                                  if r.get("signals", {}).get(k)) / len(out_records)
                               if out_records else 0),
                "mean_toks": (sum(len(r["signals"].get(k, []))
                                   for r in out_records) / len(out_records)
                              if out_records else 0),
            }
            for k in signal_keys
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[labels] wrote {summary_path}")


if __name__ == "__main__":
    main()