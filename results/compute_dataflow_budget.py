#!/usr/bin/env python3
"""Offline recompute-budget estimator for dataflow-driven head recompute (P1' P0).

Direction B (dataflow): instead of recomputing a uniform fraction (R32) or the
chunk's interface prefix (Direction A), recompute only the tokens that reference
SYMBOLS DEFINED IN ANOTHER CHUNK of the same file. The hypothesis: tokens that
reference cross-chunk globals (e.g. ``np``, ``pd``, ``self.attr``) are the ones
whose KV state is most likely to drift when the upstream definition changes; body
tokens that reference only intra-chunk locals are stable and safe to copy.

This script does NOT modify sglang runtime. It asks the cheap signal question:

    For each chunk C in the pool, what fraction of tokens reference a name
    that is DEFINED in another chunk of the same source file?

If that fraction is meaningfully different from uniform frac (0.15/0.30/0.45)
and from node-kind interface, then per-chunk dataflow is doing real work that
uniform FRAC cannot replicate -> proceed P1. If it's uniform-shaped, the lever
is structurally the same as uniform -> falsify direction B.

Also emits:
  - per-chunk ``cross_use_tokens`` and ``cross_use_frac``
  - per-anchor-type breakdown
  - comparison table vs uniform {0.15, 0.30, 0.45} and vs node-kind interface
  - top files by cross-chunk-call density (signal hotspots)

Methodology (cheap, stdlib-only):
  1. Parse the manifest. For each (slot_id, byte_start, byte_end) read the
     source file from ``--repo-root``.
  2. Build module-level symbol table per file by walking the FULL file AST:
     collect FunctionDef/AsyncFunctionDef/ClassDef/Assign/AnnAssign/Import/
     ImportFrom at module scope (ignoring nested). ``from X import Y`` adds Y.
  3. For each chunk, parse just the chunk's text. Walk its AST collecting
     ``Name`` (Load context) + the BASE ``Name`` of every ``Attribute`` chain
     (Load context). Filter out builtins / dunders. Also collect ``arg`` names
     (formal params) as DEFs for that chunk.
  4. ``cross_uses(C) = {names used in C} - {names defined in C} -
     {names defined in same file but not in pool}``
     i.e. USEs that are visible globals defined OUTSIDE this chunk.
  5. Map each Name/Attribute base to its byte range (lineno/col_offset ->
     byte via line_byte_offsets of the chunk text). Use the same byte->token
     bisect_right approach as ``compute_nodekind_budget.py``.
  6. ``cross_use_tokens(C) = sum of tokens overlapping any cross-use byte range``
     ``cross_use_frac(C) = cross_use_tokens(C) / chunk_token_len``

Output:
  - ``<pool>/dataflow_budget.json`` — aggregate summary
  - ``<pool>/dataflow_budget_per_chunk.jsonl`` — per-chunk detail

Usage:
  python3 results/compute_dataflow_budget.py \\
      --manifest results/codebase_kv/pandas_15case_v1/manifest.jsonl \\
      --repo-root results/giant_codebase/pandas_src \\
      --model Qwen/Qwen2.5-Coder-7B-Instruct
"""
from __future__ import annotations
import argparse
import ast
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/gfy/CodeMAS_Project/sglang-kvflow")
sys.path.insert(0, str(ROOT / "python/sglang/srt/mem_cache"))


# A conservative list of Python builtins / dunders that count as "non-dataflow"
# when they appear as ``Name``. These don't carry cross-chunk info and shouldn't
# inflate the cross-use count.
_BUILTINS = frozenset({
    "True", "False", "None",
    "NotImplemented", "Ellipsis",
    "print", "len", "range", "enumerate", "zip", "map", "filter", "reversed",
    "list", "dict", "set", "tuple", "frozenset", "str", "int", "float", "bool",
    "bytes", "bytearray", "memoryview",
    "isinstance", "issubclass", "type", "callable", "hasattr", "getattr",
    "setattr", "delattr", "super", "object", "classmethod", "staticmethod",
    "property", "len", "abs", "min", "max", "sum", "round", "pow", "divmod",
    "sorted", "any", "all",
    "open", "iter", "next", "repr", "hash", "id", "dir", "vars", "globals",
    "locals", "exec", "eval", "compile",
    "__name__", "__file__", "__doc__", "__dict__", "__class__", "__bases__",
    "__init__", "__repr__", "__str__", "__len__", "__iter__", "__next__",
    "__call__", "__getitem__", "__setitem__", "__delitem__", "__contains__",
    "__enter__", "__exit__", "__add__", "__sub__", "__mul__", "__truediv__",
    "__floordiv__", "__mod__", "__pow__", "__neg__", "__pos__", "__abs__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__hash__",
    "__bool__", "__nonzero__", "__new__", "__del__", "__getattr__",
    "__setattr__", "__delattr__", "__getattribute__", "__reduce__",
    "__reduce_ex__", "__getstate__", "__setstate__", "__slots__", "__all__",
    "__annotations__", "__module__", "__qualname__", "__wrapped__",
    "__missing__", "__length_hint__", "__aiter__", "__anext__", "__await__",
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--out-dir", default=None, type=Path,
                   help="output dir (default: manifest dir)")
    return p.parse_args()


def load_tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def token_count_to_byte(text: str, tok, byte_pos: int) -> int:
    """Mirror RadixCache._lookup_byte_offset: number of tokens whose END byte
    <= byte_pos = bisect_right on sorted token end offsets."""
    out = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = out["offset_mapping"] if isinstance(out, dict) else out
    if hasattr(offsets, "data") and "offset_mapping" in getattr(offsets, "data", {}):
        offsets = offsets["offset_mapping"]
    if offsets and hasattr(offsets[0], "offsets"):
        offsets = offsets[0].offsets
    elif offsets and not isinstance(offsets[0], tuple):
        offsets = list(offsets[0])
    ends = sorted(en for (_st, en) in offsets if en is not None and en >= 0)
    return bisect.bisect_right(ends, byte_pos)


def line_byte_offsets(text: str) -> list[int]:
    """Return byte offset of each line's start in text (per splitlines())."""
    out = []
    running = 0
    for line in text.splitlines():
        out.append(running)
        running += len(line) + 1
    return out


def node_byte_range(node: ast.AST, lbo: list[int]) -> tuple[int, int]:
    """Return (byte_start, byte_end) of an AST node using line/col offsets."""
    ln = getattr(node, "lineno", 0) or 0
    cn = getattr(node, "col_offset", 0) or 0
    eln = getattr(node, "end_lineno", ln) or ln
    ecn = getattr(node, "end_col_offset", cn + 1) or (cn + 1)
    if not (0 < ln <= len(lbo)):
        return (-1, -1)
    bs = lbo[ln - 1] + cn
    be = lbo[min(eln - 1, len(lbo) - 1)] + ecn if 0 < eln <= len(lbo) else bs + 1
    return (bs, be)


def collect_module_globals(tree: ast.Module) -> set[str]:
    """Return names defined at module scope (FunctionDef/ClassDef/Assign LHS/
    AnnAssign LHS/Import/ImportFrom-as). Nested defs are not module globals."""
    out: set[str] = set()
    for node in getattr(tree, "body", []) or []:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out.add(tgt.id)
                elif isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            out.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.add(alias.asname or alias.name)
    return out


def collect_chunk_uses(tree: ast.AST) -> set[str]:
    """Return the set of bare names USED (load context) in this chunk,
    EXCLUDING builtins/dunders. Includes Name(Load) and the BASE Name of
    Attribute(Load) chains."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in _BUILTINS:
                out.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id not in _BUILTINS:
                out.add(base.id)
    return out


def collect_chunk_local_defs(tree: ast.AST) -> set[str]:
    """Return names defined anywhere in this chunk (formal params + nested
    defs/assigns) — used to subtract LOCAL DEFs from the cross-use set."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out.add(tgt.id)
                elif isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            out.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
        elif isinstance(node, ast.arguments):
            for a in (list(node.args) + list(node.kwonlyargs) +
                      list(node.posonlyargs)):
                out.add(a.arg)
            if node.vararg:
                out.add(node.vararg.arg)
            if node.kwarg:
                out.add(node.kwarg.arg)
    return out


def collect_use_byte_ranges(tree: ast.AST, lbo: list[int],
                            cross_uses: set[str]) -> list[tuple[int, int]]:
    """For each Name/Attribute-base in cross_uses, return its byte range within
    the chunk text. A chunk's byte ranges are relative to chunk start (=0)."""
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in cross_uses:
                bs, be = node_byte_range(node, lbo)
                if bs >= 0:
                    ranges.append((bs, be))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in cross_uses:
                # Attribute covers ``base.attr1.attr2...``; mark the WHOLE chain
                # as dataflow-affected (any token inside is sensitive to base's
                # binding change).
                bs, be = node_byte_range(node, lbo)
                if bs >= 0:
                    ranges.append((bs, be))
    return ranges


def merge_byte_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort + merge overlapping ranges so each byte is counted at most once."""
    if not ranges:
        return []
    s = sorted(ranges)
    merged = [s[0]]
    for st, en in s[1:]:
        if st <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], en))
        else:
            merged.append((st, en))
    return merged


def cross_use_tokens(byte_ranges: list[tuple[int, int]], text: str,
                     tok, chunk_len: int) -> int:
    """Count unique tokens (in chunk-text coordinates) covered by any byte range
    in byte_ranges. Uses bisect_right on token end offsets."""
    if not byte_ranges or chunk_len <= 0:
        return 0
    out = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = out["offset_mapping"] if isinstance(out, dict) else out
    if hasattr(offsets, "data") and "offset_mapping" in getattr(offsets, "data", {}):
        offsets = offsets["offset_mapping"]
    if offsets and hasattr(offsets[0], "offsets"):
        offsets = offsets[0].offsets
    elif offsets and not isinstance(offsets[0], tuple):
        offsets = list(offsets[0])
    # token starts
    starts = sorted(st for (st, _en) in offsets if st is not None and st >= 0)
    n = 0
    seen_idx: set[int] = set()
    for st, en in byte_ranges:
        # tokens overlapping [st, en): token.start < en and token.end > st
        lo = bisect.bisect_left(starts, st)
        hi = bisect.bisect_right(starts, en)
        # But we have end offsets too — refine. For cheap estimate, count tokens
        # whose START byte is in [st, en]; safer count is via end offsets.
        # Use both:
        idx = lo
        for j, (s_byte, e_byte) in enumerate(offsets):
            if s_byte is None or e_byte is None:
                continue
            if s_byte >= en:
                break
            if e_byte <= st:
                continue
            if j not in seen_idx:
                seen_idx.add(j)
                n += 1
    return n


def main():
    args = parse_args()
    print(f"[dataflow] loading tokenizer {args.model} ...", flush=True)
    tok = load_tokenizer(args.model)

    out_dir = args.out_dir or args.manifest.parent

    recs = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    print(f"[dataflow] {len(recs)} chunks in manifest", flush=True)

    # Group by source file -> parse module AST once
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_file[r["slot_id"]].append(r)

    per_chunk = []
    per_type = defaultdict(lambda: {"n": 0, "cross_tok": 0, "chunk_len": 0,
                                    "n_with_cross": 0})
    file_cross_density = []  # for hotspot identification

    # Skip chunks whose source file is missing (mirror compute_nodekind_budget)
    file_text_cache: dict[str, str] = {}
    file_module_globals_cache: dict[str, set[str]] = {}

    n_processed = 0
    for slot_id, recs_in_file in by_file.items():
        relpath = slot_id.split(":", 1)[1]
        fpath = args.repo_root / relpath
        if not fpath.exists():
            print(f"[dataflow] WARN missing {fpath}; skipping {len(recs_in_file)}",
                  flush=True)
            continue
        if slot_id not in file_text_cache:
            file_text_cache[slot_id] = fpath.read_text()
            try:
                mod_tree = ast.parse(file_text_cache[slot_id])
                file_module_globals_cache[slot_id] = collect_module_globals(mod_tree)
            except SyntaxError:
                file_module_globals_cache[slot_id] = set()
        ftext = file_text_cache[slot_id]
        module_globals = file_module_globals_cache[slot_id]

        file_cross_tokens = 0
        file_total_tokens = 0

        for r in recs_in_file:
            n_processed += 1
            if n_processed % 20 == 0:
                print(f"  ... {n_processed}/{len(recs)}", flush=True)
            ct = ftext[r["byte_start"]:r["byte_end"]]
            if not ct.strip():
                per_chunk.append({
                    "slot_id": slot_id, "name": r["name"],
                    "anchor_type": r["anchor_type"], "n_tokens": r["n_tokens"],
                    "cross_use_tokens": 0, "cross_use_frac": 0.0,
                    "n_use_sites": 0, "n_use_names": 0,
                })
                continue
            try:
                tree = ast.parse(ct)
            except SyntaxError:
                per_chunk.append({
                    "slot_id": slot_id, "name": r["name"],
                    "anchor_type": r["anchor_type"], "n_tokens": r["n_tokens"],
                    "cross_use_tokens": 0, "cross_use_frac": 0.0,
                    "n_use_sites": 0, "n_use_names": 0,
                })
                continue
            uses = collect_chunk_uses(tree)
            local_defs = collect_chunk_local_defs(tree)
            # cross_uses = uses that are module-level globals (defined elsewhere)
            # AND not locally redefined in this chunk
            cross_names = (uses & module_globals) - local_defs
            lbo = line_byte_offsets(ct)
            byte_ranges = collect_use_byte_ranges(tree, lbo, cross_names)
            byte_ranges = merge_byte_ranges(byte_ranges)
            cross_tok = cross_use_tokens(byte_ranges, ct, tok, r["n_tokens"])

            per_chunk.append({
                "slot_id": slot_id, "name": r["name"],
                "anchor_type": r["anchor_type"], "n_tokens": r["n_tokens"],
                "cross_use_tokens": cross_tok,
                "cross_use_frac": (cross_tok / r["n_tokens"]
                                   if r["n_tokens"] > 0 else 0.0),
                "n_use_sites": len(byte_ranges),
                "n_use_names": len(cross_names),
            })

            t = r["anchor_type"]
            per_type[t]["n"] += 1
            per_type[t]["cross_tok"] += cross_tok
            per_type[t]["chunk_len"] += r["n_tokens"]
            if cross_tok > 0:
                per_type[t]["n_with_cross"] += 1
            file_cross_tokens += cross_tok
            file_total_tokens += r["n_tokens"]

        if file_total_tokens > 0:
            file_cross_density.append({
                "slot_id": slot_id,
                "relpath": relpath,
                "n_chunks": len(recs_in_file),
                "cross_frac": file_cross_tokens / file_total_tokens,
                "cross_tok": file_cross_tokens,
                "total_tok": file_total_tokens,
            })

    # ---- Aggregate ----------------------------------------------------------
    n_chunks = len(per_chunk)
    n_with_cross = sum(1 for c in per_chunk if c["cross_use_tokens"] > 0)
    total_cross = sum(c["cross_use_tokens"] for c in per_chunk)
    total_len = sum(c["n_tokens"] for c in per_chunk)
    overall_frac = total_cross / total_len if total_len else 0.0

    # Per-anchor-type
    per_type_summary = {}
    for t, d in per_type.items():
        per_type_summary[t] = {
            "n": d["n"],
            "n_with_cross": d["n_with_cross"],
            "cross_fire_rate": (d["n_with_cross"] / d["n"] if d["n"] else 0.0),
            "cross_tok": d["cross_tok"],
            "chunk_len": d["chunk_len"],
            "cross_frac": (d["cross_tok"] / d["chunk_len"]
                           if d["chunk_len"] else 0.0),
        }

    # Distribution of per-chunk cross_use_frac
    fracs = sorted(c["cross_use_frac"] for c in per_chunk if c["n_tokens"] > 0)
    if fracs:
        def pct(p):
            if not fracs:
                return 0.0
            k = max(0, min(len(fracs) - 1, int(round(p / 100.0 * (len(fracs) - 1)))))
            return fracs[k]
        dist = {
            "min": fracs[0], "p10": pct(10), "p25": pct(25), "median": pct(50),
            "p75": pct(75), "p90": pct(90), "p95": pct(95), "max": fracs[-1],
            "mean": sum(fracs) / len(fracs),
        }
    else:
        dist = {}

    # Compare to baselines
    baselines = {
        "uniform_frac_0.15": 0.15,
        "uniform_frac_0.30": 0.30,
        "uniform_frac_0.45": 0.45,
        "nodekind_interface": 0.2610192837465565,  # from existing budget
        "nodekind_signature": 0.10618214227839896,
    }
    # B at each config = baseline * total_len
    b_compare = {
        name: round(frac * total_len)
        for name, frac in baselines.items()
    }
    b_compare["dataflow_actual"] = total_cross

    # Top hotspots: files with highest cross_use_frac
    hotspots = sorted(file_cross_density,
                      key=lambda x: -x["cross_frac"])[:10]

    summary = {
        "manifest": str(args.manifest),
        "model": args.model,
        "n_chunks": n_chunks,
        "n_chunks_with_cross": n_with_cross,
        "cross_fire_rate": n_with_cross / n_chunks if n_chunks else 0.0,
        "total_cross_tokens": total_cross,
        "total_chunk_tokens": total_len,
        "dataflow_overall_frac": overall_frac,
        "frac_distribution": dist,
        "per_anchor_type": per_type_summary,
        "B_vs_baselines": b_compare,
        "dataflow_divergence_from_uniform_0.30": abs(overall_frac - 0.30),
        "dataflow_divergence_from_nodekind_iface": abs(overall_frac - 0.261),
        "top_hotspot_files": hotspots,
    }

    print("\n=== dataflow recompute budget (P1' P0 cheap signal) ===")
    print(f"chunks: {n_chunks}  (with cross-use: {n_with_cross}, "
          f"fire rate {summary['cross_fire_rate']*100:.1f}%)")
    print(f"B_dataflow = {total_cross}  /  total chunk_len = {total_len}")
    print(f"dataflow overall FRAC = {overall_frac:.4f}")
    print(f"  -> vs uniform 0.15: |d|={abs(overall_frac - 0.15):.4f}  "
          f"B_at_uniform_0.15={round(0.15 * total_len)}")
    print(f"  -> vs uniform 0.30: |d|={abs(overall_frac - 0.30):.4f}  "
          f"B_at_uniform_0.30={round(0.30 * total_len)}")
    print(f"  -> vs uniform 0.45: |d|={abs(overall_frac - 0.45):.4f}  "
          f"B_at_uniform_0.45={round(0.45 * total_len)}")
    print(f"  -> vs node-kind iface (0.261): |d|={abs(overall_frac - 0.261):.4f}  "
          f"B_at_nodekind_iface={round(0.261 * total_len)}")
    print("\nper-anchor-type:")
    print(f"  {'type':>10} {'n':>5} {'fire%':>6} {'cross_tok':>10} "
          f"{'chunk_len':>10} {'frac':>7}")
    for t, d in sorted(per_type_summary.items()):
        print(f"  {t:>10} {d['n']:>5} {d['cross_fire_rate']*100:>5.1f}% "
              f"{d['cross_tok']:>10.0f} {d['chunk_len']:>10.0f} "
              f"{d['cross_frac']:>7.4f}")
    print("\nfrac distribution (per chunk):")
    for k, v in dist.items():
        print(f"  {k:>8} = {v:.4f}" if isinstance(v, float) else f"  {k:>8} = {v}")
    print("\ntop hotspot files:")
    for h in hotspots:
        print(f"  {h['relpath']:>60}  n_chunks={h['n_chunks']:>3}  "
              f"cross_frac={h['cross_frac']:.4f}")

    # Cheap-signal verdict (the user-facing summary)
    print("\n=== P0 cheap-signal verdict ===")
    signal_strength = abs(overall_frac - 0.30) + abs(overall_frac - 0.261)
    # Heuristic: if dataflow FRAC is within 0.05 of BOTH uniform 0.30 AND
    # node-kind interface, the dataflow lever is structurally similar to
    # what we already have -> falsify. Otherwise proceed to P1.
    if abs(overall_frac - 0.30) < 0.05 and abs(overall_frac - 0.261) < 0.05:
        verdict = "WEAK_SIGNAL"
        reason = ("dataflow overall FRAC within 0.05 of both uniform 0.30 "
                  "AND node-kind interface — lever structurally similar to "
                  "what we already tested. Per-chunk distribution may differ "
                  "(inspect dist['std']) but the BUDGET is not novel.")
    elif n_with_cross / max(n_chunks, 1) < 0.5:
        verdict = "PARTIAL"
        reason = (f"only {n_with_cross}/{n_chunks} chunks have any cross-use "
                  "tokens — most chunks are local-only, lever has limited "
                  "headroom")
    else:
        verdict = "STRONG_SIGNAL"
        reason = ("dataflow FRAC meaningfully different from uniform and "
                  "node-kind, and most chunks have cross-uses -> proceed P1")
    print(f"verdict: {verdict}")
    print(f"reason:  {reason}")

    summary["verdict"] = verdict
    summary["verdict_reason"] = reason

    summary_path = out_dir / "dataflow_budget.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    detail_path = out_dir / "dataflow_budget_per_chunk.jsonl"
    detail_path.write_text(
        "\n".join(json.dumps(c) for c in per_chunk) + "\n"
    )
    print(f"\n[dataflow] wrote {summary_path}")
    print(f"[dataflow] wrote {detail_path}")


if __name__ == "__main__":
    main()