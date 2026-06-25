#!/usr/bin/env python3
"""Aggregate 8-case pass@1 (forceevict reretest) into a passrate_table.csv.

This synthesises the same CSV schema as
results/swe_generated_patch_kvcomm/qwen2_5_7b_json_30/passrate_table.csv
so the existing table_passrate() and fig_passrate_main.pdf renderer in
paper/scripts/generate_paper_figures.py can consume it without changes.

The 8-case run is the 100-case pass@1 expansion's 8-case discriminative
subset (the only cases from the 100-case manifest that built gold pass
+ base smoke fail). The reretest logs are the post-fix verified pass/fail
results — i.e. the 0/8 number that the paper reports in §sec:passrate.

Inputs (read):
  - ../qwen2_5_7b_json_8_forceevict/summary.json
      (8 results, 3 modes each; synth/apply/test_rc/cached/elapsed)
  - ./*.log  (18 reretest logs ending in {"returncode": N})

Output (written):
  - ./passrate_table.csv
      (16 rows: 8 cases × 2 modes; columns match 28-case schema)

Usage:
  python3 aggregate_passrate.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
SUMMARY = PARENT / "qwen2_5_7b_json_8_forceevict" / "summary.json"
OUT = HERE / "passrate_table.csv"

# Map log filename suffix -> mode name (must match summary.json mode strings)
LOG_SUFFIX_TO_MODE = {
    "lossless": "lossless",
    "lossy": "lossy",
    # lossy_prefetch is a 3rd mode in the 8-case run; the 28-case table only
    # carries lossless + lossy, so we drop lossy_prefetch rows from the CSV.
}


def parse_returncode(log_path: Path) -> int | None:
    """Pull the trailing {"returncode": N} JSON object from a reretest log."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r'"returncode"\s*:\s*(-?\d+)', text)
    if not matches:
        return None
    return int(matches[-1])


def derive_apply_clean(apply_rc: int | None, synth_ok: bool) -> bool:
    """apply_clean: True iff the patch applied cleanly (rc==0).

    Mirrors the 28-case semantics: if synth failed, apply_clean is False.
    """
    if not synth_ok:
        return False
    return apply_rc == 0


def main() -> int:
    if not SUMMARY.exists():
        print(f"ERROR: missing {SUMMARY}", file=sys.stderr)
        return 1
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    instances = summary.get("results", [])

    # Pre-scan all reretest logs into {(case, mode): returncode}.
    rc_map: dict[tuple[str, str], int | None] = {}
    for log in sorted(HERE.glob("*.log")):
        stem = log.stem  # e.g. django__django-11149_lossless
        for suffix, mode in LOG_SUFFIX_TO_MODE.items():
            if stem.endswith("_" + suffix):
                case = stem[: -(len(suffix) + 1)]
                rc_map[(case, mode)] = parse_returncode(log)
                break

    rows: list[dict[str, str]] = []
    for inst in instances:
        case = inst["instance_id"]
        repo = inst["repo"]
        for mode_entry in inst.get("modes", []):
            mode = mode_entry["mode"]
            if mode == "lossy_prefetch":
                # Skip — the 28-case schema is lossless + lossy only.
                continue
            synth = mode_entry.get("patch_synthesis", {}) or {}
            apply = mode_entry.get("apply_check", {}) or {}
            diff_extracted = bool(mode_entry.get("diff_extracted"))
            synth_ok = bool(synth.get("ok"))
            apply_rc = apply.get("returncode")
            apply_clean = derive_apply_clean(apply_rc, synth_ok)
            cached = mode_entry.get("cached_tokens") or 0
            elapsed = mode_entry.get("elapsed_ms") or 0.0
            repair = mode_entry.get("repair_elapsed_ms")

            # Reretest returncode is the verified pass@1 signal.
            reret_rc = rc_map.get((case, mode))
            if reret_rc is None:
                # No reretest log: either synth failed upstream (django-11138,
                # matplotlib-20676) or reretest wasn't needed (lossy_prefetch).
                # In either case, there is no clean pass@1 signal — leave empty.
                pass1 = ""
                candidate_rc = ""
            else:
                candidate_rc = reret_rc
                # pass1 requires synth + apply + test all clean.
                pass1 = bool(synth_ok and apply_clean and reret_rc == 0)

            match_reason = (
                "exact_code_content_signature" if mode == "lossy" and synth_ok else ""
            )
            reuse_allowed = "True" if mode == "lossy" and synth_ok else ""
            lossy_count = "1" if mode == "lossy" and synth_ok else ""

            rows.append(
                {
                    "instance_id": case,
                    "repo": repo,
                    "mode": mode,
                    "elapsed_ms": f"{elapsed:.2f}",
                    "repair_elapsed_ms": f"{repair:.2f}" if repair is not None else "",
                    "cached_tokens": str(cached),
                    "diff_extracted": str(diff_extracted),
                    "synthesis_ok": str(synth_ok),
                    "apply_clean": str(apply_clean),
                    "pass1": str(pass1) if pass1 != "" else "",
                    "candidate_rc": str(candidate_rc) if candidate_rc != "" else "",
                    "match_reason": match_reason,
                    "reuse_allowed": reuse_allowed,
                    "lossy_candidate_count": lossy_count,
                }
            )

    fieldnames = [
        "instance_id",
        "repo",
        "mode",
        "elapsed_ms",
        "repair_elapsed_ms",
        "cached_tokens",
        "diff_extracted",
        "synthesis_ok",
        "apply_clean",
        "pass1",
        "candidate_rc",
        "match_reason",
        "reuse_allowed",
        "lossy_candidate_count",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")
    # Sanity: count pass1=True
    pass_count = sum(1 for r in rows if r["pass1"] == "True")
    print(f"  pass@1: {pass_count}/{len(rows)} (0/8 expected for forceevict 100-case subset)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
