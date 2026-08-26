"""Mode helpers for dual-island prefix + lossy copy + template prefetch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

ARM = "coding_natural_code_cost"
# Isolated speed vs Dense: prefix_only and lossy_only.
# dual = both reuse algorithms, prefetch off.
# combined = dual + template prefetch (legacy device-resident dual).
# Ablation with a fair host-resident copy baseline:
#   lossy_host → prefix_prefetch → template_prefetch.
MODES = (
    "dense",
    "prefix_only",
    "lossy_only",
    "dual",
    "combined",
    "lossy_host",
    "prefix_prefetch",
    "template_prefetch",
)
ABLATION_MODES = ("dense", "lossy_host", "prefix_prefetch", "template_prefetch")
MODEL_7B = "Qwen2.5-Coder-7B-Instruct"
ROPE_7B = {"rotary_dim": 128, "base": 1_000_000, "is_neox_style": True}
ROPE_30B = {"rotary_dim": 128, "base": 10_000_000, "is_neox_style": True}


def rope_for_model(model: str) -> dict[str, object]:
    if "Qwen2.5-Coder-7B" in str(model):
        return dict(ROPE_7B)
    return dict(ROPE_30B)


def parse_modes(raw: str) -> list[str]:
    modes = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise ValueError(f"unknown prefetch modes: {unknown}")
    if not modes:
        raise ValueError("empty prefetch modes")
    return modes


def arm_json(artifact: Path, mode: str) -> Path:
    return artifact / f"{mode}.json"


def ledger_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    def count(event: str) -> int:
        return sum(row.get("event") == event for row in rows)

    return {
        "copy_events": count("target_copied"),
        "fallback_events": count("target_fallback"),
        "prefetch_submitted": count("template_prefetch_submitted"),
        "spill_events": count("source_spilled_host"),
        "prefetch_miss": count("template_prefetch_miss"),
        "ordinary_prefix_matched": sum(
            row.get("event") == "target_ordinary_prefix_matched"
            and int(row.get("ordinary_prefix_tokens") or 0) > 0
            for row in rows
        ),
        "middle_left_dense": count("target_middle_left_dense"),
        "source_prerotation_events": sum(
            int(row.get("applied_pre_rotate_delta") or 0) != 0 for row in rows
        ),
    }


def mode_env(mode: str) -> dict[str, str]:
    if mode == "dense":
        return {
            "SGLANG_KVCOMM_CORE": "0",
            "SGLANG_CODING_AWARE_LOSSY": "0",
            "SGLANG_KV_PREFETCH": "0",
        }
    if mode == "prefix_only":
        return {
            "SGLANG_KVCOMM_CORE": "1",
            "SGLANG_CODING_AWARE_LOSSY": "1",
            "SGLANG_KV_PREFETCH": "0",
        }
    if mode == "lossy_only":
        return {
            "SGLANG_KVCOMM_CORE": "1",
            "SGLANG_CODING_AWARE_LOSSY": "1",
            "SGLANG_KV_PREFETCH": "0",
        }
    if mode == "dual":
        return {
            "SGLANG_KVCOMM_CORE": "1",
            "SGLANG_CODING_AWARE_LOSSY": "1",
            "SGLANG_KV_PREFETCH": "0",
        }
    if mode == "combined":
        return {
            "SGLANG_KVCOMM_CORE": "1",
            "SGLANG_CODING_AWARE_LOSSY": "1",
            "SGLANG_KV_PREFETCH": "1",
            "SGLANG_KV_PREFETCH_MIDDLE": "1",
        }
    if mode == "lossy_host":
        return {
            "SGLANG_KVCOMM_CORE": "1",
            "SGLANG_CODING_AWARE_LOSSY": "1",
            "SGLANG_KV_PREFETCH": "0",
            "SGLANG_KV_PREFETCH_MIDDLE": "0",
        }
    if mode == "prefix_prefetch":
        return {
            "SGLANG_KVCOMM_CORE": "1",
            "SGLANG_CODING_AWARE_LOSSY": "1",
            "SGLANG_KV_PREFETCH": "1",
            "SGLANG_KV_PREFETCH_MIDDLE": "0",
        }
    if mode == "template_prefetch":
        return {
            "SGLANG_KVCOMM_CORE": "1",
            "SGLANG_CODING_AWARE_LOSSY": "1",
            "SGLANG_KV_PREFETCH": "1",
            "SGLANG_KV_PREFETCH_MIDDLE": "1",
        }
    raise ValueError(mode)


def mode_manifest(
    output: Path, group: dict[str, Any], model: str, mode: str
) -> dict[str, Any]:
    cases = list(group.get("cases") or [])
    if mode == "prefix_only":
        # Keep attach/prefix staging; skip the shifted island copy.
        cases = [
            {**row, "reuse_enabled": True, "copy_middle": False} for row in cases
        ]
    prefetch_on = mode in {"combined", "prefix_prefetch", "template_prefetch"}
    prefix_reuse = mode in {
        "prefix_only",
        "dual",
        "combined",
        "prefix_prefetch",
        "template_prefetch",
    }
    # Fair host baseline for the staircase: copy from host unless prefetched.
    host_overlap = mode in {
        "combined",
        "lossy_host",
        "prefix_prefetch",
        "template_prefetch",
    }
    prefetch_middle = mode in {"combined", "template_prefetch"}
    server_arm = "dense" if mode == "dense" else "reuse"
    rope = ROPE_7B if MODEL_7B in model else {
        "rotary_dim": 128,
        "base": 10_000_000,
        "is_neox_style": True,
    }
    return {
        "version": 3,
        "model_id": model,
        "cache_dtype": "bfloat16",
        "lease_ttl_s": 900,
        "ledger_path": str(output / f"server/{server_arm}/SERVER_LEDGER.jsonl"),
        "rope": rope,
        "sources": group.get("sources") or [],
        "cases": cases,
        "release_source_ids": [],
        "arm": ARM,
        "host_overflow_enabled": True,
        "prefer_host_sources": host_overlap,
        "ordinary_prefix_reuse_enabled": prefix_reuse,
        "ordinary_prefix_repair_tokens": 0,
        "ordinary_prefix_target_only": False,
        "prefetch_spill_device": prefetch_on,
        "prefetch_deadline_s": 30.0,
        "prefetch_wait_s": 90.0,
        "prefetch_middle": prefetch_middle,
    }


def staircase_increments(latency: dict[str, dict[str, float]]) -> dict[str, float]:
    """Prefetch staircase vs the host-resident lossy copy baseline."""
    lossy = float(latency["lossy_host"]["cache_ready_speedup_ratio_of_means"])
    prefix = float(latency["prefix_prefetch"]["cache_ready_speedup_ratio_of_means"])
    templ = float(latency["template_prefetch"]["cache_ready_speedup_ratio_of_means"])
    if lossy <= 0 or prefix <= 0:
        raise ValueError("staircase baselines must be positive")
    return {
        "lossy_vs_dense": lossy,
        "prefix_prefetch_vs_dense": prefix,
        "template_prefetch_vs_dense": templ,
        "prefix_prefetch_vs_lossy": prefix / lossy,
        "template_prefetch_vs_prefix": templ / prefix,
        "template_prefetch_vs_lossy": templ / lossy,
    }


def combined_vs_coding_speedup(latency: dict[str, dict[str, float]]) -> float:
    """Prefetch increment: combined vs dual (copy+prefix, prefetch off)."""
    baseline_key = "dual" if "dual" in latency else "coding"
    coding = float(latency[baseline_key]["cache_ready_speedup_ratio_of_means"])
    combined = float(latency["combined"]["cache_ready_speedup_ratio_of_means"])
    if coding <= 0:
        raise ValueError("coding speedup must be positive")
    return combined / coding
