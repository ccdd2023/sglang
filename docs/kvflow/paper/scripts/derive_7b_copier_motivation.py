#!/usr/bin/env python3
"""PLAN + COMPLETE 137400 outputs: extra spans and one-token disagreement.

No GPU. File-module islands versus unconstrained KVCOMM-style spans.
Decodes extra tokens (tokenizer only) to label tool log vs assistant wrap.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ART = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_7b_sota_copiers_20260824"
)
CODING_REUSE = Path(
    "/home/gfy/CodeMAS_Project/kvflow-artifacts/"
    "impactkv_swebench_7b_file_modules_prefixkey_20260824/reuse.json"
)
TOKENIZER_CANDIDATES = (
    Path("/home/gfy/models/Qwen2.5-Coder-7B-Instruct/tokenizer.json"),
    Path(
        "/home/gfy/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/"
        "c03e6d358207e414f1eca0bb1891e29f1db0e242/tokenizer.json"
    ),
)
SPAN_CLASSES = ("tool_log", "tool_command", "assistant", "chat_glue")
_CLASS_PATTERNS = (
    (
        "tool_log",
        re.compile(
            r"<tool_response>.*?</tool_response>"
            r"|<returncode>\s*-?\d+\s*</returncode>"
            r"|<output>.*?</output>",
            re.S | re.I,
        ),
    ),
    (
        "tool_command",
        re.compile(
            r"<tool_call>.*?</tool_call>"
            r"|<function=.*?</function>"
            r"|<parameter=.*?</parameter>",
            re.S | re.I,
        ),
    ),
    (
        "chat_glue",
        re.compile(
            r"(?:^|\n)(?:assistant|user|system|tool)\n"
            r"|<\|im_start\|>|<\|im_end\|>",
            re.I,
        ),
    ),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_mask(length: int, cases: list[dict[str, Any]]) -> list[bool]:
    mask = [False] * length
    for case in cases:
        start = int(case["target_start"])
        end = start + int(case["length"])
        for index in range(max(0, start), min(length, end)):
            mask[index] = True
    return mask


def extra_spans(
    file_mask: list[bool], clone_mask: list[bool]
) -> list[tuple[int, int, bool]]:
    """KVCOMM-only intervals. Third value is True if the span abuts a file island."""
    spans: list[tuple[int, int, bool]] = []
    index = 0
    n = len(file_mask)
    while index < n:
        if clone_mask[index] and not file_mask[index]:
            end = index
            while end < n and clone_mask[end] and not file_mask[end]:
                end += 1
            left = index > 0 and file_mask[index - 1]
            right = end < n and file_mask[end]
            spans.append((index, end, left or right))
            index = end
        else:
            index += 1
    return spans


def classify_decoded_extra(text: str) -> dict[str, int]:
    """Character coverage. Tool XML first; leftover prose is assistant."""
    n = len(text)
    tagged: list[str | None] = [None] * n
    for name, pattern in _CLASS_PATTERNS:
        for match in pattern.finditer(text):
            for pos in range(match.start(), match.end()):
                if tagged[pos] is None:
                    tagged[pos] = name
    index = 0
    while index < n:
        if tagged[index] is not None:
            index += 1
            continue
        end = index
        while end < n and tagged[end] is None:
            end += 1
        chunk = text[index:end]
        label = "chat_glue" if re.fullmatch(r"[\s>]*", chunk or "") else "assistant"
        for pos in range(index, end):
            tagged[pos] = label
        index = end
    counts = Counter(tagged)
    return {name: int(counts.get(name, 0)) for name in SPAN_CLASSES}


def tokens_from_char_counts(
    n_tokens: int, char_counts: dict[str, int]
) -> dict[str, int]:
    """Largest-remainder allocation of n_tokens across SPAN_CLASSES."""
    out = {name: 0 for name in SPAN_CLASSES}
    if n_tokens <= 0:
        return out
    total = sum(int(char_counts.get(name, 0)) for name in SPAN_CLASSES)
    if total <= 0:
        out["chat_glue"] = n_tokens
        return out
    raw = {
        name: n_tokens * int(char_counts.get(name, 0)) / total for name in SPAN_CLASSES
    }
    out = {name: int(raw[name]) for name in SPAN_CLASSES}
    remainder = n_tokens - sum(out.values())
    order = sorted(
        SPAN_CLASSES,
        key=lambda name: (raw[name] - out[name], SPAN_CLASSES.index(name)),
        reverse=True,
    )
    for name in order:
        if remainder == 0:
            break
        out[name] += 1
        remainder -= 1
    return out


def analyze_plans(
    coding_plan: dict[str, Any], kvcomm_plan: dict[str, Any]
) -> dict[str, Any]:
    coding = {int(row["group_index"]): row for row in coding_plan["groups"]}
    kvcomm = {int(row["group_index"]): row for row in kvcomm_plan["groups"]}
    if set(coding) != set(kvcomm):
        raise ValueError("coding/KVCOMM group sets differ")
    extra = 0
    shared = 0
    coding_only = 0
    kvcomm_tokens = 0
    coding_tokens = 0
    extra_groups = 0
    for index in coding:
        target_len = len(coding[index]["target_input_ids"])
        if target_len != len(kvcomm[index]["target_input_ids"]):
            raise ValueError(f"target length drifted group {index}")
        file_mask = coverage_mask(target_len, coding[index]["cases"])
        clone_mask = coverage_mask(target_len, kvcomm[index]["cases"])
        file_n = sum(file_mask)
        clone_n = sum(clone_mask)
        both = sum(a and b for a, b in zip(file_mask, clone_mask))
        extra_n = clone_n - both
        coding_tokens += file_n
        kvcomm_tokens += clone_n
        shared += both
        extra += extra_n
        coding_only += file_n - both
        if extra_n:
            extra_groups += 1
    return {
        "groups": len(coding),
        "file_module_copied_tokens": coding_tokens,
        "kvcomm_copied_tokens": kvcomm_tokens,
        "shared_copied_tokens": shared,
        "kvcomm_extra_tokens": extra,
        "file_only_tokens": coding_only,
        "groups_with_kvcomm_extra": extra_groups,
    }


def group_extra_series(
    coding_plan: dict[str, Any], kvcomm_plan: dict[str, Any]
) -> dict[str, list[int]]:
    """Per-group file / KVCOMM / extra tokens. Sorted by extra. No GPU."""
    coding = {int(row["group_index"]): row for row in coding_plan["groups"]}
    kvcomm = {int(row["group_index"]): row for row in kvcomm_plan["groups"]}
    if set(coding) != set(kvcomm):
        raise ValueError("coding/KVCOMM group sets differ")
    rows: list[tuple[int, int, int]] = []
    for index in coding:
        target_len = len(coding[index]["target_input_ids"])
        if target_len != len(kvcomm[index]["target_input_ids"]):
            raise ValueError(f"target length drifted group {index}")
        file_mask = coverage_mask(target_len, coding[index]["cases"])
        clone_mask = coverage_mask(target_len, kvcomm[index]["cases"])
        file_n = sum(file_mask)
        clone_n = sum(clone_mask)
        both = sum(a and b for a, b in zip(file_mask, clone_mask))
        rows.append((clone_n - both, file_n, clone_n))
    rows.sort()
    return {
        "extra": [row[0] for row in rows],
        "file": [row[1] for row in rows],
        "kvcomm": [row[2] for row in rows],
    }


def measured_pairs(arm: dict[str, Any]) -> dict[tuple[int, int], str]:
    return {
        (int(row["group_index"]), int(row["round_index"])): str(
            row.get("output_text") or ""
        )
        for row in arm["targets"]
        if not row["warmup"]
    }


def analyze_agreement(
    dense: dict[str, Any],
    coding: dict[str, Any],
    kvcomm: dict[str, Any],
    cacheblend: dict[str, Any],
) -> dict[str, Any]:
    d = measured_pairs(dense)
    c = measured_pairs(coding)
    k = measured_pairs(kvcomm)
    b = measured_pairs(cacheblend)
    if set(d) != set(c) or set(d) != set(k) or set(d) != set(b):
        raise ValueError("paired targets differ across copier arms")
    n = len(d)
    coding_ok = sum(d[key] == c[key] for key in d)
    kvcomm_ok = sum(d[key] == k[key] for key in d)
    blend_ok = sum(d[key] == b[key] for key in d)
    file_saves = sum(d[key] == c[key] != k[key] for key in d)
    kvcomm_saves = sum(d[key] == k[key] != c[key] for key in d)
    file_saves_blend = sum(d[key] == c[key] != b[key] for key in d)
    return {
        "pairs": n,
        "file_module_agrees": coding_ok,
        "kvcomm_agrees": kvcomm_ok,
        "cacheblend_agrees": blend_ok,
        "file_agrees_kvcomm_differs": file_saves,
        "kvcomm_agrees_file_differs": kvcomm_saves,
        "file_agrees_cacheblend_differs": file_saves_blend,
        "not_accuracy": True,
    }


def _tally_extra(
    coding: dict[int, dict[str, Any]],
    kvcomm: dict[int, dict[str, Any]],
    group_indices: list[int],
    decode,
) -> dict[str, Any]:
    extra = 0
    adjacent = 0
    disjoint = 0
    by_class: Counter[str] = Counter()
    for group_index in group_indices:
        row = coding[group_index]
        clone = kvcomm[group_index]
        ids = [int(tok) for tok in row["target_input_ids"]]
        file_mask = coverage_mask(len(ids), row["cases"])
        clone_mask = coverage_mask(len(ids), clone["cases"])
        for start, end, abuts in extra_spans(file_mask, clone_mask):
            length = end - start
            extra += length
            if abuts:
                adjacent += length
            else:
                disjoint += length
            text = decode(ids[start:end])
            by_class.update(tokens_from_char_counts(length, classify_decoded_extra(text)))
    class_counts = {name: int(by_class.get(name, 0)) for name in SPAN_CLASSES}
    if sum(class_counts.values()) != extra:
        raise ValueError("span-type token allocation drifted from extra length")
    return {
        "groups": len(group_indices),
        "extra_tokens": extra,
        "adjacent_to_file_island": adjacent,
        "disjoint_from_file_island": disjoint,
        "by_class": class_counts,
    }


def load_tokenizer():
    from tokenizers import Tokenizer

    for path in TOKENIZER_CANDIDATES:
        if path.exists():
            return Tokenizer.from_file(str(path)), str(path)
    raise FileNotFoundError("Qwen2.5-Coder-7B-Instruct tokenizer.json")


def analyze_span_types(
    coding_plan: dict[str, Any],
    kvcomm_plan: dict[str, Any],
    dense: dict[str, Any],
    coding: dict[str, Any],
    kvcomm: dict[str, Any],
    tokenizer_path: str | None = None,
) -> dict[str, Any]:
    tokenizer, used_path = (
        (__import__("tokenizers").Tokenizer.from_file(tokenizer_path), tokenizer_path)
        if tokenizer_path
        else load_tokenizer()
    )
    coding_groups = {int(row["group_index"]): row for row in coding_plan["groups"]}
    kvcomm_groups = {int(row["group_index"]): row for row in kvcomm_plan["groups"]}
    d = measured_pairs(dense)
    c = measured_pairs(coding)
    k = measured_pairs(kvcomm)
    file_save_keys = [key for key in d if d[key] == c[key] != k[key]]
    file_save_groups = sorted({group for group, _ in file_save_keys})
    rounds_per_group = Counter(group for group, _ in file_save_keys)
    decode = tokenizer.decode
    campaign = _tally_extra(
        coding_groups, kvcomm_groups, sorted(coding_groups), decode
    )
    subset = _tally_extra(coding_groups, kvcomm_groups, file_save_groups, decode)
    subset["pairs"] = len(file_save_keys)
    subset["all_three_measured_rounds"] = all(
        rounds_per_group[group] == 3 for group in file_save_groups
    )
    return {
        "status": "DERIVED_FROM_FROZEN_137400",
        "not_a_new_gpu_arm": True,
        "not_accuracy": True,
        "not_admitted_repository_code": True,
        "tokenizer": used_path,
        "method": (
            "decode KVCOMM-only extra spans; XML/role regex; leftover prose "
            "counts as assistant; token counts proportional to character coverage"
        ),
        "campaign": campaign,
        "file_agrees_kvcomm_differs": subset,
    }


def analyze(art: Path = ART, coding_reuse: Path = CODING_REUSE) -> dict[str, Any]:
    coding_plan = read_json(art / "PLAN.coding.json")
    kvcomm_plan = read_json(art / "PLAN.kvcomm.json")
    dense = read_json(art / "dense.json")
    coding = read_json(coding_reuse)
    kvcomm = read_json(art / "kvcomm.json")
    cacheblend = read_json(art / "cacheblend.json")
    spans = analyze_plans(coding_plan, kvcomm_plan)
    agreement = analyze_agreement(dense, coding, kvcomm, cacheblend)
    span_types = analyze_span_types(
        coding_plan, kvcomm_plan, dense, coding, kvcomm
    )
    if span_types["campaign"]["extra_tokens"] != spans["kvcomm_extra_tokens"]:
        raise ValueError("span-type extra tokens drifted from PLAN coverage")
    return {
        "schema_version": 1,
        "status": "DERIVED_FROM_FROZEN_137400",
        "not_a_new_gpu_arm": True,
        "not_native_cacheblend_or_kvcomm_stack": True,
        "spans": spans,
        "agreement": agreement,
        "span_types": span_types,
    }


def main() -> None:
    value = analyze()
    (ART / "COPIER_MOTIVATION.json").write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
