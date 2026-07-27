#!/usr/bin/env python3
"""Exact-cache control for the CL1/P6-H output-equality guardrail.

The guardrail compares a full dense prefill against a request that only has to
prefill the final prompt token. This control removes approximate KV entirely:
the second arm is served by an ordinary exact radix cache hit, so its KV is
bitwise identical to a dense prefill by construction.

If the two arms still disagree, the guardrail is measuring prefill-path
numerical nondeterminism rather than recovery quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.approx_kv.phase6.runner import (
    flush_cache,
    generate,
    launch_server,
    stop_server,
    wait_ready,
)


def token_stats(dense: list[int], other: list[int]) -> dict:
    width = min(len(dense), len(other))
    matched = sum(dense[index] == other[index] for index in range(width))
    prefix = 0
    for index in range(width):
        if dense[index] != other[index]:
            break
        prefix += 1
    return {
        "compared": width,
        "matched": matched,
        "rate": matched / width if width else None,
        "prefix": prefix,
        "first_token_match": bool(width and dense[0] == other[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--port", type=int, default=30013)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunked-prefill-size", type=int, default=1024)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--header-tokens", type=int, default=64)
    parser.add_argument("--body-tokens", default="1024,2048")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    body_values = [int(value) for value in args.body_tokens.split(",")]
    server = launch_server(
        model=args.model,
        model_revision=args.model_revision,
        port=args.port,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=args.chunked_prefill_size,
        policy="lru",
        log_path=args.log,
        plugin_env={},
    )
    rows = []
    try:
        wait_ready(server, port=args.port, timeout_s=600)
        for body_tokens in body_values:
            for round_index in range(args.rounds):
                flush_cache(args.port)
                header = [
                    36_000 + ((round_index * 89 + offset) % 4_000)
                    for offset in range(args.header_tokens)
                ]
                body = [
                    1_000 + ((round_index * 101 + offset) % 30_000)
                    for offset in range(body_tokens)
                ]
                prompt = header + body + [901]

                dense = generate(
                    port=args.port,
                    input_ids=prompt,
                    max_new_tokens=args.max_new_tokens,
                    extra_key=f"ctl-dense-{body_tokens}-{round_index}",
                )

                warm_namespace = f"ctl-exact-{body_tokens}-{round_index}"
                warm = generate(
                    port=args.port,
                    input_ids=prompt[:-1],
                    max_new_tokens=1,
                    extra_key=warm_namespace,
                )
                exact = generate(
                    port=args.port,
                    input_ids=prompt,
                    max_new_tokens=args.max_new_tokens,
                    extra_key=warm_namespace,
                )
                rows.append(
                    {
                        "body_tokens": body_tokens,
                        "round_index": round_index,
                        "dense_cached_tokens": dense["cached_tokens"],
                        "warm_cached_tokens": warm["cached_tokens"],
                        "exact_cached_tokens": exact["cached_tokens"],
                        "dense_output_ids": dense["output_ids"],
                        "exact_output_ids": exact["output_ids"],
                        "comparison": token_stats(
                            dense["output_ids"], exact["output_ids"]
                        ),
                    }
                )
    finally:
        stop_server(server)

    summary = {}
    for body_tokens in body_values:
        selected = [row for row in rows if row["body_tokens"] == body_tokens]
        summary[str(body_tokens)] = {
            "rounds": len(selected),
            "first_token_match_rate": sum(
                row["comparison"]["first_token_match"] for row in selected
            )
            / len(selected),
            "exact_output_match_rate": sum(
                row["dense_output_ids"] == row["exact_output_ids"] for row in selected
            )
            / len(selected),
            "mean_token_consistency_rate": sum(
                row["comparison"]["rate"] for row in selected
            )
            / len(selected),
            "exact_cached_tokens": sorted(
                {row["exact_cached_tokens"] for row in selected}
            ),
        }
    payload = {
        "schema_version": 1,
        "control": "exact_radix_cache_hit_versus_dense_prefill",
        "purpose": (
            "isolate prefill-path numerical nondeterminism from approximate "
            "KV recovery error in the CL1 and P6-H output-equality guardrail"
        ),
        "chunked_prefill_size": args.chunked_prefill_size,
        "header_tokens": args.header_tokens,
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
