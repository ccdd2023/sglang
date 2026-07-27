#!/usr/bin/env python3
"""Poll the approximate-KV store gauges until the server dies.

Answers one question: at the moment an ordinary prefill runs out of device
slots, is the approximate store still holding device bytes that the exact
pressure path should have been able to reclaim?

If it is, make_room failed to release reclaimable memory and the OOM is our
bug. If the store is already empty, the workload genuinely exceeds capacity.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

WATCH = (
    "sglang:approx_kv_store_device_bytes",
    "sglang:approx_kv_store_host_bytes",
    "sglang:approx_kv_store_records",
    "sglang:approx_kv_store_leases",
    "sglang:approx_kv_store_orphans",
    "sglang:cross_store_reservation_failures_total",
    "sglang:cross_store_demoted_bytes_total",
    "sglang:approx_kv_dense_fallback_total",
    "sglang:max_total_num_tokens",
    "sglang:num_used_tokens",
    "sglang:token_usage",
)


def parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0]
        if not name.startswith("sglang:approx_kv") and not name.startswith(
            "sglang:cross_store"
        ):
            name = name.split("{", 1)[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30011)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.4)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    samples = 0
    last_ok: dict[str, float] | None = None
    seen_server = False

    with args.output.open("w", encoding="utf-8") as handle:
        while time.time() - started < args.max_seconds:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{args.port}/metrics", timeout=2
                )
                snapshot = parse_prometheus(response.text)
            except requests.RequestException:
                if seen_server and last_ok is not None:
                    handle.write(
                        json.dumps(
                            {
                                "event": "server_unreachable",
                                "elapsed_s": round(time.time() - started, 2),
                                "last_good_sample": last_ok,
                            }
                        )
                        + "\n"
                    )
                    handle.flush()
                    print("=== A SERVER DIED. LAST GOOD SAMPLE ===")
                    print(json.dumps(last_ok, indent=1, sort_keys=True))
                    # Keep polling: the pilot launches one server per cell, so
                    # later cells would otherwise never be observed.
                    seen_server = False
                    last_ok = None
                time.sleep(args.interval)
                continue

            seen_server = True
            watched = {name: snapshot.get(name) for name in WATCH}
            for key, value in snapshot.items():
                if key.startswith("sglang:approx_kv") or key.startswith(
                    "sglang:cross_store"
                ):
                    watched[key] = value
            if watched != last_ok:
                handle.write(
                    json.dumps(
                        {
                            "elapsed_s": round(time.time() - started, 2),
                            "metrics": watched,
                        }
                    )
                    + "\n"
                )
                handle.flush()
            last_ok = watched
            samples += 1
            time.sleep(args.interval)

    print(f"poller finished without observing a death, samples={samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
