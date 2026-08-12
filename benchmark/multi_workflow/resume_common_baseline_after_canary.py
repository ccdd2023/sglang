#!/usr/bin/env python3
"""Resume the common native campaign after a quarantined Canary4 arm.

The original monitor intentionally does not overwrite artifacts.  When one
completed Slurm job is later shown to contain a backend infrastructure failure,
this resumable monitor records a replacement job ID, revalidates every Canary4
arm, and continues with formal images, Fresh24, and exact-token replay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark.multi_workflow import monitor_common_baseline_campaign as base


def save(path: Path, state: dict) -> None:
    state["updated_at_utc"] = base.utc_now()
    base.atomic_json(path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement-job", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    home = Path.home()
    project = home / "CodeMAS_Project/worktrees/sglang-common-agent"
    campaign = (
        home
        / "CodeMAS_Project/kvflow-artifacts/impactkv_common_agent_baselines_fresh24_20260812"
    )
    logs = home / "impactkv-runtime/logs/common-baselines"
    status_path = campaign / "AUTOMATED_CAMPAIGN_STATUS.json"
    state = base.read_json(status_path)

    try:
        if state.get("state") in {"canary4_submitted", "canary4_recovery_required"}:
            state["jobs"]["canary_kvcomm_reuse_all"] = args.replacement_job
            state["replaced_invalid_job"] = "74238"
            state["replacement_reason"] = (
                "KVCOMM cross-task GPU state retention plus eager-attention OOM"
            )
            state["state"] = "canary4_recovery_submitted"
            state["active_jobs"] = ["canary_kvcomm_reuse_all"]
            save(status_path, state)

        if state["state"] == "canary4_recovery_submitted":
            base.wait_jobs(state, list(state["active_jobs"]), args.poll_seconds)
            passed, reason = base.validate_native_runs(campaign, "canary", None)
            if not passed:
                raise RuntimeError(reason)
            state["canary4_gate"] = reason
            name = "formal_images"
            state["jobs"][name] = base.submit(
                script=(
                    project
                    / "benchmark/multi_workflow/slurm/common_agent_prepare_images.sbatch"
                ),
                logs=logs,
                exports={"IMPACTKV_COMMON_SCOPE": "formal"},
                dependency=args.replacement_job,
            )
            state["active_jobs"] = [name]
            state["state"] = "formal_images_submitted"
            save(status_path, state)

        if state["state"] == "formal_images_submitted":
            base.wait_jobs(state, list(state["active_jobs"]), args.poll_seconds)
            names = base.submit_native_stage(
                state,
                project=project,
                logs=logs,
                scope="formal",
                instance=None,
            )
            state["active_jobs"] = names
            state["state"] = "fresh24_submitted"
            save(status_path, state)

        if state["state"] == "fresh24_submitted":
            base.wait_jobs(state, list(state["active_jobs"]), args.poll_seconds)
            passed, reason = base.validate_native_runs(campaign, "formal", None)
            if not passed:
                raise RuntimeError(reason)
            state["fresh24_gate"] = reason
            source_ledger = (
                campaign / "runs/formal/cacheblend_dense/all/BACKEND_LEDGER.jsonl"
            )
            dependency = state["jobs"][state["active_jobs"][-1]]
            names = []
            for backend in ("cacheblend", "kvcomm"):
                name = f"fresh24_exact_{backend}"
                state["jobs"][name] = base.submit(
                    script=(
                        project
                        / "benchmark/multi_workflow/slurm/common_exact_prompt_replay.sbatch"
                    ),
                    logs=logs,
                    exports={
                        "IMPACTKV_COMMON_BACKEND": backend,
                        "IMPACTKV_COMMON_SOURCE_LEDGER": str(source_ledger),
                        "IMPACTKV_COMMON_REPLAY_LABEL": "fresh24",
                        "IMPACTKV_COMMON_REPLAY_LIMIT": "16",
                    },
                    dependency=dependency,
                )
                names.append(name)
                dependency = state["jobs"][name]
            state["active_jobs"] = names
            state["state"] = "fresh24_exact_submitted"
            save(status_path, state)

        if state["state"] == "fresh24_exact_submitted":
            base.wait_jobs(state, list(state["active_jobs"]), args.poll_seconds)
            for backend in ("cacheblend", "kvcomm"):
                result = base.read_json(
                    campaign / f"exact_prompt_replay/fresh24/{backend}/RESULT.json"
                )
                if result.get("status") != "PASS":
                    raise RuntimeError(
                        f"formal exact replay failed for {backend}: {result}"
                    )
            base.run(
                [
                    sys.executable,
                    str(
                        project
                        / "benchmark/multi_workflow/summarize_common_baseline_campaign.py"
                    ),
                    "--campaign",
                    str(campaign),
                ]
            )
            state["state"] = "complete"
            state["finished_at_utc"] = base.utc_now()
            save(status_path, state)
    except Exception as error:
        state["state"] = "blocked"
        state["error"] = f"{type(error).__name__}: {error}"
        save(status_path, state)
        raise


if __name__ == "__main__":
    main()
