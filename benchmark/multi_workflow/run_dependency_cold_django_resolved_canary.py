#!/usr/bin/env python3
"""Second outcome-selected resolved mechanism canary (Django q14)."""

from pathlib import Path

from benchmark.multi_workflow import (
    run_dependency_cold_resolved_mechanism_canary as mechanism,
)


ARTIFACTS = Path("/home/gfy/CodeMAS_Project/kvflow-artifacts")
mechanism.SOURCE_ROOT = (
    ARTIFACTS / "impactkv_natural_code_cost_agent_20260808/online/dense/full_9"
)
mechanism.SOURCE_SNAPSHOT = (
    ARTIFACTS / "impactkv_natural_code_cost_agent_20260808/FROZEN_FRESH9.json"
)
mechanism.DEFAULT_OUTPUT = (
    ARTIFACTS
    / "impactkv_dependency_cold_resolved_mechanism_20260810/django13343_q14"
)
mechanism.ARM_PORTS = {
    "dense": 32640,
    mechanism.POLICY_ARM: 32641,
}
mechanism.FORKS = {
    "django__django-13343": {
        "request": 14,
        "planned_tokens": 383,
        "predicted_saving_ms": 11.504,
    }
}


if __name__ == "__main__":
    mechanism.main()
