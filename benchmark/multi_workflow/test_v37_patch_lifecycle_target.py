from benchmark.multi_workflow import (
    motivate_v33_state_transition_target as v33a,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    coding_patch_lifecycle_target_reasons,
)
from benchmark.multi_workflow.context_bounded_litellm_model import (
    ContextBoundedLitellmModel,
)
from benchmark.multi_workflow.motivate_v37_patch_lifecycle_target import (
    ROLLING_GROUPS,
    coding_patch_lifecycle_reasons,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
)


def test_v37_motivation_and_serving_rules_match_every_frozen_request() -> None:
    compared = 0
    for paths in v33a._trajectories().values():
        for path in paths:
            trajectory = read_json(path)
            calls = int(trajectory["info"]["model_stats"]["api_calls"])
            groups = ContextBoundedLitellmModel._turn_groups(
                trajectory["messages"][2:]
            )
            for completed_index, group in enumerate(groups, start=1):
                if completed_index >= calls:
                    break
                if not any(
                    message.get("role") == "tool" for message in group
                ):
                    continue
                if completed_index < ROLLING_GROUPS:
                    continue
                rolling = groups[
                    max(0, completed_index - ROLLING_GROUPS) :
                    completed_index
                ]
                assert coding_patch_lifecycle_target_reasons(
                    rolling
                ) == coding_patch_lifecycle_reasons(rolling)
                compared += 1
    assert compared == 245
