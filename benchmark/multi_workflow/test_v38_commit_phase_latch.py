from benchmark.multi_workflow import (
    motivate_v33_state_transition_target as v33a,
)
from benchmark.multi_workflow.coding_reuse_policy import (
    repository_commit_phase_event,
)
from benchmark.multi_workflow.context_bounded_litellm_model import (
    ContextBoundedLitellmModel,
)
from benchmark.multi_workflow.motivate_v38_commit_phase_latch import (
    ROLLING_GROUPS,
    is_repository_mutation,
)
from benchmark.multi_workflow.run_frozen_trajectory_replay_v18 import (
    read_json,
)


def test_v38_motivation_and_serving_latch_events_match() -> None:
    compared = 0
    eligible = 0
    exploration = 0
    commit = 0
    for paths in v33a._trajectories().values():
        for path in paths:
            trajectory = read_json(path)
            calls = int(trajectory["info"]["model_stats"]["api_calls"])
            groups = ContextBoundedLitellmModel._turn_groups(
                trajectory["messages"][2:]
            )
            latched = False
            for completed_index, group in enumerate(groups, start=1):
                if completed_index >= calls:
                    break
                assert repository_commit_phase_event(
                    group
                ) == is_repository_mutation(group)
                compared += 1
                latched |= repository_commit_phase_event(group)
                if not any(
                    message.get("role") == "tool" for message in group
                ):
                    continue
                if completed_index < ROLLING_GROUPS:
                    continue
                eligible += 1
                if latched:
                    commit += 1
                else:
                    exploration += 1
    assert compared > eligible
    assert (eligible, exploration, commit) == (245, 96, 149)
