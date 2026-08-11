from __future__ import annotations

from benchmark.multi_workflow.analyze_single_island_action_divergence import audit


def test_audit_computes_paired_losses() -> None:
    result = {
        "decision": "ACTION_TARGET_TOO_SPARSE",
        "unique_selected_splices": 36,
        "candidate_divergence_fraction": 0.5,
        "within_case_candidate_variation": {"cases": 7, "tasks": 7},
        "signal_to_action_distance_spearman": {"fixed_probe": 0.3},
        "arms": {
            "current_recency": {"exact_dense_match_fraction": 0.5},
            "fixed_probe_min": {
                "exact_dense_match_fraction": 0.6,
                "vs_recency": {"win_fraction": 0.4, "tie_fraction": 0.3},
            },
            "module_attention_oracle": {
                "exact_dense_match_fraction": 0.6,
                "vs_recency": {"win_fraction": 0.5, "tie_fraction": 0.2},
            },
            "seeded_random": {
                "exact_dense_match_fraction": 0.6,
                "vs_recency": {"win_fraction": 0.2, "tie_fraction": 0.4},
            },
        },
    }
    value = audit(result)
    assert value["paired_vs_recency"]["fixed_probe_min"]["losses"] == 0.3
    assert value["within_case_variation"]["cases"] == 7
