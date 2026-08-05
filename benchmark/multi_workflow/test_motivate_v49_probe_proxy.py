from benchmark.multi_workflow.motivate_v49_probe_proxy import (
    _config_metrics,
)


def test_configuration_metrics_prefers_scores_aligned_with_harm():
    rows = []
    for case_index in range(3):
        candidates = []
        for candidate_index in range(3):
            candidates.append(
                {
                    "candidate_id": f"c{candidate_index}",
                    "causal_splice_logit_js": float(candidate_index),
                    "configurations": [
                        {
                            "layer": 17,
                            "head_tokens": 32,
                            "score": float(candidate_index),
                        }
                    ],
                }
            )
        rows.append(
            {
                "case_id": f"case-{case_index}",
                "candidates": candidates,
                "v46_candidate_ids": ["c0", "c1", "c2"],
                "v46_composed": {
                    "causal_splice_logit_js": float(case_index),
                    "answer_first_token_nll_delta": float(case_index),
                },
            }
        )
    metrics = _config_metrics(rows, 17, 32)
    assert metrics["single_global_js_spearman"] == 1.0
    assert metrics["single_mean_within_case_js_spearman"] == 1.0
