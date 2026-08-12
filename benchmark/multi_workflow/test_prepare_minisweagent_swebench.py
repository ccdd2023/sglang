from benchmark.multi_workflow.prepare_minisweagent_swebench import (
    normalize_model_patch,
)


def test_normalize_model_patch_preserves_unified_diff():
    patch = """diff --git a/pkg/a.py b/pkg/a.py
index 1111111..2222222 100644
--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1 +1 @@
-old
+new
"""

    assert normalize_model_patch(patch) == (patch, "unified_diff")


def test_normalize_model_patch_keeps_empty_submission_empty():
    assert normalize_model_patch("\n") == ("", "empty")


def test_normalize_model_patch_drops_submission_prose():
    assert normalize_model_patch("No changes made\n") == (
        "",
        "invalid_non_diff_dropped",
    )


def test_normalize_model_patch_requires_a_hunk():
    patch = "diff --git a/pkg/a.py b/pkg/a.py\n"
    assert normalize_model_patch(patch) == ("", "invalid_non_diff_dropped")
