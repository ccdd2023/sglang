"""SWE-bench grading rules executed in pre-imported Enroot instance images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.multi_workflow.enroot_environment import EnrootEnvironment
from benchmark.multi_workflow.prepare_enroot_images import load_dataset
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    TESTS_TIMEOUT,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec


GIT_APPLY_COMMANDS = (
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {row[KEY_INSTANCE_ID]: row for row in rows}
    value = json.loads(text)
    if isinstance(value, list):
        return {row[KEY_INSTANCE_ID]: row for row in value}
    return value


def _stage_text_file(
    environment: EnrootEnvironment, source: Path, target: str
) -> None:
    result = environment.write_text(
        target,
        source.read_text(encoding="utf-8"),
    )
    if result["returncode"] != 0:
        raise RuntimeError(f"failed to stage {source} in Enroot: {result}")


def evaluate_instance(
    *,
    instance: dict[str, Any],
    prediction: dict[str, Any],
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    instance_dir = output_dir / "instances" / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    test_spec = make_test_spec(instance, namespace="swebench")
    image = f"docker.io/{test_spec.instance_image_key}"
    patch_file = instance_dir / "patch.diff"
    patch_file.write_text(prediction.get(KEY_PREDICTION) or "", encoding="utf-8")
    eval_file = instance_dir / "eval.sh"
    eval_file.write_text(test_spec.eval_script, encoding="utf-8")
    metadata = {
        "instance_id": instance_id,
        "image": image,
        "base_commit": instance.get("base_commit"),
        "model_name_or_path": prediction.get(KEY_MODEL),
        "completed": False,
        "resolved": False,
        "patch_applied": False,
    }
    environment: EnrootEnvironment | None = None
    try:
        environment = EnrootEnvironment(
            image=image,
            cwd="/testbed",
            timeout=max(timeout, 120),
            container_timeout=f"{max(timeout + 600, 1800)}s",
            # SWE-bench's Docker path runs the eval script in a non-login
            # shell. A login shell can prepend image-specific /root/bin
            # wrappers and change which compiler is selected.
            interpreter=["bash", "-c"],
        )
        _stage_text_file(environment, patch_file, "/tmp/patch.diff")
        apply_logs: list[dict[str, Any]] = []
        for command in GIT_APPLY_COMMANDS:
            result = environment.execute(
                {"command": f"{command} /tmp/patch.diff"}, cwd="/testbed"
            )
            apply_logs.append({"command": command, **result})
            if result["returncode"] == 0:
                metadata["patch_applied"] = True
                break
        (instance_dir / "apply.json").write_text(
            json.dumps(apply_logs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not metadata["patch_applied"]:
            raise RuntimeError(APPLY_PATCH_FAIL)

        before = environment.execute(
            {"command": "git -c core.fileMode=false diff"}, cwd="/testbed"
        )
        (instance_dir / "git_diff_before.patch").write_text(
            before["output"], encoding="utf-8"
        )
        _stage_text_file(environment, eval_file, "/eval.sh")
        test_result = environment.execute(
            {"command": "/bin/bash /eval.sh"}, cwd="/testbed", timeout=timeout
        )
        test_output = test_result["output"]
        if test_result["returncode"] == -1:
            test_output += f"\n{TESTS_TIMEOUT}\n"
        test_output_path = instance_dir / "test_output.txt"
        test_output_path.write_text(test_output, encoding="utf-8")
        metadata["test_returncode"] = test_result["returncode"]
        metadata["test_exception_info"] = test_result["exception_info"]

        after = environment.execute(
            {"command": "git -c core.fileMode=false diff"}, cwd="/testbed"
        )
        (instance_dir / "git_diff_after.patch").write_text(
            after["output"], encoding="utf-8"
        )
        report = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=test_output_path,
            include_tests_status=True,
        )
        (instance_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata["completed"] = True
        metadata["resolved"] = bool(report[instance_id]["resolved"])
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if environment is not None:
            environment.cleanup()
    (instance_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def run_enroot_evaluation(
    *,
    dataset_path: Path,
    predictions_path: Path,
    output_dir: Path,
    run_id: str,
    instance_ids: list[str] | None = None,
    timeout: int = 3600,
) -> dict[str, Any]:
    dataset = load_dataset(dataset_path)
    selected = set(instance_ids or [row[KEY_INSTANCE_ID] for row in dataset])
    dataset = [row for row in dataset if row[KEY_INSTANCE_ID] in selected]
    missing_dataset = selected - {row[KEY_INSTANCE_ID] for row in dataset}
    if missing_dataset:
        raise ValueError(f"instances absent from dataset: {sorted(missing_dataset)}")
    predictions = load_predictions(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for instance in dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        prediction = predictions.get(instance_id)
        if prediction is None:
            results.append(
                {
                    "instance_id": instance_id,
                    "completed": False,
                    "resolved": False,
                    "error": "missing prediction",
                }
            )
            continue
        if prediction.get(KEY_PREDICTION) in (None, ""):
            results.append(
                {
                    "instance_id": instance_id,
                    "completed": False,
                    "resolved": False,
                    "empty_patch": True,
                }
            )
            continue
        results.append(
            evaluate_instance(
                instance=instance,
                prediction=prediction,
                output_dir=output_dir,
                timeout=timeout,
            )
        )

    completed = sorted(
        row["instance_id"] for row in results if row.get("completed")
    )
    resolved = sorted(
        row["instance_id"] for row in results if row.get("resolved")
    )
    errors = sorted(
        row["instance_id"] for row in results if row.get("error")
    )
    empty = sorted(
        row["instance_id"] for row in results if row.get("empty_patch")
    )
    unresolved = sorted(set(completed) - set(resolved))
    report = {
        "schema_version": 2,
        "container_backend": "enroot",
        "run_id": run_id,
        "total_instances": len(dataset),
        "submitted_instances": len(dataset),
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": len(empty),
        "error_instances": len(errors),
        "completed_ids": completed,
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
        "empty_patch_ids": empty,
        "error_ids": errors,
        "instances": results,
    }
    report_path = output_dir / f"impactkv__{run_id}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"returncode": 0 if not errors else 1, "report_path": str(report_path), "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--instances", help="comma-separated instance IDs")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    instance_ids = (
        [value for value in (args.instances or "").split(",") if value]
        or None
    )
    value = run_enroot_evaluation(
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        output_dir=args.output,
        run_id=args.run_id,
        instance_ids=instance_ids,
        timeout=args.timeout,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2))
    raise SystemExit(value["returncode"])


if __name__ == "__main__":
    main()
