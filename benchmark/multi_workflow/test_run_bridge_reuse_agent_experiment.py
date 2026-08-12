from pathlib import Path

from benchmark.multi_workflow import run_bridge_reuse_agent_experiment as runner


def test_mini_command_routes_native_requests_to_launched_sglang_port(
    tmp_path: Path,
) -> None:
    command = runner.mini_command(
        run_dir=tmp_path / "run",
        arm="coding_dependency_graph_cold_lcb",
        manifest=tmp_path / "manifest.json",
        port=34567,
        instance_filter=None,
        container_backend="enroot",
    )

    configs = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--config"
    ]
    assert "model.native_backend_url=http://127.0.0.1:34567" in configs
    assert (
        "model.native_backend_name=sglang-coding_dependency_graph_cold_lcb"
        in configs
    )
    assert "model.model_kwargs.api_base=http://127.0.0.1:34567/v1" in configs
