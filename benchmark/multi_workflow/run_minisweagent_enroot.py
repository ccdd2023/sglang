#!/usr/bin/env python3
"""Run mini-SWE-agent's SWE-bench CLI with the repository Enroot backend."""

from __future__ import annotations

from typing import Any

from jinja2 import StrictUndefined, Template

from benchmark.multi_workflow.enroot_environment import EnrootEnvironment
from minisweagent.run.benchmarks import swebench


def get_enroot_environment(config: dict[str, Any], instance: dict[str, Any]):
    env_config = config.setdefault("environment", {})
    env_config["image"] = swebench.get_swebench_docker_image_name(instance)
    env = EnrootEnvironment(**env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(
            startup_command, undefined=StrictUndefined
        ).render(**instance)
        output = env.execute({"command": startup_command})
        if output["returncode"] != 0:
            env.cleanup()
            raise RuntimeError(f"Error executing startup command: {output}")
    return env


def main() -> None:
    swebench.get_sb_environment = get_enroot_environment
    swebench.app()


if __name__ == "__main__":
    main()
