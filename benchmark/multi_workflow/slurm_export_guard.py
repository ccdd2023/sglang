#!/usr/bin/env python3
"""Refuse comma-valued IMPACTKV_PREFETCH_MODES on sbatch --export.

Job 114807 PARTIAL: `--export=ALL,IMPACTKV_PREFETCH_MODES=prefetch_only,combined`
is comma-split by Slurm, so the job only saw prefetch_only. Modes with a
comma must be written to a file that the sbatch script reads. Never put
IMPACTKV_PREFETCH_MODES on the --export command line.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MODES_EXPORT_KEY = "IMPACTKV_PREFETCH_MODES"
DEFAULT_MODES_FILE = Path(__file__).resolve().parent / "slurm" / "prefetch_modes.txt"


def slurm_split_export(flag: str) -> list[str]:
    if not flag.startswith("--export="):
        raise ValueError(flag)
    return flag[len("--export=") :].split(",")


def export_carries_comma_modes(flag: str) -> bool:
    for token in slurm_split_export(flag):
        if token.startswith(f"{MODES_EXPORT_KEY}="):
            value = token.split("=", 1)[1]
            if "," in value or value not in {
                "dense",
                "coding",
                "prefetch_only",
                "prefix_only",
                "lossy_only",
                "dual",
                "combined",
            }:
                return True
        elif token == "combined" or token.endswith("_only"):
            # orphan token after a comma-split value, as in 114807
            return True
    return False


def sbatch_export_argv(env: dict[str, str]) -> list[str]:
    """Build `--export=ALL[,KEY=VALUE...]`. Raises if any value contains a comma."""
    parts = ["ALL"]
    for key, value in env.items():
        if key == MODES_EXPORT_KEY:
            raise ValueError(
                "IMPACTKV_PREFETCH_MODES must not be passed via sbatch --export; "
                "write it to prefetch_modes.txt"
            )
        if "," in str(value):
            raise ValueError(
                f"sbatch --export cannot carry comma in {key}={value!r}; "
                "write the value into the sbatch script or a modes file"
            )
        parts.append(f"{key}={value}")
    return [f"--export={','.join(parts)}"]


def write_modes_file(path: Path, modes: str) -> None:
    cleaned = ",".join(part.strip() for part in modes.split(",") if part.strip())
    if not cleaned:
        raise ValueError("empty modes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cleaned + "\n", encoding="utf-8")


def build_sbatch_argv(
    *,
    modes: str,
    sbatch: str | Path,
    modes_file: Path,
    extra_export: dict[str, str] | None = None,
) -> list[str]:
    write_modes_file(modes_file, modes)
    argv = ["sbatch"]
    argv.extend(sbatch_export_argv(dict(extra_export or {})))
    argv.append(str(sbatch))
    export_flags = [item for item in argv if item.startswith("--export=")]
    for flag in export_flags:
        if export_carries_comma_modes(flag) or f"{MODES_EXPORT_KEY}=" in flag:
            raise ValueError(f"modes leaked onto --export: {flag}")
        if re.search(r"IMPACTKV_PREFETCH_MODES=", flag):
            raise ValueError(flag)
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="combined")
    parser.add_argument("--sbatch", type=Path, required=True)
    parser.add_argument("--modes-file", type=Path, default=DEFAULT_MODES_FILE)
    parser.add_argument("--print-argv", action="store_true")
    args = parser.parse_args()
    argv = build_sbatch_argv(
        modes=args.modes, sbatch=args.sbatch, modes_file=args.modes_file
    )
    if args.print_argv:
        print(" ".join(argv))
    else:
        print(args.modes_file.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
