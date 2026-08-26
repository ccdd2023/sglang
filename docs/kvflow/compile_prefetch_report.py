#!/usr/bin/env python3
"""Compile PREFETCH_M3_REPORT_CN.tex with LuaLaTeX + Fandol."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "PREFETCH_M3_REPORT_CN.tex"
PDF = HERE / "PREFETCH_M3_REPORT_CN.pdf"


def main() -> int:
    env = dict(os.environ)
    env["PATH"] = "/home/gfy/texlive/2026/bin/x86_64-linux:" + env.get("PATH", "")
    env["OSFONTDIR"] = (
        "/home/gfy/texlive/2026/texmf-dist/fonts/opentype/public/fandol"
    )
    log_tail = ""
    for _ in range(2):
        proc = subprocess.run(
            [
                "lualatex",
                "-interaction=nonstopmode",
                "-file-line-error",
                TEX.name,
            ],
            cwd=HERE,
            env=env,
            capture_output=True,
            text=True,
        )
        log_tail = (proc.stdout or "")[-2500:]
    if not PDF.exists():
        print(log_tail)
        print(proc.stderr[-1000:] if proc.stderr else "")
        print("FAIL: PDF not written")
        return 1
    log = (HERE / "PREFETCH_M3_REPORT_CN.log").read_text(errors="replace")
    pages = re.findall(r"Output written on .* \((\d+) pages", log)
    print("Wrote", PDF, "pages:", pages[-1] if pages else "?")
    bangs = [ln for ln in log.splitlines() if ln.startswith("!")]
    print("latex errors:", bangs[:8] or "none")
    return 0 if not bangs else 1


if __name__ == "__main__":
    sys.exit(main())
