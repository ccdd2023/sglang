#!/usr/bin/env bash
# ASPLOS 2027: pdflatex -> bibtex (ACM-Reference-Format) -> pdflatex x2
set -euo pipefail
if [ -d /home/gfy/texlive/2026/bin/x86_64-linux ]; then
  export PATH="/home/gfy/texlive/2026/bin/x86_64-linux:${PATH}"
fi
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -file-line-error main.tex || true
bibtex main
pdflatex -interaction=nonstopmode -file-line-error main.tex || true
pdflatex -interaction=nonstopmode -file-line-error main.tex || true
echo "Done: $(pwd)/main.pdf"
ls -l main.pdf
# Body page budget is 11 excluding references and appendix.
python3 - <<'PY'
from pathlib import Path
log = Path("main.log").read_text(errors="replace")
for key in ("Output written", "Error", "undefined", "Citation"):
    pass
import re
m = re.findall(r"Output written on main.pdf \((\d+) pages", log)
print("pages:", m[-1] if m else "?")
undef = [ln for ln in log.splitlines() if "undefined" in ln.lower() or ln.startswith("!")]
print("flags:", undef[:12] or "none")
PY
