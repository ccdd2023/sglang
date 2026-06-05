#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 scripts/generate_paper_figures.py

if command -v latexmk >/dev/null 2>&1; then
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
else
  echo "Neither latexmk nor pdflatex is available." >&2
  exit 127
fi
