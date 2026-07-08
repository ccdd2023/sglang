# Refresh a date-stamped snapshot of the progress report.
#
# Usage:
#   make report                  # today's date, YYYYMMDD
#   DATE=20260707 make report    # explicit date (rebuild / backfill)
#
# Requires TeX Live binaries on PATH:
#   export PATH=/home/gfy/texlive/2026/bin/x86_64-linux:$PATH
#
# Build outputs land in build/; .tex + .pdf + .synctex.gz stay in reports/.

DATE    ?= $(shell date +%Y%m%d)
TEX      = code_aware_lossy_kv_progress_$(DATE).tex
PDF      = code_aware_lossy_kv_progress_$(DATE).pdf
LATEXMK ?= latexmk

.PHONY: report archive clean distclean
report: $(PDF)

$(PDF): $(TEX) figures/
	@mkdir -p build
	$(LATEXMK) -xelatex -interaction=nonstopmode -outdir=build $(TEX)
	@cp build/$(PDF) $(PDF)
	@cp build/$(TEX) $(TEX) 2>/dev/null || true

archive:
	@mkdir -p archive
	cp $(PDF) archive/$(PDF)

clean:
	$(LATEXMK) -C -outdir=build $(TEX) 2>/dev/null || true
	rm -rf build

distclean: clean
	rm -f code_aware_lossy_kv_progress_*.pdf
	rm -f code_aware_lossy_kv_progress_*.synctex.gz