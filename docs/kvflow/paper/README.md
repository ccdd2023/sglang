# ImpactKV ASPLOS 2027 paper

Sources for *ImpactKV: Coding-Aware Lossy KV Reuse for Shifted File-Island Prefill*.

This tree lives inside the SGLang fork. Collaborators only need this repo
(`ccdd2023/sglang`, branch `integration/template-prefetch-swebench`).

Takeover (frozen contract, first day, what to edit):
[`../../../HANDOFF.md`](../../../HANDOFF.md).  
Commands: [`../../../IMPACTKV.md`](../../../IMPACTKV.md).

Chinese walkthrough of the argument and figures: [`PAPER_LOGIC_CN.md`](PAPER_LOGIC_CN.md).

## Check claims

Frozen JSON is **not** a cluster mount. From the repo root:

```bash
python benchmark/multi_workflow/fetch_impactkv_artifacts.py
export IMPACTKV_ARTIFACTS="$PWD/impactkv-artifacts"
cd docs/kvflow/paper
python3 scripts/check_asplos_claims.py
python3 -m pytest -q scripts/
```

## Compile

```bash
bash compile.sh    # pdflatex + bibtex; body must stay ≤ 11 pages
```

`main.pdf` is the submission PDF. Do not squeeze with `\vspace` (checker fails).

## Do not

- Edit `RESULT.json` numbers for jobs 137185 / 96092.
- Put `1.375×` in `tab:eval-summary`, or `96.5` in the 7B body.
- Bill N=4 (`0.905`, `0.841`, `tab:nuse`).
- Write SOTA. Mix 7B TTFT with another family's official Accuracy.
