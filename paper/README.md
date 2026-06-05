# CodeMAS EuroSys Draft

This directory contains an anonymous EuroSys-style LaTeX paper draft for the
current CodeMAS code-base segment reuse paper.

## Current Handoff Status

The current draft is framed around three CodeMAS contributions:

1. Workflow Template Generation for coding MAS.
2. Template-derived code-base segment hints for cache scheduling.
3. Exact-content code segment reuse with conservative safety gating.

KVFlow and KVCOMM are treated as referenced prior work and implementation
inspiration, not as contributions claimed by this paper.

The older paper draft lives at:

```text
/home/gfy/Paper_CodeMAS/CodeAgent_UCM_HKBU
```

Reusable ideas already migrated from that draft:

- AgentTemplateKV framing: agent DAGs, roles, and tool edges are serving-time
  signals.
- Template synthesis process: role discovery, tool assignment, DAG validation,
  refinement.
- KV-object metadata/lifecycle framing.
- Related work coverage for vLLM, SGLang, Parrot, Tokencake, RelayCaching,
  Continuum, KVFlow, KVCOMM, and multi-agent software-engineering surveys.

The current paper also includes:

- TikZ conceptual figures used in the paper:
  - `figures/fig_system_architecture_tikz.tex`
  - `figures/fig_coding_prefetch_tikz.tex`
  - `figures/fig_kvcomm_mechanism_tikz.tex`
- GPT-image2 conceptual PNG backups:
  - `figures/fig_system_architecture.png`
  - `figures/fig_coding_prefetch.png`
  - `figures/fig_kvcomm_mechanism.png`
- Old-draft template synthesis figure:
  - `figures/fig_template_synthesis_process.png`
- Data figures generated from local experiment CSV/JSON:
  - `figures/fig_gate_false_accepts.pdf`
  - `figures/fig_rope_and_logits.pdf`
  - `figures/fig_prefetch_modes.pdf`
  - `figures/fig_passrate_main.pdf`
  - `figures/fig_scalability.pdf`

## Regenerate Data Figures

From the repository root:

```bash
python3 paper/scripts/generate_paper_figures.py
```

The script uses only Python standard-library modules and writes:

- `paper/figures/*.pdf`
- `paper/tables/*.tex`
- `paper/data_manifest.json`

Conceptual architecture PNGs are retained as bitmap backups. The paper body
currently uses the TikZ versions for a more submission-style figure appearance.

## Build

The user's local environment can compile this paper with:

```bash
cd /home/gfy/CodeMAS_Project/sglang-kvflow/paper
./compile.sh
```

The current Codex shell does not expose `latexmk` or `pdflatex`, so Codex-side
execution may stop with:

```text
Neither latexmk nor pdflatex is available.
```

If compilation looks wrong in the user's local environment, inspect:

- `paper/main.log`
- `paper/main.pdf`
- the most recent warning lines for `Overfull`, `Citation`, `undefined`, and
  `Class acmart Warning`.

## Safety Wording

Keep these invariants in any future edits:

- AST/anchor metadata is only a locator.
- Exact-content segment reuse is allowed only with
  `exact_code_content_signature`.
- "Lossy" means cross-position RoPE-aligned KV reuse, not approximate code reuse.
- SWE-bench pass@1 should be interpreted as lossless-vs-lossy delta evidence
  because the current 7B JSON-edit patch generator has low absolute pass@1.
- Do not phrase KVFlow or KVCOMM as our contribution. Phrase them as prior work
  or as a baseline/reference design.

## Next Paper Tasks

High priority:

- Continue polishing the TikZ concept figures if more space becomes available;
  the paper currently uses them instead of the GPT-image2 PNG backups.
- Expand the system narrative in Motivation, Design, and Implementation so the
  paper reads less like a short report and more like a systems submission.
- Keep the H10 pass@1 wording focused on lossless-vs-lossy delta, not absolute
  Qwen2.5-7B patch quality.
- Keep H12 host-backed prefetch as a limitation until the HiCache file backend
  memory-checker issue is fixed.

Medium priority:

- Consider migrating old-draft figures such as `kv-lifecycle.png`,
  `policy-runtime.png`, or `kv-reuse.png` only if they clearly support the new
  three-contribution story.
- Add a stronger coder model or improved patch schema experiment before serious
  submission.
- Clean unused BibTeX entries before camera-ready.
