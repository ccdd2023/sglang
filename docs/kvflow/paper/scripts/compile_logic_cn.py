#!/usr/bin/env python3
"""Compile PAPER_LOGIC_CN.md to a Chinese PDF via XeLaTeX/ctex."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
MD = PAPER / "PAPER_LOGIC_CN.md"
TEX = PAPER / "PAPER_LOGIC_CN.tex"
PDF = PAPER / "PAPER_LOGIC_CN.pdf"


def escape_text(text: str) -> str:
    math: list[str] = []

    def hold_math(match: re.Match[str]) -> str:
        math.append(match.group(0))
        return f"@@MATH{len(math) - 1}@@"

    text = re.sub(r"\$\$[\s\S]+?\$\$|\$[^$]+\$", hold_math, text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    text = "".join(out)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    for i, blob in enumerate(math):
        inner = blob.strip("$")
        if blob.startswith("$$"):
            text = text.replace(f"@@MATH{i}@@", f"\\[{inner}\\]")
        else:
            text = text.replace(f"@@MATH{i}@@", f"${inner}$")
    return text


def convert_inline(text: str) -> str:
    return escape_text(text)


def convert_table(block: str) -> str:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return ""
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells if c):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return ""
    ncol = max(len(r) for r in rows)
    body = []
    for i, row in enumerate(rows):
        row = (row + [""] * ncol)[:ncol]
        line = " & ".join(convert_inline(c) for c in row) + r" \\"
        if i == 0:
            body.append(r"\toprule")
            body.append(line)
            body.append(r"\midrule")
        else:
            body.append(line)
    body.append(r"\bottomrule")
    spec = "l" * ncol
    return (
        "\\begin{center}\\small\\setlength{\\tabcolsep}{4pt}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        f"\\begin{{tabular}}{{{spec}}}\n"
        + "\n".join(body)
        + "\n\\end{tabular}%\n}\n\\end{center}\n"
    )


def convert_md(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_quote = False
    list_type: str | None = None

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            out.append(r"\end{" + list_type + "}")
            list_type = None

    def close_quote() -> None:
        nonlocal in_quote
        if in_quote:
            out.append(r"\end{quote}")
            in_quote = False

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if stripped.startswith("|") and i + 1 < len(lines) and re.search(r"---", lines[i + 1]):
            close_list()
            close_quote()
            block = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            out.append(convert_table("\n".join(block)))
            continue

        if stripped == "---":
            close_list()
            close_quote()
            out.append(r"\par\noindent\rule{\linewidth}{0.4pt}\par")
            i += 1
            continue

        if stripped.startswith("```"):
            close_list()
            close_quote()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append(r"\begin{verbatim}")
            out.extend(code)
            out.append(r"\end{verbatim}")
            continue

        img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if img:
            close_list()
            close_quote()
            caption, path = img.group(1), img.group(2)
            cap = convert_inline(caption)
            if path.endswith(".tikz"):
                out.append(
                    "\\begin{center}\\resizebox{\\linewidth}{!}{%\n"
                    f"\\input{{{path}}}%\n"
                    "}\\end{center}\n"
                    f"\\captionof{{figure}}{{{cap}}}\n"
                )
            else:
                out.append(
                    "\\begin{center}\n"
                    f"\\includegraphics[width=\\linewidth]{{{path}}}\n"
                    "\\end{center}\n"
                    f"\\captionof{{figure}}{{{cap}}}\n"
                )
            i += 1
            continue

        if stripped.startswith("> "):
            close_list()
            if not in_quote:
                out.append(r"\begin{quote}\small")
                in_quote = True
            out.append(convert_inline(stripped[2:]))
            i += 1
            continue
        if in_quote and stripped:
            out.append(convert_inline(stripped))
            i += 1
            continue
        if in_quote and not stripped:
            close_quote()
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            close_list()
            close_quote()
            level = len(m.group(1))
            title = convert_inline(m.group(2))
            cmd = {1: "section*", 2: "section*", 3: "subsection*", 4: "subsubsection*"}[level]
            if level == 1:
                out.append(r"\begin{center}{\LARGE\bfseries " + title + r"}\end{center}")
            else:
                out.append(f"\\{cmd}{{{title}}}")
            i += 1
            continue

        ul = re.match(r"^[-*]\s+(.*)$", stripped)
        ol = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ul or ol:
            close_quote()
            kind = "itemize" if ul else "enumerate"
            if list_type != kind:
                close_list()
                out.append(r"\begin{" + kind + r"}[leftmargin=1.4em,itemsep=0.2em]")
                list_type = kind
            out.append(r"\item " + convert_inline((ul or ol).group(1)))
            i += 1
            continue

        if not stripped:
            close_list()
            close_quote()
            out.append("")
            i += 1
            continue

        close_list()
        close_quote()
        out.append(convert_inline(stripped))
        i += 1

    close_list()
    close_quote()
    return "\n".join(out)


PREAMBLE = r"""
\documentclass[UTF8,a4paper,11pt,fontset=fandol]{ctexart}
\usepackage[margin=18mm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs,tabularx,array}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{tikz}
\usepackage{hyperref}
\usetikzlibrary{arrows.meta,positioning,fit,calc,shapes.geometric}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\graphicspath{{./}{figures/}}
\pagestyle{plain}
\begin{document}
"""


def main() -> int:
    md = MD.read_text(encoding="utf-8")
    body = convert_md(md)
    TEX.write_text(PREAMBLE + body + "\n\\end{document}\n", encoding="utf-8")
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PATH"] = "/home/gfy/texlive/2026/bin/x86_64-linux:" + env.get("PATH", "")
    env["OSFONTDIR"] = (
        "/home/gfy/texlive/2026/texmf-dist/fonts/opentype/public/fandol"
    )
    for _ in range(2):
        proc = subprocess.run(
            ["lualatex", "-interaction=nonstopmode", "-file-line-error", TEX.name],
            cwd=PAPER,
            env=env,
            capture_output=True,
            text=True,
        )
    log = (PAPER / "PAPER_LOGIC_CN.log").read_text(errors="replace") if (PAPER / "PAPER_LOGIC_CN.log").exists() else proc.stdout
    if not PDF.exists():
        print(proc.stdout[-2000:])
        print(proc.stderr[-1000:])
        print("FAIL: PDF not written")
        return 1
    pages = re.findall(r"Output written on .* \((\d+) pages", log)
    print("Wrote", PDF, "pages:", pages[-1] if pages else "?")
    overfull = [ln for ln in log.splitlines() if "Overfull \\hbox" in ln]
    print("overfull hbox:", len(overfull))
    return 0


if __name__ == "__main__":
    sys.exit(main())
