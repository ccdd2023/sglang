#!/usr/bin/env python3
"""Generate HTML report and PNG charts for Multi-Agent TTFT experiment."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "sglang-kvflow" / "results" / "ma_ttft"

def load_data():
    return json.loads((OUT / "results.json").read_text())

def b64(path):
    import base64
    return base64.b64encode(path.read_bytes()).decode()

def generate_charts(data):
    files = [r["name"] for r in data]
    x = np.arange(len(files))
    width = 0.25

    # Chart 1: TTFT by Agent
    for agent_idx, agent in enumerate(["a1", "a2", "a3"]):
        fig, ax = plt.subplots(figsize=(10, 5))
        nr = [r["no_reuse"].get(agent, {}).get("ttft_ms", 0) or 0 for r in data]
        fr = [r["full_reuse"].get(agent, {}).get("ttft_ms", 0) or 0 for r in data]
        lr = [r["lossy_reuse"].get(agent, {}).get("ttft_ms", 0) or 0 for r in data]

        ax.bar(x - width, nr, width, label='No-Reuse', color='#e74c3c')
        ax.bar(x, fr, width, label='Full-Reuse', color='#2ecc71')
        ax.bar(x + width, lr, width, label='Lossy-Reuse', color='#3498db')
        ax.set_ylabel('TTFT (ms)')
        ax.set_title(f'Time To First Token — {agent.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(files, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"chart_ttft_{agent}.png", dpi=150)
        plt.close()

    # Chart 2: Total Latency by Agent
    for agent in ["a1", "a2", "a3"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        nr = [r["no_reuse"].get(agent, {}).get("total_ms", 0) or 0 for r in data]
        fr = [r["full_reuse"].get(agent, {}).get("total_ms", 0) or 0 for r in data]
        lr = [r["lossy_reuse"].get(agent, {}).get("total_ms", 0) or 0 for r in data]

        ax.bar(x - width, nr, width, label='No-Reuse', color='#e74c3c')
        ax.bar(x, fr, width, label='Full-Reuse', color='#2ecc71')
        ax.bar(x + width, lr, width, label='Lossy-Reuse', color='#3498db')
        ax.set_ylabel('Total Latency (ms)')
        ax.set_title(f'Total Latency — {agent.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(files, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"chart_total_{agent}.png", dpi=150)
        plt.close()

    # Chart 3: Cached Tokens (A2/A3 only, A1 should be 0)
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent, color, label in [("a2", "#2ecc71", "A2 Full"), ("a3", "#9b59b6", "A3 Full")]:
        fr = [r["full_reuse"].get(agent, {}).get("cached_tokens", 0) for r in data]
        ax.plot(files, fr, 'o-', label=label, color=color, linewidth=2)
    for agent, color, label in [("a2", "#3498db", "A2 Lossy"), ("a3", "#e67e22", "A3 Lossy")]:
        lr = [r["lossy_reuse"].get(agent, {}).get("cached_tokens", 0) for r in data]
        ax.plot(files, lr, 's--', label=label, color=color, linewidth=2)
    ax.set_ylabel('Cached Tokens')
    ax.set_title('KV Cache Reuse (Cached Tokens)')
    ax.set_xticklabels(files, rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "chart_cached_tokens.png", dpi=150)
    plt.close()

    # Chart 4: BLEU Accuracy
    fig, ax = plt.subplots(figsize=(10, 5))
    for agent, color, label in [("a2", "#2ecc71", "A2 Full"), ("a3", "#9b59b6", "A3 Full")]:
        fr = [r["full_reuse"].get("bleu", {}).get(agent, 0) for r in data]
        ax.plot(files, fr, 'o-', label=label, color=color, linewidth=2)
    for agent, color, label in [("a2", "#3498db", "A2 Lossy"), ("a3", "#e67e22", "A3 Lossy")]:
        lr = [r["lossy_reuse"].get("bleu", {}).get(agent, 0) for r in data]
        ax.plot(files, lr, 's--', label=label, color=color, linewidth=2)
    ax.set_ylabel('BLEU Score')
    ax.set_title('Accuracy Preservation (BLEU vs Ground Truth)')
    ax.set_xticklabels(files, rotation=15, ha='right')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUT / "chart_bleu.png", dpi=150)
    plt.close()

    # Chart 5: TTFT Speedup % (A2/A3)
    fig, ax = plt.subplots(figsize=(8, 5))
    agents = ["a2", "a3"]
    speedups_full = []
    speedups_lossy = []
    for agent in agents:
        nr_vals = [r["no_reuse"].get(agent, {}).get("ttft_ms", 0) or 0 for r in data]
        fr_vals = [r["full_reuse"].get(agent, {}).get("ttft_ms", 0) or 0 for r in data]
        lr_vals = [r["lossy_reuse"].get(agent, {}).get("ttft_ms", 0) or 0 for r in data]
        avg_nr = sum(nr_vals) / len(nr_vals) if nr_vals else 0
        avg_fr = sum(fr_vals) / len(fr_vals) if fr_vals else 0
        avg_lr = sum(lr_vals) / len(lr_vals) if lr_vals else 0
        speedups_full.append(((avg_nr - avg_fr) / avg_nr * 100) if avg_nr else 0)
        speedups_lossy.append(((avg_nr - avg_lr) / avg_nr * 100) if avg_nr else 0)

    x = np.arange(len(agents))
    ax.bar(x - width/2, speedups_full, width, label='Full-Reuse', color='#2ecc71')
    ax.bar(x + width/2, speedups_lossy, width, label='Lossy-Reuse', color='#3498db')
    ax.set_ylabel('TTFT Reduction (%)')
    ax.set_title('TTFT Acceleration vs No-Reuse')
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in agents])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "chart_ttft_speedup.png", dpi=150)
    plt.close()

def generate_html(data):
    charts = {
        "ttft_a1": b64(OUT / "chart_ttft_a1.png"),
        "ttft_a2": b64(OUT / "chart_ttft_a2.png"),
        "ttft_a3": b64(OUT / "chart_ttft_a3.png"),
        "total_a2": b64(OUT / "chart_total_a2.png"),
        "total_a3": b64(OUT / "chart_total_a3.png"),
        "cached": b64(OUT / "chart_cached_tokens.png"),
        "bleu": b64(OUT / "chart_bleu.png"),
        "speedup": b64(OUT / "chart_ttft_speedup.png"),
    }

    def avg(agent, key, mode):
        vals = [r[mode].get(agent, {}).get(key, 0) or 0 for r in data]
        return sum(vals) / len(vals) if vals else 0

    def avg_bleu(agent, mode):
        vals = [r[mode].get("bleu", {}).get(agent, 0) for r in data]
        return sum(vals) / len(vals) if vals else 0

    speedup_fr_a2 = ((avg("a2", "ttft_ms", "no_reuse") - avg("a2", "ttft_ms", "full_reuse")) / avg("a2", "ttft_ms", "no_reuse") * 100) if avg("a2", "ttft_ms", "no_reuse") else 0
    speedup_lr_a2 = ((avg("a2", "ttft_ms", "no_reuse") - avg("a2", "ttft_ms", "lossy_reuse")) / avg("a2", "ttft_ms", "no_reuse") * 100) if avg("a2", "ttft_ms", "no_reuse") else 0
    speedup_fr_a3 = ((avg("a3", "ttft_ms", "no_reuse") - avg("a3", "ttft_ms", "full_reuse")) / avg("a3", "ttft_ms", "no_reuse") * 100) if avg("a3", "ttft_ms", "no_reuse") else 0
    speedup_lr_a3 = ((avg("a3", "ttft_ms", "no_reuse") - avg("a3", "ttft_ms", "lossy_reuse")) / avg("a3", "ttft_ms", "no_reuse") * 100) if avg("a3", "ttft_ms", "no_reuse") else 0

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Agent TTFT Acceleration Report</title>
<style>
  :root {{ --bg: #0d1117; --fg: #c9d1d9; --accent: #58a6ff; --ok: #3fb950; --warn: #d29922; --danger: #f85149; --card: #161b22; --border: #30363d; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background: var(--bg); color: var(--fg); line-height: 1.7; max-width: 1100px; margin: 0 auto; padding: 2rem 1rem; }}
  h1, h2, h3 {{ color: #e6edf3; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }}
  h1 {{ font-size: 2rem; }}
  h2 {{ font-size: 1.5rem; margin-top: 2.5rem; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; text-align: center; }}
  .metric {{ font-size: 2.2rem; font-weight: 700; color: var(--accent); }}
  .metric-label {{ font-size: 0.9rem; color: #8b949e; }}
  img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 8px; margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.92rem; }}
  th, td {{ border: 1px solid var(--border); padding: 0.5rem 0.7rem; text-align: left; }}
  th {{ background: var(--card); color: #e6edf3; font-weight: 600; }}
  tr:nth-child(even) {{ background: rgba(110,118,129,0.05); }}
  .tag {{ display: inline-block; background: var(--card); border: 1px solid var(--border); border-radius: 999px; padding: 0.15rem 0.6rem; font-size: 0.8rem; margin-right: 0.4rem; color: var(--accent); }}
  .tag.ok {{ color: var(--ok); }}
  .tag.danger {{ color: var(--danger); }}
  @media (max-width: 800px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Multi-Agent Intermediate-Context KV Reuse</h1>
<p>
  <span class="tag">Analyzer</span>
  <span class="tag">Implementer</span>
  <span class="tag">Reviewer</span>
  <span class="tag ok">TTFT Acceleration</span>
  <span class="tag">Lossy Reuse</span>
</p>
<p>
  本实验测量多 Agent 工作流中<strong>复用中间上下文</strong>带来的 TTFT（Time To First Token）加速效果。
  通过将前序 Agent 的输出作为后续 Agent 的 prompt 前缀，利用 RadixCache 前缀匹配实现 KV 复用。
</p>

<h2>核心指标</h2>
<div class="grid-2">
  <div class="card">
    <div class="metric">{speedup_fr_a2:.1f}%</div>
    <div class="metric-label">A2 TTFT Reduction (Full-Reuse)</div>
  </div>
  <div class="card">
    <div class="metric">{speedup_lr_a2:.1f}%</div>
    <div class="metric-label">A2 TTFT Reduction (Lossy-Reuse)</div>
  </div>
  <div class="card">
    <div class="metric">{speedup_fr_a3:.1f}%</div>
    <div class="metric-label">A3 TTFT Reduction (Full-Reuse)</div>
  </div>
  <div class="card">
    <div class="metric">{speedup_lr_a3:.1f}%</div>
    <div class="metric-label">A3 TTFT Reduction (Lossy-Reuse)</div>
  </div>
</div>

<h2>TTFT 对比</h2>
<p>Agent 1 (Analyzer) 是冷启动，作为 baseline。Agent 2/3 展示复用效果。</p>
<p><img src="data:image/png;base64,{charts['ttft_a1']}" alt="TTFT A1"></p>
<p><img src="data:image/png;base64,{charts['ttft_a2']}" alt="TTFT A2"></p>
<p><img src="data:image/png;base64,{charts['ttft_a3']}" alt="TTFT A3"></p>

<h2>总延迟对比</h2>
<p><img src="data:image/png;base64,{charts['total_a2']}" alt="Total A2"></p>
<p><img src="data:image/png;base64,{charts['total_a3']}" alt="Total A3"></p>

<h2>KV Cache 复用量</h2>
<p><img src="data:image/png;base64,{charts['cached']}" alt="Cached Tokens"></p>

<h2>Accuracy 保持 (BLEU)</h2>
<p><img src="data:image/png;base64,{charts['bleu']}" alt="BLEU"></p>

<h2>TTFT 加速汇总</h2>
<p><img src="data:image/png;base64,{charts['speedup']}" alt="TTFT Speedup"></p>

<h2>详细数据</h2>
<table>
  <tr><th>File</th><th>Agent</th><th>No-Reuse TTFT</th><th>Full TTFT</th><th>Lossy TTFT</th>
      <th>No-Reuse Total</th><th>Full Total</th><th>Lossy Total</th>
      <th>Full Cache</th><th>Lossy Cache</th><th>BLEU Full</th><th>BLEU Lossy</th></tr>
"""

    for r in data:
        for agent in ["a1", "a2", "a3"]:
            nr = r["no_reuse"].get(agent, {})
            fr = r["full_reuse"].get(agent, {})
            lr = r["lossy_reuse"].get(agent, {})
            bleu_fr = r["full_reuse"].get("bleu", {}).get(agent, 0)
            bleu_lr = r["lossy_reuse"].get("bleu", {}).get(agent, 0)
            html += f"""
  <tr>
    <td>{r['name']}</td><td>{agent.upper()}</td>
    <td>{nr.get('ttft_ms', 'N/A')}</td><td>{fr.get('ttft_ms', 'N/A')}</td><td>{lr.get('ttft_ms', 'N/A')}</td>
    <td>{nr.get('total_ms', 'N/A')}</td><td>{fr.get('total_ms', 'N/A')}</td><td>{lr.get('total_ms', 'N/A')}</td>
    <td>{fr.get('cached_tokens', 0)}</td><td>{lr.get('cached_tokens', 0)}</td>
    <td>{bleu_fr:.3f}</td><td>{bleu_lr:.3f}</td>
  </tr>
"""

    html += """
</table>

<hr>
<p style="color:#8b949e; font-size:0.85rem">Generated from /tmp/ma_ttft/results.json</p>
</body>
</html>
"""

    (OUT / "report.html").write_text(html, encoding='utf-8')
    print("Report saved to", OUT / "report.html")

if __name__ == "__main__":
    data = load_data()
    generate_charts(data)
    generate_html(data)
