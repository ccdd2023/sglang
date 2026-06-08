#!/usr/bin/env python3
"""Generate HTML report and PNG charts for Code KV Reuse experiments."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# Load data
codebase_data = json.loads((OUT / "codebase_reuse/results.json").read_text())
kv_replace_data = json.loads((Path("/tmp/kv_replacement_results/results.json").read_text()))

# ========================================================================
# Chart 1: KV Reuse Volume — Large Codebase Multi-Agent
# ========================================================================
fig, ax = plt.subplots(figsize=(10, 6))
files = [r["name"] for r in codebase_data]
x = np.arange(len(files))
width = 0.35

a2_lossy = [r["a2_lossy"]["kv_reuse_mb"] for r in codebase_data]
a2_lossless = [r["a2_lossless"]["kv_reuse_mb"] for r in codebase_data]
a3_lossy = [r["a3_lossy"]["kv_reuse_mb"] for r in codebase_data]
a3_lossless = [r["a3_lossless"]["kv_reuse_mb"] for r in codebase_data]

bars1 = ax.bar(x - width/2, a2_lossy, width, label='A2 lossy', color='#3498db')
bars2 = ax.bar(x + width/2, a2_lossless, width, label='A2 lossless', color='#2ecc71')
ax.set_ylabel('KV Reuse (MB)')
ax.set_title('Large Codebase × Multi-Agent: KV Reuse Volume (Agent 2)')
ax.set_xticks(x)
ax.set_xticklabels(files, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "chart_kv_reuse_a2.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, a3_lossy, width, label='A3 lossy', color='#e74c3c')
bars2 = ax.bar(x + width/2, a3_lossless, width, label='A3 lossless', color='#9b59b6')
ax.set_ylabel('KV Reuse (MB)')
ax.set_title('Large Codebase × Multi-Agent: KV Reuse Volume (Agent 3)')
ax.set_xticks(x)
ax.set_xticklabels(files, rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "chart_kv_reuse_a3.png", dpi=150)
plt.close()

# ========================================================================
# Chart 2: Reuse Ratio Comparison
# ========================================================================
fig, ax = plt.subplots(figsize=(10, 6))
a2_lossy_ratio = [r["a2_lossy"]["reuse_ratio"] for r in codebase_data]
a2_lossless_ratio = [r["a2_lossless"]["reuse_ratio"] for r in codebase_data]
a3_lossy_ratio = [r["a3_lossy"]["reuse_ratio"] for r in codebase_data]
a3_lossless_ratio = [r["a3_lossless"]["reuse_ratio"] for r in codebase_data]

ax.plot(files, a2_lossy_ratio, 'o-', label='A2 lossy', color='#3498db', linewidth=2)
ax.plot(files, a2_lossless_ratio, 's--', label='A2 lossless', color='#2ecc71', linewidth=2)
ax.plot(files, a3_lossy_ratio, '^-', label='A3 lossy', color='#e74c3c', linewidth=2)
ax.plot(files, a3_lossless_ratio, 'd--', label='A3 lossless', color='#9b59b6', linewidth=2)
ax.set_ylabel('Reuse Ratio (%)')
ax.set_title('KV Reuse Ratio across Files and Agents')
ax.set_xticklabels(files, rotation=15, ha='right')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(OUT / "chart_reuse_ratio.png", dpi=150)
plt.close()

# ========================================================================
# Chart 3: Prefill Time Saved — KV Replacement
# ========================================================================
fig, ax = plt.subplots(figsize=(10, 6))
descs = [d["desc"] for d in kv_replace_data]
saved = [d.get("saved_prefill_ms", 0) for d in kv_replace_data]
colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in saved]

bars = ax.barh(descs, saved, color=colors)
ax.set_xlabel('Saved Prefill Time (ms)')
ax.set_title('KV Tensor Replacement: Prefill Time Saved per Block')
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, saved):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.0f}ms', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "chart_prefill_saved.png", dpi=150)
plt.close()

# ========================================================================
# Chart 4: Gate Effectiveness (SWE-bench summary)
# ========================================================================
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Rejected', 'Accepted']
sizes = [45, 5]
colors = ['#e74c3c', '#2ecc71']
explode = (0.05, 0.1)

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
ax.set_title('SWE-bench Lite Gate Decisions (50 tasks)', fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "chart_gate_swe.png", dpi=150)
plt.close()

# ========================================================================
# Chart 5: End-to-end Latency Comparison (SWE-bench)
# ========================================================================
fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Accepted (5 tasks)', 'Rejected (45 tasks)']
lossy = [1924, 2176]
lossless = [1960, 2220]

x = np.arange(len(categories))
width = 0.3
bars1 = ax.bar(x - width/2, lossy, width, label='Lossy', color='#3498db')
bars2 = ax.bar(x + width/2, lossless, width, label='Lossless', color='#95a5a6')

ax.set_ylabel('Avg Latency (ms)')
ax.set_title('SWE-bench Lite: End-to-End Latency')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{bar.get_height():.0f}',
            ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{bar.get_height():.0f}',
            ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(OUT / "chart_latency_swe.png", dpi=150)
plt.close()

# ========================================================================
# Chart 6: Large Block Gate Decisions
# ========================================================================
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Reject (27)', 'Accept (18)']
sizes = [27, 18]
colors = ['#e74c3c', '#2ecc71']
explode = (0.05, 0.1)

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
ax.set_title('Large Code-block Gate Decisions (45 tasks)', fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "chart_gate_large.png", dpi=150)
plt.close()

print("All charts saved to", OUT)
