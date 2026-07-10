#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inject code-structure-driven selective recompute narrative into the deck.
Order: shift old 18-28 -> 19-29 (+1) -> insert new slide 18 -> rewrite slide 17
bottom (不足②) -> fix 4 prose refs -> add novelty bullets to TL;DR + 总结 ->
fix title count. Prints verification for every step."""
import re, sys

P = 'results/CODE_AWARE_LOSSY_KV_PROGRESS_R28_R39.html'
s = open(P, encoding='utf-8').read()
orig_len = len(s)
log = []

# ---- 1. SHIFT: old slides 18-28 -> 19-29 (+1), bump /28 -> /29 ----
def shift_di(m):
    n = int(m.group(1))
    nn = n + 1 if n >= 18 else n
    return f'data-index="主体 {nn} / 29 · ({nn})"'
s, c1 = re.subn(r'data-index="主体 (\d+) / 28 · \(\d+\)"', shift_di, s)

def shift_h1(m):
    n = int(m.group(1))
    nn = n + 1 if n >= 18 else n
    return f'class="num">({nn})</span>'
s, c2 = re.subn(r'class="num">\((\d+)\)</span>', shift_h1, s)
log.append(f"shift: data-index {c1}, h1 {c2} (expect 28 each)")

# ---- 2. REWRITE slide 17 bottom (距离真实 CacheBlend ... -> 不足② novelty) ----
SLIDE17_BOTTOM = '''  <h2 style="margin:6px 0 4px 20px;">不足② · Novelty 缺口（更根本）</h2>
  <div class="card gold" style="margin:0 0 0 20px;padding:5px 14px;">
    <p style="margin:2px 0;font-size:12.5px;"><b>诊断</b>：<b>code structure (AST) 只驱动「在哪切」-&gt; 影响复用率(速度)</b>；accuracy 杠杆 <b>FRAC =「recompute 前 N token」= 位置驱动、代码无关</b>。code structure 从未碰过 accuracy 侧 -&gt; 方法本质 = <b>KVCOMM(pool) + CacheBlend-lite(位置代理)</b>，无 code-aware accuracy 杠杆。</p>
    <p style="margin:2px 0;font-size:12.5px;">R34/R40-P3 type-aware FRAC 想补，但 gate 挂在 <b>type annotation（罕见）</b>上 -&gt; pandas 0.x no-op retired（slide 29）。<b style="color:var(--accent);">补救见 slide (18)</b>：让 code structure 驱动「recompute 什么」。</p>
  </div>
'''
anchor17 = '<h2 style="margin:6px 0 4px 20px;">距离真实 CacheBlend 还差什么 + 下一步</h2>'
i1 = s.find(anchor17)
assert i1 != -1, "slide17 anchor not found"
i2 = s.find('</section>', i1)
assert i2 != -1, "slide17 </section> not found"
s = s[:i1] + SLIDE17_BOTTOM + s[i2:]   # keep the </section>
log.append("slide17 bottom rewritten (不足② novelty)")

# ---- 3. INSERT new slide 18 (补救路线图) right after slide 17's </section> ----
NEW_SLIDE = '''<section class="slide" data-tag="补救路线图" data-tag-class="verified" data-index="主体 18 / 29 · (18)">
  <h1><span class="num">(18)</span> 补救路线图 · code-structure-driven selective recompute</h1>
  <h2 style="margin:4px 0 2px 20px;">让 code structure 决定「recompute 什么」，不只是「在哪切」</h2>
  <p class="sub" style="margin:0 0 6px 20px;">现状（slide 17 不足②）：AST 只驱动 chunking(速度)；FRAC =「前 N token」= 位置驱动、代码无关 -&gt; 无 code-aware accuracy 杠杆。补救 = recompute 决策从<b>位置</b>换成<b>代码结构</b>。</p>
  <table style="margin:0 0 0 20px;font-size:12px;">
    <tr><th style="width:14%;">方向</th><th style="width:33%;">机制</th><th style="width:21%;">信号</th><th style="width:11%;">可行性</th><th>novelty vs CacheBlend</th></tr>
    <tr style="background:#122117;"><td><b>A · node-kind FRAC</b></td><td>recompute 签名/控制流节点（def/class/if/return），copy docstring/boilerplate</td><td>AST 节点类型（<b>永远存在</b>）</td><td><span class="good">今晚</span></td><td>按<b>语义角色</b> gate，非 per-layer KV deviation</td></tr>
    <tr><td><b>B · dataflow</b></td><td>只 recompute 引用「上游已变 symbol」的 token，其余 copy</td><td>symbol def-use chain</td><td>中（静态分析）</td><td><b>高</b>：无 KV 方法用 code dataflow</td></tr>
    <tr><td><b>C · task-cycle</b></td><td>AST-diff 跨迭代，未变区域复用 KV，diff 区域 recompute</td><td>agent 迭代周期</td><td>中（接 R40-P2）</td><td><b>高</b>：跨 agent lifecycle</td></tr>
  </table>
  <div class="card gold" style="margin:6px 0 0 20px;padding:4px 12px;">
    <p style="margin:1px 0;font-size:12px;"><b>R34 教训（避免重蹈）</b>：R34/R40-P3 把 gate 挂在 <b>type annotation（罕见）</b>上 -&gt; pandas 0.x no-op retired（slide 29）。修法：① gate 在 <b>AST node kind</b>（永远存在）；② <b>先验证信号</b>；③ <b>等预算消融</b>（R34 缺消融 -&gt; 被当 global bump）。</p>
  </div>
  <h2 style="margin:6px 0 2px 20px;">决定性实验（novelty 成立与否）+ 先验证</h2>
  <p style="margin:0 0 0 20px;font-size:12px;">固定总 recompute budget <b>B</b>，等预算比 accuracy：<b>R32</b>(uniform) · <b>R38b</b>(position) · <b>A</b>(node-kind ✓) · <b>B</b>(dataflow ✓✓)。若 A/B &gt; R32/R38b @ equal B -&gt; 证明 code structure 买精度。<b>先验证（今晚）</b>：扩展 <code>measure_hkvd_by_position.py</code>（slide 8）测 <b>HKVD-by-node-kind</b>，若签名节点 deviation &gt; body（类比 pos1&gt;pos5 +7.2%）-&gt; 信号真实，policy 有据。</p>
</section>
'''
# find slide 17 </section> again (positions changed after step 2)
i1b = s.find('data-tag="现状反思"')
assert i1b != -1, "现状反思 section not found"
i2b = s.find('</section>', i1b)
assert i2b != -1
ins = i2b + len('</section>')
s = s[:ins] + '\n' + NEW_SLIDE + s[ins:]
log.append("new slide 18 (补救路线图) inserted")

# ---- 4. FIX 4 prose refs (scale-15 19->20, R40-P2 27->28) ----
for old, new in [('slide 19', 'slide 20'), ('slide 27', 'slide 28')]:
    c = s.count(old)
    s = s.replace(old, new)
    log.append(f"prose ref {old!r} -> {new!r}: {c} replaced")

# ---- 5. TL;DR: add novelty bullet before 局限 bullet ----
TLDR_BULLET = '<li><b>Novelty 路径</b>（slide 18）：当前 = KVCOMM + CacheBlend-lite，code structure 只驱动速度未驱动精度。补救 = <b>code-structure-driven selective recompute</b>（让 AST/dataflow 决定 recompute 什么），等预算消融为决定性实验。</li>\n    '
m = re.search(r'(</li>\s*)(<li><span class="warn">局限</span>)', s)
assert m, "TL;DR 局限 anchor not found"
s = s[:m.start()] + m.group(1) + TLDR_BULLET + m.group(2) + s[m.end():]
log.append("TL;DR novelty bullet added")

# ---- 6. 总结: prepend code-structure bullet as first future-direction ----
SUM_BULLET = '<li><b>★ 首要：code-structure-driven selective recompute</b>（slide 18）- 让 AST/dataflow 决定 recompute 什么，补 novelty 缺口。先跑 HKVD-by-node-kind 验证信号，再等预算消融（uniform vs position vs node-kind vs dataflow）。</li>\n    '
m2 = re.search(r'(  <ul>\n)(    <li>扩数据集)', s)
assert m2, "总结 future-direction anchor not found"
s = s[:m2.start()] + m2.group(1) + SUM_BULLET + m2.group(2) + s[m2.end():]
log.append("总结 future-direction bullet prepended")

# ---- 7. Title count: 主体 27 张 -> 主体 29 张 ----
c7 = s.count('主体 27 张')
s = s.replace('主体 27 张', '主体 29 张')
log.append(f"title count 主体 27 张 -> 29 张: {c7} replaced")

open(P, 'w', encoding='utf-8').write(s)
log.append(f"written: {orig_len} -> {len(s)} bytes (+{len(s)-orig_len})")
print('\n'.join(log))
