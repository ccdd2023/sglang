# AgentTemplateKV 论文图表生成 Prompt

用于 AI 图像生成模型（如 GPT-Image、DALL-E、Midjourney、Stable Diffusion 等）

---

## Figure 1: 系统架构图 (System Architecture)

**文件**: `figures/architecture.tex`

```
A clean, professional systems architecture diagram for an academic paper about LLM serving infrastructure.

Three-layer architecture diagram with boxes and arrows:

TOP LAYER "Orchestration Layer" (light blue background):
- Left box: "Template Synthesizer" - uses LLM to generate agent DAG templates
- Right box: "Template-Aware Planner" - selects and adapts templates for tasks
- Arrow from Synthesizer to Planner

MIDDLE LAYER "KV Management Layer" (light green background):
- Center box: "Agent-Aware KV Manager" with three sub-components:
  • Retention Policies
  • Reuse Mappings
  • Preallocation
- Arrow from Planner down to KV Manager

BOTTOM LAYER "Serving Engine Layer" (light orange background):
- Left box: "Tool & Resource Monitor" with icons for CPU, GPU, network
- Right box: "PagedAttention Engine (vLLM)"
- Arrows from KV Manager down to both boxes
- Arrow connecting Monitor to KV Manager

Clean white background, minimal style, professional academic look.
Font: Helvetica or Arial style labels.
Color scheme: Blue (#4A90E2) for orchestration, Green (#27AE60) for KV management, Orange (#F39C12) for serving.
Arrows: solid gray lines with arrowheads.
Include subtle drop shadows on boxes.
Layout: vertical stacking with clear separation.
```

**提示**: 英文prompt效果最佳，可根据需要调整

---

## Figure 2: KV 生命周期状态图 (KV Lifecycle)

**文件**: `figures/kv-lifecycle.tex`

```
A clean finite state machine diagram for academic paper showing KV cache lifecycle.

Six oval/rectangular states arranged in a flow:

CREATED (blue outline) -> ACTIVE (green fill) -> PINNED (yellow fill) -> ACTIVE
     |                        |                      |
     v                        v                      v
  (back to)              REUSED                  EVICTED
   ACTIVE                 (purple)                 (red)
     |                        |                      |
     v                        v                      v
  (reuse)                  (back)               (offload)
```

具体布局：
```
                    ┌──────────────┐
                    │   ACTIVE     │
                    │   (green)    │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────────┐
    │ CREATED │    │  PINNED  │    │   REUSED     │
    │ (blue)  │───▶│ (yellow) │───▶│  (purple)    │
    └──────────┘    └─────┬────┘    └──────────────┘
                          │
                          ▼ TTL Expires
                    ┌──────────┐
                    │ EVICTED  │
                    │  (red)   │
                    └──────────┘
```

Transition labels:
- Created -> Active: "prefill"
- Active -> Pinned: "tool call"
- Pinned -> Active: "resume"
- Active -> Reused: "cross-agent reuse"
- Pinned -> Evicted: "TTL expire"
- Active -> Offloaded: "memory pressure"

Style: Clean state diagram, white background, academic paper quality.
Arrows with labels in small gray text.
```

---

## Figure 3: Agent DAG 示例 (Bug Fixing Template)

**文件**: `figures/dag-example.tex`

```
A directed acyclic graph (DAG) diagram for an academic paper showing multi-agent code generation workflow.

Five circular nodes in a hierarchical layout:

Center-left: P (Planner) - analyzes issue description
Below P: C (Coder) - generates code patches, connects to compiler/linter tools
Right of C: T (Tester) - runs tests, connects to test-runner
Below C: R (Reviewer) - reviews code, connects to static analyzer
Below R: D (Debugger) - fixes issues if needed

Arrows with labels:
- P -> C: "issue analysis"
- C -> T: "patch" (upper arrow)
- C -> R: "patch" (lower arrow)
- T -> D: "test results"
- R -> D: "review comments"
- D -> C: "fixes" (dashed curved arrow looping back)

Style: Clean DAG diagram, light pastel colors for nodes.
- Planner: light blue circle
- Coder: light green circle
- Tester: light red/coral circle
- Reviewer: light purple circle
- Debugger: light orange circle

White background, professional academic diagram style.
Arrows in dark gray with small text labels.
Include subtle icons or text labels inside circles (P, C, T, R, D).
```

---

## Figure 4: GPU 内存使用对比 (Memory Comparison)

**文件**: `figures/eval-memory.tex`

```
A clean bar chart for academic paper comparing GPU memory usage across systems.

Chart specifications:
- Title: "GPU Memory Usage Comparison" at top, bold, 14pt
- Y-axis: "GPU Memory (GB)" from 0 to 50
- X-axis: four bars labeled "vLLM", "Continuum", "RelayCaching", "AgentTemplateKV"
- Four vertical bars of different colors:
  • vLLM: 42.3 GB - blue bar
  • Continuum: 38.1 GB - green bar
  • RelayCaching: 35.2 GB - orange bar
  • AgentTemplateKV: 28.7 GB - red/coral bar (highlighted as best)
- Values displayed above each bar
- Clean white background
- Minimal grid lines (light gray)
- Professional academic style, no 3D effects
- Legend in bottom right corner

Color palette: Blues and warm tones
- vLLM: #3498DB (blue)
- Continuum: #2ECC71 (green)
- RelayCaching: #E67E22 (orange)
- AgentTemplateKV: #E74C3C (red) - make this bar slightly taller/different to emphasize improvement
```

---

## Figure 5: 端到端延迟对比 (Latency Comparison)

**文件**: `figures/eval-latency.tex`

```
A clean bar chart for academic paper comparing end-to-end latency across systems.

Chart specifications:
- Title: "End-to-End Latency Comparison" at top, bold, 14pt
- Subtitle or note: "Average Job Completion Time" in smaller text
- Y-axis: "Average JCT (seconds)" from 0 to 180
- X-axis: four bars labeled "vLLM", "Continuum", "RelayCaching", "AgentTemplateKV"
- Four vertical bars:
  • vLLM: 156.2s - blue bar (highest)
  • Continuum: 89.4s - green bar
  • RelayCaching: 72.1s - orange bar
  • AgentTemplateKV: 48.3s - red/coral bar (lowest, highlighted)
- Values displayed above each bar
- Clean white background
- Minimal grid lines
- Professional academic style
- Consider adding a dashed line or marker showing improvement percentage

Color scheme matching Figure 4:
- vLLM: #3498DB
- Continuum: #2ECC71
- RelayCaching: #E67E22
- AgentTemplateKV: #E74C3C
```

---

## Figure 6: TTL 公式可视化 (TTL Formula Illustration)

**文件**: `figures/ttl-formula.tex`

```
A clean academic diagram illustrating the TTL (Time-To-Live) calculation formula for KV cache management.

Layout: Horizontal flow diagram showing formula components

Center equation (large, prominent):
TTL(kv) = TTL_base × α_role × α_centrality × α_resource

Four boxes below equation showing each multiplier:

┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ α_role      │   │ α_centrality│   │ α_resource  │   │ TTL_base    │
│             │   │             │   │             │   │             │
│ Agent type  │   │ DAG position│   │ GPU memory  │   │ Tool latency│
│ multiplier  │   │ multiplier  │   │ multiplier  │   │ baseline    │
│             │   │             │   │             │   │             │
│ Range: 1.0- │   │ Range: 0.8- │   │ Range: 0.5- │   │ e.g., 10s   │
│ 2.0         │   │ 1.5         │   │ 1.5         │   │             │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘

Arrows pointing from boxes to equation terms.

Color coding:
- α_role: Blue box
- α_centrality: Green box
- α_resource: Orange box
- TTL_base: Gray box

Style: Clean, minimal, academic paper quality.
White background.
Use mathematical notation style fonts.
Include small explanation text below each box.
```

---

## Figure 7: 消融实验结果 (Ablation Study)

**文件**: `figures/eval-ablation.tex` (新建)

```
A grouped bar chart for academic paper showing ablation study results.

Chart showing incremental improvements from adding components:

Five grouped bars (4 metrics each):

1. Baseline (vLLM): Blue - Memory 42.3GB, JCT 156.2s, Reuse 12%
2. + Template: Light blue - Memory 36.1GB, JCT 98.4s, Reuse 52%
3. + Tool Profiling: Light green - Memory 32.8GB, JCT 72.1s, Reuse 68%
4. + Error-Aware: Light yellow - Memory 30.2GB, JCT 58.7s, Reuse 78%
5. + Resource Feedback: Red/coral - Memory 28.7GB, JCT 48.3s, Reuse 85%

Show clear downward trend for Memory and JCT bars.
Show clear upward trend for Reuse Rate.

Layout: Grouped bar chart with four groups (one per metric) or three separate small multiples.
Include legend showing each configuration.
Clean white background, minimal styling.
```

---

## Figure 8: Multi-Turn Agent Timeline (Multi-Turn交互时序图)

**文件**: `figures/multi-turn-timeline.tex` (新建)

描述: 展示多轮Agent交互中KV cache的保留和复用时机

```
A clean sequence diagram for academic paper showing multi-turn agent interaction and KV cache management over time.

Timeline layout (horizontal):

Time flows left to right with vertical dashed lines marking turns.

Row 1 (Agent Coder):
├── Turn 1: [LLM: Generate Code] ────▶ [Tool: Compile] ────▶ KV cached
├── Turn 2: [LLM: Review Error] ◀─── [KV reused] ◀─────────────┘
└── Turn 3: [LLM: Fix Code] ────▶ [Tool: Compile]

Row 2 (Agent Tester):
├── Turn 1: [LLM: Analyze Tests] ◀─── [KV reused: code from Coder]
└── Turn 2: [LLM: Report Results]

Row 3 (GPU Memory):
├── Shows KV blocks being allocated, pinned during tool call, reused, evicted

Show colored KV blocks:
- Green: Active KV
- Yellow: Pinned (during tool call)
- Purple: Reused by another agent
- Red outline: Evicted

Style: Clean sequence diagram, white background.
Use subtle arrows and boxes.
Time arrows at top.
Memory row at bottom with block diagram.
```

---

## Figure 9: Template Synthesis Process (模板合成流程图)

**文件**: `figures/template-synthesis.tex` (新建)

描述: 展示LLM多轮对话生成Agent模板的完整流程

```
A clean flowchart for academic paper showing the multi-round LLM planning process for template synthesis.

Flowchart layout top to bottom:

1. Start: Task Category C + Available Tools T
          │
          ▼
2. Build Template Prompt
   LLM Prompt includes: task description, tool list, role definitions
          │
          ▼
3. Round 1: LLM generates initial roles and edges
          │
          ▼
4. Role Extraction + Edge Extraction
          │
          ▼
5. Round 2: LLM refines with tool assignments
          │
          ▼
6. More roles and edges added (union operation)
          │
          ▼
7. Round N: Continue until convergence or max rounds
          │
          ▼
8. Construct DAG from accumulated roles and edges
          │
          ▼
9. Decision: Validate DAG (acyclic? tool coverage?)
   ├── Yes → Return Agent Template G
   └── No → Refine DAG (fix cycles, add missing tools)
          │
          └──▶ Return Agent Template G

Style: Clean flowchart with rounded rectangles for processes,
diamonds for decisions, parallelograms for inputs/outputs.
Use subtle color coding for iteration rounds (lighter to darker blue).
Include small icons: LLM brain icon, DAG graph icon.
White background, professional academic style.
```

---

## Figure 10: Cross-Agent KV Reuse示意图

**文件**: `figures/kv-reuse.tex` (新建)

描述: 展示跨Agent的KV cache复用机制

```
A conceptual diagram for academic paper showing how KV caches are reused across agents.

Layout: Three columns showing the reuse flow

LEFT COLUMN (Agent Coder):
┌─────────────────────────────────┐
│ Code generated by Coder        │
│ "function bubble_sort(arr):"   │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ KV Cache Block for          │ │
│ │ "function bubble_sort..."   │ │
│ │ Layer 0-32, Tokens 1-150   │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘

CENTER (Reuse Engine):
┌─────────────────────────────────┐
│     KV Reuse Mapping            │
│                                 │
│  "function bubble_sort..." ────▶│
│        │                        │
│        ▼                        │
│  Sparse recomputation check     │
│  (positions 1-3, 45, 89)       │
│        │                        │
│        ▼                        │
│  Match: 98% similarity          │
└─────────────────────────────────┘

RIGHT COLUMN (Agent Tester):
┌─────────────────────────────────┐
│ Agent Tester receives code      │
│ "function bubble_sort(arr):"    │
│                                 │
│ ┌─────────────────────────────┐ │
│ │ Prefill with Reused KV      │ │
│ │ ✓ 98% of tokens reused     │ │
│ │ ◐ 2% recomputed            │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘

Arrows: Left → Center: "Extract KV signatures"
        Center → Right: "Map to downstream agent"

Style: Clean conceptual diagram, white background.
Use boxes with subtle borders.
Include small text showing token counts and percentages.
Color: Green for reused portions, Yellow for recomputed portions.
```

---

## Figure 11: Tool分类与特性矩阵 (Tool Classification Matrix)

**文件**: `figures/tool-classification.tex` (新建)

描述: 可视化展示工具分类体系和特性

```
A clean matrix/grid diagram for academic paper showing tool classification.

Layout: 2D grid with tool types and characteristics

Y-axis (Tool Types - rows):
├── Compiler
├── Unit Test
├── Static Analyzer
├── Linter
├── CI/CD API
└── Dependency Install

X-axis (Characteristics - columns):
├── Locus (Local/Remote)
├── Modality (CPU/IO/Network)
├── Latency (Low/Medium/High)
└── Error Mode (Semantic/Network/Timeout)

Cell content: Colored icons or symbols

Compiler:      Local | CPU | Low | Semantic (✓)
Unit Test:     Local | CPU/IO | Medium | Assertion (⚠)
Static Analyzer: Local | CPU | Low | Warning (△)
Linter:        Local | CPU | Low | Syntax (⚡)
CI/CD API:     Remote | Network | High | Network/Timeout (☁)
Dependency:    Local | IO | Medium | Network/Env (⬇)

Legend at bottom:
- Color coding for Latency: Green=Low, Yellow=Medium, Red=High
- Icons for Error Mode

Style: Clean table/matrix visualization, white background.
Use subtle grid lines.
Include tool icons if possible.
Professional academic diagram style.
```

---

## Figure 12: Error-Mode-Aware Retention决策流程

**文件**: `figures/error-retention.tex` (新建)

描述: 展示错误类型如何影响KV retention策略

```
A clean decision flowchart for academic paper showing error-mode-aware KV retention adjustment.

Top section: Input
┌─────────────────────────────────────┐
│ KV Object + Tool Result + Tool Profile │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│    Classify Error Type      │
└─────────────┬───────────────┘
              │
     ┌────────┼────────┐
     │        │        │
     ▼        ▼        ▼
┌─────────┐ ┌─────────┐ ┌─────────────────┐
│Network  │ │Semantic │ │Resource         │
│Failure  │ │Failure  │ │Exhaustion       │
│(transient)│ │(compiler│ │(OOM, CPU max) │
└────┬────┘ │ error)  │ └────────┬────────┘
     │       └────┬────┘          │
     │            │               │
     │            │               │
     ▼            ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ TTL × 1.5       │ │ TTL × 2.0       │ │ TTL × 0.5       │
│ Priority + 1    │ │ Priority + 2    │ │ Priority - 1    │
│ "Wait for       │ │ "Agent will     │ │ "May fail       │
│  retry/backoff" │ │  retry fixes"   │ │  again"         │
└─────────────────┘ └─────────────────┘ └─────────────────┘
     │                    │                    │
     └────────────────────┼────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Return Adjusted TTL   │
              │ and Priority to KV    │
              │ Manager               │
              └───────────────────────┘

Style: Clean decision tree/flowchart, white background.
Use different colors for different error types:
- Network: Blue
- Semantic: Yellow/Orange
- Resource: Red
Include small icons for each error type.
White background, professional academic style.
```

---

## Figure 13: Resource Feedback与KV调度的闭环控制

**文件**: `figures/resource-feedback.tex` (新建)

描述: 展示硬件资源监控如何影响KV管理决策

```
A clean control loop diagram for academic paper showing the closed-loop resource feedback system.

Layout: Circular feedback loop with four components

TOP: Resource Monitor
┌─────────────────────────────────────────┐
│ Tool & Resource Monitor                 │
│                                         │
│ Metrics:                                │
│ • CPU utilization                       │
│ • GPU memory pressure                   │
│ • IO queue depth                        │
│ • Network RTT                           │
└─────────────────────────────────────────┘
         │
         │ Metrics signals
         ▼
┌─────────────────────────────────────────┐
│    Agent-Aware KV Manager               │
│                                         │
│ Decisions:                              │
│ • Shorten TTLs when CPU/IO high        │
│ • Evict low-priority KV when memory    │
│ • Adjust reuse aggressiveness          │
│ • Offload cold KV to SSD               │
└─────────────────────────────────────────┘
         │
         │ KV allocation/eviction
         ▼
┌─────────────────────────────────────────┐
│    PagedAttention Serving Engine        │
│                                         │
│ GPU Memory Blocks:                      │
│ [█████ Active KV █████]                │
│ [████ Pinned KV ████]                  │
│ [██ Reused KV ██]                      │
│ [░░░░░░░░░░░░░░░░░░░░░░░░] (free)     │
└─────────────────────────────────────────┘
         │
         │ Tool execution / LLM inference
         ▼
Back to Resource Monitor (feedback loop)

Style: Clean control system diagram, white background.
Use curved arrows for the feedback loop.
Include gauge/meter icons for metrics.
Color code: Green for healthy, Yellow for warning, Red for critical.
Professional academic diagram style.
```

---

## Figure 14: Fragmentation-Aware KV Allocation

**文件**: `figures/fragmentation.tex` (新建)

描述: 展示模板感知的KV block分配策略对比

```
A conceptual diagram comparing naive vs template-aware KV allocation.

Layout: Two side-by-side diagrams

LEFT DIAGRAM: Naive Allocation (inefficient)
┌─────────────────────────────────────────┐
│ GPU Memory Block Layout                 │
├─────────────────────────────────────────┤
│ ██ Agent Coder (long-lived) ████████████│
│ ██ Agent Tester ██                      │
│ ██ Agent Coder (continued) ██████████████│
│ ██ Agent Reviewer ██                    │
│ ██ Agent Coder (continued) ██████████████│
│ ██ Agent Tester ██                      │
│                                         │
│ Problems:                               │
│ • External fragmentation (white space) │
│ • Non-contiguous blocks                │
│ • Poor locality                         │
└─────────────────────────────────────────┘

RIGHT DIAGRAM: Template-Aware Allocation (efficient)
┌─────────────────────────────────────────┐
│ GPU Memory Block Layout                 │
├─────────────────────────────────────────┤
│ ████████████████████ Agent Coder ████████│
│ ████████████████████ (large prealloc)  │
├─────────────────────────────────────────┤
│ ██ Agent Tester ██ │ ██ Agent Reviewer ██│
├─────────────────────────────────────────┤
│ ██ Agent Tester ██ │ ██ Agent Reviewer ██│
├─────────────────────────────────────────┤
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ (Compacted short-lived agents)          │
│                                         │
│ Benefits:                               │
│ • Contiguous long-lived blocks          │
│ • Compact short-lived agents           │
│ • Minimal fragmentation                 │
└─────────────────────────────────────────┘

Style: Memory block diagram with colored rectangles.
Use hatch patterns or colors to distinguish agents.
Include annotations for problems/benefits.
White background, professional academic style.
```

---

## Figure 15: RelayCaching vs AgentTemplateKV对比

**文件**: `figures/comparison.tex` (新建)

描述: 对比RelayCaching和AgentTemplateKV在KV reuse上的差异

```
A comparison diagram for academic paper showing key differences between RelayCaching and AgentTemplateKV.

Layout: Two columns with feature comparison

LEFT COLUMN: RelayCaching
┌─────────────────────────────────────────┐
│ RelayCaching                            │
├─────────────────────────────────────────┤
│ ✓ KV Reuse:                             │
│   Content-level matching                │
│   Sparse recomputation                  │
│                                         │
│ ✗ Missing:                              │
│   • Agent role awareness                │
│   • DAG structure knowledge             │
│   • Tool semantics                      │
│   • Error mode handling                 │
│   • Resource monitoring                 │
│                                         │
│ KV cache seen as opaque text            │
│ Reuse based on string matching only     │
└─────────────────────────────────────────┘

RIGHT COLUMN: AgentTemplateKV
┌─────────────────────────────────────────┐
│ AgentTemplateKV                         │
├─────────────────────────────────────────┤
│ ✓ KV Reuse:                             │
│   Template-aware matching               │
│   DAG structure optimization            │
│   Tool-category scheduling              │
│   Error-mode retention                  │
│   Resource feedback                     │
│                                         │
│ KV cache enriched with:                 │
│   • Agent role (Coder/Test/Reviewer)    │
│   • DAG position (depth, fan-in/out)   │
│   • Tool category                       │
│   • Error history                       │
│   • Resource state                      │
│                                         │
│ Multi-dimensional policy optimization   │
└─────────────────────────────────────────┘

Bottom: Arrow or bridge connecting both showing "Extension"

Style: Clean comparison table, white background.
Use checkmarks (green) and crosses (red) for features.
Include small icons for each capability.
AgentTemplateKV column should appear more feature-rich.
```

---

## 通用风格指南

### 配色方案
```
Primary colors:
- Blue: #4A90E2 (orchestration, headers)
- Green: #27AE60 (KV management, positive results)
- Orange: #F39C12 (serving layer, warnings)
- Red: #E74C3C (eviction, best performance)
- Purple: #9B59B6 (reuse, special states)

Neutral:
- White background: #FFFFFF
- Light gray grid: #E0E0E0
- Dark text: #333333
- Arrow color: #666666
```

### 图表规格
```
- Format: PNG or PDF with transparent/white background
- Resolution: 300 DPI minimum
- Dimensions: 8-10 inches width for single column, 12-16 inches for double column
- Font: Sans-serif (Helvetica, Arial, or system default)
- Title: 14pt bold
- Labels: 10-12pt
- Arrow heads: standard triangle style
```

### AI图像生成技巧
1. 使用英文prompt
2. 指定 "academic paper style", "clean diagram", "professional"
3. 避免生成文字错误，重要标签单独添加
4. 使用简单背景
5. 指定宽高比适合LaTeX使用
