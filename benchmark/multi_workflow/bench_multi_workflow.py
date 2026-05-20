"""
Multi-Workflow KVFlow Benchmark: Tests Priority vs LRU under realistic
multi-agent multi-workflow scenarios.

Key improvements over bench_priority.py:
  1. Multi-workflow concurrency: truly parallel workflows, not just
     sequential steps within one workflow
  2. Realistic code-generation prefixes: uses authentic Python/JS code fragments
     (import statements, function signatures, docstrings) that naturally share
     across agents/roles, enabling measurable cross-workflow KV reuse.
  3. Multi-level sharing structure:
       Tier-0 (universal):    system instructions shared by ALL agents
       Tier-1 (role-based):  role-specific imports+signatures shared across workflows
       Tier-2 (workflow):     task context unique to one workflow
       Tier-3 (dynamic):      per-request suffix (random seed varies each round)
  4. KVFlow-aware scheduling: passes next_agent_hint to server so it can
     proactively prefetch the next agent's Tier-1 prefix.
  5. write_back HiCache: avoids the locked-node deadlock that plagued
     write_through + prefetch in the original benchmarks
  6. Per-round detailed metrics: tracks which prefixes were cached and why
  7. Cross-workflow sharing analysis: reports the actual KV reuse rate
     (what fraction of KV compute is avoided by sharing).

Sharing analysis:
  - Tier-0 universal (e.g. "You are a helpful assistant...") is shared by ALL agents
    -> 100% KV reuse across all workflows
  - Tier-1 role-based (e.g. "from typing import Dict, List") shared by same role
    -> N_workflows × 1/K sharing
  - Tier-2 workflow-specific -> 0% cross-workflow sharing (but Priority helps)
  - Tier-3 dynamic suffix -> 0% sharing (truly unique per round)

Cache pressure calculation:
  - Each workflow puts ~(T0 + T1 + T2 + T3) tokens per agent step
  - N workflows × M agents × N rounds = total cache pressure
  - With sharing: T0 tokens + N×T1 tokens + N×M×T2 tokens
    (Tier-3 has no cache benefit, it's always unique)

Usage:
  # Terminal 1: start server with Priority
  python -m sglang.launch_server \
    --model-path /path/to/Qwen3-8B \
    --port 30300 \
    --radix-eviction-policy priority \
    --max-total-tokens 60000 \
    --enable-cache-report

  # Terminal 2: run multi-workflow benchmark
  python -m benchmark.multi_workflow.bench_multi_workflow \
    --config kvflow \
    --host 127.0.0.1 --port 30300 \
    --num-workflows 4 --agents-per-workflow 5 \
    --tier0-len 512 --tier1-len 1024 --tier2-len 512 \
    --suffix-len 64 --output-len 64 \
    --num-rounds 5 --warmup-rounds 1 \
    --output-dir /tmp/kvflow_results

  # Compare with LRU baseline (run on separate server instance):
  # (same command with --config lru_nocache, server with --radix-eviction-policy lru)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/bench_multi_workflow.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """An agent with a multi-tier prefix for realistic MAS code-generation sharing.

    Tier-0 (universal): system prompt shared by ALL agents across ALL workflows
    Tier-1 (role-based): role-specific imports+signatures shared by same role
                         across workflows
    Tier-2 (workflow): task context unique to this workflow
    Tier-3 (dynamic): per-request suffix, varies each round (always unique)
    """
    workflow_id: int
    agent_id: str           # e.g. "w0-a0", "w1-a3"
    role: str               # "PLANNER", "ARCHITECT", etc.
    tier0_text: str         # universal system prompt
    tier1_text: str         # role-based imports+signature (shared across workflows)
    tier2_text: str         # workflow-specific task context
    tier0_tokens: int
    tier1_tokens: int
    tier2_tokens: int

    def build_full_prefix(self, suffix_text: str) -> str:
        """Assemble full prompt for this agent step."""
        return f"{self.tier0_text}\n\n{self.tier1_text}\n\n{self.tier2_text}\n\nUser: {suffix_text}"


@dataclass
class StepResult:
    workflow_id: int
    agent_id: str
    round_idx: int
    step_idx: int
    ttft_ms: float
    e2e_ms: float
    output_tokens: int
    priority: Optional[int] = None
    # KV reuse analysis
    kv_reused_tokens: int = 0   # tokens loaded from cache (L3 hit)
    kv_total_tokens: int = 0     # total KV tokens processed
    cache_hit: bool = False       # whether this was a cache hit


@dataclass
class WorkflowResult:
    workflow_id: int
    round_results: List[List[StepResult]] = field(default_factory=list)


@dataclass
class SharedCounter:
    """Thread-safe shared counter for global step numbering across workflows."""
    value: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def get_and_increment(self, increment: int = 1) -> int:
        async with self.lock:
            result = self.value
            self.value += increment
            return result


# ---------------------------------------------------------------------------
# DAG Workflow Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """DAG 中的单个节点配置"""
    node_id: str
    role: str
    dependencies: List[str]
    parallel_group: Optional[str] = None
    description: str = ""

    def __hash__(self):
        return hash(self.node_id)


@dataclass
class DAGConfig:
    """DAG workflow 完整配置"""
    name: str
    description: str
    nodes: Dict[str, DAGNode]
    execution_order: List[List[str]]  # 按执行顺序分组的节点列表
    tier1_templates: Dict[str, int]  # role -> template index

    def get_successors(self, node_id: str) -> List[str]:
        """获取直接后继节点列表"""
        successors = []
        for nid, node in self.nodes.items():
            if node_id in node.dependencies:
                successors.append(nid)
        return successors

    def get_predecessors(self, node_id: str) -> List[str]:
        """获取直接前驱节点列表"""
        return self.nodes[node_id].dependencies.copy()

    def is_ready(self, node_id: str, completed: set) -> bool:
        """检查节点是否就绪（所有依赖已完成）"""
        return all(dep in completed for dep in self.nodes[node_id].dependencies)

    def get_depth(self, node_id: str) -> int:
        """计算节点在 DAG 中的深度（从根节点开始的最长路径）"""
        visited = set()
        def _get_depth_recursive(nid: str, visited: set) -> int:
            if nid in visited:
                return 0
            deps = self.nodes[nid].dependencies
            if not deps:
                return 0
            visited.add(nid)
            max_depth = 0
            for dep in deps:
                max_depth = max(max_depth, _get_depth_recursive(dep, visited.copy()))
            return max_depth + 1
        return _get_depth_recursive(node_id, visited)

    def get_critical_path_length(self, node_id: str) -> int:
        """计算从该节点到叶子的最长路径"""
        successors = self.get_successors(node_id)
        if not successors:
            return 1
        return 1 + max(self.get_critical_path_length(s) for s in successors)

    def get_downstream_count(self, node_id: str) -> int:
        """计算依赖该节点的下游节点数量"""
        count = 0
        for nid, node in self.nodes.items():
            if node_id in node.dependencies:
                count += 1
                count += self.get_downstream_count(nid)
        return count

    def get_critical_path_distance(self, node_id: str) -> int:
        """计算从节点到叶子节点的最长路径距离（步数）。

        这个值表示节点在 DAG 中的"关键路径距离"：
        - 值越大 → 离叶子越远 → 越早被处理 → 越早可以驱逐
        - 值越小 → 离叶子越近 → 越晚被处理 → 越晚需要 → 应保护

        例如 diamond_6agent：
        - PLANNER: distance=3（到叶子 TESTER 最长 3 步）
        - ARCHITECT/REVIEWER: distance=2（到叶子 2 步）
        - IMPLEMENTER: distance=1（到叶子 1 步）
        - TESTER: distance=0（就是叶子）
        """
        return self.get_critical_path_length(node_id)

    def calculate_priority(self, node_id: str, global_step_counter: int) -> int:
        """计算节点的优先级（基于 KVFlow 论文公式）。

        KVFlow 论文公式: priority = global_step_counter + steps_to_execution

        PriorityStrategy 驱逐逻辑:
        - 返回 (-effective_priority, -last_access_time)
        - effective_priority 越大 → -effective_priority 越小 → 越早被驱逐
        - 所以 priority 值越大 = 越早被驱逐

        在 DAG 执行中:
        - global_step_counter 是当前全局 step 计数（递增）
        - steps_to_execution 是从该节点到叶子节点的最长路径步数
        - TESTER（叶子）: steps_to_execution=1，最小，最后被驱逐
        - PLANNER（根）: steps_to_execution=4，最大，最早被驱逐

        公式: priority = global_step_counter + steps_to_execution
        """
        steps_to_execution = self.get_critical_path_length(node_id)
        # KVFlow 论文公式
        priority = global_step_counter + steps_to_execution
        return priority

    def get_convergence_factor(self, node_id: str) -> int:
        """获取节点的汇聚因子（已弃用，保留接口兼容）"""
        downstream = self.get_downstream_count(node_id)
        return downstream * 10 + (20 if downstream > 0 else 0)


@dataclass
class DAGAgentConfig:
    """DAG workflow 中的 agent 配置"""
    workflow_id: int
    node_id: str
    agent_id: str
    role: str
    parallel_group: Optional[str]
    tier0_text: str
    tier1_text: str
    tier2_text: str
    tier0_tokens: int
    tier1_tokens: int
    tier2_tokens: int
    execution_depth: int = 0

    def build_full_prefix(self, suffix_text: str) -> str:
        """组装完整的 prompt"""
        return f"{self.tier0_text}\n\n{self.tier1_text}\n\n{self.tier2_text}\n\nUser: {suffix_text}"


def load_dag_config(config_path: str) -> DAGConfig:
    """从 JSON 文件加载 DAG 配置"""
    with open(config_path, 'r') as f:
        data = json.load(f)

    nodes = {}
    for node_id, node_data in data["nodes"].items():
        nodes[node_id] = DAGNode(
            node_id=node_id,
            role=node_data["role"],
            dependencies=node_data["dependencies"],
            parallel_group=node_data.get("parallel_group"),
            description=node_data.get("description", ""),
        )

    return DAGConfig(
        name=data["dag_name"],
        description=data.get("description", ""),
        nodes=nodes,
        execution_order=data["execution_order"],
        tier1_templates=data.get("tier1_templates", {}),
    )


# ---------------------------------------------------------------------------
# Real Template Support
# ---------------------------------------------------------------------------

@dataclass
class RealTemplate:
    """Real template from MAScoder export."""
    template_id: str
    task_type: str
    task_family: str
    workflow_key: str
    agent_schedule: List[str]
    tool_plan: List[str]
    structural_fingerprint: str
    role_prefixes: Dict[str, str]  # role -> prefix text
    tier1_tokens: Dict[str, int]  # role -> token count


def load_real_templates(
    template_path: str,
    tokenizer=None,
) -> Tuple[List[RealTemplate], Dict[str, str]]:
    """加载真实 MAScoder 模板用于 KVFlow benchmark。

    Args:
        template_path: 导出模板 JSON 文件的路径
        tokenizer: 可选的 tokenizer 用于计算 token 数

    Returns:
        - templates: 模板列表
        - role_prefixes: role -> prefix text 的映射
    """
    templates = []

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load real templates from {template_path}: {e}")
        return [], {}

    # Extract templates
    template_list = data.get("templates", [])
    logger.info(f"Loaded {len(template_list)} templates from {template_path}")

    # Collect all role prefixes
    all_role_prefixes: Dict[str, str] = {}
    role_token_counts: Dict[str, int] = {}

    for tpl_data in template_list:
        role_prefixes = {}
        for rp in tpl_data.get("role_prefixes", []):
            role = rp.get("role", "")
            prefix_text = rp.get("prefix_text", "")
            if role and prefix_text:
                role_prefixes[role] = prefix_text
                if role not in all_role_prefixes:
                    all_role_prefixes[role] = prefix_text
                    if tokenizer:
                        tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
                        role_token_counts[role] = len(tokens)
                    else:
                        role_token_counts[role] = rp.get("tier1_tokens", 100)

        templates.append(RealTemplate(
            template_id=tpl_data.get("template_id", ""),
            task_type=tpl_data.get("task_type", "unknown"),
            task_family=tpl_data.get("task_family", ""),
            workflow_key=tpl_data.get("workflow_key", ""),
            agent_schedule=tpl_data.get("agent_schedule", []),
            tool_plan=tpl_data.get("tool_plan", []),
            structural_fingerprint=tpl_data.get("structural_fingerprint", ""),
            role_prefixes=role_prefixes,
            tier1_tokens=role_token_counts,
        ))

    return templates, all_role_prefixes


def get_real_template_tier1(
    role: str,
    real_templates: List[RealTemplate],
    default_text: str,
    default_tokens: int,
) -> Tuple[str, int]:
    """从真实模板获取 Tier-1 prefix。

    Args:
        role: Agent role name
        real_templates: 可用的真实模板列表
        default_text: 默认 prefix 文本
        default_tokens: 默认 token 数

    Returns:
        Tuple of (prefix_text, token_count)
    """
    if not real_templates:
        return default_text, default_tokens

    # Try to find a template with this role
    for tpl in real_templates:
        if role in tpl.role_prefixes:
            tokens = tpl.tier1_tokens.get(role, default_tokens)
            return tpl.role_prefixes[role], tokens

    return default_text, default_tokens


def setup_dag_agents(
    tokenizer,
    num_workflows: int,
    dag_config: DAGConfig,
    tier0_len: int,
    tier1_len: int,
    tier2_len: int,
    seed: int,
    real_template_prefixes: Optional[Dict[str, str]] = None,
) -> Tuple[List[DAGAgentConfig], str, Dict[str, DAGAgentConfig]]:
    """为 DAG workflow 创建所有 agent 配置

    Args:
        tokenizer: Tokenizer for text generation
        num_workflows: Number of workflows
        dag_config: DAG workflow configuration
        tier0_len: Length of Tier-0 universal prefix
        tier1_len: Length of Tier-1 role-based prefix
        tier2_len: Length of Tier-2 workflow-specific prefix
        seed: Random seed
        real_template_prefixes: Optional dict mapping role -> prefix text from real templates

    Returns:
        - all_agents: 所有 workflow 的所有 agent
        - tier0_text: 共享的 Tier-0 文本
        - dag_agents_by_node: node_id -> DAGAgentConfig（单 workflow，用于执行）
    """
    random.seed(seed)

    # Tier-0: 全局共享
    tier0_text = generate_tier0(tokenizer, tier0_len)

    # 预先为每个 role 生成 Tier-1 文本（跨 workflow 共享）
    role_to_tier1 = {}
    for role in set(node.role for node in dag_config.nodes.values()):
        template_idx = dag_config.tier1_templates.get(role, 0)
        # Use real template prefix if available
        if real_template_prefixes and role in real_template_prefixes:
            real_prefix = real_template_prefixes[role]
            tier1_text = _calibrate_and_pad(tokenizer, real_prefix, tier1_len)
            logger.info(f"Using real Tier-1 for DAG role {role}: {len(tier1_text)} chars")
        else:
            tier1_text = generate_tier1_for_role(tokenizer, role, tier1_len)
        role_to_tier1[role] = tier1_text

    all_agents = []
    dag_agents_by_node = {}

    for wf_id in range(num_workflows):
        # Tier-2: workflow 级别上下文
        tier2_text = generate_tier2(tokenizer, wf_id, tier2_len)

        for node_id, node in dag_config.nodes.items():
            # 为每个 node 创建 agent 配置
            dag_agent = DAGAgentConfig(
                workflow_id=wf_id,
                node_id=node_id,
                agent_id=f"w{wf_id}-{node_id}",
                role=node.role,
                parallel_group=node.parallel_group,
                tier0_text=tier0_text,
                tier1_text=role_to_tier1[node.role],
                tier2_text=tier2_text,
                tier0_tokens=tier0_len,
                tier1_tokens=tier1_len,
                tier2_tokens=tier2_len,
                execution_depth=dag_config.get_depth(node_id),
            )
            all_agents.append(dag_agent)

            # 保存第一个 workflow 的 agent 配置（用于执行引擎）
            if wf_id == 0:
                dag_agents_by_node[node_id] = dag_agent

    return all_agents, tier0_text, dag_agents_by_node




# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = {
    # ── Ablation configs ────────────────────────────────────────────────────
    # Each component can be tested independently by toggling these flags.
    # Server args needed for each:
    #
    #   lru_nocache     --radix-eviction-policy lru    (no HiCache, GPU-only)
    #   lru_wb_only     --radix-eviction-policy lru --hicache-write-policy write_back
    #   lru_wb_pf       --radix-eviction-policy lru --hicache-write-policy write_back --enable-hierarchical-cache
    #   priority_wb_only --radix-eviction-policy priority --hicache-write-policy write_back
    #   kvflow          --radix-eviction-policy priority --hicache-write-policy write_back --enable-hierarchical-cache
    #
    "lru_nocache": {
        "label": "LRU, no HiCache (absolute baseline, no write-back, no prefetch)",
        "use_priority": False,
        "hicache": False,
        "write_back": False,
        "prefetch": False,
        "server_note": "Server: --radix-eviction-policy lru  (no HiCache)",
    },
    "lru_wb_only": {
        "label": "LRU + write_back (no prefetch, isolates write_back benefit)",
        "use_priority": False,
        "hicache": True,
        "write_back": True,
        "prefetch": False,
        "server_note": "Server: --radix-eviction-policy lru --hicache-write-policy write_back",
    },
    "lru_wb_pf": {
        "label": "LRU + write_back + prefetch (isolates prefetch benefit over LRU)",
        "use_priority": False,
        "hicache": True,
        "write_back": True,
        "prefetch": True,
        "server_note": "Server: --radix-eviction-policy lru --hicache-write-policy write_back --enable-hierarchical-cache",
    },
    "priority_wb_only": {
        "label": "Priority + write_back (no prefetch, isolates Priority benefit)",
        "use_priority": True,
        "hicache": True,
        "write_back": True,
        "prefetch": False,
        "server_note": "Server: --radix-eviction-policy priority --hicache-write-policy write_back",
    },
    # ── DAG-optimized configs ─────────────────────────────────────────────────
    "priority_dag": {
        "label": "Priority + DAG-aware convergence protection (no prefetch)",
        "use_priority": True,
        "hicache": True,
        "write_back": True,
        "prefetch": False,
        "dag_optimized": True,
        "server_note": "Server: --radix-eviction-policy priority --hicache-write-policy write_back",
    },
    "priority_pf_lock": {
        "label": "Priority + DAG-aware + Prefetch lock (DAG-optimized with prefetch)",
        "use_priority": True,
        "hicache": True,
        "write_back": True,
        "prefetch": True,
        "dag_optimized": True,
        "server_note": "Server: --radix-eviction-policy priority --hicache-write-policy write_back --enable-hicache-prefetch",
    },
    # ── Full configurations ─────────────────────────────────────────────────
    "kvflow": {
        "label": "Priority + write_back + prefetch (full KVFlow)",
        "use_priority": True,
        "hicache": True,
        "write_back": True,
        "prefetch": True,
        "server_note": "Server: --radix-eviction-policy priority --hicache-write-policy write_back --enable-hierarchical-cache",
    },
    # Legacy aliases (kept for backward compatibility)
    "hicache": {
        "label": "LRU + HiCache write_through (legacy baseline)",
        "use_priority": False,
        "hicache": True,
        "write_back": False,
        "prefetch": False,
        "server_note": "Server: --radix-eviction-policy lru --hicache-write-policy write_through",
    },
    "hicache90k": {
        "label": "LRU + HiCache write_back (fair comparison baseline)",
        "use_priority": False,
        "hicache": True,
        "write_back": True,
        "prefetch": False,  # LRU server doesn't support prefetch hint
        "server_note": "Server: --radix-eviction-policy lru --hicache-write-policy write_back",
    },
}


# ---------------------------------------------------------------------------
# Realistic code-generation prefix pools
# These produce authentic code fragments that naturally share across agents/roles.
# ---------------------------------------------------------------------------

# Tier-0: Universal system prompts (shared by ALL agents, ALL workflows)
TIER0_TEMPLATES = [
    "You are an expert software engineer assisting with multi-agent code generation. "
    "You have access to tools for reading files, searching codebases, running shell commands, "
    "and executing Python code. Follow the instructions carefully and reason step by step.",

    "You are a helpful AI coding assistant specialized in large-scale software engineering. "
    "You can read files, search code, execute shell commands, and run Python scripts. "
    "Think carefully before taking action and provide well-structured code.",

    "You are a senior software engineer working in a collaborative multi-agent environment. "
    "You have access to file system tools, code search, and execution capabilities. "
    "Always verify your changes and maintain code quality standards.",
]

# Tier-1: Role-specific code templates (shared by same role across workflows)
# Each role uses authentic Python/JS imports and function signatures.
TIER1_ROLES = {
    "PLANNER": [
        (
            "import ast, re, json, sys\nfrom pathlib import Path\n"
            "from typing import List, Dict, Any, Optional, Tuple, Set\n\n\n"
            "def parse_requirements(filepath: str) -> Dict[str, str]:\n"
            "    \"\"\"Parse requirements.txt into a dependency dict.\"\"\"\n"
            "    deps = {}\n"
            "    for line in open(filepath):\n"
            "        line = line.strip()\n"
            "        if line and not line.startswith('#'):\n"
            "            pkg = line.split('>=')[0].split('==')[0].split('<=')[0].strip()\n"
            "            deps[pkg] = line\n"
            "    return deps\n\n"
            "def plan_task_decomposition(task: str) -> List[Dict[str, Any]]:\n"
            "    \"\"\"Break down a high-level task into subtasks.\"\"\"\n"
        ),
        (
            "from dataclasses import dataclass, field\n"
            "from typing import List, Dict, Optional, Any\n"
            "import json\n\n\n"
            "@dataclass\n"
            "class SubTask:\n"
            "    task_id: str\n"
            "    description: str\n"
            "    dependencies: List[str] = field(default_factory=list)\n"
            "    priority: int = 1\n"
            "    estimated_lines: int = 0\n\n\n"
            "def build_execution_order(tasks: List[SubTask]) -> List[SubTask]:\n"
            "    \"\"\"Topologically sort tasks by dependencies.\"\"\"\n"
        ),
    ],
    "ARCHITECT": [
        (
            "from abc import ABC, abstractmethod\n"
            "from typing import Dict, List, Type, Any\n\n\n"
            "class ModuleInterface(ABC):\n"
            "    @abstractmethod\n"
            "    def execute(self, *args, **kwargs) -> Any: pass\n\n\n"
            "class ModuleRegistry:\n"
            "    _modules: Dict[str, Type[ModuleInterface]] = {}\n\n\n"
            "    @classmethod\n"
            "    def register(cls, name: str, module_cls: Type[ModuleInterface]) -> None:\n"
            "        cls._modules[name] = module_cls\n\n\n"
            "    @classmethod\n"
            "    def get(cls, name: str) -> ModuleInterface:\n"
            "        return cls._modules[name]()\n"
        ),
        (
            "from typing import Dict, List, Optional, Callable, Any\n"
            "from dataclasses import dataclass, field\n"
            "import logging\n\n\n"
            "@dataclass\n"
            "class ModuleSpec:\n"
            "    name: str\n"
            "    inputs: List[str]\n"
            "    outputs: List[str]\n"
            "    deps: List[str] = field(default_factory=list)\n\n\n"
            "class ArchitectureGraph:\n"
            "    def __init__(self):\n"
            "        self.nodes: Dict[str, ModuleSpec] = {}\n"
            "        self.logger = logging.getLogger(__name__)\n\n\n"
            "    def add_module(self, spec: ModuleSpec) -> None:\n"
            "        self.nodes[spec.name] = spec\n"
        ),
    ],
    "IMPLEMENTER": [
        (
            "from typing import List, Optional, Dict, Any\n"
            "from collections import defaultdict\n"
            "import re\n\n\n"
            "def extract_function_calls(code: str) -> List[str]:\n"
            "    \"\"\"Extract all function calls from Python source code.\"\"\"\n"
            "    pattern = r'\\b([a-zA-Z_][a-zA-Z0-9_]*)\\s*\\('\n"
            "    return re.findall(pattern, code)\n\n\n"
            "def group_by_prefix(items: List[str]) -> Dict[str, List[str]]:\n"
            "    \"\"\"Group strings by their common prefix.\"\"\"\n"
            "    groups: Dict[str, List[str]] = defaultdict(list)\n"
            "    for item in items:\n"
            "        prefix = item.split('_')[0] if '_' in item else item\n"
            "        groups[prefix].append(item)\n"
            "    return dict(groups)\n"
        ),
        (
            "import hashlib\n"
            "from typing import List, Tuple, Optional\n\n\n"
            "def deduplicate_by_hash(items: List[str]) -> Tuple[List[str], int]:\n"
            "    \"\"\"Remove duplicates while preserving order. Returns (unique, num_removed).\"\"\"\n"
            "    seen = set()\n"
            "    unique = []\n"
            "    for item in items:\n"
            "        h = hashlib.md5(item.encode()).hexdigest()\n"
            "        if h not in seen:\n"
            "            seen.add(h)\n"
            "            unique.append(item)\n"
            "    return unique, len(items) - len(unique)\n\n\n"
            "def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:\n"
            "    \"\"\"Split a list into chunks of given size.\"\"\"\n"
            "    return [lst[i:i+size] for i in range(0, len(lst), size)]\n"
        ),
    ],
    "REVIEWER": [
        (
            "import ast\n"
            "from typing import List, Dict, Set, Tuple, Optional\n\n\n"
            "class CodeIssue:\n"
            "    SEVERITY_ERROR = 'error'\n"
            "    SEVERITY_WARNING = 'warning'\n"
            "    SEVERITY_INFO = 'info'\n\n\n"
            "def check_imports(tree: ast.AST) -> List[Dict[str, str]]:\n"
            "    \"\"\"Find all imports and check for unused ones.\"\"\"\n"
            "    imports: List[Dict[str, str]] = []\n"
            "    for node in ast.walk(tree):\n"
            "        if isinstance(node, ast.Import):\n"
            "            for alias in node.names:\n"
            "                imports.append({'name': alias.name, 'asname': alias.asname})\n"
            "    return imports\n"
        ),
        (
            "from typing import List, Dict, Set, Optional\n"
            "import re\n\n\n"
            "ISSUE_PATTERNS = {\n"
            "    'long-line': re.compile(r'.{120,}'),\n"
            "    'trailing-whitespace': re.compile(r'[ \\t]+$\\n'),\n"
            "    'todo': re.compile(r'#\\s*TODO:'),\n"
            "    'debug-print': re.compile(r'print\\s*\\('),\n"
            "}\n\n\n"
            "def scan_file(path: str) -> Dict[str, List[str]]:\n"
            "    \"\"\"Scan a file for common code issues.\"\"\"\n"
            "    issues: Dict[str, List[str]] = {k: [] for k in ISSUE_PATTERNS}\n"
            "    try:\n"
            "        with open(path) as f:\n"
            "            for i, line in enumerate(f, 1):\n"
            "                for name, pat in ISSUE_PATTERNS.items():\n"
            "                    if pat.search(line):\n"
            "                        issues[name].append(f'line {i}: {line.rstrip()}')\n"
            "    except Exception:\n"
            "        pass\n"
            "    return issues\n"
        ),
    ],
    "TESTER": [
        (
            "import unittest\n"
            "from typing import List, Callable, Any\n"
            "from dataclasses import dataclass, field\n\n\n"
            "@dataclass\n"
            "class TestCase:\n"
            "    name: str\n"
            "    func: Callable\n"
            "    args: List[Any] = field(default_factory=list)\n"
            "    expected: Any = None\n"
            "    tolerance: float = 1e-6\n\n\n"
            "class TestRunner:\n"
            "    def __init__(self):\n"
            "        self.cases: List[TestCase] = []\n"
            "        self.passed = 0\n"
            "        self.failed = 0\n\n\n"
            "    def add(self, tc: TestCase) -> None:\n"
            "        self.cases.append(tc)\n"
        ),
        (
            "from typing import List, Dict, Optional, Callable, Any\n"
            "import time\n\n\n"
            "def benchmark(func: Callable, args: List[Any], kwargs: Dict[str, Any], runs: int = 100) -> Dict[str, float]:\n"
            "    \"\"\"Benchmark a function over multiple runs.\"\"\"\n"
            "    times = []\n"
            "    for _ in range(runs):\n"
            "        start = time.perf_counter()\n"
            "        func(*args, **kwargs)\n"
            "        times.append(time.perf_counter() - start)\n"
            "    return {\n"
            "        'mean': sum(times) / len(times),\n"
            "        'min': min(times),\n"
            "        'max': max(times),\n"
            "        'p95': sorted(times)[int(len(times) * 0.95)],\n"
            "    }\n"
        ),
    ],
}


def _calibrate_and_pad(tokenizer, text: str, target_tokens: int) -> str:
    """Pad or trim text to exactly target_tokens using tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) == target_tokens:
        return text
    if len(tokens) > target_tokens:
        return tokenizer.decode(tokens[:target_tokens])
    # Need to pad: append space-padded token cycle
    pad_text = text
    while len(tokenizer.encode(pad_text, add_special_tokens=False)) < target_tokens:
        pad_text += " ."
    return tokenizer.decode(tokenizer.encode(pad_text, add_special_tokens=False)[:target_tokens])


def generate_tier0(tokenizer, target_tokens: int) -> str:
    """Tier-0: universal system prompt shared by ALL agents."""
    t0 = TIER0_TEMPLATES[0]  # deterministic: same for every run
    return _calibrate_and_pad(tokenizer, t0, target_tokens)


def generate_tier1_for_role(tokenizer, role: str, target_tokens: int) -> str:
    """Tier-1: role-specific code imports+signatures shared across workflows.
    
    Same role (e.g. "IMPLEMENTER") uses the same code template pool across
    ALL workflows. This is the core of cross-workflow KV sharing:
    when multiple workflows run the same role simultaneously, their Tier-1
    prefixes produce IDENTICAL token sequences -> 100% KV reuse for Tier-1.
    """
    templates = TIER1_ROLES.get(role, TIER1_ROLES["IMPLEMENTER"])
    template = templates[0]  # deterministic: same for every run
    return _calibrate_and_pad(tokenizer, template, target_tokens)


def generate_tier2(tokenizer, workflow_id: int, target_tokens: int) -> str:
    """Tier-2: workflow-specific task context (unique to this workflow).
    
    Different workflows get different task descriptions, so Tier-2 tokens
    are NOT shared across workflows. But Priority eviction still helps:
    within a workflow, the Tier-2 context persists across rounds.
    """
    task_contexts = [
        f"You are working on a web-scraping framework. The goal is to build "
        f"a robust crawler that handles JavaScript-rendered pages using asyncio.",
        f"You are building a data pipeline for ETL processing. Focus on handling "
        f"large CSV and JSON files with streaming transformations.",
        f"You are developing a REST API framework. The design should support "
        f"async handlers, middleware chaining, and OpenAPI documentation.",
        f"You are creating a machine learning utility library. Focus on common "
        f"preprocessing steps and model evaluation helpers.",
        f"You are working on a configuration management system. It should support "
        f"YAML/TOML/JSON configs with validation and environment overrides.",
    ]
    context = task_contexts[workflow_id % len(task_contexts)]
    return _calibrate_and_pad(tokenizer, context, target_tokens)


def generate_suffix_dynamic(tokenizer, workflow_id: int, round_idx: int, agent_idx: int, target_tokens: int) -> str:
    """Tier-3: per-request dynamic suffix (always unique).
    
    Uses workflow/round/agent indices as part of the seed so every
    (workflow, round, step) combination produces a DIFFERENT suffix,
    even though they share the same target token count. This prevents
    accidental cache hits from repeated identical prompts.
    """
    # Use distinct seeds so suffixes are never identical across rounds
    seed = workflow_id * 10000 + round_idx * 100 + agent_idx
    rng = random.Random(seed)
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)
    tokens = []
    for _ in range(target_tokens + 20):
        tid = rng.randint(0, vocab_size - 1)
        if tid not in special_ids:
            tokens.append(tid)
        if len(tokens) >= target_tokens + 20:
            break
    text = tokenizer.decode(tokens[:target_tokens], skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))
    while actual > target_tokens and len(tokens) > 10:
        tokens = tokens[len(tokens) - 5:]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
    return text


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    priority: Optional[int] = None,
    next_agent_hint: Optional[str] = None,
    role_type: int = 0,
    convergence_factor: int = 0,
    critical_path_distance: int = 1,
    timeout: int = 600,
) -> Tuple[float, float, int]:
    """Send a streaming chat completion request.
    Returns (ttft_ms, e2e_ms, num_output_tokens).
    """
    payload: Dict[str, Any] = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "top_k": 1,
    }
    if priority is not None:
        payload["priority"] = priority
    if next_agent_hint is not None:
        # KVFlow-aware hint: next_agent_hint tells the server the exact text of the
        # next agent's prefix, enabling proactive prefetch (CPU->GPU KV load-back)
        # before the next request arrives.
        payload["next_agent_prefix"] = next_agent_hint
    if role_type > 0:
        # Token-type awareness: role_type is set via InsertParams.role_type on the server
        # side and used by PriorityStrategy to boost retention of Tier-0/1 prefixes.
        payload["role_type"] = role_type
    if convergence_factor > 0:
        # DAG-aware convergence protection (deprecated, kept for compatibility)
        payload["convergence_factor"] = convergence_factor
    if critical_path_distance > 1:
        # DAG-aware critical path distance for PriorityStrategy v3
        # PLANNER=3, ARCHITECT/REVIEWER=2, IMPLEMENTER/TESTER=1
        payload["critical_path_distance"] = critical_path_distance

    start = time.perf_counter()
    ttft: Optional[float] = None
    num_tokens = 0

    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout, sock_read=300)
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error(f"Request failed ({resp.status}): {body[:500]}")
                e2e = (time.perf_counter() - start) * 1000
                return (e2e, e2e, 0)

            async for raw_line in resp.content:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        num_tokens += 1
                        if ttft is None:
                            ttft = (time.perf_counter() - start) * 1000
    except asyncio.TimeoutError:
        e2e = (time.perf_counter() - start) * 1000
        logger.error(f"Request timed out after {timeout}s")
        return (e2e, e2e, 0)

    e2e = (time.perf_counter() - start) * 1000
    return (ttft or e2e, e2e, num_tokens)


async def fetch_loads(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch load metrics from /v1/loads endpoint."""
    try:
        async with session.get(f"{url}/v1/loads") as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception as e:
        logger.warning(f"Could not fetch /v1/loads: {e}")
    return {}


# ---------------------------------------------------------------------------
# Agent and workflow setup
# ---------------------------------------------------------------------------

ROLES = ["PLANNER", "ARCHITECT", "IMPLEMENTER", "REVIEWER", "TESTER"]


def setup_agents(
    tokenizer,
    num_workflows: int,
    agents_per_workflow: int,
    tier0_len: int,
    tier1_len: int,
    tier2_len: int,
    seed: int,
    real_template_prefixes: Optional[Dict[str, str]] = None,
) -> Tuple[List[AgentConfig], str]:
    """Create all agents with multi-tier prefix structure.

    Args:
        tokenizer: Tokenizer for text generation
        num_workflows: Number of concurrent workflows
        agents_per_workflow: Number of agents per workflow
        tier0_len: Length of Tier-0 universal prefix
        tier1_len: Length of Tier-1 role-based prefix
        tier2_len: Length of Tier-2 workflow-specific prefix
        seed: Random seed
        real_template_prefixes: Optional dict mapping role -> prefix text from real templates

    Returns (agents, tier0_text).
    Each agent gets:
      - Tier-0 (universal): same for ALL agents (shared by all workflows)
      - Tier-1 (role-based): same for same ROLE across workflows
      - Tier-2 (workflow): unique to this workflow
    """
    random.seed(seed)

    # Tier-0: one universal system prompt for all
    tier0_text = generate_tier0(tokenizer, tier0_len)

    agents: List[AgentConfig] = []
    for wf_id in range(num_workflows):
        # Tier-2: workflow-specific task context
        tier2_text = generate_tier2(tokenizer, wf_id, tier2_len)

        for a_id in range(agents_per_workflow):
            role = ROLES[a_id % len(ROLES)]

            # Tier-1: role-based code template
            # Use real template prefix if available, otherwise generate synthetic
            if real_template_prefixes and role in real_template_prefixes:
                # Use real prefix and calibrate to target length
                real_prefix = real_template_prefixes[role]
                tier1_text = _calibrate_and_pad(tokenizer, real_prefix, tier1_len)
                logger.info(f"Using real Tier-1 for {role}: {len(tier1_text)} chars")
            else:
                # Tier-1: role-based code template (same for this role across ALL workflows)
                tier1_text = generate_tier1_for_role(tokenizer, role, tier1_len)

            agents.append(AgentConfig(
                workflow_id=wf_id,
                agent_id=f"w{wf_id}-a{a_id}",
                role=role,
                tier0_text=tier0_text,
                tier1_text=tier1_text,
                tier2_text=tier2_text,
                tier0_tokens=tier0_len,
                tier1_tokens=tier1_len,
                tier2_tokens=tier2_len,
            ))

    return agents, tier0_text


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------

async def run_one_workflow(
    session: aiohttp.ClientSession,
    url: str,
    workflow_id: int,
    agents: List[AgentConfig],
    num_rounds: int,
    output_len: int,
    suffix_len: int,
    use_priority: bool,
    enable_prefetch_hint: bool,
    tokenizer,
    counter: SharedCounter,
) -> WorkflowResult:
    """Run one workflow: execute its agents round-robin for num_rounds."""
    result = WorkflowResult(workflow_id=workflow_id)
    num_agents = len(agents)

    for round_idx in range(num_rounds):
        round_results: List[StepResult] = []

        for step_idx, agent in enumerate(agents):
            # KVFlow-aware prefetch hint
            next_agent_hint: Optional[str] = None
            if enable_prefetch_hint and step_idx < num_agents - 1:
                next_agent = agents[step_idx + 1]
                next_agent_hint = next_agent.tier1_text

            suffix_text = generate_suffix_dynamic(
                tokenizer, workflow_id, round_idx, step_idx, suffix_len
            )
            prompt = agent.build_full_prefix(suffix_text)

            priority = None
            if use_priority:
                priority = await counter.get_and_increment()

            ttft_ms, e2e_ms, n_tokens = await send_request(
                session=session,
                url=url,
                prompt=prompt,
                max_tokens=output_len,
                priority=priority,
                next_agent_hint=next_agent_hint,
            )

            step_result = StepResult(
                workflow_id=workflow_id,
                agent_id=agent.agent_id,
                round_idx=round_idx,
                step_idx=step_idx,
                ttft_ms=ttft_ms,
                e2e_ms=e2e_ms,
                output_tokens=n_tokens,
                priority=priority,
            )
            round_results.append(step_result)

            logger.info(
                f"  wf={workflow_id} round={round_idx} step={step_idx} "
                f"agent={agent.agent_id}({agent.role}) "
                f"ttft={ttft_ms:.1f}ms e2e={e2e_ms:.1f}ms tokens={n_tokens}"
                + (f" pri={priority}" if priority is not None else "")
                + (" [prefetch_hint]" if next_agent_hint else "")
            )

        result.round_results.append(round_results)

    return result


# ---------------------------------------------------------------------------
# DAG Workflow Runner
# ---------------------------------------------------------------------------

async def execute_dag_node(
    session: aiohttp.ClientSession,
    url: str,
    workflow_id: int,
    dag_agent: DAGAgentConfig,
    round_idx: int,
    step_counter: int,
    output_len: int,
    suffix_len: int,
    use_priority: bool,
    priority_value: Optional[int],
    next_tier1_prefix: Optional[str],
    tokenizer,
    convergence_factor: int = 0,
    critical_path_distance: int = 1,
) -> Tuple[StepResult, int]:
    """Execute a single DAG node.

    Args:
        next_tier1_prefix: Pre-computed Tier-1 prefix for the next stage's nodes.
                           Used for prefetch hint. Pass None if no next stage.
        convergence_factor: DAG-aware convergence protection factor (deprecated).
        critical_path_distance: DAG-aware critical path distance. PLANNER=3,
                           ARCHITECT/REVIEWER=2, IMPLEMENTER/TESTER=1.
                           Used by PriorityStrategy v3 for critical-path protection.
    """

    # Full prompt: [Tier-0] [Tier-1 role] [Tier-2 workflow] [Tier-3 dynamic]
    suffix_text = generate_suffix_dynamic(
        tokenizer, workflow_id, round_idx,
        step_counter, suffix_len
    )
    prompt = dag_agent.build_full_prefix(suffix_text)

    # Priority debug logging (KVFlow formula: priority = global_step_counter + steps_to_execution)
    if priority_value is not None:
        logger.debug(
            f"[Priority-Debug] wf={workflow_id} node={dag_agent.node_id} "
            f"step_counter={step_counter} steps_to_exec={critical_path_distance} "
            f"priority={priority_value} (formula: {step_counter} + {critical_path_distance})"
        )

    ttft_ms, e2e_ms, n_tokens = await send_request(
        session=session,
        url=url,
        prompt=prompt,
        max_tokens=output_len,
        priority=priority_value,
        next_agent_hint=next_tier1_prefix,
        convergence_factor=convergence_factor,
        critical_path_distance=critical_path_distance,
    )

    step_result = StepResult(
        workflow_id=workflow_id,
        agent_id=dag_agent.agent_id,
        round_idx=round_idx,
        step_idx=step_counter,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        output_tokens=n_tokens,
        priority=priority_value,
    )

    return step_result, step_counter + 1


async def run_dag_workflow(
    session: aiohttp.ClientSession,
    url: str,
    workflow_id: int,
    dag_config: DAGConfig,
    dag_agents: Dict[str, DAGAgentConfig],
    num_rounds: int,
    output_len: int,
    suffix_len: int,
    use_priority: bool,
    enable_prefetch_hint: bool,
    tokenizer,
    counter: SharedCounter,
) -> WorkflowResult:
    """Execute one DAG workflow.

    Executes by stages (execution_order):
    1. Nodes in the same stage run in parallel
    2. Wait for all to complete before advancing to next stage

    Global step counter ensures proper Priority ordering across all workflows.
    """
    result = WorkflowResult(workflow_id=workflow_id)
    step_counter = 0

    role_to_tier1: Dict[str, str] = {}
    for node_id, agent in dag_agents.items():
        role_to_tier1[agent.role] = agent.tier1_text

    for round_idx in range(num_rounds):
        round_results: List[StepResult] = []

        for stage_idx, parallel_nodes in enumerate(dag_config.execution_order):
            next_stage_idx = stage_idx + 1
            next_tier1_hint: Optional[str] = None
            if enable_prefetch_hint and next_stage_idx < len(dag_config.execution_order):
                next_nodes = dag_config.execution_order[next_stage_idx]
                if next_nodes:
                    first_next_node = next_nodes[0]
                    next_tier1_hint = role_to_tier1.get(dag_agents[first_next_node].role)

            tasks = []
            for node_id in parallel_nodes:
                dag_agent = dag_agents[node_id]
                wf_agent = DAGAgentConfig(
                    workflow_id=workflow_id,
                    node_id=node_id,
                    agent_id=f"w{workflow_id}-{node_id}",
                    role=dag_agent.role,
                    parallel_group=dag_agent.parallel_group,
                    tier0_text=dag_agent.tier0_text,
                    tier1_text=dag_agent.tier1_text,
                    tier2_text=dag_agent.tier2_text,
                    tier0_tokens=dag_agent.tier0_tokens,
                    tier1_tokens=dag_agent.tier1_tokens,
                    tier2_tokens=dag_agent.tier2_tokens,
                    execution_depth=dag_agent.execution_depth,
                )

                # 计算 priority 和 steps_to_execution (使用 KVFlow 公式)
                priority_value = None
                steps_to_execution = 1  # 默认值，TESTER=1
                if use_priority:
                    # 正确传入 global step_counter，使用 KVFlow 公式
                    priority_value = dag_config.calculate_priority(node_id, step_counter)
                    steps_to_execution = dag_config.get_critical_path_length(node_id)
                    # steps_to_execution: PLANNER=4, ARCHITECT/REVIEWER=3, IMPLEMENTER=2, TESTER=1

                convergence_factor = dag_config.get_convergence_factor(node_id)

                tasks.append(execute_dag_node(
                    session=session,
                    url=url,
                    workflow_id=workflow_id,
                    dag_agent=wf_agent,
                    round_idx=round_idx,
                    step_counter=step_counter,
                    output_len=output_len,
                    suffix_len=suffix_len,
                    use_priority=use_priority,
                    priority_value=priority_value,
                    next_tier1_prefix=next_tier1_hint,
                    tokenizer=tokenizer,
                    convergence_factor=convergence_factor,
                    critical_path_distance=steps_to_execution,
                ))

            stage_results = await asyncio.gather(*tasks)

            for step_result, new_counter in stage_results:
                # priority and convergence_factor are already set in execute_dag_node call
                round_results.append(step_result)
                step_counter = new_counter

            logger.info(
                f"  wf={workflow_id} round={round_idx} stage={stage_idx} "
                f"nodes={parallel_nodes} completed={len(round_results)} steps"
                + (" [prefetch_hint]" if next_tier1_hint else "")
            )

        round_results.sort(key=lambda x: x.step_idx)
        result.round_results.append(round_results)

    return result


# ---------------------------------------------------------------------------
# Results printing and JSON output
# ---------------------------------------------------------------------------

def compute_aggregate(
    all_results: List[WorkflowResult],
    args: argparse.Namespace,
) -> Tuple[List[float], List[float], List[float], Dict]:
    """Compute aggregate metrics from all workflow results.

    Returns (ttfts, e2es, round_e2es, per_step_stats).
    per_step_stats maps step_idx -> dict with:
        - warmup_ttft, warmup_e2e: round 0 baselines
        - stable_ttfts: list of TTFTs from rounds 1+ (stable-state)
        - stable_e2es: list of E2Es from rounds 1+
        - ttft_speedup_per_step: warmup_ttft / avg(stable_ttfts)
        - e2e_speedup_per_step: warmup_e2e / avg(stable_e2es)
    """
    warmup = args.warmup_rounds
    ttfts, e2es, round_e2es = [], [], []
    per_step_stats: Dict[int, Dict] = {}

    for wf in all_results:
        for round_idx, round_steps in enumerate(wf.round_results):
            if round_idx < warmup:
                continue
            for s in round_steps:
                ttfts.append(s.ttft_ms)
                e2es.append(s.e2e_ms)
            round_e2es.append(sum(s.e2e_ms for s in round_steps))

    # Per-step stable-state stats (rounds 1+), excluding warmup
    for wf in all_results:
        for round_idx, round_steps in enumerate(wf.round_results):
            for s in round_steps:
                if s.step_idx not in per_step_stats:
                    per_step_stats[s.step_idx] = {
                        "warmup_ttft": [],
                        "warmup_e2e": [],
                        "stable_ttfts": [],
                        "stable_e2es": [],
                    }
                if round_idx < warmup:
                    per_step_stats[s.step_idx]["warmup_ttft"].append(s.ttft_ms)
                    per_step_stats[s.step_idx]["warmup_e2e"].append(s.e2e_ms)
                else:
                    per_step_stats[s.step_idx]["stable_ttfts"].append(s.ttft_ms)
                    per_step_stats[s.step_idx]["stable_e2es"].append(s.e2e_ms)

    return ttfts, e2es, round_e2es, per_step_stats


def print_and_save_results(
    all_results: List[WorkflowResult],
    loads_before: dict,
    loads_after: dict,
    args: argparse.Namespace,
    elapsed_seconds: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[dict]:
    """Print formatted results and return the output dict for comparison.

    Key improvements:
    - stable_ttft: avg TTFT from rounds 1+ (excludes warmup AND round 0),
      which is the true steady-state latency uninfluenced by cold-start overhead.
    - per-step speedup: per-step stable-state TTFT speedup vs warmup baseline
      (more precise than global average).
    - cache_hit_rate: estimated from TTFT ratio (stable vs warmup).

    """
    warmup = args.warmup_rounds
    total_rounds = warmup + args.num_rounds
    config_name = args.config
    if cfg is None:
        cfg = CONFIGS.get(args.config, {})

    ttfts, e2es, round_e2es, per_step_stats = compute_aggregate(all_results, args)

    # Compute stable-state avg TTFT/E2E (rounds 1+, excludes warmup round)
    all_stable_ttfts = [s.ttft_ms for wf in all_results
                        for rnd in wf.round_results[warmup:]
                        for s in rnd]
    all_stable_e2es = [s.e2e_ms for wf in all_results
                       for rnd in wf.round_results[warmup:]
                       for s in rnd]
    stable_ttft_avg = sum(all_stable_ttfts) / len(all_stable_ttfts) if all_stable_ttfts else 0.0
    stable_e2e_avg = sum(all_stable_e2es) / len(all_stable_e2es) if all_stable_e2es else 0.0

    # Warmup round (round 0) baselines
    all_warmup_ttfts = [s.ttft_ms for wf in all_results if len(wf.round_results) > 0
                        for s in wf.round_results[0]]
    all_warmup_e2es = [s.e2e_ms for wf in all_results if len(wf.round_results) > 0
                       for s in wf.round_results[0]]
    warmup_ttft_avg = sum(all_warmup_ttfts) / len(all_warmup_ttfts) if all_warmup_ttfts else 0.0
    warmup_e2e_avg = sum(all_warmup_e2es) / len(all_warmup_e2es) if all_warmup_e2es else 0.0

    # Per-step stable-state speedup (more precise than global avg)
    per_step_speedup_ttft, per_step_speedup_e2e = {}, {}
    for step_idx, stats in per_step_stats.items():
        if stats["warmup_ttft"] and stats["stable_ttfts"]:
            w_t = sum(stats["warmup_ttft"]) / len(stats["warmup_ttft"])
            s_t = sum(stats["stable_ttfts"]) / len(stats["stable_ttfts"])
            per_step_speedup_ttft[step_idx] = w_t / s_t if s_t > 0 else 0.0
            per_step_speedup_e2e[step_idx] = (
                (sum(stats["warmup_e2e"]) / len(stats["warmup_e2e"]))
                / (sum(stats["stable_e2es"]) / len(stats["stable_e2es"]))
                if stats["stable_e2es"] else 0.0
            )

    # Estimated cache hit rate from TTFT speedup:
    # If stable_ttft = warmup_ttft * (1 - hit_rate), then hit_rate = 1 - stable/warmup
    avg_speedup_ttft = (warmup_ttft_avg / stable_ttft_avg) if stable_ttft_avg > 0 else 0.0
    est_hit_rate_ttft = max(0.0, 1.0 - (stable_ttft_avg / warmup_ttft_avg)) if warmup_ttft_avg > 0 else 0.0

    logger.info("=" * 72)
    logger.info(f"Multi-Workflow KVFlow Results [{config_name}]")
    total_prefix = args.tier0_len + args.tier1_len + args.tier2_len
    logger.info("=" * 72)
    # Dynamically compute total_agents from actual results
    num_workflows = len(all_results)
    steps_per_workflow = len(all_results[0].round_results[0]) if all_results and all_results[0].round_results else 0
    total_agents_computed = num_workflows * steps_per_workflow
    agents_per_workflow_computed = steps_per_workflow
    logger.info(
        f"  {num_workflows} workflows × {agents_per_workflow_computed} agents "
        f"= {total_agents_computed} total agents"
    )
    logger.info(
        f"  tier0={args.tier0_len}, tier1={args.tier1_len}, tier2={args.tier2_len}, "
        f"suffix={args.suffix_len}, output={args.output_len}"
    )
    logger.info(
        f"  {total_rounds} rounds ({warmup} warmup + {args.num_rounds} measured), "
        f"total_runtime={elapsed_seconds:.1f}s"
    )
    kv_pressure = num_workflows * agents_per_workflow_computed * total_prefix
    logger.info(f"  Total KV pressure per round: {kv_pressure} tokens")
    logger.info("-" * 72)

    # Per-round breakdown
    round_data: Dict[int, Dict] = {}
    for round_idx in range(total_rounds):
        r_ttfts, r_e2es = [], []
        for wf in all_results:
            if round_idx < len(wf.round_results):
                for s in wf.round_results[round_idx]:
                    r_ttfts.append(s.ttft_ms)
                    r_e2es.append(s.e2e_ms)
        if r_ttfts:
            round_data[round_idx] = {
                "avg_ttft": sum(r_ttfts) / len(r_ttfts),
                "avg_e2e": sum(r_e2es) / len(r_e2es),
            }
            tag = " [warmup]" if round_idx < warmup else ""
            logger.info(
                f"  Round {round_idx}: avg TTFT={round_data[round_idx]['avg_ttft']:.1f}ms, "
                f"avg E2E={round_data[round_idx]['avg_e2e']:.1f}ms{tag}"
            )

    logger.info("-" * 72)

    # Aggregate stats
    avg_ttft = avg_e2e = avg_round_e2e = 0.0
    p50_ttft = p90_ttft = p50_e2e = p90_e2e = 0.0

    if ttfts:
        ttfts.sort()
        e2es.sort()
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_e2e = sum(e2es) / len(e2es)
        p50_ttft = ttfts[len(ttfts) // 2]
        p90_ttft = ttfts[int(len(ttfts) * 0.9)]
        p50_e2e = e2es[len(e2es) // 2]
        p90_e2e = e2es[int(len(e2es) * 0.9)]

    if round_e2es:
        round_e2es.sort()
        avg_round_e2e = sum(round_e2es) / len(round_e2es)

    # Speedup vs warmup baseline
    # Global avg speedup (comparing all measured rounds vs warmup round)
    global_speedup_ttft = (warmup_ttft_avg / stable_ttft_avg) if stable_ttft_avg > 0 else 0.0
    global_speedup_e2e = (warmup_e2e_avg / stable_e2e_avg) if stable_e2e_avg > 0 else 0.0

    logger.info(
        f"Stable-state TTFT (rounds 1-{warmup + args.num_rounds - 1}): "
        f"avg={stable_ttft_avg:.2f}ms  vs  warmup={warmup_ttft_avg:.1f}ms  "
        f"(speedup={global_speedup_ttft:.2f}x, est_hit_rate={est_hit_rate_ttft:.1%})"
    )
    logger.info(
        f"Stable-state E2E: avg={stable_e2e_avg:.2f}ms  vs  warmup={warmup_e2e_avg:.1f}ms  "
        f"(speedup={global_speedup_e2e:.2f}x)"
    )
    if per_step_speedup_ttft:
        logger.info(f"Per-step TTFT speedup (stable vs warmup): "
            + ", ".join(f"step{k}={v:.2f}x" for k, v in sorted(per_step_speedup_ttft.items())[:5])
            + (" ..." if len(per_step_speedup_ttft) > 5 else ""))
    logger.info("=" * 72)
    logger.info("Multi-tier KV Sharing Analysis:")
    logger.info(
        f"  Tier-0 (universal, all agents): {args.tier0_len} tokens "
        f"= {100*args.tier0_len/(args.tier0_len+args.tier1_len+args.tier2_len):.0f}% "
        f"of prefix -- 100% cross-workflow KV reuse"
    )
    logger.info(
        f"  Tier-1 (role-based, same role): {args.tier1_len} tokens "
        f"= {100*args.tier1_len/(args.tier0_len+args.tier1_len+args.tier2_len):.0f}% "
        f"of prefix -- ~{min(100, args.num_workflows*100)}% reuse within same role"
    )
    logger.info(
        f"  Tier-2 (workflow-specific): {args.tier2_len} tokens "
        f"= {100*args.tier2_len/(args.tier0_len+args.tier1_len+args.tier2_len):.0f}% "
        f"of prefix -- unique to each workflow"
    )
    # Theoretical KV saving vs no sharing
    no_sharing_total = args.tier0_len + args.tier1_len + args.tier2_len
    # With sharing: tier0 (once) + tier1 (once per role, not per workflow) + tier2 (once per workflow)
    # For 5 roles uniformly distributed: tier1 once per 5 agents
    sharing_total = args.tier0_len + (args.tier1_len / min(agents_per_workflow_computed, 5)) + args.tier2_len
    theoretical_saving = (1 - sharing_total / no_sharing_total) * 100
    logger.info(
        f"  Theoretical KV saving: {theoretical_saving:.1f}% "
        f"(compared to no sharing at all)"
    )

    # Build output dict
    aggregate: Dict[str, Any] = {
        "label": cfg["label"],
        # Legacy aggregate (all measured rounds, rounds 1+)
        "ttft_avg_ms": avg_ttft,
        "ttft_p50_ms": p50_ttft,
        "ttft_p90_ms": p90_ttft,
        "e2e_avg_ms": avg_e2e,
        "e2e_p50_ms": p50_e2e,
        "e2e_p90_ms": p90_e2e,
        "round_e2e_avg_ms": avg_round_e2e,
        # Stable-state metrics (rounds 1+, excludes warmup round 0)
        "stable_ttft_avg_ms": stable_ttft_avg,
        "stable_e2e_avg_ms": stable_e2e_avg,
        "warmup_ttft_avg_ms": warmup_ttft_avg,
        "warmup_e2e_avg_ms": warmup_e2e_avg,
        # Speedups (stable vs warmup)
        "stable_speedup_ttft": global_speedup_ttft,
        "stable_speedup_e2e": global_speedup_e2e,
        # Estimated cache hit rate from TTFT speedup
        "est_ttft_hit_rate": est_hit_rate_ttft,
        # Multi-tier sharing analysis
        "theoretical_kv_saving_pct": theoretical_saving,
        "tier0_tokens": args.tier0_len,
        "tier1_tokens": args.tier1_len,
        "tier2_tokens": args.tier2_len,
        "tier3_tokens": args.suffix_len,
    }

    # Round-by-round data
    round_summaries = {}
    for ridx, rdata in round_data.items():
        round_summaries[f"round_{ridx}"] = rdata

    output: Dict[str, Any] = {
        "config": {
            k: v for k, v in vars(args).items()
            if k not in ("baseline_json",)
        },
        "warmup_rounds": warmup,
        "workflow_type": args.workflow_type,
        "aggregate": aggregate,
        "round_summaries": round_summaries,
        "loads_before": loads_before,
        "loads_after": loads_after,
        "elapsed_seconds": elapsed_seconds,
        "results": [
            {
                "workflow_id": wf.workflow_id,
                "rounds": [
                    [
                        {
                            "agent_id": s.agent_id,
                            "round_idx": s.round_idx,
                            "step_idx": s.step_idx,
                            "ttft_ms": s.ttft_ms,
                            "e2e_ms": s.e2e_ms,
                            "output_tokens": s.output_tokens,
                            "priority": s.priority,
                        }
                        for s in round_steps
                    ]
                    for round_steps in wf.round_results
                ],
            }
            for wf in all_results
        ],
    }

    # Speedup vs external baseline JSON
    baseline_data = None
    speedup_info = {}
    if args.baseline_json:
        try:
            with open(args.baseline_json, "r") as f:
                baseline_data = json.load(f)
            bl_ttft = baseline_data["aggregate"].get("stable_ttft_avg_ms",
                            baseline_data["aggregate"].get("ttft_avg_ms"))
            bl_e2e = baseline_data["aggregate"].get("stable_e2e_avg_ms",
                            baseline_data["aggregate"].get("round_e2e_avg_ms"))
            logger.info("-" * 72)
            logger.info("Speedup vs external baseline (stable-state):")
            if stable_ttft_avg > 0:
                ext_ttft_speedup = bl_ttft / stable_ttft_avg
                speedup_info["ttft_vs_baseline"] = ext_ttft_speedup
                logger.info(
                    f"  TTFT: {ext_ttft_speedup:.2f}x "
                    f"(baseline={bl_ttft:.1f}ms, current={stable_ttft_avg:.1f}ms)"
                )
            if stable_e2e_avg > 0:
                ext_e2e_speedup = bl_e2e / stable_e2e_avg
                speedup_info["round_e2e_vs_baseline"] = ext_e2e_speedup
                logger.info(
                    f"  Round E2E: {ext_e2e_speedup:.2f}x "
                    f"(baseline={bl_e2e:.1f}ms, current={stable_e2e_avg:.1f}ms)"
                )
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load baseline JSON: {e}")

    if speedup_info:
        output["speedup"] = speedup_info

    # Per-step stable-state speedup
    if per_step_speedup_ttft:
        output["per_step_stable_speedup"] = {
            "ttft": {f"step_{k}": round(v, 4) for k, v in per_step_speedup_ttft.items()},
            "e2e": {f"step_{k}": round(v, 4) for k, v in per_step_speedup_e2e.items()},
        }

    # Write JSON
    os.makedirs(args.output_dir, exist_ok=True)
    total_prefix = args.tier0_len + args.tier1_len + args.tier2_len
    # Use computed values (works for both linear and DAG workflows)
    total_agents_out = num_workflows * agents_per_workflow_computed
    output_file = os.path.join(
        args.output_dir,
        f"mwf_{config_name}_{total_agents_out}agents_"
        f"t{args.tier0_len}p{args.tier1_len}p{args.tier2_len}_"
        f"{args.num_rounds}rounds_{num_workflows}wf.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {output_file}")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Multi-Workflow KVFlow Benchmark for SGLang Priority vs LRU"
    )
    parser.add_argument(
        "--config", type=str, required=True, choices=list(CONFIGS.keys()),
        help="Test configuration: baseline (LRU) or priority (Priority)",
    )
    parser.add_argument(
        "--workflow-type", type=str, default="linear",
        choices=["linear", "dag"],
        help="Workflow topology: linear chain or DAG (default: linear)",
    )
    parser.add_argument(
        "--dag-config", type=str, default=None,
        help="Path to DAG configuration JSON file (required for --workflow-type dag)",
    )
    parser.add_argument(
        "--num-workflows", type=int, default=4,
        help="Number of concurrent workflows (default: 4)",
    )
    parser.add_argument(
        "--agents-per-workflow", type=int, default=5,
        help="Agents per workflow (default: 5, only for linear workflow)",
    )
    parser.add_argument(
        "--tier0-len", type=int, default=512,
        help="Length of Tier-0 universal prefix in tokens (default: 512, shared by ALL)",
    )
    parser.add_argument(
        "--tier1-len", type=int, default=1024,
        help="Length of Tier-1 role-based prefix in tokens (default: 1024, shared by role)",
    )
    parser.add_argument(
        "--tier2-len", type=int, default=512,
        help="Length of Tier-2 workflow-specific prefix in tokens (default: 512, unique to workflow)",
    )
    parser.add_argument(
        "--suffix-len", type=int, default=64,
        help="Length of Tier-3 dynamic suffix in tokens (default: 64, always unique)",
    )
    parser.add_argument(
        "--output-len", type=int, default=64,
        help="Max output tokens per step (default: 64)",
    )
    parser.add_argument(
        "--num-rounds", type=int, default=5,
        help="Number of measured rounds (default: 5)",
    )
    parser.add_argument(
        "--warmup-rounds", type=int, default=1,
        help="Number of warmup rounds (default: 1)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=30300,
        help="Server port (default: 30300)",
    )
    parser.add_argument(
        "--model", type=str,
        default="/home/comp/25480812/models/hub/models--Qwen--Qwen3-8B",
        help="Model path for tokenizer",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/home/comp/25480812/CodeMAS_Project/logs/kvflow-multi-workflow/results",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--baseline-json", type=str, default=None,
        help="Path to baseline JSON for speedup comparison",
    )
    parser.add_argument(
        "--agents-seed", type=int, default=42,
        help="Random seed for agent prefix generation (default: 42)",
    )
    parser.add_argument(
        "--real-templates", type=str, default=None,
        help="Path to JSON file containing real MAScoder templates for KVFlow benchmark",
    )
    parser.add_argument(
        "--real-templates-mode", type=str, default="mix",
        choices=["mix", "dominant"],
        help="How to use real templates: 'mix' = mix with synthetic, 'dominant' = use only real (default: mix)",
    )
    args = parser.parse_args()

    # Validate DAG config
    if args.workflow_type == "dag" and not args.dag_config:
        parser.error("--dag-config is required when --workflow-type is dag")

    cfg = CONFIGS[args.config]
    use_priority = cfg["use_priority"]
    enable_prefetch_hint = cfg.get("prefetch", False)
    logger.info(f"Config: {args.config} -- {cfg['label']}")
    logger.info(f"  {cfg['server_note']}")
    logger.info(f"Workflow type: {args.workflow_type}")

    tier0 = args.tier0_len
    tier1 = args.tier1_len
    tier2 = args.tier2_len
    total_prefix = tier0 + tier1 + tier2

    random.seed(args.seed)
    url = f"http://{args.host}:{args.port}"

    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load real templates if specified
    real_templates = []
    real_template_prefixes: Dict[str, str] = {}
    if args.real_templates:
        logger.info(f"Loading real templates from {args.real_templates}...")
        real_templates, real_template_prefixes = load_real_templates(args.real_templates, tokenizer)
        if real_templates:
            logger.info(f"Loaded {len(real_templates)} real templates")
            logger.info(f"Role prefixes: {list(real_template_prefixes.keys())}")
        else:
            logger.warning("No real templates loaded, using synthetic templates")
    else:
        logger.info("Using synthetic templates for benchmark")

    # Load DAG config if needed
    dag_config = None
    if args.workflow_type == "dag":
        logger.info(f"Loading DAG config from {args.dag_config}")
        dag_config = load_dag_config(args.dag_config)
        logger.info(f"DAG: {dag_config.name}")
        logger.info(f"  Nodes: {list(dag_config.nodes.keys())}")
        logger.info(f"  Execution order: {dag_config.execution_order}")

        # Calculate total agents in DAG
        total_agents = args.num_workflows * len(dag_config.nodes)
        logger.info(f"Scenario (DAG multi-tier sharing design):")
        logger.info(f"  {args.num_workflows} workflows × {len(dag_config.nodes)} nodes = {total_agents} total agents")
    else:
        total_agents = args.num_workflows * args.agents_per_workflow
        logger.info(f"Scenario (linear multi-tier sharing design):")
        logger.info(f"  {args.num_workflows} workflows × {args.agents_per_workflow} agents = {total_agents} total")

    logger.info(f"  Tier-0 (universal, all agents): {tier0} tokens -- 100% cross-workflow KV reuse")
    logger.info(f"  Tier-1 (role-based, same role): {tier1} tokens -- cross-workflow reuse by role")
    logger.info(f"  Tier-2 (workflow-specific): {tier2} tokens -- unique to each workflow")
    logger.info(f"  Tier-3 (dynamic suffix): {args.suffix_len} tokens -- always unique per request")
    logger.info(f"  Per-agent total KV: {total_prefix} tokens")
    logger.info(f"  KVFlow-aware prefetch hint: {'enabled' if enable_prefetch_hint else 'disabled'}")

    # Theoretical KV reuse rate analysis
    reuse_rate_theory = (tier0 + tier1 + tier2) / total_prefix
    sharing_saving_pct = 100 * (1 - 1 / reuse_rate_theory) if reuse_rate_theory > 0 else 0
    logger.info(f"  Theoretical KV reuse rate: {sharing_saving_pct:.1f}% (compared to no sharing)")

    total_rounds = args.warmup_rounds + args.num_rounds
    counter = SharedCounter()

    async with aiohttp.ClientSession() as session:
        # Connection check
        try:
            async with session.get(f"{url}/health_generate",
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.error(f"Health check failed: {resp.status}")
                    return
        except Exception as e:
            logger.error(f"Cannot connect to server at {url}: {e}")
            return

        logger.info("Server connection verified")

        loads_before = await fetch_loads(session, url)

        logger.info(
            f"Running {args.num_workflows} concurrent workflows, "
            f"{total_rounds} rounds each "
            f"({args.warmup_rounds} warmup + {args.num_rounds} measured)..."
        )

        start_time = time.perf_counter()
        tasks = []

        if args.workflow_type == "dag":
            # DAG workflow execution
            logger.info(f"Setting up DAG agents...")
            all_agents, tier0_text, dag_agents_by_node = setup_dag_agents(
                tokenizer,
                args.num_workflows,
                dag_config,
                args.tier0_len,
                args.tier1_len,
                args.tier2_len,
                args.agents_seed,
                real_template_prefixes=real_template_prefixes if real_templates else None,
            )
            logger.info(
                f"Created {args.num_workflows} workflows × {len(dag_config.nodes)} nodes "
                f"= {total_agents} total DAG agents"
            )
            if real_templates:
                logger.info(f"Using {len(real_templates)} real templates for DAG Tier-1 prefixes")

            for wf_id in range(args.num_workflows):
                # Create workflow-specific agent mapping
                wf_agents = {}
                for node_id, base_agent in dag_agents_by_node.items():
                    wf_agents[node_id] = DAGAgentConfig(
                        workflow_id=wf_id,
                        node_id=node_id,
                        agent_id=f"w{wf_id}-{node_id}",
                        role=base_agent.role,
                        parallel_group=base_agent.parallel_group,
                        tier0_text=base_agent.tier0_text,
                        tier1_text=base_agent.tier1_text,
                        tier2_text=generate_tier2(tokenizer, wf_id, args.tier2_len),
                        tier0_tokens=base_agent.tier0_tokens,
                        tier1_tokens=base_agent.tier1_tokens,
                        tier2_tokens=base_agent.tier2_tokens,
                        execution_depth=base_agent.execution_depth,
                    )

                tasks.append(
                    run_dag_workflow(
                        session=session,
                        url=url,
                        workflow_id=wf_id,
                        dag_config=dag_config,
                        dag_agents=wf_agents,
                        num_rounds=total_rounds,
                        output_len=args.output_len,
                        suffix_len=args.suffix_len,
                        use_priority=use_priority,
                        enable_prefetch_hint=enable_prefetch_hint,
                        tokenizer=tokenizer,
                        counter=counter,
                    )
                )
        else:
            # Linear workflow execution (original logic)
            agents, tier0_text = setup_agents(
                tokenizer,
                args.num_workflows,
                args.agents_per_workflow,
                args.tier0_len,
                args.tier1_len,
                args.tier2_len,
                args.agents_seed,
                real_template_prefixes=real_template_prefixes if real_templates else None,
            )
            logger.info(
                f"Created {args.num_workflows} workflows × {args.agents_per_workflow} agents "
                f"= {total_agents} total linear agents"
            )
            if real_templates:
                logger.info(f"Using {len(real_templates)} real templates for Tier-1 prefixes")

            for wf_id in range(args.num_workflows):
                wf_agents = [a for a in agents if a.workflow_id == wf_id]
                tasks.append(
                    run_one_workflow(
                        session=session,
                        url=url,
                        workflow_id=wf_id,
                        agents=wf_agents,
                        num_rounds=total_rounds,
                        output_len=args.output_len,
                        suffix_len=args.suffix_len,
                        use_priority=use_priority,
                        enable_prefetch_hint=enable_prefetch_hint,
                        tokenizer=tokenizer,
                        counter=counter,
                    )
                )

        logger.info(f"Priority: {'enabled' if use_priority else 'disabled (baseline LRU)'}")
        logger.info(f"Prefetch hint: {'enabled' if enable_prefetch_hint else 'disabled'}")
        logger.info(f"Target server: {url}")

        all_results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time

        await asyncio.sleep(1)
        loads_after = await fetch_loads(session, url)

        print_and_save_results(all_results, loads_before, loads_after, args, elapsed, cfg)


if __name__ == "__main__":
    asyncio.run(main())
