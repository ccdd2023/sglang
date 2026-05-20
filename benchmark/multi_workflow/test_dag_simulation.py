#!/usr/bin/env python3
"""
DAG Workflow 模拟测试

模拟服务器响应，验证 DAG 执行逻辑。不需要真实 GPU。
"""

import asyncio
import json
import sys
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_multi_workflow


class MockServer:
    """模拟 SGLang 服务器，用于验证 DAG 执行逻辑"""
    
    def __init__(self):
        self.request_count = 0
        self.prefix_cache = {}  # 模拟 KV 缓存
        self.requests_log = []
        
    async def chat_completion(self, prompt: str, priority: Optional[int] = None) -> Dict:
        """模拟 chat completion 请求"""
        self.request_count += 1
        
        # 模拟 TTFT（首次生成时间）
        # 如果 prompt 已在缓存中，返回更快的 TTFT
        cache_key = hash(prompt[:100])  # 简化缓存键
        if cache_key in self.prefix_cache:
            ttft_ms = 10.0  # 缓存命中
        else:
            ttft_ms = 100.0  # 缓存未命中
            self.prefix_cache[cache_key] = True
        
        result = {
            "request_id": self.request_count,
            "prompt": prompt[:100],
            "ttft_ms": ttft_ms,
            "e2e_ms": ttft_ms + 50,
            "output_tokens": 10,
            "priority": priority,
        }
        
        self.requests_log.append(result)
        return result


async def test_dag_execution():
    """测试 DAG 执行逻辑"""
    print("\n" + "="*60)
    print("DAG 执行逻辑测试")
    print("="*60)
    
    # 加载 DAG 配置
    config_path = os.path.join(
        os.path.dirname(__file__),
        "dag_configs",
        "diamond_6agent.json"
    )
    dag_config = bench_multi_workflow.load_dag_config(config_path)
    
    print(f"\nDAG: {dag_config.name}")
    print(f"Execution Order: {dag_config.execution_order}")
    
    # 模拟 tokenizer（不需要真实 tokenizer）
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()[:128]  # 简化
        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(tokens)
        vocab_size = 100000
        all_special_ids = []
    
    tokenizer = MockTokenizer()
    
    # 设置 DAG agents
    all_agents, tier0_text, dag_agents_by_node = bench_multi_workflow.setup_dag_agents(
        tokenizer,
        num_workflows=2,
        dag_config=dag_config,
        tier0_len=256,
        tier1_len=512,
        tier2_len=256,
        seed=42,
    )
    
    print(f"\nCreated {len(all_agents)} agents for 2 workflows")
    
    # 模拟执行一个 workflow
    mock_server = MockServer()
    workflow_id = 0
    num_rounds = 2
    completed = set()
    step_counter = 0
    
    print(f"\n--- Workflow {workflow_id} Execution ---")
    
    for round_idx in range(num_rounds):
        print(f"\nRound {round_idx}:")
        completed = set()
        
        for stage_idx, parallel_nodes in enumerate(dag_config.execution_order):
            # 检查哪些节点就绪
            ready_nodes = [
                node_id for node_id in parallel_nodes
                if dag_config.is_ready(node_id, completed)
            ]
            
            if not ready_nodes:
                continue
            
            print(f"  Stage {stage_idx}: {ready_nodes}")
            
            # 模拟并行执行
            for node_id in ready_nodes:
                dag_agent = dag_agents_by_node[node_id]
                wf_agent = bench_multi_workflow.DAGAgentConfig(
                    workflow_id=workflow_id,
                    node_id=node_id,
                    agent_id=f"w{workflow_id}-{node_id}",
                    role=dag_agent.role,
                    parallel_group=dag_agent.parallel_group,
                    tier0_text=dag_agent.tier0_text,
                    tier1_text=dag_agent.tier1_text,
                    tier2_text=f"Workflow {workflow_id} context",
                    tier0_tokens=256,
                    tier1_tokens=512,
                    tier2_tokens=256,
                    execution_depth=dag_agent.execution_depth,
                )
                
                # 模拟请求
                priority = dag_config.calculate_priority(node_id, step_counter)
                result = await mock_server.chat_completion(
                    prompt=wf_agent.build_full_prefix("test suffix"),
                    priority=priority
                )
                
                print(f"    {node_id}: ttft={result['ttft_ms']:.1f}ms, "
                      f"priority={priority}, depth={wf_agent.execution_depth}")
                
                completed.add(node_id)
                step_counter += 1
    
    # 打印结果
    print("\n" + "="*60)
    print("执行统计")
    print("="*60)
    print(f"总请求数: {mock_server.request_count}")
    print(f"缓存命中: {len(mock_server.prefix_cache)}")
    print(f"预期 DAG 节点数: {len(dag_config.nodes) * num_rounds}")
    
    # 验证
    expected_requests = len(dag_config.nodes) * num_rounds
    assert mock_server.request_count == expected_requests, \
        f"请求数不匹配: {mock_server.request_count} vs {expected_requests}"
    
    print("\n  [PASS] DAG 执行逻辑正确")


async def test_linear_vs_dag():
    """测试线性 vs DAG 的差异"""
    print("\n" + "="*60)
    print("线性 vs DAG 对比分析")
    print("="*60)
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        "dag_configs",
        "diamond_6agent.json"
    )
    dag_config = bench_multi_workflow.load_dag_config(config_path)
    
    # 线性配置
    linear_agents_per_workflow = 5  # 默认线性配置
    
    # DAG 配置
    dag_nodes = len(dag_config.nodes)
    
    print(f"\n线性配置:")
    print(f"  每 workflow agent 数: {linear_agents_per_workflow}")
    print(f"  拓扑: 链式 (1 -> 2 -> 3 -> 4 -> 5)")
    
    print(f"\nDAG 配置:")
    print(f"  每 workflow 节点数: {dag_nodes}")
    print(f"  拓扑: 菱形")
    for stage_idx, stage in enumerate(dag_config.execution_order):
        print(f"    Stage {stage_idx}: {stage}")
    
    # 计算并行度
    linear_parallelism = 1  # 链式无并行
    dag_max_parallelism = max(len(stage) for stage in dag_config.execution_order)
    
    print(f"\n最大并行度:")
    print(f"  线性: {linear_parallelism}")
    print(f"  DAG: {dag_max_parallelism}")
    
    # 计算 Prefetch 机会
    linear_prefetch_opportunities = linear_agents_per_workflow - 1
    dag_prefetch_opportunities = sum(
        len(stage) * len(dag_config.get_successors(node_id))
        for stage in dag_config.execution_order
        for node_id in stage
    )
    
    print(f"\nPrefetch 机会 (DAG-aware):")
    print(f"  线性: {linear_prefetch_opportunities} (每对相邻节点)")
    print(f"  DAG: {dag_prefetch_opportunities} (多后继并行)")
    
    print("\n  [PASS] 对比分析完成")


async def test_priority_analysis():
    """测试优先级分析"""
    print("\n" + "="*60)
    print("DAG Priority 优势分析")
    print("="*60)
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        "dag_configs",
        "diamond_6agent.json"
    )
    dag_config = bench_multi_workflow.load_dag_config(config_path)
    
    print("\n优先级计算分析:")
    print("-" * 50)
    print(f"{'节点':<12} {'CriticalPath':<14} {'Downstream':<12} {'Priority':<10}")
    print("-" * 50)
    
    priorities = {}
    for stage in dag_config.execution_order:
        for node_id in stage:
            critical_path = dag_config.get_critical_path_length(node_id)
            downstream = dag_config.get_downstream_count(node_id)
            priority = dag_config.calculate_priority(node_id, 0)
            priorities[node_id] = priority
            print(f"{node_id:<12} {critical_path:<14} {downstream:<12} {priority:<10}")
    
    print("-" * 50)
    
    # 分析
    print("\n关键发现:")
    
    # 1. reviewer 优先级最低（最重要，保留到最后）
    reviewer_priority = priorities["reviewer"]
    planner_priority = priorities["planner"]
    print(f"\n1. reviewer 优先级: {reviewer_priority} (最低 = 最重要)")
    print(f"   planner 优先级: {planner_priority} (最高 = 可驱逐)")
    
    # 2. 共享节点保护
    shared_nodes = ["retriever", "architect", "searcher"]
    shared_avg = sum(priorities[n] for n in shared_nodes) / len(shared_nodes)
    print(f"\n2. 分支节点平均优先级: {shared_avg:.1f}")
    print(f"   这些节点被 impl_1 和 impl_2 依赖")
    
    # 3. Priority vs LRU 差异
    print(f"\n3. Priority vs LRU 预期差异:")
    print(f"   - LRU: 基于最近访问时间驱逐")
    print(f"   - Priority: 基于 DAG 结构驱逐")
    print(f"   - DAG-aware: reviewer 最晚驱逐 (关键路径)")
    
    print("\n  [PASS] 优先级分析完成")


async def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("DAG Workflow 模拟测试套件")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        await test_dag_execution()
        tests_passed += 1
    except Exception as e:
        print(f"\n  [FAIL] DAG 执行: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    try:
        await test_linear_vs_dag()
        tests_passed += 1
    except Exception as e:
        print(f"\n  [FAIL] 对比分析: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    try:
        await test_priority_analysis()
        tests_passed += 1
    except Exception as e:
        print(f"\n  [FAIL] 优先级分析: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # 总结
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"  通过: {tests_passed}")
    print(f"  失败: {tests_failed}")
    print(f"  总计: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n  所有模拟测试通过!")
        return 0
    else:
        print(f"\n  {tests_failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
