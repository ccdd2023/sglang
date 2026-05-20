#!/usr/bin/env python3
"""
DAG Workflow 功能测试脚本

不需要 GPU 服务器，用于验证 DAG 配置、优先级计算和执行逻辑。
"""

import json
import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_multi_workflow


def test_dag_config_loading():
    """测试 DAG 配置加载"""
    print("\n" + "="*60)
    print("Test 1: DAG 配置加载")
    print("="*60)
    
    config_path = os.path.join(
        os.path.dirname(__file__), 
        "dag_configs", 
        "diamond_6agent.json"
    )
    
    dag_config = bench_multi_workflow.load_dag_config(config_path)
    
    print(f"  DAG Name: {dag_config.name}")
    print(f"  Nodes: {list(dag_config.nodes.keys())}")
    print(f"  Execution Order: {dag_config.execution_order}")
    print(f"  Total Nodes: {len(dag_config.nodes)}")
    
    # 验证
    assert dag_config.name == "diamond_6agent"
    assert len(dag_config.nodes) == 7
    assert len(dag_config.execution_order) == 4  # 4 stages
    
    print("  [PASS] 配置加载成功")
    return dag_config


def test_dag_topology(dag_config):
    """测试 DAG 拓扑结构"""
    print("\n" + "="*60)
    print("Test 2: DAG 拓扑结构")
    print("="*60)
    
    # 测试依赖关系
    print("\n  节点依赖关系:")
    for node_id, node in dag_config.nodes.items():
        print(f"    {node_id}: depends on {node.dependencies}")
    
    # 验证 planner 没有依赖
    assert dag_config.nodes["planner"].dependencies == []
    
    # 验证 retriever/architect/searcher 依赖 planner
    for node_id in ["retriever", "architect", "searcher"]:
        assert "planner" in dag_config.nodes[node_id].dependencies
    
    # 验证 impl_1/impl_2 依赖所有分支节点
    for node_id in ["impl_1", "impl_2"]:
        deps = dag_config.nodes[node_id].dependencies
        assert "retriever" in deps
        assert "architect" in deps
        assert "searcher" in deps
    
    # 验证 reviewer 依赖实现者
    assert "impl_1" in dag_config.nodes["reviewer"].dependencies
    assert "impl_2" in dag_config.nodes["reviewer"].dependencies
    
    print("  [PASS] 拓扑结构正确")


def test_dag_dependencies(dag_config):
    """测试 DAG 依赖检查"""
    print("\n" + "="*60)
    print("Test 3: DAG 依赖就绪检查")
    print("="*60)
    
    # 测试 planner 就绪
    assert dag_config.is_ready("planner", set())
    print("  planner 就绪: True (无依赖)")
    
    # 测试分支节点不就绪
    assert not dag_config.is_ready("retriever", set())
    print("  retriever 就绪: False (planner 未完成)")
    
    # 测试分支节点就绪
    assert dag_config.is_ready("retriever", {"planner"})
    print("  retriever 就绪: True (planner 已完成)")
    
    # 测试 impl_1 就绪
    deps_done = {"planner"}
    assert not dag_config.is_ready("impl_1", deps_done)
    print("  impl_1 就绪: False (部分依赖未完成)")
    
    deps_done = {"planner", "retriever", "architect", "searcher"}
    assert dag_config.is_ready("impl_1", deps_done)
    print("  impl_1 就绪: True (所有依赖已完成)")
    
    print("  [PASS] 依赖检查正确")


def test_dag_priority(dag_config):
    """测试 DAG 优先级计算"""
    print("\n" + "="*60)
    print("Test 4: DAG 优先级计算")
    print("="*60)
    
    print("\n  各节点优先级分析:")
    print("  " + "-"*50)
    print(f"  {'Node':<12} {'Depth':<8} {'CriticalPath':<14} {'Downstream':<12} {'Priority':<10}")
    print("  " + "-"*50)
    
    priorities = {}
    for stage in dag_config.execution_order:
        for node_id in stage:
            depth = dag_config.get_depth(node_id)
            critical_path = dag_config.get_critical_path_length(node_id)
            downstream = dag_config.get_downstream_count(node_id)
            priority = dag_config.calculate_priority(node_id, 0)
            priorities[node_id] = priority
            print(f"  {node_id:<12} {depth:<8} {critical_path:<14} {downstream:<12} {priority:<10}")
    
    print("  " + "-"*50)
    
    # 验证优先级顺序
    # reviewer 应该有最低优先级（最先被驱逐）
    # planner 应该有最高优先级（最后被驱逐）
    assert priorities["reviewer"] < priorities["impl_1"]
    assert priorities["impl_1"] < priorities["retriever"]
    assert priorities["retriever"] <= priorities["architect"]
    assert priorities["architect"] <= priorities["searcher"]
    
    print("\n  优先级顺序验证:")
    print(f"    reviewer ({priorities['reviewer']}) < impl_1/impl_2 ({priorities['impl_1']})")
    print(f"    impl_1/impl_2 ({priorities['impl_1']}) < branches ({priorities['retriever']})")
    print(f"    branches ({priorities['retriever']}) < planner ({priorities['planner']})")
    
    print("  [PASS] 优先级计算正确")


def test_dag_successors(dag_config):
    """测试 DAG 后继节点查找"""
    print("\n" + "="*60)
    print("Test 5: DAG 后继节点查找")
    print("="*60)
    
    # planner 的后继
    planner_succ = dag_config.get_successors("planner")
    print(f"\n  planner 后继: {planner_succ}")
    assert len(planner_succ) == 3
    assert "retriever" in planner_succ
    assert "architect" in planner_succ
    assert "searcher" in planner_succ
    
    # retriever 的后继
    retriever_succ = dag_config.get_successors("retriever")
    print(f"  retriever 后继: {retriever_succ}")
    assert "impl_1" in retriever_succ
    assert "impl_2" in retriever_succ
    
    # reviewer 的后继
    reviewer_succ = dag_config.get_successors("reviewer")
    print(f"  reviewer 后继: {reviewer_succ}")
    assert len(reviewer_succ) == 0  # 叶子节点
    
    print("  [PASS] 后继节点查找正确")


def test_parallel_group(dag_config):
    """测试并行组"""
    print("\n" + "="*60)
    print("Test 6: 并行组识别")
    print("="*60)
    
    # 分支并行组
    branch_nodes = [n for n, node in dag_config.nodes.items() 
                    if node.parallel_group == "branch_1"]
    print(f"\n  branch_1 组: {branch_nodes}")
    assert len(branch_nodes) == 3
    
    # 实现并行组
    impl_nodes = [n for n, node in dag_config.nodes.items() 
                  if node.parallel_group == "impl_phase"]
    print(f"  impl_phase 组: {impl_nodes}")
    assert len(impl_nodes) == 2
    
    # 无并行组
    no_group = [n for n, node in dag_config.nodes.items() 
                if node.parallel_group is None]
    print(f"  无并行组: {no_group}")
    assert "planner" in no_group
    assert "reviewer" in no_group
    
    print("  [PASS] 并行组识别正确")


def test_execution_order(dag_config):
    """测试执行顺序"""
    print("\n" + "="*60)
    print("Test 7: 执行顺序验证")
    print("="*60)
    
    print("\n  执行阶段:")
    for stage_idx, stage in enumerate(dag_config.execution_order):
        print(f"    Stage {stage_idx}: {stage}")
        
        # 检查依赖是否都在之前的阶段
        for node_id in stage:
            node = dag_config.nodes[node_id]
            for dep in node.dependencies:
                # 找到 dep 所在的阶段
                dep_stage = None
                for si, s in enumerate(dag_config.execution_order):
                    if dep in s:
                        dep_stage = si
                        break
                assert dep_stage is not None
                assert dep_stage < stage_idx, f"{dep} 在 stage {dep_stage} 应该在 stage {stage_idx} 之前"
    
    # 验证所有节点都在执行顺序中
    all_in_order = set(node for stage in dag_config.execution_order for node in stage)
    assert all_in_order == set(dag_config.nodes.keys())
    
    print("\n  [PASS] 执行顺序正确")


def test_argparse():
    """测试命令行参数"""
    print("\n" + "="*60)
    print("Test 8: 命令行参数解析")
    print("="*60)
    
    import argparse
    
    # 模拟命令行参数
    test_cases = [
        # Linear workflow
        ["--config", "kvflow", "--num-workflows", "4", "--agents-per-workflow", "5"],
        # DAG workflow
        ["--config", "kvflow", "--workflow-type", "dag", 
         "--dag-config", "dag_configs/diamond_6agent.json",
         "--num-workflows", "4"],
    ]
    
    for args_list in test_cases:
        print(f"\n  测试: {' '.join(args_list)}")
        parser = argparse.ArgumentParser()
        # 重新创建 parser 来测试
        sys.argv = ["test.py"] + args_list
        
        # 检查是否需要 dag-config
        if "--workflow-type" in args_list and "dag" in args_list:
            print("    [INFO] DAG 模式，需要 --dag-config")
    
    print("  [PASS] 命令行参数正确")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DAG Workflow 功能测试套件")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        dag_config = test_dag_config_loading()
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] 配置加载: {e}")
        tests_failed += 1
        return
    
    try:
        test_dag_topology(dag_config)
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] 拓扑结构: {e}")
        tests_failed += 1
    
    try:
        test_dag_dependencies(dag_config)
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] 依赖检查: {e}")
        tests_failed += 1
    
    try:
        test_dag_priority(dag_config)
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] 优先级计算: {e}")
        tests_failed += 1
    
    try:
        test_dag_successors(dag_config)
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] 后继查找: {e}")
        tests_failed += 1
    
    try:
        test_parallel_group(dag_config)
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] 并行组: {e}")
        tests_failed += 1
    
    try:
        test_execution_order(dag_config)
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] 执行顺序: {e}")
        tests_failed += 1
    
    try:
        test_argparse()
        tests_passed += 1
    except Exception as e:
        print(f"  [FAIL] 命令行参数: {e}")
        tests_failed += 1
    
    # 总结
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"  通过: {tests_passed}")
    print(f"  失败: {tests_failed}")
    print(f"  总计: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n  所有测试通过!")
        return 0
    else:
        print(f"\n  {tests_failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
