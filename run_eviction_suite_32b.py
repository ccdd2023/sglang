#!/usr/bin/env python3
"""Quick eviction suite runner for 32B GPTQ-Int4 model.

Usage:
  python run_eviction_suite_32b.py --output-dir /path/to/output
"""
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    base_port = 32700

    policies = ['lru', 'priority', 'tiered']
    for i, policy in enumerate(policies):
        port = base_port + i
        cmd = [
            sys.executable,
            'benchmark/multi_workflow/run_serial_eviction_suite.py',
            '--policies', policy,
            '--output-dir', args.output_dir,
            '--num-unique', '60',
            '--model-path', '/home/gfy/models/Qwen2.5-32B-Instruct-GPTQ-Int4',
            '--mem-fraction-static', '0.85',
            '--max-total-tokens', '8000',
            '--chunked-prefill-size', '1024',
            '--max-prefill-tokens', '2048',
            '--hicache-ratio', '1.0',
            '--real-templates', '/home/gfy/Paper_CodeMAS/CodeServing/paper_figures/data/real_templates_export.json',
            '--real-template-role', 'planner',
            '--real-templates-mode', 'mix',
            '--server-python', sys.executable,
            '--base-port', str(port),
        ]
        print(f'Running eviction suite for {policy} on port {port}...')
        subprocess.run(cmd, check=True, cwd='/home/gfy/CodeMAS_Project/sglang-kvflow')

if __name__ == '__main__':
    main()
