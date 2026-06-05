#!/usr/bin/env python3
"""Multi-agent workflow + large code-block KV reuse test.

Tests two hypotheses:
  H1: In multi-agent chain, Agent 2's KV reuse is gated by similarity to Agent 1's code.
  H2: Larger code blocks (20-50 lines) give better matcher separation than tiny 5-line ones.

Workflow: Agent1(planner) → Agent2(implementer) → Agent3(reviewer)
  Each agent sends a prompt that includes the code block.
  Agent1 fills cache. Agent2/3's requests go through lossy KV reuse gating.
"""
from __future__ import annotations
import argparse, asyncio, json, os, signal, subprocess, sys, time, random
from pathlib import Path
import aiohttp

for entry in (str(Path(__file__).resolve().parents[3]/"MAScoder"/"src"),
              str(Path(__file__).resolve().parents[3]/"sglang-kvflow"/"python")):
    if entry not in sys.path: sys.path.insert(0, entry)
from mascoder.code_anchor import build_code_anchor_payload

PORT=30000; MODEL="/home/gfy/models/Qwen2.5-3B-Instruct"
ROOT=Path(__file__).resolve().parents[3]/"sglang-kvflow"
PY="/home/gfy/.conda/envs/sglang-kvflow/bin/python"

# ---- Server ----
def kill():
    try:
        with open("/proc/net/tcp") as f:
            for r in f.readlines()[1:]:
                p=r.split()
                if p[1].endswith(f":{PORT:04X}") and p[3]=="0A": ino=p[9]; break
            else: return
    except: return
    for pid in sorted(filter(str.isdigit,os.listdir("/proc")),key=int):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.readlink(f"/proc/{pid}/fd/{fd}")==f"socket:[{ino}]":
                    os.kill(int(pid),signal.SIGTERM); time.sleep(2); return
        except: pass

def launch():
    e=os.environ.copy()
    e["PYTHONPATH"]=str(ROOT/"python")+(":"+e.get("PYTHONPATH","") if e.get("PYTHONPATH") else "")
    return subprocess.Popen([PY,"-m","sglang.launch_server","--model-path",MODEL,"--port",str(PORT),
        "--tp-size","1","--mem-fraction-static","0.85","--max-total-tokens","32768",
        "--chunked-prefill-size","4096","--max-prefill-tokens","8192","--radix-eviction-policy","priority",
        "--enable-hierarchical-cache","--hicache-ratio","1.5","--hicache-write-policy","write_back",
        "--enable-cache-report","--disable-cuda-graph","--log-level","error"],
        env=e,stdout=open("/tmp/sglang_ma.log","w"),stderr=subprocess.STDOUT,cwd=str(ROOT))

def wait_ready(t=120):
    import urllib.request; d=time.monotonic()+t
    while time.monotonic()<d:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate",timeout=3) as r:
                if r.status==200: return True
        except: time.sleep(3)
    return False

# ---- Large code blocks (20-50 lines) ----
LARGE_CODE = {
    "avl_tree_insert": r'''class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

def avl_tree_insert(root, key):
    if not root:
        return AVLNode(key)
    if key < root.key:
        root.left = avl_tree_insert(root.left, key)
    elif key > root.key:
        root.right = avl_tree_insert(root.right, key)
    else:
        return root
    root.height = 1 + max(get_height(root.left), get_height(root.right))
    balance = get_balance(root)
    if balance > 1 and key < root.left.key:
        return right_rotate(root)
    if balance < -1 and key > root.right.key:
        return left_rotate(root)
    if balance > 1 and key > root.left.key:
        root.left = left_rotate(root.left)
        return right_rotate(root)
    if balance < -1 and key < root.right.key:
        root.right = right_rotate(root.right)
        return left_rotate(root)
    return root

def get_height(node):
    return node.height if node else 0

def get_balance(node):
    return get_height(node.left) - get_height(node.right) if node else 0

def right_rotate(y):
    x = y.left
    T2 = x.right
    x.right = y
    y.left = T2
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    x.height = 1 + max(get_height(x.left), get_height(x.right))
    return x

def left_rotate(x):
    y = x.right
    T2 = y.left
    y.left = x
    x.right = T2
    x.height = 1 + max(get_height(x.left), get_height(x.right))
    y.height = 1 + max(get_height(y.left), get_height(y.right))
    return y''',

    "rbtree_insert": r'''class RBNode:
    def __init__(self, key, color="RED"):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None

def rbtree_insert(root, key):
    node = RBNode(key)
    y = None
    x = root
    while x is not None:
        y = x
        if node.key < x.key:
            x = x.left
        else:
            x = x.right
    node.parent = y
    if y is None:
        node.color = "BLACK"
        return node
    elif node.key < y.key:
        y.left = node
    else:
        y.right = node
    if node.parent.parent is None:
        return root
    return _fix_rb_insert(root, node)

def _fix_rb_insert(root, k):
    while k.parent is not None and k.parent.color == "RED":
        if k.parent == k.parent.parent.right:
            u = k.parent.parent.left
            if u is not None and u.color == "RED":
                u.color = "BLACK"
                k.parent.color = "BLACK"
                k.parent.parent.color = "RED"
                k = k.parent.parent
            else:
                if k == k.parent.left:
                    k = k.parent
                    root = _rb_right_rotate(root, k)
                k.parent.color = "BLACK"
                k.parent.parent.color = "RED"
                root = _rb_left_rotate(root, k.parent.parent)
        else:
            u = k.parent.parent.right
            if u is not None and u.color == "RED":
                u.color = "BLACK"
                k.parent.color = "BLACK"
                k.parent.parent.color = "RED"
                k = k.parent.parent
            else:
                if k == k.parent.right:
                    k = k.parent
                    root = _rb_left_rotate(root, k)
                k.parent.color = "BLACK"
                k.parent.parent.color = "RED"
                root = _rb_right_rotate(root, k.parent.parent)
        if k == root:
            break
    root.color = "BLACK"
    return root

def _rb_left_rotate(root, x):
    y = x.right
    x.right = y.left
    if y.left is not None:
        y.left.parent = x
    y.parent = x.parent
    if x.parent is None:
        root = y
    elif x == x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y
    y.left = x
    x.parent = y
    return root

def _rb_right_rotate(root, y):
    x = y.left
    y.left = x.right
    if x.right is not None:
        x.right.parent = y
    x.parent = y.parent
    if y.parent is None:
        root = x
    elif y == y.parent.right:
        y.parent.right = x
    else:
        y.parent.left = x
    x.right = y
    y.parent = x
    return root''',

    "merge_sort": r'''def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    merge_sort(left)
    merge_sort(right)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        arr[k] = left[i]
        i += 1; k += 1
    while j < len(right):
        arr[k] = right[j]
        j += 1; k += 1
    return arr''',

    "heap_sort": r'''def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        _heapify(arr, i, 0)
    return arr

def _heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[largest] < arr[left]:
        largest = left
    if right < n and arr[largest] < arr[right]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        _heapify(arr, n, largest)''',

    "dijkstra": r'''import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)
        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances''',

    "bfs_shortest": r'''from collections import deque

def bfs_shortest_path(graph, start, target):
    visited = set()
    queue = deque([(start, [start])])
    visited.add(start)
    while queue:
        current_node, path = queue.popleft()
        if current_node == target:
            return path
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append((neighbor, new_path))
    return None''',
}

SYSTEM = "You are a Python coding assistant. Analyze the code."

# Multi-agent workflow scenarios
WORKFLOWS = [
    # (desc, agent1_code, agent2_code, expected)
    ("same-func-large (AVL insert vs AVL insert)", "avl_tree_insert", "avl_tree_insert", "exact_code_content_signature"),
    ("diff-impl-same-task (AVL vs RedBlack)", "avl_tree_insert", "rbtree_insert", "code_content_signature_mismatch"),
    ("diff-task-same-domain (merge_sort vs heap_sort)", "merge_sort", "heap_sort", "code_content_signature_mismatch"),
    ("diff-domain (AVL tree vs Dijkstra)", "avl_tree_insert", "dijkstra", "no_anchor_overlap"),
    ("diff-paradigm (Dijkstra vs BFS)", "dijkstra", "bfs_shortest", "code_content_signature_mismatch"),
    ("unrelated (AVL tree vs merge_sort)", "avl_tree_insert", "merge_sort", "no_anchor_overlap"),
]

def pld(model, code_text, agent_role, mt, rm, temp=0.0):
    a=build_code_anchor_payload(code_text,language="python")
    msg = f"[Agent: {agent_role}]\n\n{code_text}\n\nAnalyze this code in one sentence."
    return {"model":model,"messages":[
        {"role":"system","content":SYSTEM},
        {"role":"user","content":msg}],
        "max_tokens":mt,"temperature":temp,
        "code_anchor_signature":a.get("ast_anchor_signature",""),
        "code_content_signature":a.get("code_content_signature",""),
        "code_anchor_spans":a.get("code_anchor_spans",[]),
        "reuse_mode":rm,"lossy_alignment_method":"kvcomm"}

async def req(sess,base,payload):
    s=time.perf_counter()
    async with sess.post(f"{base}/v1/chat/completions",json=payload,timeout=aiohttp.ClientTimeout(total=180)) as r:
        b=await r.json()
    return {"total_ms":(time.perf_counter()-s)*1000,"body":b}

def get_text(b):
    try: return b["choices"][0]["message"]["content"]
    except: return ""
def get_cached(b):
    try: return b["usage"]["prompt_tokens_details"].get("cached_tokens",0)
    except: return 0
def get_meta(b):
    try: return b["metadata"]["lossy_reuse"]
    except: return {}

async def run_workflow(desc, k1, k2, exp_matcher, base, model, mt):
    code1 = LARGE_CODE[k1]; code2 = LARGE_CODE[k2]
    n1 = len(code1.splitlines()); n2 = len(code2.splitlines())

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        # Agent 1 (planner): fill cache with code1
        r1 = await req(sess,base,pld(model,code1,"planner",mt,"lossless"))
        meta1 = get_meta(r1["body"]) if r1["body"] else {}

        # Agent 2 (implementer): test lossy reuse
        r2_lossy = await req(sess,base,pld(model,code2,"implementer",mt,"lossy"))
        m2 = get_meta(r2_lossy["body"]) if r2_lossy["body"] else {}
        r2_lossless = await req(sess,base,pld(model,code2,"implementer",mt,"lossless"))

        # Agent 3 (reviewer): same code as agent2 but with shared cache
        r3_lossy = await req(sess,base,pld(model,code2,"reviewer",mt,"lossy"))
        m3 = get_meta(r3_lossy["body"]) if r3_lossy["body"] else {}

    return {
        "desc": desc,
        "code1": k1, "code2": k2,
        "lines1": n1, "lines2": n2,
        "a1_ms": round(r1["total_ms"],0),
        "expected_matcher": exp_matcher,
        "a2_matcher": m2.get("lossy_first_match_reason",""),
        "a2_allowed": m2.get("lossy_first_reuse_allowed",""),
        "a2_cached_ly": get_cached(r2_lossy["body"]),
        "a2_cached_lr": get_cached(r2_lossless["body"]),
        "a2_ly_ms": round(r2_lossy["total_ms"],0),
        "a2_lr_ms": round(r2_lossless["total_ms"],0),
        "a3_matcher": m3.get("lossy_first_match_reason",""),
        "a3_allowed": m3.get("lossy_first_reuse_allowed",""),
        "a3_cached": get_cached(r3_lossy["body"]),
    }

def report(rs,od):
    od=Path(od); od.mkdir(parents=True,exist_ok=True)
    (od/"results.json").write_text(json.dumps(rs,indent=2,ensure_ascii=False))
    lines=["# Multi-Agent + Large Code-block KV Reuse",
           f"",
           "| workflow | lines(A1→A2) | expected | a2_matcher | a2_allow | a2_cache | a2_ly | a2_lr | a3_allow |",
           "|---|---:|---|---|---:|---:|---:|---:|"]
    for r in rs:
        a2_m = r['a2_matcher'] or 'no_overlap'
        a2_g = 'allow' if r['a2_allowed'] else 'reject'
        a3_g = 'allow' if r['a3_allowed'] else 'reject'
        lines.append(f"| {r['desc'][:50]} | {r['lines1']}/{r['lines2']} | {r['expected_matcher']} | {a2_m} | {a2_g} | {r['a2_cached_ly']}/{r['a2_cached_lr']} | {r['a2_ly_ms']:.0f} | {r['a2_lr_ms']:.0f} | {a3_g} |")
    (od/"summary.md").write_text("\n".join(lines)+"\n")

async def main(args):
    rs=[]; base=f"http://127.0.0.1:{PORT}"
    for i,(desc,k1,k2,exp) in enumerate(WORKFLOWS):
        print(f"\n[{i+1}/{len(WORKFLOWS)}] {desc[:60]}")
        kill(); time.sleep(2); p=launch()
        if not wait_ready(): print("  fail"); p.terminate(); continue
        r = await run_workflow(desc,k1,k2,exp,base,args.model,args.max_tokens)
        rs.append(r)
        print(f"  A2: matcher={r['a2_matcher'] or 'no_overlap'} allow={r['a2_allowed']} "
              f"cache_ly={r['a2_cached_ly']}/lr={r['a2_cached_lr']} "
              f"t_ly={r['a2_ly_ms']:.0f}ms t_lr={r['a2_lr_ms']:.0f}ms "
              f"A3: allow={r['a3_allowed']}")
        p.terminate(); time.sleep(3); kill()
    report(rs,args.output_dir)
    print(f"\nDone → {args.output_dir}/")

def pa():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=MODEL); p.add_argument("--max-tokens",type=int,default=128)
    p.add_argument("--output-dir",default="/tmp/ma_large_blocks")
    return p.parse_args()

if __name__=="__main__": asyncio.run(main(pa()))
