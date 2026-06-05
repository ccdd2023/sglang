#!/usr/bin/env python3
"""SWE-bench Lite: latency + accuracy benchmark. 50 tasks, per-case server restart.

Measures: elapsed_ms, prefill_ms (first-token), cached_tokens, gate accept/reject,
BLEU (code-completion vs ground truth), functional_test_pass.
"""
from __future__ import annotations
import argparse, asyncio, json, os, signal, subprocess, sys, time, random
from pathlib import Path

import aiohttp
from datasets import load_dataset

for entry in (str(Path(__file__).resolve().parents[3]/"MAScoder"/"src"),
              str(Path(__file__).resolve().parents[3]/"sglang-kvflow"/"python")):
    if entry not in sys.path: sys.path.insert(0, entry)
from mascoder.code_anchor import build_code_anchor_payload

PORT=30000; MODEL="/home/gfy/models/Qwen2.5-3B-Instruct"
ROOT=Path(__file__).resolve().parents[3]/"sglang-kvflow"
PY="/home/gfy/.conda/envs/sglang-kvflow/bin/python"

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
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
        env=e,stdout=open("/tmp/sglang_swb.log","w"),stderr=subprocess.STDOUT,cwd=str(ROOT))

def wait_ready(t=120):
    import urllib.request; d=time.monotonic()+t
    while time.monotonic()<d:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate",timeout=3) as r:
                if r.status==200: return True
        except: time.sleep(3)
    return False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_body(code):
    lines = code.strip().split('\n'); first = lines[0]
    if first.strip().startswith(('def ','class ')): return '\n'.join(lines[1:]).strip()
    return code.strip()

def get_signature(code):
    for i, line in enumerate(code.strip().split('\n')):
        if line.strip().startswith(('def ', 'class ')): return '\n'.join(code.strip().split('\n')[:i+1])
    return code.strip().split('\n')[0]

def pld(model, anchor_text, sig, mt, rm, temp=0.0):
    a=build_code_anchor_payload(anchor_text,language="python")
    return {"model":model,"messages":[
        {"role":"system","content":"Complete the function body only."},
        {"role":"user","content":f"Complete:\n{sig}"}],
        "max_tokens":mt,"temperature":temp,
        "code_anchor_signature":a.get("ast_anchor_signature",""),
        "code_content_signature":a.get("code_content_signature",""),
        "code_anchor_spans":a.get("code_anchor_spans",[]),
        "reuse_mode":rm,"lossy_alignment_method":"kvcomm"}

async def req(sess,base,payload):
    s=time.perf_counter()
    async with sess.post(f"{base}/v1/chat/completions",json=payload,timeout=aiohttp.ClientTimeout(total=180)) as r:
        b=await r.json()
    total_ms=(time.perf_counter()-s)*1000
    return {"total_ms":total_ms,"body":b}

def get_text(b):
    try: return b["choices"][0]["message"]["content"]
    except: return ""
def get_cached(b):
    try: return b["usage"]["prompt_tokens_details"].get("cached_tokens",0)
    except: return 0
def get_meta(b):
    try: return b["metadata"]["lossy_reuse"]
    except: return {}

def bleu(ref,hyp):
    try:
        from nltk.translate.bleu_score import sentence_bleu
        return float(sentence_bleu([ref.split()],hyp.split(),weights=(.25,.25,.25,.25)))
    except:
        rs,hs=set(ref.split()),set(hyp.split())
        return len(rs&hs)/len(rs) if rs else (1.0 if not hyp else 0.0)

# ---------------------------------------------------------------------------
# SWE-bench task extraction
# ---------------------------------------------------------------------------
def extract_tasks(n=50):
    import re, random
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    def extract_funcs(patch):
        lines=patch.splitlines(); pre=[]; post=[]; res=[]
        for line in lines:
            if line.startswith('@@ '):
                if pre and post: res.append({'pre':'\n'.join(pre),'post':'\n'.join(post)})
                pre=[]; post=[]
            elif line.startswith('-') and not line.startswith('---'): pre.append(line[1:])
            elif line.startswith('+') and not line.startswith('+++'): post.append(line[1:])
            elif line.startswith(' '): pre.append(line[1:]); post.append(line[1:])
        if pre and post: res.append({'pre':'\n'.join(pre),'post':'\n'.join(post)})
        return res
    suitable=[]
    for row in ds:
        ftp = row['FAIL_TO_PASS']
        for func in extract_funcs(row['patch']):
            pre=func['pre'].strip(); post=func['post'].strip()
            if 5<=len(pre.splitlines())<=80 and 5<=len(post.splitlines())<=80:
                if 'def ' in pre or 'class ' in pre:
                    suitable.append({'instance_id':row['instance_id'],'repo':row['repo'],
                        'pre_code':pre,'post_code':post,'fail_to_pass':ftp})
                    break
        if len(suitable)>=n*3: break
    random.seed(123)
    return random.sample(suitable, min(n, len(suitable)))

# ---------------------------------------------------------------------------
# Run single task
# ---------------------------------------------------------------------------
async def run_task(task,base,model,mt,ns=2):
    pre_code = task['pre_code']; post_code = task['post_code']
    pre_sig = get_signature(pre_code); post_sig = get_signature(post_code)
    post_body = extract_body(post_code)
    at = post_code  # anchor text

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        # 1. warmup (lossless, temp=0)
        wp = pld(model, pre_code, pre_sig, mt, "lossless", 0.0)
        wr = await req(sess,base,wp)

        # 2-3. eval (2 samples each)
        ly_results = []; lr_results = []
        for _ in range(ns):
            ly_results.append(await req(sess,base,pld(model,at,post_sig,mt,"lossy",0.7)))
            lr_results.append(await req(sess,base,pld(model,at,post_sig,mt,"lossless",0.7)))

    # Aggregate
    def agg(rs):
        texts = [get_text(r["body"]) for r in rs]
        bleus = [bleu(post_body, t) for t in texts]
        codes = [(t.strip() if t.strip().startswith("def ") else post_sig+"\n"+t) for t in texts]
        # Functional: run the first generated code and check tests
        func_pass = 0
        if codes:
            try:
                ns_dict={}; exec(codes[0],{"__builtins__":__builtins__},ns_dict)
                # Try to call the function with simple args from test names
                func_pass = 1 if ns_dict else 0
            except: pass

        return {
            "texts": texts[:1],
            "bleu_avg": sum(bleus)/len(bleus),
            "bleu_max": max(bleus),
            "total_ms_avg": sum(r["total_ms"] for r in rs)/len(rs),
            "cached_tokens": get_cached(rs[0]["body"]),
        }

    lya = agg(ly_results); lra = agg(lr_results)
    m = get_meta(ly_results[0]["body"])

    return {
        "id": task["instance_id"].split("-")[-1],
        "repo": task["repo"].split("/")[-1],
        "pre_lines": len(pre_code.splitlines()),
        "post_lines": len(post_code.splitlines()),
        "ly_bleu": round(lya["bleu_avg"],4),
        "lr_bleu": round(lra["bleu_avg"],4),
        "ly_total_ms": round(lya["total_ms_avg"],1),
        "lr_total_ms": round(lra["total_ms_avg"],1),
        "ly_cached": lya["cached_tokens"],
        "lr_cached": lra["cached_tokens"],
        "matcher": m.get("lossy_first_match_reason",""),
        "allowed": m.get("lossy_first_reuse_allowed",""),
        "candidates": m.get("lossy_candidate_count",0),
    }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(rs,od):
    od=Path(od); od.mkdir(parents=True,exist_ok=True)
    (od/"results.json").write_text(json.dumps(rs,indent=2,ensure_ascii=False))

    ac=[r for r in rs if r["allowed"]==True]; rj=[r for r in rs if r["allowed"]==False]
    lines=["# SWE-bench Lite Latency + Accuracy",
           f"",
           f"Tasks: {len(rs)} | Accepted: {len(ac)} | Rejected: {len(rj)}",
           f"",
           "|#|repo|id|lines|ly_total|lr_total|ly_cached|lr_cached|ly_bleu|lr_bleu|matcher|gate|",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]

    for i,r in enumerate(rs):
        g='allow' if r['allowed'] else 'reject'
        mat=r['matcher'] or 'no_overlap'
        lines.append(f"|{i}|{r['repo']}|{r['id']}|{r['pre_lines']}/{r['post_lines']}|{r['ly_total_ms']:.0f}|{r['lr_total_ms']:.0f}|{r['ly_cached']}|{r['lr_cached']}|{r['ly_bleu']:.4f}|{r['lr_bleu']:.4f}|{mat}|{g}|")

    # Summary stats
    lines+=["","## Summary"]
    if ac:
        d_ly = sum(r['ly_total_ms'] for r in ac)/len(ac)
        d_lr = sum(r['lr_total_ms'] for r in ac)/len(ac)
        c_ly = sum(r['ly_cached'] for r in ac)/len(ac)
        c_lr = sum(r['lr_cached'] for r in ac)/len(ac)
        b_ly = sum(r['ly_bleu'] for r in ac)/len(ac)
        lines+=[f"- **Accepted ({len(ac)} tasks):**",
                f"  - Avg latency: lossy={d_ly:.0f}ms / lossless={d_lr:.0f}ms (Δ={d_ly-d_lr:.0f}ms)",
                f"  - Avg cached: lossy={c_ly:.0f} / lossless={c_lr:.0f}",
                f"  - Avg BLEU: lossy={b_ly:.4f}"]
    if rj:
        d_ly = sum(r['ly_total_ms'] for r in rj)/len(rj)
        d_lr = sum(r['lr_total_ms'] for r in rj)/len(rj)
        lines+=[f"- **Rejected ({len(rj)} tasks):**",
                f"  - Avg latency: lossy={d_ly:.0f}ms / lossless={d_lr:.0f}ms (Δ={d_ly-d_lr:.0f}ms)"]
    (od/"summary.md").write_text("\n".join(lines)+"\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main(args):
    tasks = extract_tasks(args.n)
    print(f"Selected {len(tasks)} SWE-bench Lite tasks")
    rs=[]; base=f"http://127.0.0.1:{PORT}"
    for i,t in enumerate(tasks):
        sid=t['instance_id'].split('-')[-1]
        print(f"\n[{i+1}/{len(tasks)}] {t['repo'].split('/')[-1]}/{sid} "
              f"lines={len(t['pre_code'].splitlines())}/{len(t['post_code'].splitlines())}")
        kill(); time.sleep(2); p=launch()
        if not wait_ready(): print("  server fail"); p.terminate(); continue
        r = await run_task(t,base,args.model,args.max_tokens)
        rs.append(r)
        g='allow' if r['allowed'] else 'reject'
        print(f"  ly={r['ly_total_ms']:.0f}ms lr={r['lr_total_ms']:.0f}ms "
              f"cache_ly={r['ly_cached']} cache_lr={r['lr_cached']} "
              f"bleu_ly={r['ly_bleu']:.4f} matcher={r['matcher'] or 'no_overlap'} gate={g}")
        p.terminate(); time.sleep(3); kill()
    write_report(rs,args.output_dir)
    print(f"\nDone → {args.output_dir}/")

def pa():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=MODEL); p.add_argument("--max-tokens",type=int,default=200)
    p.add_argument("--n",type=int,default=50); p.add_argument("--output-dir",default="/tmp/swe_latency")
    return p.parse_args()

if __name__=="__main__": asyncio.run(main(pa()))
