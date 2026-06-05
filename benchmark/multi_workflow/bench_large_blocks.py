#!/usr/bin/env python3
"""Large code-block KV reuse test. Reads /tmp/large_tasks.json (SWE Verified + CodeHub)."""
from __future__ import annotations
import argparse, asyncio, json, os, signal, subprocess, sys, time
from pathlib import Path
import aiohttp

for entry in (str(Path(__file__).resolve().parents[3]/"MAScoder"/"src"),
              str(Path(__file__).resolve().parents[3]/"sglang-kvflow"/"python")):
    if entry not in sys.path: sys.path.insert(0, entry)
from mascoder.code_anchor import build_code_anchor_payload

PORT=30000; MODEL="/home/gfy/models/Qwen2.5-3B-Instruct"
ROOT=Path(__file__).resolve().parents[3]/"sglang-kvflow"
PY="/home/gfy/.conda/envs/sglang-kvflow/bin/python"

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
        env=e,stdout=open("/tmp/sglang_lb.log","w"),stderr=subprocess.STDOUT,cwd=str(ROOT))

def wait_ready(t=120):
    import urllib.request; d=time.monotonic()+t
    while time.monotonic()<d:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate",timeout=3) as r:
                if r.status==200: return True
        except: time.sleep(3)
    return False

def get_sig(code):
    for line in code.strip().split('\n'):
        if line.strip().startswith(('def ','class ')): return line.strip()
    return code.split('\n')[0][:60]

def pld(model, code, sig, mt, rm):
    a=build_code_anchor_payload(code,language="python")
    return {"model":model,"messages":[{"role":"system","content":"Analyze this code in one sentence."},
        {"role":"user","content":f"```\n{code}```"}],
        "max_tokens":mt,"temperature":0.0,
        "code_anchor_signature":a.get("ast_anchor_signature",""),
        "code_content_signature":a.get("code_content_signature",""),
        "code_anchor_spans":a.get("code_anchor_spans",[]),
        "reuse_mode":rm,"lossy_alignment_method":"kvcomm"}

async def req(sess,base,payload):
    s=time.perf_counter()
    async with sess.post(f"{base}/v1/chat/completions",json=payload,timeout=aiohttp.ClientTimeout(total=180)) as r:
        b=await r.json()
    return {"total_ms":(time.perf_counter()-s)*1000,"body":b}

def get_meta(b):
    try: return b["metadata"]["lossy_reuse"]
    except: return {}

async def run(task,base,model,mt):
    pre,post=task['pre_code'],task['post_code']
    src=task['source'];lbl=f"{src}/{task['repo']}/{task['id']}"[:60]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        await req(sess,base,pld(model,pre,get_sig(pre),mt,"lossless"))
        rly=await req(sess,base,pld(model,post,get_sig(post),mt,"lossy"))
        rlr=await req(sess,base,pld(model,post,get_sig(post),mt,"lossless"))
    m=get_meta(rly["body"]) if rly["body"] else {}
    return {"label":lbl,"source":src,"pre_ln":task['pre_lines'],"post_ln":task['post_lines'],
            "a2_matcher":m.get("lossy_first_match_reason",""),
            "a2_allowed":m.get("lossy_first_reuse_allowed",""),
            "a2_cand":m.get("lossy_candidate_count",0),
            "ly_ms":round(rly["total_ms"],0),"lr_ms":round(rlr["total_ms"],0)}

def report(rs,od):
    od=Path(od); od.mkdir(parents=True,exist_ok=True)
    (od/"results.json").write_text(json.dumps(rs,indent=2,ensure_ascii=False))
    ac=[r for r in rs if r["a2_allowed"]==True]; rj=[r for r in rs if r["a2_allowed"]==False]
    l=["# Large Code-block KV Reuse (≥15 lines)","",
       f"Tasks: {len(rs)} | Accept: {len(ac)} ({len(ac)/len(rs)*100:.0f}%) | Reject: {len(rj)} ({len(rj)/len(rs)*100:.0f}%)","",
       "|#|source|pre/post|matcher|gate|ly_ms|lr_ms|cand|",
       "|---|---|---:|---|---|---:|---:|---:|"]
    for i,r in enumerate(rs):
        g='allow' if r['a2_allowed'] else 'reject'
        l.append(f"|{i}|{r['source']}|{r['pre_ln']}/{r['post_ln']}|{r['a2_matcher'] or 'no_overlap'}|{g}|{r['ly_ms']:.0f}|{r['lr_ms']:.0f}|{r['a2_cand']}|")
    if ac: l+=["",f"- Accept avg lat: ly={sum(r['ly_ms'] for r in ac)/len(ac):.0f}ms lr={sum(r['lr_ms'] for r in ac)/len(ac):.0f}ms"]
    if rj: l+=["",f"- Reject avg lat: ly={sum(r['ly_ms'] for r in rj)/len(rj):.0f}ms lr={sum(r['lr_ms'] for r in rj)/len(rj):.0f}ms"]
    (od/"summary.md").write_text("\n".join(l)+"\n")

async def main(args):
    tasks=json.load(open("/tmp/large_tasks.json"))
    print(f"Loaded {len(tasks)} large tasks (≥15 lines)")
    rs=[]; base=f"http://127.0.0.1:{PORT}"
    for i,t in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {t['source']}/{t['repo']}: {t['pre_lines']}→{t['post_lines']} lines")
        kill(); time.sleep(2); p=launch()
        if not wait_ready(): print("  fail"); p.terminate(); continue
        r=await run(t,base,args.model,args.max_tokens); rs.append(r)
        print(f"  matcher={r['a2_matcher'] or 'no_overlap'} allow={r['a2_allowed']} "
              f"ly={r['ly_ms']:.0f}ms lr={r['lr_ms']:.0f}ms cand={r['a2_cand']}")
        p.terminate(); time.sleep(3); kill()
    report(rs,args.output_dir)
    print(f"\nDone → {args.output_dir}/")

def pa():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=MODEL); p.add_argument("--max-tokens",type=int,default=128)
    p.add_argument("--output-dir",default="/tmp/large_block_results")
    return p.parse_args()
if __name__=="__main__": asyncio.run(main(pa()))
