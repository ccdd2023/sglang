#!/usr/bin/env python3
"""Benchmark lossy KV reuse accuracy — pass@3 edition.

Each case restarts server → warmup → 3×lossy + 3×lossless at temp=0.7.
Metric: functional_test_pass@3.
"""
from __future__ import annotations
import argparse, asyncio, json, os, signal, subprocess, sys, time
from pathlib import Path
import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(PROJECT_ROOT/"MAScoder"/"src"), str(PROJECT_ROOT/"sglang-kvflow"/"python")):
    if entry not in sys.path: sys.path.insert(0, entry)
from mascoder.code_anchor import build_code_anchor_payload

WARMUP = "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n"

CASES = [
    {"id":"exact_same","desc":"完全相同","sig":"def factorial(n: int) -> int:",
     "body":"    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
     "tests":[("factorial(0)",1),("factorial(5)",120),("factorial(7)",5040)],"exp":"exact_code_content_signature"},
    {"id":"rename_vars","desc":"变量重命名 n→k","sig":"def factorial(k: int) -> int:",
     "body":"    if k <= 1:\n        return 1\n    return k * factorial(k - 1)",
     "tests":[("factorial(0)",1),("factorial(5)",120),("factorial(7)",5040)],"exp":"code_content_signature_mismatch"},
    {"id":"diff_algo","desc":"同功能不同算法(递归→迭代)","sig":"def factorial_iter(n: int) -> int:",
     "body":"    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",
     "tests":[("factorial_iter(0)",1),("factorial_iter(5)",120)],"exp":"code_content_signature_mismatch"},
    {"id":"diff_func","desc":"不同功能(factorial→fibonacci)","sig":"def fibonacci(n: int) -> int:",
     "body":"    if n <= 1:\n        return n\n    return fibonacci(n-1)+fibonacci(n-2)",
     "tests":[("fibonacci(0)",0),("fibonacci(6)",8)],"exp":"no_anchor_overlap"},
    {"id":"structure","desc":"结构改写(递归→条件表达式)","sig":"def factorial_ternary(n: int) -> int:",
     "body":"    return 1 if n <= 1 else n * factorial_ternary(n - 1)",
     "tests":[("factorial_ternary(0)",1),("factorial_ternary(5)",120)],"exp":"span_overlap_high"},
    {"id":"unrelated","desc":"完全不相关","sig":"def is_palindrome(s: str) -> bool:",
     "body":"    return s == s[::-1]",
     "tests":[("is_palindrome('racecar')",True),("is_palindrome('hello')",False)],"exp":"no_anchor_overlap"},
]
PORT=30000; MODEL="/home/gfy/models/Qwen2.5-3B-Instruct"; ROOT=PROJECT_ROOT/"sglang-kvflow"
PY="/home/gfy/.conda/envs/sglang-kvflow/bin/python"

def kill(): 
    import os,signal;ino=None
    try:
        with open("/proc/net/tcp") as f:
            for r in f.readlines()[1:]:
                p=r.split()
                if p[1].endswith(f":{PORT:04X}") and p[3]=="0A":ino=p[9];break
    except:pass
    if not ino:return
    for pid in sorted(filter(str.isdigit,os.listdir("/proc")),key=int):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.readlink(f"/proc/{pid}/fd/{fd}")==f"socket:[{ino}]":os.kill(int(pid),signal.SIGTERM);time.sleep(2);return
        except:pass

def launch():
    e=os.environ.copy();e["PYTHONPATH"]=str(ROOT/"python")+(":"+e.get("PYTHONPATH","") if e.get("PYTHONPATH") else "")
    return subprocess.Popen([PY,"-m","sglang.launch_server","--model-path",MODEL,"--port",str(PORT),"--tp-size","1","--mem-fraction-static","0.85","--max-total-tokens","32768","--chunked-prefill-size","4096","--max-prefill-tokens","8192","--radix-eviction-policy","priority","--enable-hierarchical-cache","--hicache-ratio","1.5","--hicache-write-policy","write_back","--enable-cache-report","--disable-cuda-graph","--log-level","info"],env=e,stdout=open("/tmp/sglang_acc.log","w"),stderr=subprocess.STDOUT,cwd=str(ROOT))

def wait_ready(t=120):
    import urllib.request;d=time.monotonic()+t
    while time.monotonic()<d:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health_generate",timeout=3) as r:
                if r.status==200:return True
        except:time.sleep(3)
    return False

def pld(model,code_text,sig,mt,rm,temp=0.0):
    a=build_code_anchor_payload(code_text,language="python")
    return {"model":model,"messages":[{"role":"system","content":"Write only the requested function body, no explanation."},{"role":"user","content":f"Complete this function:\n{sig}"}],"max_tokens":mt,"temperature":temp,"code_anchor_signature":a.get("ast_anchor_signature",""),"code_content_signature":a.get("code_content_signature",""),"code_anchor_spans":a.get("code_anchor_spans",[]),"reuse_mode":rm,"lossy_alignment_method":"kvcomm"}

async def req(sess,base,payload):
    s=time.perf_counter()
    async with sess.post(f"{base}/v1/chat/completions",json=payload) as r: b=await r.json()
    return {"elapsed":(time.perf_counter()-s)*1000,"body":b}

def get_text(b): 
    try: return b["choices"][0]["message"]["content"]
    except: return ""
def get_meta(b):
    try: return b["metadata"]["lossy_reuse"]
    except: return {}

def run_tests(code,tests):
    p=0
    for expr,exp in tests:
        try:
            ns={};exec(code,{"__builtins__":__builtins__},ns);r=eval(expr,{"__builtins__":__builtins__},ns)
            if r==exp:p+=1
        except:pass
    return p,len(tests)

async def run(task,base,model,mt,ns=3):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:
        # Warmup uses the actual WARMUP code as anchor, not the task's sig+body
        wp=pld(model,WARMUP,WARMUP.split("\n")[0],mt,"lossless",0.0)
        await req(sess,base,wp)
        at=task["sig"]+"\n"+task["body"]
        ly,lr=[],[]
        for _ in range(ns):
            ly.append(await req(sess,base,pld(model,at,task["sig"],mt,"lossy",0.7)))
            lr.append(await req(sess,base,pld(model,at,task["sig"],mt,"lossless",0.7)))
    def cnt(rs):
        p=0
        for r in rs:
            t=get_text(r["body"]);c=t.strip() if t.strip().startswith("def ") else task["sig"]+"\n"+t
            pp,_=run_tests(c,task["tests"])
            if pp==len(task["tests"]):p+=1
        return p
    ly_p=cnt(ly);lr_p=cnt(lr);m=get_meta(ly[0]["body"])
    return {"id":task["id"],"desc":task["desc"],"exp":task["exp"],"ns":ns,"ly_pass":ly_p,"lr_pass":lr_p,"nt":len(task["tests"]),"ly_avg_ms":sum(r["elapsed"]for r in ly)/ns,"lr_avg_ms":sum(r["elapsed"]for r in lr)/ns,"matcher":m.get("lossy_first_match_reason","?"),"allowed":m.get("lossy_first_reuse_allowed","?")}

def report(rs,od):
    od=Path(od);od.mkdir(parents=True,exist_ok=True)
    (od/"results.json").write_text(json.dumps(rs,indent=2,ensure_ascii=False))
    l=[f"# Lossy KV Reuse Accuracy — pass@{rs[0]['ns']}","","|case|desc|lossy_pass|lossless_pass|total|matcher|allowed|lossy_ms|lossless_ms|","|---|---|---:|---:|---:|---|---|---:|---:|"]
    for r in rs:l.append(f"|{r['id']}|{r['desc']}|{r['ly_pass']}|{r['lr_pass']}|{r['nt']}|{r['matcher']}|{r['allowed']}|{r['ly_avg_ms']:.0f}|{r['lr_avg_ms']:.0f}|")
    ac=[r for r in rs if r.get("allowed")];rj=[r for r in rs if r.get("allowed") is False]
    l+=["","## Summary",f"- Matcher accept: {len(ac)}/{len(rs)}",f"- Matcher reject: {len(rj)}/{len(rs)}"]
    if ac:l+=[f"- Accepted avg pass@{rs[0]['ns']}: {sum(r['ly_pass']/r['ns'] for r in ac)/len(ac):.0%}"]
    (od/"summary.md").write_text("\n".join(l)+"\n")

async def main(args):
    rs=[];base=f"http://127.0.0.1:{PORT}"
    for i,t in enumerate(CASES):
        print(f"\n--- {i+1}/{len(CASES)}: {t['id']} ---")
        kill();time.sleep(2);p=launch()
        if not wait_ready():print("server fail");p.terminate();continue
        r=await run(t,base,args.model,args.max_tokens);rs.append(r)
        print(f"  lossy:{r['ly_pass']}/{r['ns']} lossless:{r['lr_pass']}/{r['ns']} matcher:{r['matcher']} allowed:{r['allowed']}")
        p.terminate();time.sleep(3);kill()
    report(rs,args.output_dir);print(f"\nDone. {args.output_dir}/")

def pa():p=argparse.ArgumentParser();p.add_argument("--model",default=MODEL);p.add_argument("--max-tokens",type=int,default=256);p.add_argument("--output-dir",default="/tmp/lossy_accuracy_v4");return p.parse_args()
if __name__=="__main__":asyncio.run(main(pa()))
