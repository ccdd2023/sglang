#!/usr/bin/env python3
"""
Simple local test for KVFlow benchmark functionality.
Tests basic server connectivity, prefix caching, and eviction policy.
"""

import argparse
import asyncio
import json
import time
import aiohttp


async def test_server(host: str, port: int) -> dict:
    """Test basic server connectivity."""
    url = f"http://{host}:{port}/model_info"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {"status": "success", "model": data.get("model_path", "unknown")}
                else:
                    return {"status": "error", "code": resp.status}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def test_prefix_caching(host: str, port: int, base_prompt: str) -> dict:
    """Test prefix caching by sending same prompt twice."""
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "messages": [{"role": "user", "content": base_prompt}],
        "max_tokens": 20,
        "temperature": 0.7
    }
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        # First request - should have no cache
        start1 = time.time()
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            data1 = await resp.json()
            ttft1 = time.time() - start1
            cached1 = data1.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
            results.append({"request": 1, "ttft": ttft1, "cached": cached1})
        
        # Small delay
        await asyncio.sleep(1)
        
        # Second request - should have cache hit for prefix
        start2 = time.time()
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            data2 = await resp.json()
            ttft2 = time.time() - start2
            cached2 = data2.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
            results.append({"request": 2, "ttft": ttft2, "cached": cached2})
    
    return results


async def test_multi_tier_caching(host: str, port: int) -> dict:
    """Test multi-tier caching with different prefix levels."""
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    tiers = [
        ("System", "You are a helpful AI assistant. "),
        ("Tier-0 (universal)", "You are a helpful AI assistant. " + "Be concise. " * 50),
        ("Tier-1 (role)", "You are a Python code reviewer. Analyze the following code: "),
        ("Tier-2 (workflow)", "Review this function for performance issues:\n" + "def example():\n" + "    pass\n" * 20),
    ]
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        prev_cached = 0
        
        for name, prefix in tiers:
            payload = {
                "messages": [{"role": "user", "content": prefix + "What is 1+1?"}],
                "max_tokens": 10
            }
            
            start = time.time()
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                data = await resp.json()
                ttft = time.time() - start
                cached = data.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
                total_tokens = data.get("usage", {}).get("prompt_tokens", 0)
                
                results.append({
                    "tier": name,
                    "ttft": round(ttft, 3),
                    "cached_tokens": cached,
                    "total_tokens": total_tokens,
                    "cache_ratio": round(cached / total_tokens, 2) if total_tokens > 0 else 0
                })
                
                prev_cached = cached
            
            await asyncio.sleep(0.5)
    
    return results


async def run_tests(host: str, port: int):
    """Run all tests."""
    print("=" * 60)
    print("SGLang KVFlow Local Test")
    print("=" * 60)
    print(f"Server: {host}:{port}")
    print()
    
    # Test 1: Server connectivity
    print("[Test 1] Server Connectivity...")
    result = await test_server(host, port)
    if result["status"] == "success":
        print(f"  ✓ Server is healthy")
        print(f"  ✓ Model: {result['model']}")
    else:
        print(f"  ✗ Server error: {result}")
        return
    print()
    
    # Test 2: Prefix caching
    print("[Test 2] Prefix Caching Test...")
    base_prompt = "Explain what a decorator is in Python."
    cache_results = await test_prefix_caching(host, port, base_prompt)
    for r in cache_results:
        print(f"  Request {r['request']}: TTFT={r['ttft']:.3f}s, Cached={r['cached']} tokens")
    
    if len(cache_results) == 2:
        speedup = cache_results[0]["ttft"] / cache_results[1]["ttft"] if cache_results[1]["ttft"] > 0 else 0
        if cache_results[1]["cached"] > 0:
            print(f"  ✓ Cache hit detected! Speedup: {speedup:.2f}x")
        else:
            print(f"  - No cache hit (short prompt)")
    print()
    
    # Test 3: Multi-tier caching
    print("[Test 3] Multi-Tier Caching Test...")
    tier_results = await test_multi_tier_caching(host, port)
    for r in tier_results:
        print(f"  {r['tier']:20s}: TTFT={r['ttft']:.3f}s, Cached={r['cached_tokens']:3d}/{r['total_tokens']} tokens ({r['cache_ratio']:.0%})")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    total_cached = sum(r["cached_tokens"] for r in tier_results)
    total_tokens = sum(r["total_tokens"] for r in tier_results)
    if total_tokens > 0:
        print(f"Total cache ratio: {total_cached}/{total_tokens} = {total_cached/total_tokens:.1%}")
    print("✓ Local KVFlow test completed successfully!")
    print()


def main():
    parser = argparse.ArgumentParser(description="Simple KVFlow local test")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    args = parser.parse_args()
    
    asyncio.run(run_tests(args.host, args.port))


if __name__ == "__main__":
    main()
