#!/usr/bin/env python3
"""Smoke + load test for FRP tunnel to sglang (or any OpenAI-compatible server).

Usage:
    # With sglang:
    python test_tunnel.py --base-url http://65.108.32.175:39001/v1 --api-key test

    # With any OpenAI-compatible endpoint:
    python test_tunnel.py --base-url http://65.108.32.175:39001/v1 --api-key test --concurrency 64

    # Quick connectivity check only (no model needed):
    python test_tunnel.py --base-url http://65.108.32.175:39001 --api-key test --health-only
"""

import argparse
import asyncio
import statistics
import sys
import time

import httpx


async def health_check(base_url: str, api_key: str) -> bool:
    """Check if the endpoint is reachable (GET /models or GET /)."""
    print("[1/3] Health check...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try OpenAI /models endpoint first
            r = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code == 200:
                data = r.json()
                models = data.get("data", [])
                if models:
                    model_id = models[0].get("id", "unknown")
                    print(f"  OK - model: {model_id}")
                    return True
                print(f"  OK - endpoint reachable (no models listed)")
                return True

            # Fallback: just check connectivity
            r = await client.get(base_url)
            print(f"  OK - endpoint reachable (status {r.status_code})")
            return True
    except Exception as e:
        print(f"  FAIL - {e}")
        return False


async def single_completion(base_url: str, api_key: str, model: str) -> bool:
    """Send a single chat completion request."""
    print("[2/3] Single completion...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello in one word."}],
                    "max_tokens": 16,
                },
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  OK - response: {content[:80]}")
            return True
    except Exception as e:
        print(f"  FAIL - {e}")
        return False


async def load_test(
    base_url: str, api_key: str, model: str, concurrency: int
) -> bool:
    """Concurrent load test."""
    print(f"[3/3] Load test ({concurrency} concurrent requests)...")
    latencies: list[float] = []
    errors = 0

    async def one_request(i: int):
        nonlocal errors
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "user", "content": f"Count to {i % 10 + 1}"}
                        ],
                        "max_tokens": 32,
                    },
                )
                r.raise_for_status()
                latencies.append(time.monotonic() - t0)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Request {i} failed: {e}")

    await asyncio.gather(*[one_request(i) for i in range(concurrency)])

    if latencies:
        latencies.sort()
        p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)
        print(f"\n  Results: {len(latencies)} OK, {errors} errors")
        print(
            f"  Latency: min={min(latencies):.2f}s  max={max(latencies):.2f}s  "
            f"mean={statistics.mean(latencies):.2f}s  p99={latencies[p99_idx]:.2f}s"
        )
    else:
        print(f"\n  All {errors} requests failed!")

    return errors == 0


async def get_model_id(base_url: str, api_key: str) -> str:
    """Try to discover the model ID from /models endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code == 200:
                data = r.json()
                models = data.get("data", [])
                if models:
                    return models[0]["id"]
    except Exception:
        pass
    return "default"


async def main(args):
    # Health check
    ok = await health_check(args.base_url, args.api_key)
    if not ok:
        print("\nFAILED: endpoint not reachable")
        return 1
    if args.health_only:
        print("\nPASSED: health check OK")
        return 0

    # Discover model
    model = args.model or await get_model_id(args.base_url, args.api_key)
    print(f"  Using model: {model}")

    # Single completion
    ok = await single_completion(args.base_url, args.api_key, model)
    if not ok:
        print("\nFAILED: single completion failed")
        return 1

    # Load test
    ok = await load_test(args.base_url, args.api_key, model, args.concurrency)
    if ok:
        print("\nPASSED: all tests OK")
        return 0
    else:
        print("\nPARTIAL: some requests failed (see above)")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRP tunnel smoke + load test")
    parser.add_argument("--base-url", required=True, help="e.g. http://relay:39001/v1")
    parser.add_argument("--api-key", default="test")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--model", default="", help="Model ID (auto-detected if empty)")
    parser.add_argument(
        "--health-only", action="store_true", help="Only check connectivity"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
