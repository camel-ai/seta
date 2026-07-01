#!/usr/bin/env python3
"""Delete ONLY our Daytona sandboxes (those carrying the harbor_owner_uid label).

Creds are read from the process environment (DAYTONA_API_KEY / DAYTONA_API_URL);
nothing is printed. Dry-run by default; set DELETE=1 to actually delete.
"""
import asyncio
import os
from collections import Counter

from daytona import AsyncDaytona

OWNER_LABEL = "harbor_owner_uid"
DELETE = os.environ.get("DELETE") == "1"
CONC = int(os.environ.get("CONC", "16"))


async def _delete(d, sb):
    try:
        if hasattr(sb, "delete"):
            await asyncio.wait_for(sb.delete(), timeout=60)
        else:
            await asyncio.wait_for(d.delete(sb), timeout=60)
        return True
    except Exception:
        return False


async def _fetch_all(d):
    items, page, limit = [], 1, 100
    while True:
        res = await d.list(page=page, limit=limit)
        batch = getattr(res, "items", None) or []
        items.extend(batch)
        total_pages = getattr(res, "total_pages", 1) or 1
        if page >= total_pages or not batch:
            break
        page += 1
    return items


async def main():
    d = AsyncDaytona()
    sandboxes = await _fetch_all(d)
    ours, others = [], 0
    prefixes, states = Counter(), Counter()
    for sb in sandboxes:
        labels = getattr(sb, "labels", None) or {}
        owner = labels.get(OWNER_LABEL)
        if owner:
            ours.append(sb)
            parts = owner.split("-")
            prefix = "-".join(parts[:-2]) if len(parts) >= 3 else owner
            prefixes[prefix] += 1
            states[str(getattr(sb, "state", "?"))] += 1
        else:
            others += 1

    print(f"total sandboxes visible : {len(sandboxes)}")
    print(f"  OURS (harbor_owner_uid): {len(ours)}")
    print(f"  others (NOT touched)   : {others}")
    print(f"  ours by state          : {dict(states)}")
    print(f"  ours by env-name prefix (top 25):")
    for p, c in prefixes.most_common(25):
        print(f"      {c:4d}  {p}")

    if not DELETE:
        print("\n[dry-run] no deletions. Re-run with DELETE=1 to remove the OURS set.")
        return

    print(f"\n[delete] removing {len(ours)} sandboxes (conc={CONC})...")
    sem = asyncio.Semaphore(CONC)
    ok = fail = 0

    async def worker(sb):
        nonlocal ok, fail
        async with sem:
            if await _delete(d, sb):
                ok += 1
            else:
                fail += 1

    await asyncio.gather(*(worker(sb) for sb in ours))
    print(f"[delete] done: ok={ok} fail={fail}")


if __name__ == "__main__":
    asyncio.run(main())
