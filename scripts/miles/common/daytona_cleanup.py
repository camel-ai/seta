#!/usr/bin/env python3
"""List, and optionally delete, the Daytona sandboxes created by Harbor.

Harbor's Daytona backend labels every sandbox it creates with ``harbor_owner_uid``.
After an interrupted run such sandboxes can stay alive until Daytona's auto-stop
interval expires and count against the organization's quota. This tool lists the
labelled sandboxes (by state and by environment-name prefix) and deletes them only
with ``--delete``. Sandboxes without the label are never touched.

It sees every labelled sandbox of the organization, including those of other jobs
that are still running: delete only when no other training or evaluation job uses the
same Daytona organization.

    python scripts/miles/common/daytona_cleanup.py            # dry run: list only
    python scripts/miles/common/daytona_cleanup.py --delete   # delete them

Credentials come from DAYTONA_API_KEY (and DAYTONA_API_URL for a non-default
endpoint); nothing secret is printed. Needs the ``daytona`` SDK (installed with Harbor).
"""

import argparse
import asyncio
from collections import Counter

from daytona import AsyncDaytona

OWNER_LABEL = "harbor_owner_uid"


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


async def main(delete: bool, concurrency: int) -> None:
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

    print(f"sandboxes visible            : {len(sandboxes)}")
    print(f"  created by Harbor ({OWNER_LABEL}): {len(ours)}")
    print(f"  others (never touched)     : {others}")
    print(f"  Harbor sandboxes by state  : {dict(states)}")
    print("  Harbor sandboxes by environment-name prefix (top 25):")
    for p, c in prefixes.most_common(25):
        print(f"      {c:4d}  {p}")

    if not delete:
        print("\n[dry run] nothing deleted; re-run with --delete to remove the Harbor sandboxes.")
        return

    print(f"\n[delete] removing {len(ours)} sandboxes ({concurrency} at a time)...")
    sem = asyncio.Semaphore(concurrency)
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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--delete", action="store_true", help="delete the listed sandboxes (default: dry run)")
    parser.add_argument("--concurrency", type=int, default=16, help="parallel delete calls")
    cli = parser.parse_args()
    asyncio.run(main(cli.delete, cli.concurrency))
