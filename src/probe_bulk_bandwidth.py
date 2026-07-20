"""Bulk-download bandwidth probe for NAIP COGs on Azure blob.

Companion to :mod:`src.probe_naip_concurrency`: that probe found windowed
COG reads plateau at ~20 crops/s regardless of worker count (latency-bound,
no server throttling), so the open question is how fast this box can pull a
*whole DOQQ* in one sustained stream. If bulk MB/s is high, the winning
architecture is: download each dense tile once, crop locally.

Downloads ~150 MB of one urban DOQQ two ways — (a) a single sustained GET,
(b) 4 concurrent range streams — and reports effective crops/s at plausible
crops-per-tile densities.

Run::

    python src/probe_bulk_bandwidth.py

Read-only against the network; writes nothing to disk.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import planetary_computer
import pystac_client
import requests

CAP = 150 * 1024 * 1024  # bytes sampled per test


def main():
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1")
    # Dense urban area (Chicago Loop) — representative test-city DOQQ.
    items = list(client.search(collections=["naip"],
                               bbox=[-87.66, 41.86, -87.62, 41.89],
                               max_items=10).items())
    item = max(items, key=lambda it: it.datetime.year)
    href = planetary_computer.sign(
        (item.assets.get("image") or item.assets["visual"]).href)

    head = requests.head(href, timeout=30)
    total = int(head.headers.get("Content-Length", 0))
    print(f"item: {item.id} ({item.datetime.year})  "
          f"full tile: {total/1e6:.0f} MB")

    # (a) single sustained stream
    n, t0 = 0, time.perf_counter()
    with requests.get(href, stream=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(4 * 1024 * 1024):
            n += len(chunk)
            if n >= CAP:
                break
    dt = time.perf_counter() - t0
    print(f"single stream : {n/1e6:6.0f} MB in {dt:5.1f}s = "
          f"{n/1e6/dt:6.1f} MB/s ({n*8/1e6/dt:5.0f} Mbps)")

    # (b) concurrent range streams over the same span, sweeping stream count
    def pull(rng):
        lo, hi = rng
        got = 0
        with requests.get(href, stream=True, timeout=60,
                          headers={"Range": f"bytes={lo}-{hi}"}) as r:
            r.raise_for_status()
            for chunk in r.iter_content(4 * 1024 * 1024):
                got += len(chunk)
        return got

    span = min(CAP, total)
    mb_s = 0.0
    for nstreams in (4, 8, 16, 32):
        q = span // nstreams
        ranges = [(i * q, (i + 1) * q - 1) for i in range(nstreams)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nstreams) as pool:
            n = sum(pool.map(pull, ranges))
        dt = time.perf_counter() - t0
        mb_s = max(mb_s, n / 1e6 / dt)
        print(f"{nstreams:2d}-way ranges : {n/1e6:6.0f} MB in {dt:5.1f}s = "
              f"{n/1e6/dt:6.1f} MB/s ({n*8/1e6/dt:5.0f} Mbps)")

    print(f"\nAt {mb_s:.0f} MB/s, one {total/1e6:.0f} MB tile downloads in "
          f"{total/1e6/mb_s:.0f}s.")
    for crops in (1000, 5000, 20000):
        rate = crops / (total / 1e6 / mb_s)
        print(f"  if the tile serves {crops:>6,} crops -> effective "
              f"{rate:7,.0f} crops/s (vs ~20/s windowed)")


if __name__ == "__main__":
    main()
