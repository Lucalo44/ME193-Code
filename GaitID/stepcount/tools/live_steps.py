#!/usr/bin/env python3
"""
Count steps live from whichever ankle sensors are streaming.

    python stepcount/tools/live_steps.py
    python stepcount/tools/live_steps.py --no-lego        # MPU only

Every 0.25 s the last 10 s of both sensors are put on the 50 Hz grid and run
through StepNet. Density is only committed once a sample is LIVE_MARGIN_S old,
when the network can see enough after it, and each grid sample is committed
exactly once -- so the running total is a plain sum with no double counting.
When the sensors go quiet (or on Ctrl-C) the last LIVE_MARGIN_S is flushed too.
A sensor that goes quiet for over a second is treated as absent; the network
was trained with sensor dropout, so it keeps counting on the other one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from stepcount import config as C
from stepcount import fuse as F
from stepcount import net as N
from stepcount.sources import make_streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "stepcount", "models", "stepnet.pt"))
    ap.add_argument("--no-mpu", action="store_true")
    ap.add_argument("--no-lego", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Train one with stepcount/tools/train_steps.py first.")
    model, ck = N.load(args.model)
    norm = ck["norm"]
    streams = make_streams(use_mpu=not args.no_mpu, use_lego=not args.no_lego)
    print("Counting steps. Ctrl-C to stop.")

    committed_k = None          # last grid index (k / FS seconds) already added to the total
    last = None                 # (grid indices, density) from the newest pass, for the final flush
    total = 0.0
    margin_k = int(C.LIVE_MARGIN_S * C.FS)

    def flush():
        """Commit everything up to the newest sample (the stream stopped, so nothing more is coming)."""
        nonlocal total, committed_k, last
        if last is not None and committed_k is not None:
            ks, dens = last
            total += float(dens[ks > committed_k].sum())
            committed_k = max(committed_k, int(ks[-1]))
        last = None

    try:
        while True:
            time.sleep(0.25)
            live = {n: s for n, s in streams.items() if s.live()}
            status = "  ".join(f"{n} {'on' if n in live else 'off'}" for n in streams)
            if not live:
                flush()
                print(f"\rSteps: {total:6.0f}   waiting for sensors ({status})   ", end="", flush=True)
                continue
            newest_k = int(np.floor(min(s.last_time() for s in live.values()) * C.FS))
            ks = np.arange(newest_k - 10 * C.FS, newest_k + 1)
            grid = ks / C.FS
            data = {n: s.snapshot(grid[0] - 0.5) for n, s in live.items()}
            dens = N.predict_density(model, F.normalize(F.build(F.on_grid(data, grid)), norm))
            upto = newest_k - margin_k
            start = upto - 5 * C.FS if committed_k is None else committed_k   # first pass: start 5 s back
            total += float(dens[(ks > start) & (ks <= upto)].sum())
            committed_k = max(upto, start)
            last = (ks, dens)
            print(f"\rSteps: {total:6.0f}   ({status})   ", end="", flush=True)
    except KeyboardInterrupt:
        flush()
        print(f"\nTotal: {total:.0f} steps")
    finally:
        for s in streams.values():
            s.stop()


if __name__ == "__main__":
    main()
