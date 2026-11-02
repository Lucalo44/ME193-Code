#!/usr/bin/env python3
"""
Record one labelled take from both ankle sensors at once.

    python stepcount/tools/record_steps.py walk --seconds 60
    python stepcount/tools/record_steps.py run --steps 100          # stop after counting 100 steps
    python stepcount/tools/record_steps.py still --seconds 30 --no-lego

Before running: the UNO Q app in record mode (STREAM_TO = this laptop's IP),
the MPU-6050 on your RIGHT ankle, and the Double Motor switched on and strapped
to your LEFT ankle. Count your steps out loud while recording; you'll be asked
for the count at the end (0 for still / bike). Writes
stepcount/data/raw/<activity>_<n>/{mpu.csv, lego.csv, meta.json}.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

from stepcount import fuse as F
from stepcount.sources import make_streams


def wait_for(streams, timeout):
    t0 = time.monotonic()
    width = shutil.get_terminal_size().columns - 1          # never wrap, or \r can't overwrite the line
    while time.monotonic() - t0 < timeout:
        line = "  ".join(f"{n}: {s.status}" for n, s in streams.items())
        if "mpu" in streams and not streams["mpu"].live() and time.monotonic() - t0 > 8:
            line += "  (no MPU data: board on, app running, STREAM_TO = this laptop?)"
        print(f"\r{line[:width]:<{width}}", end="", flush=True)
        if all(s.live() for s in streams.values()):
            print()
            return []
        time.sleep(0.25)
    print()
    return [n for n, s in streams.items() if not s.live()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("activity", help="walk, run, stairs, still, bike, mixed ... (any word, used in the folder name)")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--steps", type=int, help="known step count (skips the question at the end)")
    ap.add_argument("--no-mpu", action="store_true")
    ap.add_argument("--no-lego", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "stepcount", "data", "raw"))
    args = ap.parse_args()

    streams = make_streams(use_mpu=not args.no_mpu, use_lego=not args.no_lego)
    if not streams:
        raise SystemExit("Nothing to record: both sensors switched off.")
    print("Connecting sensors (Ctrl-C to quit)...")
    missing = wait_for(streams, 30)
    if missing:
        print(f"No data from: {', '.join(missing)}.")
        if input("Record without them? [y/N] ").strip().lower() != "y":
            raise SystemExit(1)
        for n in missing:
            streams[n].stop()
            del streams[n]

    for k in (3, 2, 1):
        print(f"\rStarting in {k}...", end="", flush=True)
        time.sleep(1)
    t0 = time.monotonic()
    print(f"\rRecording '{args.activity}' for {args.seconds:.0f} s. Go! (Ctrl-C stops early)")
    try:
        while time.monotonic() - t0 < args.seconds:
            print(f"\r  {time.monotonic() - t0:5.1f} s", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    t1 = time.monotonic()
    print("\nStopped.")
    data = {}
    for n, s in streams.items():
        t, v = s.snapshot(t0)
        keep = t <= t1
        data[n] = (t[keep] - t0, v[keep])
        s.stop()
        rate = len(t[keep]) / max(t1 - t0, 1e-9)
        print(f"  {n}: {keep.sum()} samples ({rate:.1f} Hz)")

    steps = args.steps
    while steps is None:
        try:
            steps = int(input("How many steps did you take? (0 for still / bike) "))
        except ValueError:
            pass

    os.makedirs(args.out, exist_ok=True)
    n = len(glob.glob(os.path.join(args.out, f"{args.activity}_*"))) + 1
    folder = os.path.join(args.out, f"{args.activity}_{n}")
    meta = {"activity": args.activity, "steps": steps, "seconds": round(t1 - t0, 2),
            "sensors": sorted(data), "placement": {"mpu": "right ankle", "lego": "left ankle"}}
    F.save_take(folder, meta, mpu=data.get("mpu"), lego=data.get("lego"))
    print(f"Saved {folder}")


if __name__ == "__main__":
    main()
