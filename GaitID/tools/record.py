#!/usr/bin/env python3
"""
Record a labelled recording on the laptop while the board streams samples.

    python tools/record.py walk --seconds 60            # UDP from the board (default)
    python tools/record.py run --serial /dev/cu.usbmodem1101

Writes data/raw/<label>_<n>.csv (t, ax, ay, az, gx, gy, gz; g and deg/s).
Start the app on the board first (app/python/main.py, STREAM_TO = this laptop's IP).
Record several separate takes per activity -- cross-validation holds out whole
recordings, so more takes means a more honest accuracy number.
"""

from __future__ import annotations

import argparse
import glob
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from gaitid import config as C


def lines_udp(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", port))
    s.settimeout(5)
    while True:
        data, _ = s.recvfrom(65535)
        yield from data.decode().splitlines()


def lines_serial(dev):
    import serial
    with serial.Serial(dev, 115200, timeout=5) as ser:
        while True:
            yield ser.readline().decode(errors="ignore").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", choices=C.CLASSES)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--serial")
    ap.add_argument("--port", type=int, default=C.UDP_PORT)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "raw"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = len(glob.glob(os.path.join(args.out, f"{args.label}_*.csv"))) + 1
    path = os.path.join(args.out, f"{args.label}_{n}.csv")
    src = lines_serial(args.serial) if args.serial else lines_udp(args.port)

    print(f"Recording '{args.label}' for {args.seconds:.0f}s -> {path}")
    rows, t0 = [], None
    try:
        for line in src:
            p = line.split(",")
            if len(p) != 7:
                continue
            try:
                vals = [float(v) for v in p]
            except ValueError:
                continue
            t0 = vals[0] if t0 is None else t0
            vals[0] -= t0
            rows.append(vals)
            if vals[0] >= args.seconds:
                break
    except (KeyboardInterrupt, socket.timeout):
        print("Stopped early.")
    if len(rows) < C.WINDOW:
        raise SystemExit("Too little data; nothing saved.")
    with open(path, "w") as f:
        f.write(",".join(C.COLUMNS) + "\n")
        f.writelines(",".join(f"{v:.5f}" for v in r) + "\n" for r in rows)
    print(f"Saved {len(rows)} samples ({len(rows) / max(rows[-1][0], 1e-9):.1f} Hz)")


if __name__ == "__main__":
    main()
