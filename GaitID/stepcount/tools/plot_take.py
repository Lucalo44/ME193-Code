#!/usr/bin/env python3
"""
Check one take: both sensors' |a| and |gyro|, the steps the labeller chose, and (if a
model exists) StepNet's density. Use it whenever a take's labels look off.

    python stepcount/tools/plot_take.py stepcount/data/raw/walk_1
    python stepcount/tools/plot_take.py stepcount/data/raw/walk_1 --seconds 10 20

Writes <take>/check.png.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stepcount import fuse as F
from stepcount import net as N
from stepcount import training as T

COLOR = {"mpu": "#2a78d6", "lego": "#eb6834"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("--seconds", type=float, nargs=2, metavar=("FROM", "TO"))
    ap.add_argument("--model", default=os.path.join(HERE, "stepcount", "models", "stepnet.pt"))
    args = ap.parse_args()

    item = T.prepare(F.load_take(args.take))
    t = np.arange(len(item["y"])) / 50
    sel = np.ones(len(t), bool) if not args.seconds else (t >= args.seconds[0]) & (t <= args.seconds[1])
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for s, x in item["streams"].items():
        if not np.isnan(x).all():
            axes[0].plot(t[sel], np.linalg.norm(x[sel, 0:3], axis=1), color=COLOR[s], lw=1, label=f"{s} |a| (g)")
    axes[0].legend(loc="upper right")
    top = np.zeros(len(t))
    for s, x in item["streams"].items():
        if not np.isnan(x).all():
            g = np.linalg.norm(x[:, 3:6], axis=1)
            axes[1].plot(t[sel], g[sel], color=COLOR[s], lw=1, label=f"{s} |gyro| (deg/s)")
            top = np.fmax(top, np.nan_to_num(g))
    st = item["step_idx"][sel[item["step_idx"]]] if len(item["step_idx"]) else []
    axes[1].plot(t[st], top[st], "v", color="#1baf7a", ms=8, label=f"labelled steps ({len(item['step_idx'])} total)")
    axes[1].legend(loc="upper right")
    axes[2].plot(t[sel], item["y"][sel], color="#1baf7a", lw=1.5, label="target density")
    title = f"{item['name']}: {item['steps']} counted, labels by {item['label_info']['method']}"
    if os.path.exists(args.model):
        model, ck = N.load(args.model)
        dens = N.predict_density(model, F.normalize(F.build(item["streams"]), ck["norm"]))
        axes[2].plot(t[sel], dens[sel], color="#6250d6", lw=1.5, label="StepNet density")
        title += f", StepNet counts {dens.sum():.1f}"
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(args.take, "check.png")
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
