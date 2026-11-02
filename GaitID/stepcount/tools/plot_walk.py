#!/usr/bin/env python3
"""
Plot a walk the UNO Q logged on its own: the running step count, cadence, and a
zoomed 10 s window showing how StepNet's density turns gyro swings into steps.

    python stepcount/tools/plot_walk.py session_0007        # prefix of a pulled session is enough
    python stepcount/tools/plot_walk.py session_0007 --zoom 300 310

Run stepcount/tools/pull_logs.py first. Writes media/<session>.png.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stepcount import config as C
from stepcount import fuse as F
from stepcount import net as N

LOGS = os.path.join(HERE, "stepcount", "data", "board_logs")
INK, MUTED, GRID = "#0f1417", "#737c85", "#e1e0d9"
ACCENT, SIGNAL, ZOOM = "#2a78d6", "#9aa3ab", "#eda100"


def load(prefix):
    match = sorted(glob.glob(os.path.join(LOGS, prefix + "*.csv")))
    if len(match) != 1:
        raise SystemExit(f"'{prefix}' matches {len(match)} sessions in {LOGS}. Run pull_logs.py first?")
    name = os.path.basename(match[0])[:-4]
    data = np.genfromtxt(match[0], delimiter=",", skip_header=1, invalid_raise=False)
    data = data[~np.isnan(data).any(axis=1)]
    summary = {}
    if os.path.exists(match[0][:-4] + ".json"):
        with open(match[0][:-4] + ".json") as f:
            summary = json.load(f)
    return name, data[:, 0] - data[0, 0], data[:, 1:], summary


def steady_window(cadence, t, seconds=10):
    """Start of the 10 s stretch whose cadence is closest to the walk's typical cadence."""
    walking = cadence > 40
    if not walking.any():
        return t[len(t) // 2]
    target = np.median(cadence[walking])
    n = int(seconds * C.FS)
    score = np.convolve(np.abs(cadence - target) + 1000 * (~walking), np.ones(n) / n, mode="valid")
    return t[int(np.argmin(score))]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--model", default=os.path.join(HERE, "stepcount", "models", "stepnet.pt"))
    ap.add_argument("--zoom", type=float, nargs=2, metavar=("FROM", "TO"), help="seconds for the zoomed window")
    ap.add_argument("--out")
    args = ap.parse_args()

    name, t, v, summary = load(args.session)
    model, ck = N.load(args.model)
    take = {"name": name, "meta": {"steps": 0}, "mpu": (t, v), "lego": None}
    grid = F.take_grid(take)
    streams = F.on_grid(take, grid)
    dens = N.predict_density(model, F.normalize(F.build(streams), ck["norm"]))      # PyTorch, MPU only
    tg = grid - grid[0]
    total = np.cumsum(dens)
    win = int(30 * C.FS)
    cadence = np.convolve(dens, np.ones(win), mode="same") * (60 / 30)              # steps/min over 30 s
    gyro = np.linalg.norm(streams["mpu"][:, 3:6], axis=1)
    # still = the shank barely rotates (|gyro| under 40 deg/s for 3 s): steps counted there are the
    # network's floor, not real steps -- a model trained without still/bike takes never learns zero
    from scipy.ndimage import maximum_filter1d
    still = maximum_filter1d(np.nan_to_num(gyro), int(3 * C.FS)) < 40
    phantom, moving = dens[still].sum(), dens[~still].sum()
    z0, z1 = args.zoom if args.zoom else (lambda a: (a, a + 10))(steady_window(cadence, tg))
    minutes = tg[-1] / 60
    board = summary.get("steps_on_board")

    plt.rcParams.update({"font.family": "sans-serif"})
    fig = plt.figure(figsize=(13, 10.5), facecolor="white")
    gs = fig.add_gridspec(4, 1, height_ratios=[1.5, 1.0, 1.0, 1.0], hspace=0.55,
                          left=0.07, right=0.97, top=0.87, bottom=0.06)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3], sharex=ax3)

    # 1. running total
    ax1.plot(tg / 60, total, color=ACCENT, lw=2)
    ax1.axvspan(z0 / 60, z1 / 60, color=ZOOM, alpha=0.25, lw=0)
    ax1.annotate(f"{total[-1]:.0f} steps", (tg[-1] / 60, total[-1]), xytext=(-8, 6), textcoords="offset points",
                 ha="right", fontsize=11, weight="bold", color=INK)
    if board is not None:
        ax1.plot([tg[-1] / 60], [board], "o", ms=7, color="white", mec=INK, mew=1.5, zorder=5)
        ax1.annotate(f"board's live count: {board:.0f}", (tg[-1] / 60, board), xytext=(-8, -16),
                     textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    ax1.set_title(f"Running step total (sum of StepNet's density) · gray = standing still, "
                  f"where {phantom:.0f} steps were counted that didn't happen", loc="left", fontsize=11, color=INK)
    ax1.set_ylabel("steps", color=MUTED, fontsize=9)
    ax1.set_xlim(0, minutes)
    ax1.set_ylim(0, None)

    # 2. cadence
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        for ax in (ax1, ax2):
            ax.axvspan(tg[a] / 60, tg[min(b, len(tg) - 1)] / 60, color=SIGNAL, alpha=0.18, lw=0)
    ax2.fill_between(tg / 60, cadence, color=ACCENT, alpha=0.15, lw=0)
    ax2.plot(tg / 60, cadence, color=ACCENT, lw=1.5)
    ax2.axvspan(z0 / 60, z1 / 60, color=ZOOM, alpha=0.25, lw=0)
    walking = cadence > 40
    ax2.set_title(f"Cadence, 30 s average (median while walking: {np.median(cadence[walking]) if walking.any() else 0:.0f} steps/min)",
                  loc="left", fontsize=11, color=INK)
    ax2.set_ylabel("steps / min", color=MUTED, fontsize=9)
    ax2.set_xlabel("minutes into the walk", color=MUTED, fontsize=9)
    ax2.set_xlim(0, minutes)
    ax2.set_ylim(0, None)

    # 3 + 4. zoom: what the network sees, and what it outputs
    m = (tg >= z0) & (tg <= z1)
    ax3.plot(tg[m], gyro[m], color=SIGNAL, lw=1.3)
    ax3.set_title(f"Zoom {z0:.0f}-{z1:.0f} s: input, right-ankle rotation rate |ω|", loc="left", fontsize=11, color=INK)
    ax3.set_ylabel("deg/s", color=MUTED, fontsize=9)
    ax4.fill_between(tg[m], dens[m], color=ACCENT, alpha=0.25, lw=0)
    ax4.plot(tg[m], dens[m], color=ACCENT, lw=1.5)
    ax4.set_title(f"Output, step density: each bump's area is about one step ({dens[m].sum():.1f} steps in this window)",
                  loc="left", fontsize=11, color=INK)
    ax4.set_ylabel("steps per sample", color=MUTED, fontsize=9)
    ax4.set_xlabel("seconds into the walk", color=MUTED, fontsize=9)
    ax4.set_xlim(z0, z1)
    ax4.set_ylim(0, None)
    for ax in (ax1, ax2, ax3, ax4):
        style(ax)

    fig.text(0.07, 0.955, f"Walk to class: about {moving:.0f} steps ({total[-1]:.0f} counted, {phantom:.0f} of them "
             f"while standing still) in {minutes:.1f} minutes", fontsize=15, weight="bold", color=INK)
    fig.text(0.07, 0.925, f"{name} · logged on the UNO Q (MPU-6050, right ankle, {C.FS} Hz) · counted by StepNet "
             f"({os.path.basename(args.model)}) with the Double Motor absent", fontsize=9.5, color=MUTED)
    out = args.out or os.path.join(HERE, "media", f"{name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"PyTorch total {total[-1]:.1f} steps ({moving:.1f} moving, {phantom:.1f} while still), board {board}; saved {out}")


if __name__ == "__main__":
    main()
