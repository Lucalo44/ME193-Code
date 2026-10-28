#!/usr/bin/env python3
"""
One static figure showing how each trained model decides, for write-ups.

    python tools/plot_models.py              # -> media/models.png

Left:  Model 1 -- the fitted bell curve per activity over the peak of its axis,
       the range each activity wins, and every training window's peak.
Right: Model 2 -- the 12 largest logistic-regression weights per activity.
Run after tools/train.py so it reflects the current models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from gaitid import config as C
from gaitid import data as D
from gaitid import features as F

NAME = {"still": "Standing still", "walk": "Walking", "run": "Running", "bike": "Biking"}
COLOR = {"still": "#2a78d6", "walk": "#008300", "run": "#e87ba4", "bike": "#eda100"}   # colorblind-checked set
MARKER = {"still": "o", "walk": "s", "run": "^", "bike": "D"}
INK, MUTED = "#0f1417", "#737c85"
FEAT = {"amag_mean": "Mean |a|", "amag_std": "|a| variability", "amag_max": "Max |a|", "amag_min": "Min |a|",
        "amag_rms_dyn": "|a| RMS (motion)", "amag_freq": "|a| rhythm (Hz)", "amag_freq_share": "|a| rhythm regularity",
        "gmag_mean": "Mean |ω|", "gmag_std": "|ω| variability", "gmag_max": "Max |ω|",
        "gmag_freq": "|ω| rhythm (Hz)", "gmag_freq_share": "|ω| rhythm regularity"}
STAT = {"mean": "mean", "std": "variability", "peak": "peak"}


def feat_name(f):
    if f in FEAT:
        return FEAT[f]
    ch, st = f.split("_")
    return f"{ch} {STAT[st]}" + (" (tilt)" if st == "mean" and ch[0] == "a" else "")


def plot_model1(ax_c, ax_d, m1, X, y):
    axis = m1["axis"]
    theta, var, prior = (np.array(m1[k]).ravel() for k in ("theta", "var", "prior"))
    xmax = np.log1p(4.0)
    grid = np.linspace(0, xmax, 600)
    ll = -0.5 * (np.log(2 * np.pi * var) + (grid[:, None] - theta) ** 2 / var) + np.log(prior)
    pdf, win = np.exp(ll), ll.argmax(1)

    # shade the range each activity wins, label it, mark the thresholds
    edges = [0] + [i for i in range(1, len(grid)) if win[i] != win[i - 1]] + [len(grid) - 1]
    for a, b in zip(edges[:-1], edges[1:]):
        c = C.CLASSES[win[a]]
        for ax in (ax_c, ax_d):
            ax.axvspan(grid[a], grid[b], color=COLOR[c], alpha=0.07, lw=0)
        ax_c.text((grid[a] + grid[b]) / 2, 1.02, NAME[c], transform=ax_c.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=9.5, weight="bold", color=INK)
        if a:
            for ax in (ax_c, ax_d):
                ax.axvline(grid[a], color=MUTED, lw=0.8, ls=(0, (2, 3)))
            ax_c.text(grid[a] + 0.01, 0.97, f"{np.expm1(grid[a]):.2f} g", transform=ax_c.get_xaxis_transform(),
                      va="top", fontsize=8, color=MUTED, family="monospace")

    # curves; the very narrow "still" curve is capped so the others stay readable
    tops = np.sort(pdf.max(0))[::-1]
    ycap = tops[1] * 1.35 if tops[0] > 3 * tops[1] else tops[0] * 1.05
    for k, c in enumerate(C.CLASSES):
        ax_c.fill_between(grid, pdf[:, k], color=COLOR[c], alpha=0.15, lw=0)
        ax_c.plot(grid, pdf[:, k], color=COLOR[c], lw=2)
    ax_c.set_ylim(0, ycap)
    ax_c.set_yticks([])
    ax_c.set_ylabel("fitted curve", color=MUTED, fontsize=9)

    # every training window's peak, one row per activity
    f1 = np.log1p(np.array([F.peak_feature(w, axis)[0] for w in X]))
    rng = np.random.default_rng(0)
    for k, c in enumerate(C.CLASSES):
        v = np.minimum(f1[y == k], xmax)
        ax_d.scatter(v, -k + rng.uniform(-0.25, 0.25, len(v)), s=12, marker=MARKER[c], color=COLOR[c],
                     edgecolor="white", linewidth=0.4, label=NAME[c])
    ax_d.set_yticks(-np.arange(4), [NAME[c] for c in C.CLASSES], fontsize=9)
    ax_d.set_ylim(-3.6, 0.6)
    ticks = [0, 0.1, 0.25, 0.5, 1, 2, 4]
    ax_d.set_xticks(np.log1p(ticks), [str(t) for t in ticks])
    ax_d.set_xlim(0, xmax)
    ax_d.set_xlabel(f"Peak of {axis} in a 2 s window (g, compressed scale)", fontsize=9.5)
    for ax in (ax_c, ax_d):
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)


def plot_model2(ax, m2):
    coef = np.array(m2["coef"])
    top = np.argsort(-np.abs(coef).max(0))[:12]
    w = coef[:, top].T
    lim = np.abs(w).max()
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    ax.imshow(w, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (r, k), v in np.ndenumerate(w):
        ax.text(k, r, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=8.5, family="monospace",
                color="white" if abs(v) > 0.62 * lim else INK)
    ax.set_xticks(range(4), [NAME[c].replace("Standing still", "Still") for c in C.CLASSES], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(top)), [feat_name(m2["features"][i]) for i in top], fontsize=9)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("blue = a high value argues for that activity, red = against", fontsize=8.5, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "media", "models.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    m1 = json.load(open(os.path.join(HERE, "models", "model1_peak.json")))
    m2 = json.load(open(os.path.join(HERE, "models", "model2_full.json")))
    X, y, _ = D.load_dir(args.data)

    plt.rcParams.update({"font.family": "sans-serif", "axes.titlesize": 12.5, "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(14, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], height_ratios=[2.1, 1.25], wspace=0.32, hspace=0.08,
                          left=0.08, right=0.98, top=0.83, bottom=0.11)
    ax_c, ax_d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_c.sharex(ax_d)
    ax_c.tick_params(labelbottom=False)
    plot_model1(ax_c, ax_d, m1, X, y)
    plot_model2(fig.add_subplot(gs[:, 1]), m2)
    fig.text(0.08, 0.95, f"Model 1 · one number, cut into four ranges ({m1['cv_accuracy']:.1%})",
             fontsize=13, weight="bold", color=INK)
    fig.text(0.08, 0.915, "Gaussian naive Bayes on the window's peak: whichever curve is tallest wins",
             fontsize=9.5, color=MUTED)
    fig.text(0.60, 0.95, f"Model 2 · 30 numbers, weighed together ({m2['cv_accuracy']:.1%})",
             fontsize=13, weight="bold", color=INK)
    fig.text(0.60, 0.915, "Logistic regression: the 12 largest weights per activity", fontsize=9.5, color=MUTED)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
