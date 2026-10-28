#!/usr/bin/env python3
"""
Record a full demo run to video without the board: the stick figure stands,
walks, runs and bikes through all three groupmates' turns, hand-off screens
included. Uses the real tools/demo.py window, stepped at a fixed frame rate.

    python tools/record_demo.py                         # -> media/demo.mp4, 30 s per activity
    python tools/record_demo.py --goal 10 --out media/demo_short.mp4

Needs imageio-ffmpeg (in requirements.txt) for its bundled ffmpeg.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import matplotlib
matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from demo import STAGES, Demo, KeyDetector


class ScriptedDetector(KeyDetector):
    """The timeline below sets .label directly; no keyboard, no sensor."""

    def hint(self):
        return "Recorded demo run: activities scripted, no sensor attached"


def scenes(goal):
    """(seconds, activity being done, press Space afterwards) in order."""
    out = [(3.0, "still", False)]                       # first groupmate, before they start
    for i, act in enumerate(STAGES):
        if i:
            out.append((2.5, "still", False))           # next groupmate straps the sensor on
        out.append((goal + 0.3, act, False))            # the activity until the goal is reached
        out.append((5.0, "still", i < len(STAGES) - 1))  # "Wow, that was cool!" card, then Space
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", type=float, default=30, help="seconds of each activity, as in demo.py")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=100, help="the window is 8 in square, so 100 -> 800x800")
    ap.add_argument("--out", default=os.path.join(HERE, "media", "demo.mp4"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    det = ScriptedDetector()
    demo = Demo(det, args.goal)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "22"])
    plan = scenes(args.goal)
    total = sum(s for s, _, _ in plan)
    print(f"Recording {total:.0f} s at {args.fps} fps -> {args.out}")
    with writer.saving(demo.fig, args.out, dpi=args.dpi):
        for secs, label, space in plan:
            det.label = label
            for _ in range(round(secs * args.fps)):
                demo.step(1 / args.fps)
                writer.grab_frame()
            if space:
                demo.advance()
    print("Done.")


if __name__ == "__main__":
    main()
