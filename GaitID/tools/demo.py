#!/usr/bin/env python3
"""
Group demo: a window with a stick figure that copies the detected activity.

    python tools/demo.py                          # live, Model 2 on the board's stream
    python tools/demo.py --model model1_peak.json
    python tools/demo.py --sim                    # no board: keys S / W / R / B fake the activity

Each groupmate gets one activity in turn (walk, run, bike). Once the model has
detected it for --goal seconds in total, the window asks them to say "Wow, that
was cool!" and hand the device on; Space starts the next activity.

The model runs here on the laptop, so the board must be in record mode
(STREAM_TO = this laptop's IP in app/python/main.py). tools/record.py listens on
the same port, so don't run both at once.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from gaitid import config as C
from gaitid.model import Model

STAGES = ["walk", "run", "bike"]
STATUS = {"still": "You are standing still", "walk": "You are walking",
          "run": "You are running", "bike": "You are biking"}
VERB = {"walk": "walking", "run": "running", "bike": "biking"}

# Animation: seconds per stride / pedal turn, and how fast the ground scrolls by
PERIOD = {"walk": 1.1, "run": 0.62, "bike": 0.75}
SPEED = {"still": 0.0, "walk": 0.9, "run": 2.4, "bike": 2.6}
# Gait poses: thigh swing, knee bend, arm swing, elbow bend, forward lean (radians)
GAIT = {"still": (0.12, 0.0, 0.2, 0.1, 0.0),       # drawn frozen at phase pi/2: a relaxed stance
        "walk": (0.42, 0.6, 0.4, 0.3, 0.03),
        "run": (0.75, 1.4, 0.7, 1.4, 0.18)}

# Body (ground at y = 0, figure centred on x = 0)
THIGH, SHIN, TORSO, UPPER, FORE, HEAD = 0.48, 0.46, 0.68, 0.32, 0.30, 0.13
# Bike
REAR, FRONT, CRANK = np.array([-0.55, 0.32]), np.array([0.62, 0.32]), np.array([0.02, 0.36])
SEAT, STEM, BAR = np.array([-0.22, 1.0]), np.array([0.40, 0.95]), np.array([0.45, 1.18])
WHEEL_R, CRANK_R = 0.32, 0.17

INK, FAR_INK, BIKE_INK = "#222222", "#9a9a9a", "#3a7bd5"
DASH_GAP = 0.5


# ---------------- geometry ----------------
def seg(p, angle, length):
    """End of a limb leaving p at `angle` from straight down (positive = forward)."""
    return p + length * np.array([np.sin(angle), -np.cos(angle)])


def ik(a, b, l1, l2, bend):
    """Middle joint of a two-part limb reaching from a to b. bend=+1 rotates it
    counter-clockwise off the a->b line (knees forward), -1 clockwise (elbows down)."""
    d = b - a
    dist = np.clip(np.hypot(*d), abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6)
    alpha = np.arccos((l1 ** 2 + dist ** 2 - l2 ** 2) / (2 * l1 * dist))
    theta = np.arctan2(d[1], d[0]) + bend * alpha
    return a + l1 * np.array([np.cos(theta), np.sin(theta)])


def gait_pose(phase, swing, knee, arm, elbow, lean):
    """Standing / walking / running, with the lowest foot on the ground.
    Returns neck, legs [[hip, knee, foot]] x2, arms [[shoulder, elbow, hand]] x2."""
    hip = np.zeros(2)
    neck = hip + TORSO * np.array([np.sin(lean), np.cos(lean)])
    s = np.sin(phase)
    legs, arms = [], []
    for side in (1, -1):
        a = side * swing * s
        k = knee * (0.15 + max(0.0, side * np.cos(phase)))   # bend the leg swinging forward
        kn = seg(hip, a, THIGH)
        legs.append([hip, kn, seg(kn, a - k, SHIN)])
        sa = -side * arm * s
        el = seg(neck, sa, UPPER)
        arms.append([neck, el, seg(el, sa + elbow, FORE)])
    lift = np.array([0.0, -min(legs[0][2][1], legs[1][2][1])])
    return neck + lift, [[p + lift for p in l] for l in legs], [[p + lift for p in a] for a in arms]


def bike_pose(phase):
    """Rider on the bike: feet on the pedals, hands on the bar."""
    hip = SEAT + [0.0, 0.04]
    neck = hip + TORSO * np.array([np.sin(0.55), np.cos(0.55)])
    legs = []
    for off in (0.0, np.pi):
        pedal = CRANK + CRANK_R * np.array([np.cos(off - phase), np.sin(off - phase)])
        legs.append([hip, ik(hip, pedal, THIGH, SHIN, 1), pedal])
    arm = [neck, ik(neck, BAR, UPPER, FORE, -1), BAR]
    return neck, legs, [arm, arm]


def polyline(parts):
    """Several point lists -> one x, y pair with NaN breaks, for a single Line2D."""
    pts = []
    for part in parts:
        pts += [np.asarray(p, dtype=float) for p in part] + [np.full(2, np.nan)]
    xy = np.array(pts)
    return xy[:, 0], xy[:, 1]


# ---------------- activity sources ----------------
class StreamDetector:
    """Runs the model on the board's UDP stream (the packets tools/record.py reads).
    The label only changes once `consistent` predictions in a row agree."""

    def __init__(self, model, port, consistent):
        self.model = model
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", port))
        self.sock.setblocking(False)
        self.buf = deque(maxlen=C.WINDOW)
        self.recent = deque(maxlen=consistent)
        self.since = 0
        self.label = "still"
        self.last_data = None

    def poll(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
            except BlockingIOError:
                return self.label
            for line in data.decode().splitlines():
                p = line.split(",")
                if len(p) != 7:
                    continue
                try:
                    vals = [float(v) for v in p[1:]]
                except ValueError:
                    continue
                self.last_data = time.monotonic()
                self.buf.append(vals)
                self.since += 1
                if len(self.buf) == C.WINDOW and self.since >= C.HOP:
                    self.since = 0
                    self.recent.append(self.model.predict(np.array(self.buf)))
                    if len(self.recent) == self.recent.maxlen and len(set(self.recent)) == 1:
                        self.label = self.recent[0]

    def key(self, k):
        pass

    def hint(self):
        if self.last_data is None:
            return f"Waiting for sensor data on UDP port {self.sock.getsockname()[1]}..."
        if time.monotonic() - self.last_data > 3:
            return "Sensor data stopped - check the board"
        return ""


class KeyDetector:
    """--sim: the keyboard stands in for the model."""
    KEYS = {"s": "still", "w": "walk", "r": "run", "b": "bike"}

    def __init__(self):
        self.label = "still"

    def poll(self):
        return self.label

    def key(self, k):
        self.label = self.KEYS.get(k, self.label)

    def hint(self):
        return "Simulation: press S / W / R / B to fake standing / walking / running / biking"


# ---------------- window ----------------
class Demo:
    def __init__(self, detector, goal):
        self.det, self.goal = detector, goal
        self.stage, self.progress, self.mode = 0, 0.0, "active"   # mode: active | handoff | done
        self.phase, self.scroll, self.t_last = 0.0, 0.0, time.monotonic()

        for k in plt.rcParams:                     # free S / R / B etc. from matplotlib's shortcuts
            if k.startswith("keymap.") and k != "keymap.quit":
                plt.rcParams[k] = []
        plt.rcParams["toolbar"] = "None"
        self.fig = fig = plt.figure(figsize=(8, 8), facecolor="white")
        fig.canvas.manager.set_window_title("GaitID")
        fig.canvas.mpl_connect("key_press_event", self.on_key)

        ax = fig.add_axes([0, 0.32, 1, 0.58])
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-0.15, 2.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.plot([-1.6, 1.6], [0, 0], color=INK, lw=2)
        self.dashes, = ax.plot([], [], color=FAR_INK, lw=2)
        self.wheels = [ax.add_patch(Circle(c, WHEEL_R, fill=False, ec=BIKE_INK, lw=3)) for c in (REAR, FRONT)]
        self.frame, = ax.plot(*polyline([[REAR, CRANK, SEAT, REAR], [SEAT, STEM, CRANK], [STEM, FRONT], [STEM, BAR]]),
                              color=BIKE_INK, lw=4, solid_capstyle="round")
        self.spokes, = ax.plot([], [], color=BIKE_INK, lw=1.5)
        self.far, = ax.plot([], [], color=FAR_INK, lw=7, solid_capstyle="round", solid_joinstyle="round")
        self.crank, = ax.plot([], [], color=BIKE_INK, lw=4, solid_capstyle="round")
        self.near, = ax.plot([], [], color=INK, lw=7, solid_capstyle="round", solid_joinstyle="round")
        self.head = ax.add_patch(Circle((0, 0), HEAD, fc="white", ec=INK, lw=6))
        self.bike_parts = [*self.wheels, self.frame, self.spokes, self.crank]

        self.title = fig.text(0.5, 0.94, "", ha="center", fontsize=16, color=FAR_INK)
        self.status = fig.text(0.5, 0.24, "", ha="center", fontsize=30, weight="bold", color=INK)
        self.goal_text = fig.text(0.5, 0.17, "", ha="center", fontsize=17, color=INK)
        bar = fig.add_axes([0.25, 0.11, 0.5, 0.03])
        bar.set_xlim(0, 1)
        bar.set_ylim(0, 1)
        bar.axis("off")
        bar.add_patch(Rectangle((0, 0), 1, 1, fc="#e6e6e6"))
        self.fill = bar.add_patch(Rectangle((0, 0), 0, 1, fc="#2e9d5b"))
        self.hint = fig.text(0.5, 0.04, "", ha="center", fontsize=11, color=FAR_INK)

        # Hand-off screen, drawn over everything
        self.overlay = [fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                                                 fc="white", alpha=0.95, zorder=10)),
                        fig.text(0.5, 0.58, 'Say "Wow, that was cool!"', ha="center",
                                 fontsize=34, weight="bold", color="#2e9d5b", zorder=11)]
        self.handoff_text = fig.text(0.5, 0.45, "", ha="center", va="top", fontsize=19,
                                     color=INK, zorder=11, linespacing=1.6)
        self.overlay.append(self.handoff_text)

    def run(self):
        self.anim = FuncAnimation(self.fig, self.tick, interval=33, cache_frame_data=False)
        plt.show()

    def on_key(self, event):
        if event.key in (" ", "enter") and self.mode != "active":
            self.advance()
        else:
            self.det.key(event.key)

    def advance(self):
        """Space on the hand-off screen: next groupmate (or start over after the last)."""
        self.stage = self.stage + 1 if self.mode == "handoff" else 0
        self.progress, self.mode = 0.0, "active"

    def tick(self, _):
        now = time.monotonic()
        dt = min(now - self.t_last, 0.1)
        self.t_last = now
        self.step(dt)

    def step(self, dt):
        """Advance the demo by dt seconds and redraw (tools/record_demo.py calls this directly)."""
        label = self.det.poll()
        if self.mode == "active" and label == STAGES[self.stage]:
            self.progress = min(self.progress + dt, self.goal)
            if self.progress >= self.goal:
                self.mode = "handoff" if self.stage < len(STAGES) - 1 else "done"
        self.draw_figure(label, dt)
        self.draw_text(label)

    def draw_figure(self, label, dt):
        if label in PERIOD:
            self.phase = (self.phase + 2 * np.pi * dt / PERIOD[label]) % (2 * np.pi)
        self.scroll = (self.scroll + SPEED[label] * dt) % DASH_GAP
        xs = np.arange(-1.6, 1.6 + DASH_GAP, DASH_GAP) - self.scroll
        self.dashes.set_data(*polyline([[(x, -0.07), (x + 0.15, -0.07)] for x in xs]))

        bike = label == "bike"
        phase = self.phase if label in PERIOD else np.pi / 2
        neck, legs, arms = bike_pose(phase) if bike else gait_pose(phase, *GAIT[label])
        hip = legs[0][0]
        up = (neck - hip) / np.linalg.norm(neck - hip)
        self.head.center = tuple(neck + (HEAD + 0.04) * up)
        self.near.set_data(*polyline([[hip, neck], legs[0], arms[0]]))
        self.far.set_data(*polyline([legs[1], arms[1]]))

        for p in self.bike_parts:
            p.set_visible(bike)
        if bike:
            w = -1.6 * self.phase
            spokes = [[c + WHEEL_R * np.array([np.cos(a), np.sin(a)]), c - WHEEL_R * np.array([np.cos(a), np.sin(a)])]
                      for c in (REAR, FRONT) for a in (w, w + np.pi / 2)]
            self.spokes.set_data(*polyline(spokes))
            self.crank.set_data(*polyline([[legs[0][2], legs[1][2]]]))

    def draw_text(self, label):
        target = VERB[STAGES[self.stage]]
        self.title.set_text(f"Groupmate {self.stage + 1} of {len(STAGES)}: {target}")
        self.status.set_text(STATUS[label])
        if self.progress == 0:
            self.goal_text.set_text(f"Your turn: start {target}!")
        else:
            self.goal_text.set_text(f"Keep {target}!   {int(self.progress)} / {self.goal:.0f} s")
        self.fill.set_width(self.progress / self.goal)
        self.hint.set_text(self.det.hint())

        for a in self.overlay:
            a.set_visible(self.mode != "active")
        if self.mode == "handoff":
            nxt = VERB[STAGES[self.stage + 1]]
            self.handoff_text.set_text(f"Then hand the device to the next groupmate.\n"
                                       f"Next up: {nxt}.  Press Space when they're ready.")
        elif self.mode == "done":
            self.handoff_text.set_text("Everyone has had a turn!\nPress Space to start over.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model2_full.json", help="model file in models/")
    ap.add_argument("--goal", type=float, default=30, help="seconds of each activity before the hand-off")
    ap.add_argument("--consistent", type=int, default=3,
                    help="predictions in a row (one per second) that must agree to change the figure")
    ap.add_argument("--port", type=int, default=C.UDP_PORT)
    ap.add_argument("--sim", action="store_true", help="no board: fake the activity with the keyboard")
    args = ap.parse_args()

    if args.sim:
        det = KeyDetector()
    else:
        path = os.path.join(HERE, "models", args.model)
        if not os.path.exists(path):
            raise SystemExit(f"No model at {path}. Train one first (python tools/train.py), "
                             "or try the window with --sim.")
        det = StreamDetector(Model.load(path), args.port, args.consistent)
    Demo(det, args.goal).run()


if __name__ == "__main__":
    main()
