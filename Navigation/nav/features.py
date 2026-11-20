"""The observation the policy sees: one fixed-length vector per step.

Built from step-level state only (never raw IMU): how far along the walk we are, heading,
recent heading changes, the turns made so far, stairs, and movement. Training (laptop) and
navigation (board) both build it here, so the two can't drift apart.
"""
import math

import numpy as np

NAMES = [
    "steps/100", "progress", "sin(heading)", "cos(heading)", "turn now (3 steps)",
    "turn recent (8 steps)", "left turns", "right turns", "steps since turn/50", "last turn L/R",
    "P(stairs)", "stairs steps/50", "stairs flights", "P(still)", "P(walking)", "P(running)",
    "cadence/2", "x/100", "y/100",
]


def observation(state, activity_probs, route_len):
    s = state
    h = math.radians(s["heading"])
    p = activity_probs
    return np.array([
        s["steps"] / 100.0,
        s["steps"] / max(route_len, 1.0),
        math.sin(h), math.cos(h),
        s["dh3"] / 90.0,
        s["dh8"] / 90.0,
        s["n_left"] / 4.0,
        s["n_right"] / 4.0,
        min(s["since_turn"], 200) / 50.0,
        float(s["last_turn"]),
        s["p_stairs"],
        s["stairs_steps"] / 50.0,
        float(s["stairs_segments"]),
        p[0], p[1], p[2],
        s["cadence"] / 2.0,
        s["x"] / 100.0,
        s["y"] / 100.0,
    ], dtype=float)


def feature_names():
    return list(NAMES)
