"""
Synthetic two-ankle takes with known step times, for testing the pipeline
without hardware. Never report accuracy on these.

Each foot strike puts a sharp decaying burst on the ankle sensor of that foot
and a weaker echo on the other ankle; each stride swings the shank forward in
one big gyro hump during swing, with a small backward roll during stance. The MPU is sampled at ~50 Hz and clips at +/-8 g like the real one;
the Double Motor at ~66 Hz, mounted in a different orientation.
"""

from __future__ import annotations

import numpy as np

from . import config as C

ACTIVITIES = ["walk", "run", "mixed", "still", "bike"]


def _rotation(rng, max_deg):
    a = np.radians(rng.uniform(-max_deg, max_deg, 3))
    cx, cy, cz, sx, sy, sz = *np.cos(a), *np.sin(a)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def step_times(activity, seconds, rng):
    if activity in ("still", "bike"):
        return np.zeros(0)
    cad = rng.uniform(2.6, 3.0) if activity == "run" else rng.uniform(1.65, 2.05)   # steps per second
    start, end = (3.0, seconds - 3.0) if activity == "mixed" else (0.6, seconds - 0.4)
    out, t = [], start + rng.uniform(0, 0.3)
    while t < end:
        out.append(t)
        t += rng.normal(1, 0.035) * (1 + 0.07 * np.sin(2 * np.pi * t / 17)) / cad
    return np.array(out)


def make_take(activity, seconds=45.0, seed=0):
    rng = np.random.default_rng(seed)
    fine = np.arange(0, seconds, 1 / 400)
    steps = step_times(activity, seconds, rng)
    first_foot = rng.integers(2)
    foot = (np.arange(len(steps)) + first_foot) % 2            # 0 = right (MPU), 1 = left (LEGO)
    run = activity == "run"
    amp = (4.5 if run else 2.2) * rng.uniform(0.8, 1.2)
    swing = (450 if run else 250) * rng.uniform(0.8, 1.2)
    rev = rng.uniform(1.0, 1.4)                                # bike: pedal revolutions per second
    out = {}
    for k, name in enumerate(C.SENSORS):
        acc = np.zeros((len(fine), 3))
        acc[:, 2] = 1.0
        gyro = np.zeros((len(fine), 3))
        pitch = np.zeros(len(fine))
        own, other = steps[foot == k], steps[foot != k]
        for strikes, a in ((own, amp), (other, 0.25 * amp)):
            for ts in strikes:
                m = (fine >= ts) & (fine < ts + 0.2)
                d = fine[m] - ts
                burst = a * np.exp(-d / 0.04) * np.cos(2 * np.pi * 15 * d)    # ~15 Hz ring, 40 ms decay
                acc[m, 2] += burst
                acc[m, 0] += 0.5 * burst
        for a, b in zip(own[:-1], own[1:]):
            m = (fine >= a) & (fine < b)
            ph = (fine[m] - a) / (b - a)                  # 0 = this foot's strike, 1 = its next one
            stance, sw = ph < 0.6, ph >= 0.6
            shape = np.where(stance, -0.15 * np.sin(np.pi * ph / 0.6), np.sin(np.pi * (ph - 0.6) / 0.4))
            gyro[m, 1] += swing * shape                    # one big forward swing, small backward roll
            acc[m, 0] += 0.4 * np.sin(2 * np.pi * ph)
            pitch[m] += 25 * shape
        if activity == "bike":
            ph = 2 * np.pi * rev * fine + np.pi * k
            gyro[:, 1] += 90 * np.sin(ph)
            acc[:, 2] += 0.25 * np.sin(ph)
            pitch += 30 * np.sin(ph)
        acc[:, 0] += 0.01 * np.sin(2 * np.pi * 0.3 * fine)                     # sway
        if name == "lego":                                                    # mounted differently
            R = _rotation(rng, 40)
            acc, gyro = acc @ R.T, gyro @ R.T
        acc += rng.normal(0, 0.02, acc.shape)
        gyro += rng.normal(0, 2.0, gyro.shape)
        fs = 50.0 if name == "mpu" else 1000.0 / C.LEGO_NOTIFY_MS
        t = np.arange(rng.uniform(0, 0.03), seconds - 0.05, 1 / fs) + rng.normal(0, 0.002, 1)
        cols = np.column_stack([acc, gyro] + ([pitch, rng.normal(0, 1, len(fine))] if name == "lego" else []))
        v = np.column_stack([np.interp(t, fine, cols[:, j]) for j in range(cols.shape[1])])
        if name == "mpu":
            v[:, 0:3] = np.clip(v[:, 0:3], -8, 8)
        out[name] = (t, v)
    meta = {"activity": activity, "steps": int(len(steps)), "synthetic": True}
    return meta, out, steps
