"""
Synthetic recordings for testing the pipeline before you have a wearable on.
NOT a substitute for real data: it only checks that the code works end to end.
"""

from __future__ import annotations

import numpy as np

from . import config as C

# (step frequency Hz, vertical accel amplitude g, gyro amplitude dps) -- loosely ankle-like
PROFILES = {
    "still": (0.0, 0.0, 0.0),
    "walk": (1.0, 1.2, 150.0),
    "run": (1.45, 3.5, 350.0),
    "bike": (0.9, 0.5, 120.0),
}


def make_recording(label: str, seconds: float = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * C.FS)
    t = np.arange(n) / C.FS
    f, a_amp, g_amp = PROFILES[label]
    f = f * rng.uniform(0.9, 1.1)
    a_amp = a_amp * rng.uniform(0.8, 1.2)
    ph = rng.uniform(0, 2 * np.pi)
    s = np.sin(2 * np.pi * f * t + ph)
    data = rng.normal(0, 0.02, (n, 6))
    data[:, 2] += 1.0                                   # gravity on z
    if label == "bike":                                 # smooth circular pedalling
        data[:, 0] += a_amp * np.cos(2 * np.pi * f * t + ph)
        data[:, 2] += a_amp * s
        data[:, 4] += g_amp * s
    elif label != "still":                              # impacts: rectified, spiky
        spike = np.maximum(s, 0) ** 3
        data[:, 2] += a_amp * spike
        data[:, 0] += 0.3 * a_amp * s
        data[:, 4] += g_amp * s
        data[:, 3] += 0.4 * g_amp * np.cos(2 * np.pi * f * t + ph)
    return data


def write_csv(path: str, data: np.ndarray) -> None:
    t = np.arange(len(data)) / C.FS
    out = np.column_stack([t, data])
    np.savetxt(path, out, delimiter=",", header=",".join(C.COLUMNS), comments="", fmt="%.5f")
