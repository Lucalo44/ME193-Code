"""Synthetic Double Motor IMU recordings for the swing tests.

Raw units match the defaults in config.py: accel in milli-g, gyro in deg/s,
angles in decidegrees. Swings are smooth bumps in linear acceleration and gyro,
with an optional face-angle (pitch) offset to produce top/backspin.
"""

import math
import random

from hardware.swing import ImuSample

DT = 0.015


def recording(swings, rest_s=1.0, gap_s=1.0, seed=0, noise_mg=8, noise_dps=2):
    """swings: list of dicts with keys peak_g, peak_dps, pitch_deg (optional),
    yaw_deg (optional), double (optional: second bump 0.15 s later)."""
    rng = random.Random(seed)
    t = 0.0
    duration = rest_s + len(swings) * gap_s + rest_s
    centers = [rest_s + i * gap_s + gap_s / 2 for i in range(len(swings))]
    samples = []
    while t < duration:
        ax, ay, az = 0.0, 0.0, 1000.0
        gx = gy = gz = 0.0
        pitch = yaw = 0.0
        for c, sw in zip(centers, swings):
            for k, offs in enumerate([0.0] + ([0.15] if sw.get("double") else [])):
                u = (t - (c + offs)) / 0.06
                bump = math.exp(-u * u)
                ax += 1000 * sw["peak_g"] * bump * (0.7 if k else 1.0)
                gz += sw["peak_dps"] * math.exp(-((t - (c + offs)) / 0.09) ** 2)
            w = math.exp(-((t - c) / 0.15) ** 2)
            pitch += sw.get("pitch_deg", 0.0) * w
            yaw += sw.get("yaw_deg", 0.0) * (0.5 + 0.5 * math.tanh((t - c) / 0.05))
        samples.append(ImuSample(
            t, yaw * 10, pitch * 10, 0.0,
            ax + rng.gauss(0, noise_mg), ay + rng.gauss(0, noise_mg), az + rng.gauss(0, noise_mg),
            gx + rng.gauss(0, noise_dps), gy + rng.gauss(0, noise_dps), gz + rng.gauss(0, noise_dps)))
        t += DT
    return samples, centers
