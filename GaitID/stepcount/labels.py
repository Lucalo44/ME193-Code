"""
Training targets from a hand count.

Hand-timestamping every foot strike is tedious, so each take only stores the
number of steps you counted. The labeller places them with the gyroscope: every
stride, the shank swings forward and |gyro| makes one big, smooth hump (several
hundred deg/s), then drops near zero while the foot is planted. Heel-strike
spikes in |a| looked promising on synthetic data, but on real ankles the swing
itself moves |a| by 0.5-0.9 g and buries the strikes; the gyro humps stand far
above everything else.

  * both ankles present: every step is one ankle's swing, so the `steps` most
    prominent humps across both sensors (each scaled to its own 90th percentile)
  * one ankle only: its strides, with the other foot's steps halfway between
  * no usable gyro peaks: the most prominent impact peaks instead

A step is marked at its mid-swing. Each becomes a Gaussian bump of area 1
(config.STEP_SIGMA_S), so the target density integrates to the step count --
that's what the network learns.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

from . import config as C


def impact_signal(streams, rescale=True):
    """(T,) impact energy: per sensor |(|a| - its 0.5 s running median)| averaged over 100 ms and
    scaled to its 99th percentile, summed over sensors. A heel strike is only 1-2 samples wide at
    50 Hz and rings both above and below 1 g, so the absolute value plus smoothing turns every
    strike into one solid bump instead of a spike that depends on sampling phase.
    rescale=False keeps g units (the baseline counter needs an absolute threshold)."""
    total = None
    for s in C.SENSORS:
        x = streams[s]
        if np.isnan(x).all():
            continue
        amag = np.linalg.norm(np.nan_to_num(x[:, 0:3]), axis=1)
        sig = np.abs(amag - median_filter(amag, size=int(0.5 * C.FS) | 1, mode="nearest"))
        sig[np.isnan(x).any(axis=1)] = 0
        sig = uniform_filter1d(sig, int(0.1 * C.FS))
        if rescale:
            sig /= max(np.percentile(sig, 99), 1e-6)
        total = sig if total is None else total + sig
    if total is None:
        raise ValueError("no sensor data")
    return total


def swing_peaks(streams):
    """{sensor: (sample indices, prominence / that sensor's 90th percentile)} of the |gyro| humps."""
    out = {}
    for s in C.SENSORS:
        x = streams[s]
        if np.isnan(x).all():
            continue
        g = np.linalg.norm(np.nan_to_num(x[:, 3:6]), axis=1)
        g[np.isnan(x).any(axis=1)] = 0
        g = uniform_filter1d(g, 5)
        p, props = find_peaks(g, distance=max(1, int(C.MIN_STRIDE_S * C.FS)), prominence=0)
        if len(p):
            out[s] = (p, props["prominences"] / max(np.percentile(props["prominences"], 90), 1e-6))
    return out


def _fill_other_foot(strides, n_steps, length):
    """One ankle's strides -> all steps: the other foot lands halfway between."""
    gaps = np.diff(strides)
    med = np.median(gaps) if len(gaps) else C.FS
    mids = [a + g // 2 for a, g in zip(strides[:-1], gaps) if g < 1.6 * med]
    steps = np.sort(np.concatenate([strides, np.array(mids, dtype=int)]))
    while len(steps) < n_steps:                            # finish the last stride
        steps = np.append(steps, min(length - 1, steps[-1] + int(med // 2)))
    return np.sort(steps[:n_steps])


def label_steps(streams, n_steps):
    """Sample indices of the n_steps steps, and a dict describing how they were found."""
    if n_steps <= 0:
        return np.zeros(0, dtype=int), {"method": "none (0 steps)"}
    length = len(next(iter(streams.values())))
    sw = swing_peaks(streams)
    if len(sw) >= 2:
        idx = np.concatenate([p for p, _ in sw.values()])
        score = np.concatenate([q for _, q in sw.values()])
        if len(idx) > n_steps:
            order = np.argsort(-score)
            margin = float(score[order[n_steps - 1]] / max(score[order[n_steps]], 1e-6))
            return np.sort(idx[order[:n_steps]]), {"method": "gyro swings, both ankles", "margin": round(margin, 2)}
    if len(sw) == 1:
        (p, q), = sw.values()
        half = (n_steps + 1) // 2
        if len(p) > half:
            order = np.argsort(-q)
            margin = float(q[order[half - 1]] / max(q[order[half]], 1e-6))
            strides = np.sort(p[order[:half]])
            return _fill_other_foot(strides, n_steps, length), \
                {"method": "gyro swings, one ankle + midpoints", "margin": round(margin, 2)}

    # fallback: impact peaks
    sig = impact_signal(streams)
    peaks, props = find_peaks(sig, distance=max(1, int(C.MIN_STEP_GAP_S * C.FS)), prominence=0)
    order = np.argsort(-props["prominences"])
    if len(peaks) <= n_steps:
        return np.sort(peaks), {"method": f"impact peaks (only {len(peaks)} found for {n_steps} counted)"}
    return np.sort(peaks[order[:n_steps]]), {"method": "impact peaks (no usable gyro swings)"}


def density(step_idx, length, fs=C.FS):
    """(length,) sum of area-1 Gaussian bumps at step_idx. Bumps cut off at the edges keep only
    the part that falls inside, so a crop's density sums to the steps it really contains."""
    y = np.zeros(length, dtype=np.float32)
    sig = C.STEP_SIGMA_S * fs
    r = int(np.ceil(4 * sig))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    for i in step_idx:
        lo, hi = i - r, i + r + 1
        a, b = max(lo, 0), min(hi, length)
        if a < b:
            y[a:b] += k[a - lo:b - lo]
    return y
