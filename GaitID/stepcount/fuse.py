"""
Takes on disk, resampling both sensors onto one 50 Hz grid, and the network's
input matrix.

A take is a folder  data/raw/<activity>_<n>/  holding
    mpu.csv    t, ax, ay, az, gx, gy, gz                (g, deg/s)
    lego.csv   t, ax, ay, az, gx, gy, gz, pitch, roll   (g, deg/s, deg)
    meta.json  {"activity": ..., "steps": <counted by hand>, ...}
with t in seconds on one shared clock. Either CSV may be missing.

Network input, per sensor: its channels (accelerometer and angles with their
1 s running mean removed), |a|, |gyro| and a present flag
(config.SENSOR_CHANNELS + config.DERIVED). A missing sensor is all zeros with
present = 0 after normalisation, so one network handles any sensor combination.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from . import config as C


def channel_names():
    return [f"{s}_{c}" for s in C.SENSORS for c in C.SENSOR_CHANNELS[s] + C.DERIVED]


# ---------------- takes on disk ----------------
def save_take(folder, meta, mpu=None, lego=None):
    """mpu / lego: (t, values) arrays, or None if that sensor wasn't recorded."""
    os.makedirs(folder, exist_ok=True)
    for name, data in (("mpu", mpu), ("lego", lego)):
        if data is None or len(data[0]) == 0:
            continue
        t, v = data
        df = pd.DataFrame(np.column_stack([t, v]), columns=["t"] + C.SENSOR_CHANNELS[name])
        df.to_csv(os.path.join(folder, f"{name}.csv"), index=False, float_format="%.5f")
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def load_take(folder):
    take = {"name": os.path.basename(os.path.normpath(folder))}
    with open(os.path.join(folder, "meta.json")) as f:
        take["meta"] = json.load(f)
    for name in C.SENSORS:
        p = os.path.join(folder, f"{name}.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            take[name] = (df["t"].to_numpy(), df[C.SENSOR_CHANNELS[name]].to_numpy(dtype=float))
        else:
            take[name] = None
    return take


def load_dir(folder):
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if os.path.exists(os.path.join(p, "meta.json")))
    if not paths:
        raise SystemExit(f"No takes in {folder} (expected <activity>_<n>/meta.json folders)")
    return [load_take(p) for p in paths]


# ---------------- grid ----------------
def time_span(take):
    spans = [(d[0][0], d[0][-1]) for d in (take[s] for s in C.SENSORS) if d is not None and len(d[0]) > 1]
    return min(a for a, _ in spans), max(b for _, b in spans)


def on_grid(take_or_streams, grid):
    """{sensor: (len(grid), n_channels) array, NaN where the sensor has no data}."""
    out = {}
    for s in C.SENSORS:
        d = take_or_streams.get(s)
        n = len(C.SENSOR_CHANNELS[s])
        if d is None or len(d[0]) < 2:
            out[s] = np.full((len(grid), n), np.nan)
            continue
        t, v = d
        x = np.column_stack([np.interp(grid, t, v[:, j]) for j in range(n)])
        gap = 0.2                                            # don't bridge dropouts longer than this
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        holes = (grid < t[0]) | (grid > t[-1]) | (t[idx] - t[idx - 1] > gap)
        x[holes] = np.nan
        out[s] = x
    return out


def take_grid(take):
    t0, t1 = time_span(take)
    return np.arange(np.ceil(t0 * C.FS), np.floor(t1 * C.FS) + 1) / C.FS


def running_mean(x, seconds=1.0):
    """Column-wise centred moving average that ignores NaN rows."""
    from scipy.ndimage import uniform_filter1d
    ok = ~np.isnan(x)
    n = max(1, int(seconds * C.FS))
    num = uniform_filter1d(np.where(ok, x, 0.0), n, axis=0, mode="nearest")
    den = uniform_filter1d(ok.astype(float), n, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def build(streams):
    """Gridded sensor dict -> raw network input (T, C) with NaN for missing data.
    Accelerometer and angle channels have their 1 s running mean removed: gravity and
    the strap's mounting tilt change between sessions and say nothing about steps,
    while a 20 degree tilt would otherwise shift an axis by ~0.34 g. Gyro has no such
    offset and |a| keeps the full impact size, so both stay raw."""
    cols = []
    for s in C.SENSORS:
        x = streams[s].copy()
        present = (~np.isnan(x).any(axis=1)).astype(float)
        amag = np.linalg.norm(x[:, 0:3], axis=1)
        gmag = np.linalg.norm(x[:, 3:6], axis=1)
        slow = [0, 1, 2] + list(range(6, x.shape[1]))         # accel axes + pitch/roll
        x[:, slow] -= running_mean(x[:, slow])
        cols += [x, amag[:, None], gmag[:, None], present[:, None]]
    return np.concatenate(cols, axis=1)


# ---------------- normalisation ----------------
def fit_norm(X_list):
    """Per-channel mean / std, except that each sensor's accelerometer and gyro triplets share
    one mean (0) and one std (RMS over the three axes). Per-axis scaling would let a tilted
    strap turn ordinary motion on a quiet axis into a 40-sigma input."""
    import warnings
    X = np.concatenate(X_list, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # a sensor missing from every take
        mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        names = channel_names()
        for s in C.SENSORS:
            for trip in (("ax", "ay", "az"), ("gx", "gy", "gz")):
                idx = [names.index(f"{s}_{c}") for c in trip]
                rms = np.sqrt(np.nanmean(X[:, idx] ** 2))
                mean[idx], std[idx] = 0.0, rms
    for j, n in enumerate(names):
        if n.endswith("_present") or not np.isfinite(mean[j]) or not np.isfinite(std[j]):
            mean[j], std[j] = 0.0, 1.0
    return {"mean": np.nan_to_num(mean).tolist(), "std": np.where(std > 1e-6, std, 1.0).tolist()}


def normalize(X, norm):
    Z = (X - np.array(norm["mean"])) / np.array(norm["std"])
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)
