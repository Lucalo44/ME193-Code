"""Load labelled recordings and cut them into windows."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import config as C
from . import features as F


def label_of(path: str) -> str:
    """data/raw/<label>_<anything>.csv -> label"""
    name = os.path.basename(path).split("_")[0].lower()
    if name not in C.CLASSES:
        raise ValueError(f"{path}: filename must start with one of {C.CLASSES}")
    return name


def load_recording(path: str) -> np.ndarray:
    """-> (N, 6) array in CHANNELS order (the t column is not needed at fixed FS)."""
    return pd.read_csv(path)[C.CHANNELS].to_numpy(dtype=float)


def load_dir(folder: str, trim_s: float = 2.0):
    """
    Returns windows (n, WINDOW, 6), labels (n,) as class indices, groups (n,) as
    recording indices. trim_s is dropped from both ends of each file (putting the
    device on / taking it off). Cross-validate with `groups` so windows from one
    recording never sit in both train and test.
    """
    X, y, g = [], [], []
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    for gi, p in enumerate(paths):
        data = load_recording(p)
        k = int(trim_s * C.FS)
        data = data[k:len(data) - k]
        for w in F.make_windows(data):
            X.append(w)
            y.append(C.CLASSES.index(label_of(p)))
            g.append(gi)
    if not X:
        raise SystemExit(f"No usable recordings in {folder} (need <label>_*.csv, >{2*trim_s+C.WINDOW_S:.0f}s each)")
    return np.array(X), np.array(y), np.array(g)
