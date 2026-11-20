"""
Window features. numpy only, so the same file runs on the laptop and the board.

Model 1 -- ONE number per window: the dynamic peak of one accelerometer axis,
           max|a - median(a)|. Subtracting the median removes gravity/mounting
           offset so the number means "how hard did this axis get hit".
Model 2 -- every IMU output: per-channel stats on all six axes plus
           orientation-free features from |accel| and |gyro| (including the
           dominant motion frequency, which separates walk / run / bike).
"""

from __future__ import annotations

import numpy as np

from . import config as C

AX = {name: i for i, name in enumerate(C.CHANNELS)}


def peak_feature(win: np.ndarray, axis: str) -> np.ndarray:
    """Model 1 input: shape (1,). win is (n, 6) in CHANNELS order."""
    x = win[:, AX[axis]]
    return np.array([np.max(np.abs(x - np.median(x)))])


def _dominant(sig: np.ndarray, fs: float, lo=0.5, hi=6.0):
    """Frequency (Hz) and share of spectral power of the strongest peak in [lo, hi]."""
    sig = sig - sig.mean()
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any() or spec[band].sum() <= 1e-12:
        return 0.0, 0.0
    i = np.argmax(np.where(band, spec, 0.0))
    return float(freqs[i]), float(spec[i] / spec[band].sum())


def full_feature_names() -> list[str]:
    names = [f"{ch}_{s}" for ch in C.CHANNELS for s in ("mean", "std", "peak")]
    names += ["amag_mean", "amag_std", "amag_max", "amag_min", "amag_rms_dyn",
              "amag_freq", "amag_freq_share",
              "gmag_mean", "gmag_std", "gmag_max", "gmag_freq", "gmag_freq_share"]
    return names


def full_features(win: np.ndarray, fs: float = C.FS) -> np.ndarray:
    """Model 2 input: shape (len(full_feature_names()),)."""
    f = []
    for i in range(6):
        x = win[:, i]
        f += [x.mean(), x.std(), np.max(np.abs(x - np.median(x)))]
    amag = np.linalg.norm(win[:, :3], axis=1)
    gmag = np.linalg.norm(win[:, 3:], axis=1)
    af, ap = _dominant(amag, fs)
    gf, gp = _dominant(gmag, fs)
    f += [amag.mean(), amag.std(), amag.max(), amag.min(),
          np.sqrt(np.mean((amag - amag.mean()) ** 2)), af, ap,
          gmag.mean(), gmag.std(), gmag.max(), gf, gp]
    return np.array(f, dtype=float)


def make_windows(data: np.ndarray, window: int = C.WINDOW, hop: int = C.HOP):
    """Yield (n, 6) slices of data (N, 6)."""
    for s in range(0, len(data) - window + 1, hop):
        yield data[s:s + window]
