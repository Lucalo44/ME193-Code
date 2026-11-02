"""
StepNet in numpy only, for counting steps on the UNO Q without the laptop.

tools/sync_app.py copies this file to app/python/stepcount_board.py and the
exported weights (stepcount/tools/export_board.py) to app/python/models/.
It deliberately imports nothing but numpy, so the board needs no PyTorch,
scipy or pandas. On the board only the MPU-6050 exists; the Double Motor's
input channels stay at zero with present = 0, which the network learned to
handle through sensor-dropout training.

The math mirrors stepcount/fuse.py (features) and stepcount/net.py (layers);
stepcount/tests checks both give the same density.
"""

from __future__ import annotations

from collections import deque

import numpy as np

FS = 50
MARGIN_S = 1.5          # commit density this far behind the newest sample (as in live_steps.py)
CONTEXT_S = 10.0        # samples kept for each pass


def _erf(x):
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7); numpy has no erf."""
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def _running_mean(x, n=FS):
    """Centred moving average with edge padding, same window as scipy uniform_filter1d(size=n, mode='nearest')."""
    left, right = n // 2, (n - 1) // 2
    xp = np.concatenate([np.repeat(x[:1], left, 0), x, np.repeat(x[-1:], right, 0)])
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(xp, axis=0)])
    return (c[n:] - c[:-n]) / n


class BoardStepNet:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.kernel = int(d["kernel"])
        self.dilations = [int(v) for v in d["dilations"]]
        self.mean, self.std = d["mean"], d["std"]
        self.channels = [str(c) for c in d["channels"]]
        self.w = {k: d[k] for k in d.files if k.startswith(("inp.", "blocks.", "out."))}

    # ---- features: fuse.build() + fuse.normalize() for an MPU-only window ----
    def features(self, mpu):
        """mpu: (T, 6) ax, ay, az (g), gx, gy, gz (deg/s) at 50 Hz -> (T, C) normalised input."""
        x = np.asarray(mpu, dtype=float).copy()
        amag = np.linalg.norm(x[:, 0:3], axis=1)
        gmag = np.linalg.norm(x[:, 3:6], axis=1)
        x[:, 0:3] -= _running_mean(x[:, 0:3])
        cols = {f"mpu_{c}": x[:, i] for i, c in enumerate(["ax", "ay", "az", "gx", "gy", "gz"])}
        cols.update(mpu_amag=amag, mpu_gmag=gmag, mpu_present=np.ones(len(x)))
        Z = np.zeros((len(x), len(self.channels)))
        for j, name in enumerate(self.channels):
            if name in cols:                                  # the Double Motor's channels stay 0
                Z[:, j] = (cols[name] - self.mean[j]) / self.std[j]
        return Z

    # ---- layers ----
    @staticmethod
    def _conv(x, w, b, dilation=1):
        """x (C, T), w (O, C, k): PyTorch Conv1d with zero 'same' padding."""
        k = w.shape[2]
        if k == 1:
            return w[:, :, 0] @ x + b[:, None]
        p = dilation * (k - 1) // 2
        T = x.shape[1]
        xp = np.pad(x, ((0, 0), (p, p)))
        cols = np.stack([xp[:, j * dilation:j * dilation + T] for j in range(k)], axis=1)   # (C, k, T)
        return w.reshape(w.shape[0], -1) @ cols.reshape(-1, T) + b[:, None]

    def density(self, Z):
        """(T, C) normalised input -> (T,) step density."""
        w = self.w
        h = self._conv(Z.T, w["inp.weight"], w["inp.bias"])
        for i, d in enumerate(self.dilations):
            a = _gelu(self._conv(h, w[f"blocks.{i}.conv.weight"], w[f"blocks.{i}.conv.bias"], d))
            h = h + self._conv(a, w[f"blocks.{i}.mix.weight"], w[f"blocks.{i}.mix.bias"])
        z = self._conv(h, w["out.weight"], w["out.bias"])[0]
        return np.logaddexp(0.0, z)                           # softplus


class StepCounter:
    """Running step total from a stream of MPU rows. Every pass looks at the last
    CONTEXT_S seconds and commits each sample's density exactly once, MARGIN_S
    behind the newest sample (the network needs to see a little after a step)."""

    def __init__(self, net):
        self.net = net
        self.buf = deque(maxlen=int(CONTEXT_S * FS))
        self.n = 0              # samples received so far
        self.committed = 0      # samples whose density is already in the total
        self.total = 0.0

    def add(self, row):
        self.buf.append(row)
        self.n += 1

    def update(self, flush=False):
        if len(self.buf) < FS:
            return self.total
        dens = self.net.density(self.net.features(np.array(self.buf)))
        first = self.n - len(self.buf)                       # sample index of dens[0]
        upto = self.n if flush else self.n - int(MARGIN_S * FS)
        a, b = max(self.committed, first), upto
        if b > a:
            self.total += float(dens[a - first:b - first].sum())
            self.committed = b
        return self.total
