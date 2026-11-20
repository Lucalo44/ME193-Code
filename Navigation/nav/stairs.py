"""Stairs classifier: P(stairs) for each step.

One decision per step, from features of that step's IMU segment (not raw windows):
vertical acceleration (along gravity) peak, trough, spread and range, the step's duration,
and how much the device rotates during it. Stairs give a sharper vertical push and a slower,
more even cadence than flat walking. A logistic regression is plenty for 6 features; it's
trained by tools/train.py from the stairs_start / stairs_end taps in the walks.
"""
import json

import numpy as np

FEATURES = ["v_max", "v_min", "v_std", "v_range", "step_s", "gyro_mean"]


def step_features(vert, gyro, step_s):
    """vert: vertical accel (g, gravity removed) over the step; gyro: (n, 3) deg/s."""
    if len(vert) < 3:
        return np.zeros(len(FEATURES))
    return np.array([vert.max(), vert.min(), vert.std(), vert.max() - vert.min(), step_s,
                     float(np.linalg.norm(gyro, axis=1).mean()) / 100.0])


class StairsClassifier:
    def __init__(self, spec=None):
        self.spec = spec

    @classmethod
    def load(cls, path):
        try:
            with open(path) as f:
                return cls(json.load(f))
        except (OSError, ValueError):
            return cls(None)                      # untrained: always 0

    @property
    def trained(self):
        return self.spec is not None

    def predict(self, x):
        if not self.trained:
            return 0.0
        s = self.spec
        z = (np.asarray(x) - np.array(s["mean"])) / np.array(s["scale"])
        return float(1.0 / (1.0 + np.exp(-(z @ np.array(s["coef"]) + s["intercept"]))))

    @staticmethod
    def fit(X, y, l2=1e-2, iters=2000, lr=0.1):
        """Plain logistic regression with class balancing; returns a StairsClassifier."""
        X, y = np.asarray(X, float), np.asarray(y, float)
        mean, scale = X.mean(0), X.std(0) + 1e-6
        Z = (X - mean) / scale
        w_pos = 0.5 / max(y.mean(), 1e-3)
        w_neg = 0.5 / max(1 - y.mean(), 1e-3)
        sw = np.where(y > 0.5, w_pos, w_neg)
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            g = sw * (p - y)
            w -= lr * (Z.T @ g / len(y) + l2 * w)
            b -= lr * g.mean()
        return StairsClassifier({"features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(),
                                 "coef": w.tolist(), "intercept": float(b)})

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.spec, f, indent=1)
