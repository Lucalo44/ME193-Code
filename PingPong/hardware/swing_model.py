"""
hardware/swing_model.py -- small learned classifiers for swings.

Each decision about a swing -- forehand or backhand? topspin or backspin
(per stroke)? brushed left or right? -- is a two-class problem with labelled
examples from tools/calibrate_swing.py. This fits an L2-regularized logistic
regression on the standardized strike features of those swings (numpy, IRLS)
and measures it with leave-one-out cross-validation: every calibration swing
is predicted by a model trained without it, so the reported accuracy is how it
does on swings it hasn't seen.

Models are stored as plain JSON in swing_calibration.json:
    {"features": [...], "mean": [...], "std": [...], "w": [...], "b": float,
     "loo_accuracy": float, "n": int, "use": bool}
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

# Strike features used by the classifiers. yaw_at_peak is left out on purpose:
# absolute heading depends on which way the player happens to face.
FEATURES = [
    "pitch_at_peak", "roll_at_peak",
    "yaw_delta", "pitch_delta", "roll_delta",
    "gx_mean", "gy_mean", "gz_mean", "gx_int", "gy_int", "gz_int",
    "gx_pk", "gy_pk", "gz_pk", "lx_pk", "ly_pk", "lz_pk",
]
RIDGE = 1.0          # L2 strength on standardized features (fixed, so the LOO score stays honest)


def noise_floor(name: str) -> float:
    """Smallest spread a feature is treated as having: its measurement noise.
    Without it, a feature that barely varies gets blown up by standardization
    and the classifier fits noise that happens to line up with the labels."""
    if name.startswith(("lx", "ly", "lz")):
        return 0.1            # direction components (unit vector)
    if name.endswith(("_mean", "_pk")):
        return 15.0           # deg/s
    return 2.0                # degrees (angles, deltas, integrated rotation)


def _matrix(feature_dicts: Sequence[Dict[str, float]], names: Sequence[str]) -> np.ndarray:
    return np.array([[float(f.get(n, 0.0)) for n in names] for f in feature_dicts], dtype=float)


def _fit_raw(X: np.ndarray, y: np.ndarray, ridge: float = RIDGE, iters: int = 25):
    """Logistic regression by iteratively reweighted least squares. X standardized."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    P = np.eye(d + 1) * ridge
    P[-1, -1] = 0.0                                   # don't penalize the bias
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        W = p * (1 - p) + 1e-6
        H = A.T @ (A * W[:, None]) + P
        g = A.T @ (p - y) + P @ w
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w[:-1], w[-1]


def fit(positive: Sequence[Dict[str, float]], negative: Sequence[Dict[str, float]],
        names: Sequence[str] = FEATURES) -> Optional[dict]:
    """Train on feature dicts of the two classes. Returns the model (with its
    leave-one-out accuracy), or None with too few examples."""
    if len(positive) < 3 or len(negative) < 3:
        return None
    X = _matrix(list(positive) + list(negative), names)
    y = np.array([1.0] * len(positive) + [0.0] * len(negative))

    floors = np.array([noise_floor(n) for n in names])

    def standardize(Xtr, Xte):
        mu, sd = Xtr.mean(axis=0), np.maximum(Xtr.std(axis=0), floors)
        return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd

    correct = 0
    z_out = np.zeros(len(y))                          # held-out decision value of every swing
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        Xtr, Xte, _, _ = standardize(X[keep], X[i:i + 1])
        w, b = _fit_raw(Xtr, y[keep])
        z_out[i] = (Xte @ w + b)[0]
        correct += int(z_out[i] >= 0) == int(y[i])
    Xs, _, mu, sd = standardize(X, X[:1])
    w, b = _fit_raw(Xs, y)
    # Map the decision value so the average positive swing reads +1 and the
    # average negative swing -1 (the spin *amount* used in the game). Use the
    # held-out values: a model is more confident on swings it trained on than
    # on new ones, so in-sample values would make real swings read too weak.
    zp, zn = float(z_out[y == 1].mean()), float(z_out[y == 0].mean())
    return {"features": list(names), "mean": mu.tolist(), "std": sd.tolist(), "w": w.tolist(), "b": float(b),
            "z_pos": max(1e-6, zp), "z_neg": min(-1e-6, zn),
            "loo_accuracy": correct / len(y), "n": int(len(y)), "use": True}


def decision(model: dict, feats: Dict[str, float]) -> float:
    x = np.array([float(feats.get(n, 0.0)) for n in model["features"]])
    return float(((x - np.array(model["mean"])) / np.array(model["std"])) @ np.array(model["w"]) + model["b"])


def probability(model: dict, feats: Dict[str, float]) -> float:
    """P(positive class) for one swing's features."""
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, decision(model, feats)))))


def signed(model: dict, feats: Dict[str, float]) -> float:
    """-1..+1 amount: 0 on the decision boundary, +1 at the average positive
    calibration swing, -1 at the average negative one (each side scaled on its own)."""
    z = decision(model, feats)
    v = z / model.get("z_pos", 1.0) if z >= 0 else -z / model.get("z_neg", -1.0)
    return max(-1.0, min(1.0, v))


def loo_accuracy_of_terms(positive: Sequence[Dict[str, float]], negative: Sequence[Dict[str, float]],
                          fit_terms, combine_terms) -> Optional[float]:
    """Leave-one-out accuracy of the older single-feature rule, for comparison.
    fit_terms(pos, neg) -> terms; combine_terms(terms, feats) -> signed value."""
    items = [(f, 1) for f in positive] + [(f, 0) for f in negative]
    if len(positive) < 3 or len(negative) < 3:
        return None
    correct = 0
    for i, (f, label) in enumerate(items):
        pos = [g for j, (g, l) in enumerate(items) if l == 1 and j != i]
        neg = [g for j, (g, l) in enumerate(items) if l == 0 and j != i]
        terms = fit_terms(pos, neg)
        correct += int(combine_terms(terms, f) >= 0) == label if terms else 0
    return correct / len(items)
