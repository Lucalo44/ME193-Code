"""
Tiny exported classifiers. Training (scikit-learn) lives in tools/train.py; this
file only *runs* a model from its JSON, using numpy, so it works on the board.

kind "gaussian_nb": per-class mean/var of the (log1p) feature  -> Model 1
kind "logreg":      standardized multinomial logistic regression -> Model 2
"""

from __future__ import annotations

import json

import numpy as np

from . import features as F


class Model:
    def __init__(self, spec: dict):
        self.spec = spec
        self.kind = spec["kind"]
        self.classes = spec["classes"]

    # -- feature extraction -------------------------------------------------
    def featurize(self, win: np.ndarray) -> np.ndarray:
        if self.spec["input"] == "peak":
            return np.log1p(F.peak_feature(win, self.spec["axis"]))
        return F.full_features(win)

    # -- inference ----------------------------------------------------------
    def scores(self, x: np.ndarray) -> np.ndarray:
        s = self.spec
        if self.kind == "gaussian_nb":
            theta, var = np.array(s["theta"]), np.array(s["var"])
            ll = -0.5 * (np.log(2 * np.pi * var) + (x - theta) ** 2 / var).sum(axis=1)
            return ll + np.log(s["prior"])
        z = (x - np.array(s["mean"])) / np.array(s["scale"])
        return z @ np.array(s["coef"]).T + np.array(s["intercept"])

    def predict_proba(self, win: np.ndarray) -> np.ndarray:
        sc = self.scores(self.featurize(win)[None, :])[0]
        e = np.exp(sc - sc.max())
        return e / e.sum()

    def predict(self, win: np.ndarray) -> str:
        return self.classes[int(np.argmax(self.predict_proba(win)))]

    # -- io -----------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.spec, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Model":
        with open(path) as f:
            return cls(json.load(f))
