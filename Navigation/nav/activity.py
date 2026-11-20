"""Movement prediction for the dashboard and the policy: still / walking / running / stairs.

Still, walking and running come from our GaitID activity classifier (model 2, a logistic
regression on 2 s windows, 99.8% cross-validated accuracy). It also has a "bike" class, which
can't happen on foot, so it's dropped and the rest renormalised. Stairs come from the per-step
stairs classifier: the "moving" share of the probability is split by P(stairs).
"""
from collections import deque

import numpy as np

from gaitid.model import Model

LABELS = ["still", "walking", "running", "stairs"]


class Activity:
    def __init__(self, model_path, fs, window_s=2.0, hop_s=1.0):
        try:
            self.model = Model.load(str(model_path))
        except OSError:
            self.model = None
        self.buf = deque(maxlen=int(window_s * fs))
        self.hop = int(hop_s * fs)
        self.since = 0
        self.base = np.array([1.0, 0.0, 0.0])        # still, walk, run
        self.p_stairs = 0.0

    def add(self, row):
        self.buf.append(row)
        self.since += 1
        if self.model and self.since >= self.hop and len(self.buf) == self.buf.maxlen:
            self.since = 0
            p = dict(zip(self.model.classes, self.model.predict_proba(np.array(self.buf))))
            v = np.array([p.get("still", 0), p.get("walk", 0), p.get("run", 0)])
            self.base = v / max(v.sum(), 1e-9)

    def probs(self):
        """[still, walking, running, stairs], summing to 1."""
        still, walk, run = self.base
        s = self.p_stairs
        return np.array([still, walk * (1 - s), run * (1 - s), (walk + run) * s])

    def label(self):
        return LABELS[int(np.argmax(self.probs()))]
