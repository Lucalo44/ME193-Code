"""Step events from our trained StepNet (stepcount_board.py, numpy only).

StepNet outputs a step density; its running sum is the step count. This turns that into
discrete events: one Step each time the count passes the next whole number, stamped with
the time of the sample where it happened. StepNet commits samples MARGIN_S (1.5 s) behind
the newest one, so step events arrive about 1.5 s late; at walking pace that's 2 steps.
"""
from dataclasses import dataclass

import numpy as np

from nav.stepcount_board import FS, MARGIN_S, BoardStepNet, StepCounter


@dataclass
class Step:
    t: float          # time of the step (board seconds)
    index: int        # step number, 1-based
    sample: int       # index of the sample where it happened


class StepDetector:
    def __init__(self, model_path, update_s=0.5):
        self.counter = StepCounter(BoardStepNet(str(model_path)))
        self.times = []               # time of every sample fed so far (to stamp steps)
        self.update_every = max(1, int(update_s * FS))
        self.since_update = 0
        self.count = 0

    def add(self, t, row):
        self.counter.add(row)
        self.times.append(t)
        self.since_update += 1

    def poll(self, flush=False):
        """Runs StepNet if due; returns the list of new Step events."""
        if not flush and self.since_update < self.update_every:
            return []
        self.since_update = 0
        c = self.counter                 # same commit rule as StepCounter.update(), but keeps the density
        if len(c.buf) < FS:
            return []
        dens = c.net.density(c.net.features(np.array(c.buf)))
        first = c.n - len(c.buf)
        upto = c.n if flush else c.n - int(MARGIN_S * FS)
        a = max(c.committed, first)
        if upto <= a:
            return []
        seg = dens[a - first:upto - first]
        run = c.total + np.cumsum(seg)
        c.total, c.committed = float(run[-1]), upto
        steps = []
        while self.count + 1 <= run[-1]:          # each whole number crossed is one step
            self.count += 1
            j = min(int(np.searchsorted(run, self.count)), len(seg) - 1)
            s = a + j
            steps.append(Step(t=self.times[s], index=self.count, sample=s))
        return steps

    @property
    def latency_s(self):
        return MARGIN_S
