"""The whole sensing-to-decision chain, sample in, decision out. Used live on the board, by the
laptop simulator, and offline by training (which replays recorded walks through it), so a
feature computed in training is exactly the feature computed on a walk.

    raw IMU -> Heading (gravity + yaw) -> StepNet steps -> stairs (per step) + activity
            -> Odometry (turns, position) -> features.observation -> Policy -> decision
"""
from collections import deque

import numpy as np

import config as C
from nav.activity import Activity
from nav.features import observation
from nav.imu import Heading
from nav.odometry import Odometry
from nav.policy import Policy
from nav.stairs import StairsClassifier, step_features
from nav.steps import StepDetector

RAW_KEEP = C.FS * 15            # raw samples kept for the dashboard


class Navigator:
    def __init__(self, model_dir=C.MODEL_DIR, policy=None, stairs=None):
        model_dir = model_dir if hasattr(model_dir, "joinpath") else __import__("pathlib").Path(model_dir)
        self.model_dir = model_dir
        self.policy = policy if policy is not None else Policy(model_dir / C.POLICY_MODEL)
        self.stairs = stairs if stairs is not None else StairsClassifier.load(model_dir / C.STAIRS_MODEL)
        self.reset()

    def reset(self):
        self.heading = Heading(C.FS, C.GRAVITY_TAU_S)
        self.steps = StepDetector(self._base(C.STEP_MODEL), C.STEP_UPDATE_S)
        self.activity = Activity(self._base(C.ACTIVITY_MODEL), C.FS)
        self.odo = Odometry()
        self.h_hist, self.v_hist, self.g_hist = [], [], []     # per sample: heading, vertical accel, |gyro|
        self.raw = deque(maxlen=RAW_KEEP)                       # (seq, t, ax..gz)
        self.seq = 0
        self.last_step_sample = 0
        self.records = []

    def _base(self, name):
        """StepNet and the activity model are fixed inputs: fall back to models/ if a custom
        model folder (e.g. a test-trained policy) doesn't have them."""
        p = self.model_dir / name
        return p if p.exists() else C.MODEL_DIR / name

    def route_len(self):
        return self.policy.route_len

    def feed(self, rows):
        """rows: (n, 7) t, accel, gyro. Returns the new step records."""
        rows = np.asarray(rows, float).reshape(-1, 7)
        for r in rows:
            imu = r[1:7]
            h = self.heading.update(imu[None, :])
            self.h_hist.append(h)
            self.v_hist.append(float(self.heading.vertical(imu[None, :])[0]))
            self.g_hist.append(imu[3:6])
            self.steps.add(r[0], imu)
            self.activity.add(imu)
            self.raw.append((self.seq, *r.tolist()))
            self.seq += 1
        return self._steps(self.steps.poll())

    def flush(self):
        return self._steps(self.steps.poll(flush=True))

    def _steps(self, events):
        out = []
        for ev in events:
            a, b = self.last_step_sample, ev.sample + 1
            self.last_step_sample = b
            step_s = (b - a) / C.FS
            sf = step_features(np.array(self.v_hist[a:b]), np.array(self.g_hist[a:b]).reshape(-1, 3), step_s)
            p_st = self.stairs.predict(sf)
            self.activity.p_stairs = p_st
            heading = float(np.mean(self.h_hist[a:b])) if b > a else self.h_hist[ev.sample]   # mean over the step cancels sway
            state = self.odo.on_step(ev.t, heading, p_st)
            probs = self.activity.probs()
            obs = observation(state, probs, self.route_len())
            rec = {"i": ev.index, "t": ev.t, "sample": ev.sample, "state": state,
                   "stairs_features": sf.tolist(), "obs": obs.tolist(),
                   "activity": probs.tolist(), "movement": self.activity.label(),
                   "decision": self.policy.act(obs), "why": self.policy.explain(obs)}
            self.records.append(rec)
            out.append(rec)
        return out

    def raw_since(self, seq):
        return [list(r) for r in self.raw if r[0] >= seq]


def replay(imu, model_dir=C.MODEL_DIR, policy=None, stairs=None, chunk=10):
    """Runs a whole recorded walk (N, 7) through a fresh Navigator; returns its step records."""
    nav = Navigator(model_dir, policy=policy, stairs=stairs)
    for i in range(0, len(imu), chunk):
        nav.feed(imu[i:i + chunk])
    nav.flush()
    return nav.records
