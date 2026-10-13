"""
game/recorder.py -- records what happens in real games, for the play profile
(tools/game_report.py), for extra training data (calibrate_swing.py
--with-games) and, later, for modelling a player's style.

One folder per run of the game, created when the first game starts:

    game_logs/<player>/<YYYYmmdd-HHMMSS>/
        session.json   who played, with which hand / calibration / settings
        calibration.json  the swing calibration in use (the swing features depend on it)
        imu.csv        every paddle sample (same columns as imu_logs/*.csv)
        pose.csv       every camera frame: t, label, conf, then the pose features
                       (same vector as pose_data.npz, mirrored frame like training)
        swings.jsonl   one line per paddle swing: features, the paddle's and the
                       camera's stroke reading, the ball it was aimed at, timing,
                       and what the game made of it (outcome)
        events.jsonl   every game event (hits, bounces, points, misses, ...)

Times: `t` is the shared monotonic clock (time.monotonic(), the clock of the
paddle samples and the camera frames); events also carry `sim_t`, the game clock.
Recording runs on the game thread except the sample/frame listeners, which only
append to buffers; flush() (called every loop iteration) writes them out.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import threading
import time
from typing import Optional

import config as C
from hardware.swing import CSV_FIELDS

SKIP_EVENTS = {"tag_progress"}           # UI-only chatter
STRIP_FIELDS = {"trace"}                 # the HUD's swing trace -- imu.csv has the full stream


def _clean(v):
    """JSON-safe: tuples -> lists, numpy scalars/arrays -> Python, floats rounded."""
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if hasattr(v, "tolist"):
        return _clean(v.tolist())
    return v


class GameRecorder:
    def __init__(self, player: str, root: Optional[str] = None, meta: Optional[dict] = None):
        self.player = player
        self.root = root or os.path.join(C.HERE, C.GAME_LOG_DIR)
        self.meta = dict(meta or {})
        self.dir: Optional[str] = None
        self._lock = threading.Lock()
        self._imu: list = []
        self._pose: list = []
        self._files = {}
        self._imu_writer = self._pose_writer = None
        self.counts = {"swings": 0, "events": 0, "imu": 0, "pose": 0}

    # -- lifecycle ------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.dir is not None

    def open(self) -> None:
        """Create the session folder (first game of the run)."""
        if self.dir is not None:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(self.root, self.player, stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.meta.update({"player": self.player, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "clock": "time.monotonic", "t_open": time.monotonic()})
        self._write_meta()
        cal = self.meta.get("calibration_path")
        if cal and os.path.exists(cal):
            shutil.copy(cal, os.path.join(self.dir, "calibration.json"))   # the features depend on it
        for name in ("swings", "events"):
            self._files[name] = open(os.path.join(self.dir, f"{name}.jsonl"), "a", buffering=1)
        self._files["imu"] = open(os.path.join(self.dir, "imu.csv"), "w", newline="")
        self._imu_writer = csv.writer(self._files["imu"])
        self._imu_writer.writerow(CSV_FIELDS)
        self._files["pose"] = open(os.path.join(self.dir, "pose.csv"), "w", newline="")
        self._pose_writer = csv.writer(self._files["pose"])
        self._pose_header = False

    def close(self) -> None:
        if self.dir is None:
            return
        self.flush()
        self.meta["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.meta["counts"] = dict(self.counts)
        self._write_meta()
        for f in self._files.values():
            f.close()
        self._files = {}

    def _write_meta(self) -> None:
        with open(os.path.join(self.dir, "session.json"), "w") as f:
            json.dump(_clean(self.meta), f, indent=2)

    def update_meta(self, **fields) -> None:
        self.meta.update(fields)
        if self.dir is not None:
            self._write_meta()

    # -- streams (called from the paddle / vision threads) --------------------
    def on_sample(self, s) -> None:
        if self.dir is not None:
            with self._lock:
                self._imu.append([getattr(s, k) for k in CSV_FIELDS])

    def on_frame(self, msg: dict) -> None:
        if self.dir is not None and msg.get("type", "frame") == "frame":
            with self._lock:
                self._pose.append((msg["t"], msg.get("label"), msg.get("conf", 0.0), msg.get("feat")))

    def flush(self) -> None:
        if self.dir is None:
            return
        with self._lock:
            imu, self._imu = self._imu, []
            pose, self._pose = self._pose, []
        for row in imu:
            self._imu_writer.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in row])
        for t, label, conf, feat in pose:
            if feat is not None and not self._pose_header:
                self._pose_writer.writerow(["t", "label", "conf"] + [f"f{i}" for i in range(len(feat))])
                self._pose_header = True
            if self._pose_header:
                self._pose_writer.writerow([f"{t:.4f}", label or "", f"{conf:.3f}"]
                                           + ([f"{x:.5f}" for x in feat] if feat is not None else []))
        self.counts["imu"] += len(imu)
        self.counts["pose"] += len(pose)

    # -- game records (game thread) -------------------------------------------
    def event(self, name: str, t: float, sim_t: float, fields: dict) -> None:
        if self.dir is None or name in SKIP_EVENTS:
            return
        rec = {"name": name, "t": t, "sim_t": sim_t}
        rec.update({k: v for k, v in fields.items() if k not in STRIP_FIELDS})
        self._files["events"].write(json.dumps(_clean(rec)) + "\n")
        self.counts["events"] += 1

    def swing(self, rec: dict) -> None:
        if self.dir is None:
            return
        self._files["swings"].write(json.dumps(_clean(rec)) + "\n")
        self.counts["swings"] += 1


# --------------------------------------------------------------------------
# Reading recordings back for training (game/style.py, tools/calibrate_swing.py)
# --------------------------------------------------------------------------

def session_weights(n: int) -> list:
    """Training weight of each of n sessions, oldest first: the newest counts 1, each
    older one GAME_DATA_RECENCY times the next -- recent play matters a little more."""
    return [C.GAME_DATA_RECENCY ** (n - 1 - i) for i in range(n)]


def hand_timeline(events: list, swings: list):
    """hand_at(sim_t): the hand picked in the lobby at that moment of a session. From
    the logged lobby changes, game starts and swing records (older sessions log no
    changes, so their swings and game starts tell)."""
    marks = sorted([(e["sim_t"], e["hand"]) for e in events if e["name"] == "handedness"]
                   + [(e["sim_t"], e["handedness"]) for e in events if e["name"] == "game_start" and e.get("handedness")]
                   + [(sw["sim_t"], sw["handedness"]) for sw in swings if sw.get("handedness")])

    def hand_at(sim_t: float) -> Optional[str]:
        hand = marks[0][1] if marks else None
        for t, h in marks:
            if t > sim_t + 1e-6:
                break
            hand = h
        return hand
    return hand_at
