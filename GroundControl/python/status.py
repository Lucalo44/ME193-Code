"""The status state machine. Never blocks, never touches the LLM.

State is a small dict guarded by a lock. Anyone can read a snapshot at any
time (the MQTT thread answers /status from here in well under a second, even
mid-generation). Listeners get called on every change.
"""
import threading
import time

import config

STATES = ("IDLE", "PLANNING", "EXECUTING", "DANCING", "DRIVING", "ESTOP", "ERROR")


class Status:
    def __init__(self):
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._listeners = []
        self._s = {
            "state": "IDLE",
            "doing": "Waiting for a command",
            "job": None,
            "queue": 0,
            "model": config.LLM_MODEL,
            "llm_tok_s": None,
            "llm_ready": False,
            "mqtt": False,
            "car": {"connected": False, "left": 0, "right": 0},
            "lego": {"connected": False, "state": "off"},
            "imu": {"yaw": None, "pitch": None, "roll": None, "source": "none"},
            "dance": None,
            "game": {"mode": "free", "score": 0},
            "last_reply": None,
        }

    def on_change(self, fn):
        self._listeners.append(fn)

    def update(self, notify=True, **kw):
        with self._lock:
            changed = any(self._s.get(k) != v for k, v in kw.items())
            self._s.update(kw)
        if changed and notify:
            snap = self.snapshot()
            for fn in self._listeners:
                try:
                    fn(snap)
                except Exception as e:  # noqa: BLE001
                    print(f"[status] listener error: {e!r}", flush=True)

    def set_state(self, state, doing, **kw):
        assert state in STATES, state
        self.update(state=state, doing=doing, **kw)

    def get(self, key):
        with self._lock:
            return self._s.get(key)

    def snapshot(self):
        with self._lock:
            s = dict(self._s)
        s["alive"] = True
        s["uptime_s"] = round(time.time() - self._t0)
        s["t"] = round(time.time(), 3)
        return s
