"""One navigator session: the mode, the walk being recorded, and the GPS-style instruction.

Modes (picked on the phone page):
    navigate  the policy guides you ("In 30 m, turn left" ... "Turn left now")
    record    demonstration walk: you walk the route and tap each maneuver as you make it
    dagger    the policy guides you AND you tap the maneuvers you really make; wherever it
              was wrong, your taps become the corrected labels (states come from walking
              under the policy, which is the point of DAgger). "Wrong" flags a bad instruction.
"""
import threading

import config as C
from nav.pipeline import Navigator
from nav.walks import WalkWriter

TEXT = {"left": "turn left", "right": "turn right", "stairs": "take the stairs", "arrive": "arrive at " + C.GOAL}
NOW = {"left": "Turn left now", "right": "Turn right now", "stairs": "Take the stairs now",
       "arrive": "You have arrived at the " + C.GOAL}


class Session:
    def __init__(self, walk_root=C.DATA_DIR, model_dir=C.MODEL_DIR):
        self.lock = threading.RLock()
        self.walk_root = walk_root
        self.nav = Navigator(model_dir)
        self.mode, self.person = "idle", C.PEOPLE[0]
        self.writer = None
        self.t_now = 0.0
        self.taps = []
        self.dist_ema = None
        self.instruction = self._idle_instruction()
        self.arrived = False
        self.last_saved = None

    # ---------------- control (from the web page) ----------------
    def start(self, mode, person):
        with self.lock:
            self.stop()
            self.nav.reset()
            self.mode, self.person = mode, person
            self.taps, self.dist_ema, self.arrived = [], None, False
            if mode in ("record", "dagger"):
                self.writer = WalkWriter(self.walk_root, person, "demo" if mode == "record" else "dagger")
            self.instruction = self._make_instruction(None)

    def tap(self, action):
        with self.lock:
            self.taps.append({"t": self.t_now, "action": action, "step": self.nav.odo.steps})
            if self.writer:
                self.writer.tap(self.t_now, action, "correction" if self.mode == "dagger" else "tap")

    def stop(self, discard=False):
        with self.lock:
            if self.writer:
                if discard:
                    self.writer.discard()
                else:
                    self.last_saved = str(self.writer.close())
                self.writer = None
            self.mode = "idle"
            self.instruction = self._idle_instruction()

    # ---------------- data ----------------
    def feed(self, rows):
        with self.lock:
            if len(rows) == 0:
                return []
            self.t_now = float(rows[-1][0])
            if self.writer:
                self.writer.add_samples(rows)
            recs = self.nav.feed(rows)
            if recs and self.mode != "idle":
                self.instruction = self._make_instruction(recs[-1])
            return recs

    # ---------------- instruction ----------------
    def _idle_instruction(self):
        return {"primary": "Pick a mode to start", "secondary": C.ROUTE, "maneuver": None,
                "steps": None, "metres": None, "now": False}

    def _make_instruction(self, rec):
        if self.mode == "record":
            return {"primary": "Recording a demonstration",
                    "secondary": "Walk to the " + C.GOAL + " and tap each maneuver as you make it",
                    "maneuver": None, "steps": None, "metres": None, "now": False}
        if not self.nav.policy.loaded:
            return {"primary": "No navigator trained yet", "secondary": "Record demo walks, then run tools/train.py",
                    "maneuver": None, "steps": None, "metres": None, "now": False}
        if rec is None or rec["decision"] is None:
            return {"primary": "Start walking", "secondary": "Head out of the " + C.START,
                    "maneuver": "straight", "steps": None, "metres": None, "now": False}
        d = rec["decision"]
        steps = d["steps"]
        self.dist_ema = steps if self.dist_ema is None or abs(steps - self.dist_ema) > 15 else \
            0.6 * self.dist_ema + 0.4 * steps
        s = max(0.0, self.dist_ema)
        m = d["maneuver"]
        now = s <= C.NOW_STEPS
        if m == "arrive" and now:
            self.arrived = True
        metres = s * C.STRIDE_M
        if now:
            primary, secondary = NOW[m], ""
        else:
            primary = f"In {self._fmt_m(metres)}, {TEXT[m]}"
            secondary = "Continue straight" if s > 15 else f"Get ready to {TEXT[m]}"
        return {"primary": primary, "secondary": secondary, "maneuver": m, "steps": round(s, 1),
                "metres": round(metres), "now": now, "probs": d["probs"]}

    @staticmethod
    def _fmt_m(m):
        return f"{int(round(m / 10.0) * 10)} m" if m >= 20 else f"{int(round(m))} m"

    def matrix_icon(self):
        ins = self.instruction
        if self.mode == "record":
            return "record"
        if ins.get("maneuver") in (None, "straight") or (ins.get("steps") or 99) > 20:
            return "straight" if self.mode != "idle" else "idle"
        return ins["maneuver"]

    # ---------------- views ----------------
    def state(self):
        with self.lock:
            nav = self.nav
            st = nav.records[-1]["state"] if nav.records else None
            steps = nav.odo.steps
            route_len = nav.route_len()
            cadence = st["cadence"] if st else 0.0
            remaining = max(0.0, route_len - steps)
            return {
                "mode": self.mode, "person": self.person, "people": C.PEOPLE, "route": C.ROUTE,
                "recording": self.writer is not None, "steps": steps,
                "metres": round(steps * C.STRIDE_M), "heading": round(st["heading"], 1) if st else 0.0,
                "movement": nav.activity.label(), "activity": nav.activity.probs().tolist(),
                "instruction": self.instruction, "arrived": self.arrived,
                "progress": min(1.0, steps / route_len) if route_len else 0.0,
                "remaining_m": round(remaining * C.STRIDE_M),
                "eta_s": round(remaining / cadence) if cadence > 0.3 else None,
                "taps": self.taps[-8:], "model": nav.policy.loaded, "stairs_model": nav.stairs.trained,
                "trail": [[round(r["state"]["x"], 2), round(r["state"]["y"], 2)] for r in nav.records[-400:]],
                "reference": nav.policy.meta.get("reference_trail"),
                "last_saved": self.last_saved, "t": self.t_now,
            }

    def live(self, since_seq, since_step):
        """For the laptop dashboard: raw samples and step records since the given counters."""
        with self.lock:
            nav = self.nav
            raw = nav.raw_since(since_seq)[-C.FS * 10:]
            recs = [r for r in nav.records if r["i"] > since_step][-50:]
            return {"seq": nav.seq, "raw": raw, "steps": recs,
                    "yaw_rate": nav.heading.yaw_rate, "heading": nav.h_hist[-1] if nav.h_hist else 0.0}
