"""
hardware/stroke.py -- live stroke phase from the paddle's orientation (Wii-style).

A backswing turns the paddle about the vertical axis, away from the ready pose;
the forward swing turns it back through ready and on into the follow-through.
Measured on our recordings (Oct 7): forehands drew back to about +95 deg of
on-screen yaw and swung forward at 120-1000 deg/s; backhands drew back the
other way, to about -68 deg. So the on-screen paddle can follow the stroke
continuously from the yaw alone:

    phase +1   fully drawn back (the calibrated backswing for that side)
    phase  0   at the ready pose -- where the paddle meets the ball
    phase -1   follow-through (as far past ready as the backswing was before it)

The one ambiguity is that a forehand's follow-through turns the same way as a
backhand's backswing. Speed resolves it: a FAST turn back through ready is a
stroke (mode "forward"), and its follow-through keeps belonging to that stroke
until the paddle comes back near ready; a SLOW turn is a new backswing. A
stroke only starts from at least half a backswing, so the quick return from a
follow-through isn't mistaken for one -- and in rally rhythm a new stroke can
start straight out of the previous follow-through.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import config as C


class StrokeTracker:
    FAST_DPS = 150.0          # yaw rate that counts as a forward swing
    SLOW_DPS = 60.0           # below this the forward swing is over
    FORWARD_MAX_S = 0.6       # a forward swing never lasts longer than this
    SETTLE_MAX_S = 0.8        # follow-through belongs to the stroke for at most this long
    NEAR_READY_DEG = 18.0     # back within this of ready = settled
    MIN_FORWARD_PHASE = 0.5   # a fast turn only starts a stroke from at least half a backswing
                              # (the quick return from a follow-through is smaller than that)
    RATE_SMOOTH = 0.5         # EMA weight of the newest yaw-rate estimate

    def __init__(self, fh_back_deg: float = C.STROKE_FH_BACK_DEG, bh_back_deg: float = C.STROKE_BH_BACK_DEG):
        self.set_backswing(fh_back_deg, bh_back_deg)
        self.mode = "ready"        # "ready" (incl. drawing back) | "forward" | "settle"
        self.side: Optional[str] = None
        self.phase = 0.0
        self._t: Optional[float] = None
        self._yaw = 0.0
        self.rate = 0.0
        self._mode_t = 0.0
        self._slow_since: Optional[float] = None
        self._history: deque = deque(maxlen=120)     # (t, yaw, rate, mode), ~1.8 s

    def set_backswing(self, fh_back_deg: float, bh_back_deg: float) -> None:
        """Signed yaw (deg) of a full forehand / backhand backswing (opposite signs)."""
        self.back = {"forehand": fh_back_deg, "backhand": bh_back_deg}

    @staticmethod
    def _shape(abs_yaw: float, amp: float) -> float:
        """|yaw| -> phase. Linear up to a little past the full backswing (1.0 at
        `amp`, 1.2 at 1.2*amp); beyond that it eases back to 0 at 180 deg, so
        turning the paddle all the way round moves the on-screen paddle
        continuously (back, round to center at 180, back on the other side,
        home at 360) instead of jumping from one side to the other."""
        amp = max(1.0, amp)
        knee = min(1.2 * amp, 170.0)
        if abs_yaw <= knee:
            return abs_yaw / amp
        return (knee / amp) * max(0.0, (180.0 - abs_yaw) / (180.0 - knee))

    def _signed_phase(self, yaw: float) -> float:
        """Phase against the current stroke's side: + still coming through, - past ready."""
        back = self.back[self.side]
        sign = 1.0 if (yaw >= 0) == (back >= 0) else -1.0
        return sign * self._shape(abs(yaw), abs(back))

    def _side_of(self, yaw: float) -> str:
        return "forehand" if (yaw >= 0) == (self.back["forehand"] >= 0) else "backhand"

    def update(self, t: float, yaw_deg: float) -> dict:
        if self._t is not None and t > self._t:
            d = (yaw_deg - self._yaw + 180.0) % 360.0 - 180.0
            r = d / (t - self._t)
            self.rate = self.RATE_SMOOTH * r + (1 - self.RATE_SMOOTH) * self.rate
        self._t, self._yaw = t, yaw_deg

        if self.mode == "forward":
            # Keep mapping against the stroke's own side: positive while still
            # coming through, negative once past ready (the follow-through).
            self.phase = self._signed_phase(yaw_deg)
            slow = abs(self.rate) < self.SLOW_DPS
            self._slow_since = (self._slow_since if self._slow_since is not None else t) if slow else None
            # The stroke is over when the paddle stops, or turns back the other way
            # (in rally rhythm the follow-through flows straight into the next backswing).
            back_dir = 1.0 if self.back[self.side] >= 0 else -1.0
            reversed_ = self.rate * back_dir > self.SLOW_DPS
            if reversed_ or (self._slow_since is not None and t - self._slow_since > 0.1) \
                    or t - self._mode_t > self.FORWARD_MAX_S:
                self.mode, self._mode_t = "settle", t
        elif self.mode == "settle":
            self.phase = self._signed_phase(yaw_deg)
            toward_ready = self.rate * self.back[self.side] < 0
            if self.phase > self.MIN_FORWARD_PHASE and toward_ready and abs(self.rate) > self.FAST_DPS:
                # Rally rhythm: follow-through -> straight back -> next swing, never pausing at ready.
                self.mode, self._mode_t, self._slow_since = "forward", t, None
            elif abs(yaw_deg) < self.NEAR_READY_DEG or t - self._mode_t > self.SETTLE_MAX_S:
                self.mode, self._mode_t = "ready", t
        if self.mode == "ready":
            self.side = self._side_of(yaw_deg)
            self.phase = self._shape(abs(yaw_deg), abs(self.back[self.side]))
            # Turning back toward ready fast, from a real backswing = a forward swing.
            toward_ready = self.rate * self.back[self.side] < 0
            if toward_ready and abs(self.rate) > self.FAST_DPS and self.phase > self.MIN_FORWARD_PHASE:
                self.mode, self._mode_t, self._slow_since = "forward", t, None
        self.phase = max(-1.3, min(1.3, self.phase))
        self._history.append((t, yaw_deg, self.rate, self.mode))
        return self.snapshot()

    def drawing_back(self, t: float) -> bool:
        """Was the paddle being drawn back (turning away from ready, not in a
        stroke) at time t? Used to keep backswings from being reported as swings."""
        if not self._history:
            return False
        _, yaw, rate, mode = min(self._history, key=lambda h: abs(h[0] - t))
        if mode != "ready" or abs(yaw) < 5.0:
            return False
        away = rate if yaw > 0 else -rate        # deg/s away from the ready pose
        return away > self.SLOW_DPS

    def snapshot(self) -> dict:
        return {"phase": round(self.phase, 3), "side": self.side, "mode": self.mode}


def backswing_from_recording(times, yaws, peak_times, before=(0.45, 0.05)) -> Optional[float]:
    """Typical signed backswing yaw: for each swing, the yaw furthest from ready
    in the `before` window ahead of its peak; the median over all swings."""
    picks = []
    for tp in peak_times:
        w = [y for t, y in zip(times, yaws) if tp - before[0] <= t <= tp - before[1]]
        if w:
            picks.append(max(w, key=abs))
    if not picks:
        return None
    picks.sort()
    return picks[len(picks) // 2]

