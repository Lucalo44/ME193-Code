"""
hardware/sim_paddle.py -- keyboard stand-in for the Double Motor (--no-motor).

Same interface as hardware/paddle.py, so the whole game is playable without
hardware. Keys arrive from the browser (and the terminal):

    space          swing at medium strength -- or, when it's your serve and the
                   ball is in your hand, toss it (shift = higher toss)
    shift + space  hard swing
    W / S (held)   topspin / backspin, and tilts the ghost paddle face
    A / D (held)   sidespin left / right
    F / B (held)   force the judged stroke to forehand / backhand -- only used
                   with --no-camera, to test the WRONG STROKE rule
"""

from __future__ import annotations

import math
import time

from hardware.paddle import PaddleBase
from hardware.orientation import qaxis, qmul
from hardware.swing import Calibration, ImuSample, SwingEvent, TossEvent, strength_to_speed, toss_height

MEDIUM_STRENGTH = 0.45
HARD_STRENGTH = 0.95


class SimPaddle(PaddleBase):
    kind = "sim"

    def __init__(self, calibration: Calibration):
        super().__init__(calibration)
        self.status = "sim"
        self.held = set()
        self.shift = False
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self._swing_t = None          # time since the last swing, drives the on-screen stroke
        self._swing_dir = 1.0         # +1 forehand, -1 backhand

    def handle_key(self, key: str, down: bool, shift: bool = False) -> None:
        key = key.lower()
        self.shift = shift
        if key in ("w", "s", "a", "d", "f", "b"):
            (self.held.add if down else self.held.discard)(key)
        elif key in (" ", "space") and down:
            if self.toss_armed:
                strength = 0.8 if shift else 0.4
                self.toss_armed = False
                self.toss_events.put(TossEvent(time.monotonic(), strength, toss_height(strength)))
            else:
                self._swing(HARD_STRENGTH if shift else MEDIUM_STRENGTH)

    def _swing(self, strength: float) -> None:
        topspin = (1.0 if "w" in self.held else 0.0) - (1.0 if "s" in self.held else 0.0)
        sidespin = (1.0 if "d" in self.held else 0.0) - (1.0 if "a" in self.held else 0.0)
        stroke = "forehand" if "f" in self.held else "backhand" if "b" in self.held else None
        self._swing_t = 0.0
        self._swing_dir = -1.0 if stroke == "backhand" else 1.0
        t_peak = time.monotonic() - 0.03
        peak_g = 1.5 + 4.5 * strength
        raw = [ImuSample(t_peak + dt, 0, self.pitch * 10, self.roll * 10,
                         0, 0, 1000 + 1000 * peak_g * max(0.0, 1 - abs(dt) / 0.12), 0, 0, 0)
               for dt in [i * 0.015 for i in range(-10, 11)]]
        self.swing_events.put(SwingEvent(
            t_peak=t_peak, strength=strength, return_speed=strength_to_speed(strength),
            topspin=topspin * 0.8, sidespin=sidespin * 0.8,
            peak_accel_g=peak_g, peak_gyro_dps=200 + 800 * strength,
            features={}, raw=raw, stroke_hint=stroke,
        ))

    def tick(self, dt: float) -> None:
        target_pitch = (-25.0 if "w" in self.held else 0.0) + (25.0 if "s" in self.held else 0.0)
        target_roll = (-20.0 if "a" in self.held else 0.0) + (20.0 if "d" in self.held else 0.0)
        k = min(1.0, dt * 12.0)
        self.pitch += (target_pitch - self.pitch) * k
        self.roll += (target_roll - self.roll) * k
        # Stroke animation: Space is the moment of contact, so sweep from a
        # backswing through the ball into the follow-through, then recover.
        # Negative yaw swings the paddle out to the right (forehand backswing).
        self.yaw = 0.0
        if self._swing_t is not None:
            self._swing_t += dt
            u = self._swing_t
            if u < 0.08:
                yaw = -45.0 + 45.0 * (u / 0.08)              # backswing -> contact
            elif u < 0.28:
                yaw = 50.0 * ((u - 0.08) / 0.2)               # through to the follow-through
            elif u < 0.6:
                yaw = 50.0 * (1 - (u - 0.28) / 0.32)          # recover to ready
            else:
                yaw, self._swing_t = 0.0, None
            self.yaw = yaw * self._swing_dir

    def orientation(self) -> dict:
        q = qmul(qmul(qaxis((0, 1, 0), math.radians(self.yaw)), qaxis((1, 0, 0), math.radians(self.pitch))),
                 qaxis((0, 0, 1), math.radians(-self.roll)))
        return {"pitch": round(self.pitch, 2), "roll": round(self.roll, 2), "yaw": round(self.yaw, 2),
                "q": [round(q[1], 4), round(q[2], 4), round(q[3], 4), round(q[0], 4)]}

    def zero(self) -> bool:
        self.pitch = self.roll = self.yaw = 0.0
        return True

    def stroke_state(self):
        return None      # keyboard swings use the scripted on-screen stroke

    def swing_in_progress(self) -> bool:
        return False     # keyboard swings register instantly -- nothing to wait for
