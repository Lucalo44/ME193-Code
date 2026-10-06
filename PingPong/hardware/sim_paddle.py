"""
hardware/sim_paddle.py -- keyboard stand-in for the Double Motor (--no-motor).

Same interface as hardware/paddle.py, so the whole game is playable without
hardware. Keys arrive from the browser (and the terminal):

    space          swing at medium strength
    shift + space  hard swing
    W / S (held)   topspin / backspin, and tilts the ghost paddle face
    A / D (held)   sidespin left / right
    F / B (held)   force the judged stroke to forehand / backhand -- only used
                   with --no-camera, to test the WRONG STROKE rule
"""

from __future__ import annotations

import time

from hardware.paddle import PaddleBase
from hardware.swing import Calibration, ImuSample, SwingEvent, strength_to_speed

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

    def handle_key(self, key: str, down: bool, shift: bool = False) -> None:
        key = key.lower()
        self.shift = shift
        if key in ("w", "s", "a", "d", "f", "b"):
            (self.held.add if down else self.held.discard)(key)
        elif key in (" ", "space") and down:
            self._swing(HARD_STRENGTH if shift else MEDIUM_STRENGTH)

    def _swing(self, strength: float) -> None:
        topspin = (1.0 if "w" in self.held else 0.0) - (1.0 if "s" in self.held else 0.0)
        sidespin = (1.0 if "d" in self.held else 0.0) - (1.0 if "a" in self.held else 0.0)
        stroke = "forehand" if "f" in self.held else "backhand" if "b" in self.held else None
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

    def orientation(self) -> dict:
        return {"pitch": round(self.pitch, 2), "roll": round(self.roll, 2), "yaw": 0.0}

    def zero(self) -> bool:
        self.pitch = self.roll = 0.0
        return True
