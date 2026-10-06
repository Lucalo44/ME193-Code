"""
hardware/paddle.py -- the LEGO Education Double Motor as a paddle.

Connects in a background thread (same pattern as MotorController in
Pose Racecar/gesture_car_control.py) with IMU notifications every
IMU_NOTIFICATION_MS. Every IMU notification arrives through the notification
callback -- not by polling motor.imu_device -- so no samples are dropped; each
one is timestamped with time.monotonic() on arrival, kept in a ~2 s ring
buffer, and fed to the swing detector. Detected swings go on `swing_events`
for the game loop to drain.

The motors are never driven during play (they're held stopped), except for
the optional haptic buzz on a hit (HAPTICS_ENABLED).

Interface shared with hardware/sim_paddle.py:
    status, kind, calibration, swing_events, toss_events, toss_armed
    start(), shutdown(), latest(), window(t0, t1), zero(), orientation(),
    haptic(), handle_key(key, down, shift)
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Deque, List, Optional

import config as C
from hardware.swing import Calibration, ImuSample, SwingDetector, TossDetector

try:
    import legoeducation as le
except ImportError:  # the sim paddle and tests work without it
    le = None


def card_filter():
    color = getattr(le, C.CARD_COLOR, None) if (le is not None and C.CARD_COLOR) else None
    return color, C.CARD_SERIAL


class PaddleBase:
    kind = "base"

    def __init__(self, calibration: Calibration):
        self.calibration = calibration
        self.detector = SwingDetector(calibration)
        self.toss = TossDetector(self.detector)
        self.swing_events: "queue.Queue" = queue.Queue()
        self.toss_events: "queue.Queue" = queue.Queue()   # serve tosses (upward flicks)
        self.toss_armed = False                            # set by the game while you're due to toss
        self.status = "disconnected"
        self._buf: Deque[ImuSample] = deque()
        self._lock = threading.Lock()
        self.sample_count = 0
        self.raw_listeners: List = []   # callables(sample) -- used by the tools

    # -- sample plumbing --------------------------------------------------
    def _ingest(self, s: ImuSample) -> None:
        with self._lock:
            self._buf.append(s)
            while self._buf and s.t - self._buf[0].t > C.IMU_BUFFER_S:
                self._buf.popleft()
            self.sample_count += 1
        for fn in self.raw_listeners:
            fn(s)
        ev = self.detector.push(s)
        if ev is not None:
            self.swing_events.put(ev)
        toss = self.toss.push(s)
        if toss is not None:
            self.toss_events.put(toss)

    def latest(self) -> Optional[ImuSample]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def window(self, t0: float, t1: float) -> List[ImuSample]:
        with self._lock:
            return [s for s in self._buf if t0 <= s.t <= t1]

    def orientation(self) -> dict:
        s = self.latest()
        if s is None:
            return {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        rel = self.calibration.relative_angles(s)
        return {k: round(v, 2) for k, v in rel.items()}

    def zero(self) -> bool:
        """Capture the neutral ready orientation from the last 0.5 s.
        Hold the paddle still in the ready position when calling this."""
        now = time.monotonic()
        still = self.window(now - 0.5, now)
        if len(still) < 5:
            return False
        self.calibration.zero_from(still)
        self.detector.set_calibration(self.calibration)
        return True

    def set_calibration(self, cal: Calibration) -> None:
        self.calibration = cal
        self.detector.set_calibration(cal)

    # -- overridden by subclasses -----------------------------------------
    def start(self) -> None: ...
    def shutdown(self) -> None: ...
    def haptic(self) -> None: ...
    def handle_key(self, key: str, down: bool, shift: bool = False) -> None: ...
    def tick(self, dt: float) -> None: ...


class Paddle(PaddleBase):
    kind = "real"

    def __init__(self, calibration: Calibration, auto_zero: bool = True):
        super().__init__(calibration)
        self.motor = None
        self.auto_zero = auto_zero
        self._zeroed = False
        if le is None:
            self.status = "error: legoeducation not installed"

    def start(self) -> None:
        if le is None or self.status in ("connecting", "connected"):
            return
        self.status = "connecting"
        threading.Thread(target=self._connect, daemon=True, name="paddle-connect").start()

    def _connect(self) -> None:
        try:
            color, serial = card_filter()
            self.motor = le.DoubleMotor()
            self.motor.connect(card_color=color, card_serial=serial,
                               device_notification_delay=C.IMU_NOTIFICATION_MS)
            if not self.motor.connected:
                self.status = "not found"
                return
            self.motor.set_notification_callback(self._on_notification)
            try:
                self.motor.movement_stop(blocking=False)
            except Exception:
                pass
            self.status = "connected"
            if self.auto_zero:
                # Give the buffer a moment to fill, then zero on the current pose.
                time.sleep(1.0)
                self.zero()
        except Exception as exc:  # noqa: BLE001 -- surface any BLE error in the HUD
            self.status = f"error: {exc}"

    def _on_notification(self, data) -> None:
        t = time.monotonic()
        try:
            items = le.device_notification_parser(data)
        except Exception:
            return
        for item in items:
            if isinstance(item, le.ImuDeviceNotification):
                self._ingest(ImuSample(t, item.yaw, item.pitch, item.roll,
                                       item.accelerometerX, item.accelerometerY, item.accelerometerZ,
                                       item.gyroscopeX, item.gyroscopeY, item.gyroscopeZ))

    def zero(self) -> bool:
        ok = super().zero()
        if ok and self.motor is not None and self.status == "connected":
            # Reset the hardware yaw so "straight ahead" is 0 from now on.
            def reset():
                try:
                    self.motor.imu_reset_yaw_axis(0)
                    self.calibration.neutral["yaw"] = 0.0
                except Exception:
                    pass
            threading.Thread(target=reset, daemon=True).start()
        self._zeroed = self._zeroed or ok
        return ok

    def haptic(self) -> None:
        if not C.HAPTICS_ENABLED or self.motor is None or self.status != "connected":
            return

        def buzz():
            try:
                self.motor.motor_run_for_time(C.HAPTICS_PULSE_MS, speed=C.HAPTICS_SPEED, blocking=False)
            except Exception:
                pass
        threading.Thread(target=buzz, daemon=True).start()

    def shutdown(self) -> None:
        if self.motor is None:
            return
        try:
            self.motor.movement_stop(blocking=False)
        except Exception:
            pass
        try:
            self.motor.disconnect()
        except Exception:
            pass
        self.status = "disconnected"
