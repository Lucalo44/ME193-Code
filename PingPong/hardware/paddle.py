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
from hardware.orientation import OnboardOrientation, OrientationFilter, euler_yxz_deg, scene_mapping, to_scene
from hardware.stroke import StrokeTracker
from hardware.swing import (Calibration, ImuSample, SampleClock, SwingDetector, TossDetector, accel_g,
                            gyro_vec_dps)

try:
    import legoeducation as le
except ImportError:  # the sim paddle and tests work without it
    le = None


COLOR_NAMES = ["nocolor", "red", "yellow", "blue", "teal", "green", "purple",
               "white", "magenta", "orange", "azure"]          # index = LEGO_COLOR_* value


def color_name(value) -> str:
    return COLOR_NAMES[value] if isinstance(value, int) and 0 <= value < len(COLOR_NAMES) else str(value)


def card_filter():
    """(card_color, card_serial) for legoeducation's connect(), validated.
    Raises ValueError with a readable message for settings it can't use --
    an unknown color must never silently turn the filter off."""
    color = C.CARD_COLOR
    if isinstance(color, str):
        name = color.strip().lower().removeprefix("lego_color_")
        if name not in COLOR_NAMES[1:]:
            raise ValueError(f"CARD_COLOR = {C.CARD_COLOR!r} isn't a card color. Use one of: "
                             + ", ".join(COLOR_NAMES[1:]) + " (or None).")
        color = COLOR_NAMES.index(name)
    elif color is not None and not (isinstance(color, int) and 1 <= color < len(COLOR_NAMES)):
        raise ValueError(f"CARD_COLOR = {C.CARD_COLOR!r} isn't a card color.")
    serial = C.CARD_SERIAL
    if serial is not None:
        try:
            if not 0 <= int(serial) <= 9999:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"CARD_SERIAL = {C.CARD_SERIAL!r} must be the card's number, e.g. \"0994\" (or None).")
        serial = f"{int(serial):04d}"
    return color, serial


def describe_card_filter() -> str:
    color, serial = card_filter()
    if color is None and serial is None:
        return "the first Double Motor found (no Connection Card filter)"
    parts = ([f"{color_name(color)} card"] if color is not None else []) + \
            ([f"serial {serial}"] if serial is not None else [])
    return "a Double Motor with " + ", ".join(parts)


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
        # 3D orientation for the on-screen paddle: the motor's own fused angles
        # (default) or our gyro + accelerometer filter -- see hardware/orientation.py.
        self.clock = SampleClock()
        self.onboard = OnboardOrientation()
        self.orient = OrientationFilter()
        self.stroke = StrokeTracker()
        self.detector.stroke = self.stroke if C.LIVE_STROKE else None   # don't report backswings
        self._remap()

    def _remap(self) -> None:
        cal = self.calibration
        self.orient.reset(cal.gravity)
        self.onboard.reset()
        self._mapping = scene_mapping(cal.gravity, cal.close_axis)
        flip = -1.0 if C.HANDEDNESS == "left" else 1.0
        fh = cal.stroke_back_fh_deg if cal.stroke_back_fh_deg is not None else C.STROKE_FH_BACK_DEG * flip
        bh = cal.stroke_back_bh_deg if cal.stroke_back_bh_deg is not None else C.STROKE_BH_BACK_DEG * flip
        self.stroke.set_backswing(fh, bh)

    # -- sample plumbing --------------------------------------------------
    def _ingest(self, s: ImuSample) -> None:
        with self._lock:
            self._buf.append(s)
            while self._buf and s.t - self._buf[0].t > C.IMU_BUFFER_S:
                self._buf.popleft()
            self.sample_count += 1
        for fn in self.raw_listeners:
            fn(s)
        cal = self.calibration
        if C.ORIENTATION_SOURCE == "fusion":
            self.orient.update(s.t, gyro_vec_dps(s, cal), accel_g(s, cal))
        else:
            k = cal.angle_raw_per_deg
            self.onboard.update(s.yaw / k, s.pitch / k, s.roll / k)
        if C.LIVE_STROKE:
            q = self.orient.q if C.ORIENTATION_SOURCE == "fusion" else self.onboard.q
            self.stroke.update(s.t, euler_yxz_deg(to_scene(q, self._mapping))["yaw"])
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
        """Paddle orientation in the scene frame, relative to the zeroed ready pose:
        quaternion q = [x, y, z, w] (three.js order) plus Euler angles in degrees."""
        q = to_scene(self.orient.q if C.ORIENTATION_SOURCE == "fusion" else self.onboard.q, self._mapping)
        out = {k: round(v, 2) for k, v in euler_yxz_deg(q).items()}
        out["q"] = [round(q[1], 4), round(q[2], 4), round(q[3], 4), round(q[0], 4)]
        return out

    def swing_in_progress(self) -> bool:
        """A swing is under way that may not have been reported yet."""
        return self.detector._active or self.stroke.mode == "forward"

    def stroke_state(self) -> Optional[dict]:
        """Live stroke phase for the on-screen paddle (None = no live tracking)."""
        return self.stroke.snapshot() if C.LIVE_STROKE and self.sample_count else None

    def zero(self) -> bool:
        """Capture the neutral ready orientation from the last 0.5 s.
        Hold the paddle still in the ready position when calling this."""
        now = time.monotonic()
        still = self.window(now - 0.5, now)
        if len(still) < 5:
            return False
        self.calibration.zero_from(still)
        self.detector.set_calibration(self.calibration)
        self._remap()                      # the ready pose becomes "straight ahead"
        return True

    def set_calibration(self, cal: Calibration) -> None:
        self.calibration = cal
        self.detector.set_calibration(cal)
        self._remap()

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
                self.status = "not found -- run tools/find_motor.py"
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
        arrival = time.monotonic()
        try:
            items = le.device_notification_parser(data)
        except Exception:
            return
        for item in items:
            if isinstance(item, le.ImuDeviceNotification):
                t = self.clock.stamp(arrival)      # even 15 ms clock, not bursty arrival times
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
