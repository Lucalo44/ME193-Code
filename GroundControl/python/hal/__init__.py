"""Hardware abstraction. Everything above this package runs the same on the board
(bridge_hal) and on the laptop (sim_hal)."""
import math
import threading
import time

ROWS, COLS = 8, 13
V_MAX = 0.4       # m/s at wheel PWM 255 (estimate; calibrate)
TRACK = 0.12      # m between wheels (estimate; measure)


class Hal:
    """Common bookkeeping; subclasses implement _draw/_drive/_heartbeat.

    The base class dead-reckons the car's pose from the COMMANDED wheel speeds.
    On the simulator that's the ground truth; on the real car (no IMU yet) it's
    an estimate and is labelled as such, so nothing pretends to be a sensor."""
    kind = "base"
    imu_label = "dead-reckoning"

    def __init__(self):
        self._lock = threading.Lock()
        self.frame = [0] * (ROWS * COLS)
        self.left = 0
        self.right = 0
        self.connected = False
        self.on_frame = None          # viz hook: fn(frame)
        self.on_drive = None          # viz hook: fn(left, right)
        self.x = self.y = 0.0
        self.yaw = 0.0                # deg, unwrapped
        self._t = time.monotonic()
        self._pose_lock = threading.Lock()

    def _integrate(self):
        with self._pose_lock:
            now = time.monotonic()
            dt, self._t = now - self._t, now
            vl, vr = self.left / 255 * V_MAX, self.right / 255 * V_MAX
            v, w = (vl + vr) / 2, (vr - vl) / TRACK
            yaw_r = math.radians(self.yaw)
            self.x += v * math.cos(yaw_r) * dt
            self.y += v * math.sin(yaw_r) * dt
            self.yaw += math.degrees(w) * dt

    def draw(self, frame):
        frame = [max(0, min(7, int(v))) for v in frame[: ROWS * COLS]]
        with self._lock:
            self.frame = frame
            self._draw(frame)
        if self.on_frame:
            self.on_frame(frame)

    def drive(self, left, right):
        left, right = int(left), int(right)
        self._integrate()
        with self._lock:
            self.left, self.right = left, right
            self._drive(left, right)
        if self.on_drive:
            self.on_drive(left, right)

    def stop(self):
        self.drive(0, 0)

    def heartbeat(self):
        """-> True if the MCU watchdog had tripped (caller should redraw)."""
        with self._lock:
            tripped = self._heartbeat()
        if tripped:
            self.draw(self.frame)
        return tripped

    def ping(self):
        t = time.perf_counter()
        with self._lock:
            ok = self._ping()
        return ok, (time.perf_counter() - t) * 1000

    def imu(self):
        self._integrate()
        return {"yaw": round(self.yaw % 360, 1), "pitch": 0.0, "roll": 0.0, "source": self.imu_label}

    def pose(self):
        self._integrate()
        return {"x": round(self.x, 3), "y": round(self.y, 3), "yaw": round(self.yaw, 1)}

    def probe_pins(self):
        """-> {"D2": "f"|"H"|"L"|"?", ...} for the free header pins, or None if unsupported."""
        with self._lock:
            raw = self._probe_pins()
        if not raw:
            return None
        return dict(item.split(":") for item in raw.strip(",").split(",") if ":" in item)

    def i2c_scan(self):
        """-> {"Wire": ["29", ...], "Wire1": [...], "Wire2": [...]} hex addresses, or None."""
        with self._lock:
            raw = self._i2c_scan()
        if raw is None:
            return None
        out = {}
        for part in raw.split(";"):
            if ":" in part:
                bus, addrs = part.split(":", 1)
                out[bus] = addrs.split()
        return out

    # subclass hooks
    def _probe_pins(self): return None
    def _i2c_scan(self): return None
    def _draw(self, frame): ...
    def _drive(self, left, right): ...
    def _heartbeat(self): return False
    def _ping(self): return True
