"""
Live sensor streams on one clock (time.monotonic() on the laptop).

MpuStream  -- the UNO Q's UDP packets (app/python/main.py with STREAM_TO set).
              Rows carry the board's own timestamp; the board->laptop offset is
              the smallest (arrival - board time) seen, i.e. the fastest packet.
LegoStream -- the LEGO Double Motor's IMU notifications over BLE (same
              connect/notify pattern as PingPong/hardware/paddle.py). BLE
              delivers them in bursts, so samples are re-timed onto an even
              clock that never runs ahead of arrival (PingPong's SampleClock).

Both keep every sample in memory; .snapshot(t0) returns (t, values) since t0.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import deque

import numpy as np

from . import config as C

try:
    import legoeducation as le
except ImportError:   # recording / live counting with the MPU only still works
    le = None

LEGO_COLORS = ["nocolor", "red", "yellow", "blue", "teal", "green", "purple",
               "white", "magenta", "orange", "azure"]


class _Stream:
    name = "?"

    def __init__(self):
        self.t, self.v = [], []
        self.lock = threading.Lock()
        self.status = "off"

    def _add(self, t, row):
        with self.lock:
            self.t.append(t)
            self.v.append(row)

    def snapshot(self, t0=-np.inf):
        """(t, values) for samples newer than t0, as arrays (empty if none)."""
        with self.lock:
            if not self.t:
                return np.zeros(0), np.zeros((0, len(C.SENSOR_CHANNELS[self.name])))
            i = np.searchsorted(self.t, t0, side="right")
            return np.array(self.t[i:]), np.array(self.v[i:])

    def last_time(self):
        with self.lock:
            return self.t[-1] if self.t else None

    def live(self, max_age_s=1.0):
        t = self.last_time()
        return t is not None and time.monotonic() - t < max_age_s


class MpuStream(_Stream):
    name = "mpu"

    def __init__(self, port=C.UDP_PORT):
        super().__init__()
        self.port = port
        self.offset = None          # laptop time = board time + offset
        self._stop = False

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.port))
        self.sock.settimeout(0.5)
        self.status = f"waiting for UDP on port {self.port}"
        threading.Thread(target=self._run, daemon=True, name="mpu-udp").start()

    def _run(self):
        pending = deque()           # rows kept until the offset settles a little
        while not self._stop:
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            arrival = time.monotonic()
            rows = []
            for line in data.decode(errors="ignore").splitlines():
                p = line.split(",")
                if len(p) == 7:
                    try:
                        rows.append([float(x) for x in p])
                    except ValueError:
                        pass
            if not rows:
                continue
            off = arrival - rows[-1][0]                 # newest row was sampled just before arrival
            self.offset = off if self.offset is None else min(self.offset, off)
            self.status = "streaming"
            for r in rows:
                self._add(r[0] + self.offset, r[1:])

    def stop(self):
        self._stop = True


class LegoStream(_Stream):
    name = "lego"
    MAX_LAG_S = 0.4

    def __init__(self, color=C.LEGO_CARD_COLOR, serial=C.LEGO_CARD_SERIAL):
        super().__init__()
        self.color, self.serial = color, serial
        self.motor = None
        self.period = C.LEGO_NOTIFY_MS / 1000.0
        self._arrivals = deque(maxlen=200)
        self._clock = None

    def start(self):
        if le is None:
            self.status = "legoeducation not installed"
            return
        self.status = "connecting"
        threading.Thread(target=self._connect, daemon=True, name="lego-ble").start()

    def _connect(self):
        try:
            color = self.color
            if isinstance(color, str):
                color = LEGO_COLORS.index(color.strip().lower())
            serial = None if self.serial is None else f"{int(self.serial):04d}"
            self.motor = le.DoubleMotor()
            self.motor.connect(card_color=color, card_serial=serial,
                               device_notification_delay=C.LEGO_NOTIFY_MS)
            if not self.motor.connected:
                self.status = "not found (check it's on; PingPong/tools/find_motor.py lists motors)"
                return
            self.motor.set_notification_callback(self._on_notification)
            try:
                self.motor.movement_stop(blocking=False)
            except Exception:
                pass
            self.status = "streaming"
        except Exception as exc:  # noqa: BLE001 -- show any BLE error to the user
            self.status = f"error: {exc}"

    def _stamp(self, arrival):
        """Even clock: one period after the previous sample, never after arrival."""
        self._arrivals.append(arrival)
        if len(self._arrivals) >= 50:
            self.period = (self._arrivals[-1] - self._arrivals[0]) / (len(self._arrivals) - 1)
        if self._clock is None:
            self._clock = arrival
        else:
            self._clock = min(arrival, max(self._clock + self.period, arrival - self.MAX_LAG_S))
        return self._clock

    def _on_notification(self, data):
        arrival = time.monotonic()
        try:
            items = le.device_notification_parser(data)
        except Exception:
            return
        for it in items:
            if isinstance(it, le.ImuDeviceNotification):
                a, g, d = C.LEGO_ACCEL_RAW_PER_G, C.LEGO_GYRO_RAW_PER_DPS, C.LEGO_ANGLE_RAW_PER_DEG
                self._add(self._stamp(arrival),
                          [it.accelerometerX / a, it.accelerometerY / a, it.accelerometerZ / a,
                           it.gyroscopeX / g, it.gyroscopeY / g, it.gyroscopeZ / g,
                           it.pitch / d, it.roll / d])

    def stop(self):
        if self.motor is None:
            return
        try:
            self.motor.disconnect()
        except Exception:
            pass


def make_streams(use_mpu=True, use_lego=True):
    streams = {}
    if use_mpu:
        streams["mpu"] = MpuStream()
    if use_lego:
        streams["lego"] = LegoStream()
    for s in streams.values():
        s.start()
    return streams
