"""Performers: anything that can play a dance timeline.

    play(timeline, t_start, cancel, on_kf) -> [report, ...]   (blocks until done)
    stop(), state(), connected, latency_ms

A report per keyframe: {i, beat, action, t_cmd, t_sent, t_actual, motion}
  t_cmd     wall-clock time the keyframe was supposed to happen
  t_sent    when we issued the command
  t_actual  best evidence of when it really happened:
              car  -> midpoint of the Bridge call (the MCU applied the PWM);
                      the car has no encoders, so motor spin-up isn't observed
              lego -> first motor feedback showing motion (or the RPC ack for stops)
              sim  -> modelled latency + jitter
All times are time.time(): the board host and the app container share the
kernel clock, so the LEGO service on the host and the conductor in the
container agree on "now" to well under a millisecond.
"""
import json
import random
import socket
import statistics
import threading
import time

import config

MOTION = {"wheels", "pose", "run", "tank", "stop"}


def _wait_until(t, cancel):
    while True:
        rem = t - time.time()
        if rem <= 0:
            return True
        if cancel.wait(min(rem, 0.02)):
            return False


class Performer:
    name = "performer"
    kind = "base"
    connected = False
    latency_ms = 0.0

    def play(self, timeline, t_start, cancel, on_kf=None):
        raise NotImplementedError

    def stop(self):
        pass

    def state(self):
        return {"name": self.name, "kind": self.kind, "connected": self.connected,
                "latency_ms": round(self.latency_ms, 1)}


class LocalPerformer(Performer):
    """Schedules keyframes itself, firing each one early by its send-ahead."""

    def send_ahead(self):
        return self.latency_ms / 2000.0

    def play(self, timeline, t_start, cancel, on_kf=None):
        reps = []
        ahead = self.send_ahead()
        for kf in timeline:
            t_cmd = t_start + kf["t"]
            if not _wait_until(t_cmd - ahead, cancel):
                break
            t_sent = time.time()
            t_act = self._exec(kf, t_sent)
            rep = {"i": kf["i"], "beat": kf["beat"], "action": kf["action"], "t_cmd": t_cmd,
                   "t_sent": t_sent, "t_actual": t_act, "motion": kf["action"] in MOTION}
            reps.append(rep)
            if on_kf:
                on_kf(self.name, rep)
        return reps

    def _exec(self, kf, t_sent):
        raise NotImplementedError


class CarPerformer(LocalPerformer):
    name, kind = "car", "car"

    def __init__(self, hal, matrix):
        self.hal, self.matrix = hal, matrix

    @property
    def connected(self):
        return self.hal.connected

    def calibrate(self):
        rtts = [self.hal.ping()[1] for _ in range(5)]
        self.latency_ms = statistics.median(rtts)
        return self.latency_ms

    def _exec(self, kf, t_sent):
        p = kf["params"]
        if kf["action"] == "wheels":
            self.hal.drive(p["l"], p["r"])
            return (t_sent + time.time()) / 2
        if kf["action"] == "matrix":
            from skills.matrix import digit_frame
            if "digit" in p:
                self.matrix.show_frame(digit_frame(p["digit"]), {"kind": "text", "text": p["digit"]})
            elif p.get("clear"):
                self.matrix.clear()
            else:
                self.matrix.icon(p["icon"])
            return time.time()
        return time.time()

    def stop(self):
        self.hal.stop()


class SimLegoPerformer(LocalPerformer):
    """A virtual LEGO bot: used when the real one isn't connected, so a dance
    always plays (the dashboard draws it). Latency is modelled on BLE."""
    name, kind = "lego", "sim"
    connected = True

    def __init__(self, ble_ms=45.0, jitter_ms=12.0, motor_ms=30.0):
        self.ble_ms, self.jitter_ms, self.motor_ms = ble_ms, jitter_ms, motor_ms
        self.latency_ms = ble_ms * 2
        self.arms = {"a": 0, "b": 0}

    def _exec(self, kf, t_sent):
        p = kf["params"]
        if kf["action"] == "pose":
            self.arms = {"a": p["a"], "b": p["b"]}
        delay = max(5.0, random.gauss(self.ble_ms, self.jitter_ms)) + self.motor_ms
        return t_sent + delay / 1000

    def state(self):
        return super().state() | {"arms": self.arms}


def docker_host_gateway():
    """Default-route gateway = the board host, seen from inside the app container."""
    try:
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                parts = line.split()
                if parts[1] == "00000000":
                    g = int(parts[2], 16)
                    return socket.inet_ntoa(g.to_bytes(4, "little"))
    except OSError:
        pass
    return "127.0.0.1"


class LegoPerformer(Performer):
    """Client for lego_service/lego_server.py (runs on the board host, owns BLE).

    The whole compiled timeline is sent ahead with its start time; the service
    schedules it locally, measures motion from motor feedback, and reports
    back. The conductor only sends keepalives during the dance (the service
    stops the motors if they stop arriving: deadman 500 ms)."""
    name, kind = "lego", "lego"

    def __init__(self, host=None, port=None):
        self.host = host or config.LEGO_HOST or docker_host_gateway()
        self.port = port or config.LEGO_PORT
        self.sock = None
        self._rid = 0
        self._pending = {}
        self._lock = threading.Lock()
        self._reports = {}
        self.connected = False
        self.ble = False
        self.last_state = {}
        self.error = None

    # ---- connection ----
    def connect(self, timeout=2.0):
        try:
            s = socket.create_connection((self.host, self.port), timeout=timeout)
            s.settimeout(None)
            self.sock = s
            threading.Thread(target=self._reader, daemon=True, name="lego-rx").start()
            hello = self.request("hello", timeout=3)
            self.ble = bool(hello.get("ble"))
            if not self.ble:
                r = self.request("connect_ble", card=config.LEGO_CARD,
                                 color=config.LEGO_CARD_COLOR, timeout=25)
                self.ble = bool(r.get("ble"))
                self.error = r.get("error")
            self.connected = self.ble
            if self.ble:
                self.latency_ms = float(self.request("calibrate", timeout=10).get("rtt_ms", 0))
            return self.connected
        except (OSError, TimeoutError) as e:
            self.error = f"service unreachable at {self.host}:{self.port} ({e})"
            self.connected = False
            return False

    def _reader(self):
        buf = b""
        try:
            while True:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if msg.get("op") == "report":
                        ev = self._reports.get(msg.get("dance_id"))
                        if ev:
                            ev["data"] = msg
                            ev["event"].set()
                    rid = msg.get("rid")
                    if rid in self._pending:
                        self._pending[rid]["data"] = msg
                        self._pending[rid]["event"].set()
        except OSError:
            pass
        self.connected = False

    def request(self, op, timeout=5.0, **kw):
        if not self.sock:
            raise OSError("not connected")
        with self._lock:
            self._rid += 1
            rid = self._rid
            slot = {"event": threading.Event(), "data": None}
            self._pending[rid] = slot
            self.sock.sendall((json.dumps({"op": op, "rid": rid, **kw}) + "\n").encode())
        try:
            if not slot["event"].wait(timeout):
                raise TimeoutError(f"lego service: no reply to {op}")
            return slot["data"]
        finally:
            self._pending.pop(rid, None)

    def send(self, op, **kw):
        if self.sock:
            with self._lock:
                self.sock.sendall((json.dumps({"op": op, **kw}) + "\n").encode())

    # ---- performer ----
    def play(self, timeline, t_start, cancel, on_kf=None):
        dance_id = f"d{int(t_start * 1000)}"
        ev = {"event": threading.Event(), "data": None}
        self._reports[dance_id] = ev
        self.request("timeline", dance_id=dance_id, t_start=t_start, keyframes=timeline)
        t_end = t_start + (timeline[-1]["t"] if timeline else 0) + 3.0
        while not ev["event"].is_set():
            if cancel.is_set():
                self.stop()
                break
            if time.time() > t_end:
                break
            self.send("keepalive")
            ev["event"].wait(0.2)
        self._reports.pop(dance_id, None)
        reps = (ev["data"] or {}).get("keyframes", [])
        for r in reps:
            r["motion"] = r.get("action") in MOTION
            if on_kf:
                on_kf(self.name, r)
        return reps

    def stop(self):
        try:
            self.send("stop")
        except OSError:
            pass

    def state(self):
        if self.connected:
            try:
                self.last_state = self.request("state", timeout=1.0)
            except (OSError, TimeoutError):
                pass
        return super().state() | {"ble": self.ble, "error": self.error,
                                  "motors": self.last_state.get("motors"),
                                  "imu": self.last_state.get("imu")}
