#!/usr/bin/env python3
"""Host service: owns the LEGO BLE link (plays dance timelines locally) and
answers `sysinfo` (host facts for the planner's self_report skill).

Why a separate process: App Lab runs the app in a Docker container with no
D-Bus socket, so bleak/BlueZ can't be reached from inside it (docs/SPIKES.md).
This runs on the UNO Q's Linux HOST in its own venv and the app talks to it
over TCP on the Docker bridge.

    # Option A, on the board host (tools/deploy.py --lego does this):
    /home/arduino/gc-lego-venv/bin/python lego_server.py
    # Option B, laptop relay (motor paired to the laptop instead). Both clocks
    # must be NTP-synced, and the app needs GC_LEGO_HOST=<laptop IP>:
    GC_LEGO_PEERS=192.168. venv/bin/python lego_server.py

Protocol: JSON lines. Requests carry "rid", replies echo it.
  hello                         -> {ble, device}
  connect_ble {card, color}     -> {ble, error}
  calibrate                     -> {rtt_ms}       median of 15 acked RPCs
  timeline {dance_id, t_start, keyframes}  -> {ok}; later {"op":"report", dance_id, keyframes:[...]}
  keepalive                     (no reply; deadman: motors stop if none for 0.5 s during a dance)
  stop                          -> {ok}
  state                         -> {motors, imu, offset_ms}

Timing: keyframe times are wall-clock (time.time()). The host and the app
container share one kernel clock. Each keyframe is fired early by the BLE
one-way latency (rtt/2) plus an adaptive offset learned from the previous
dance's measured motion delay, so sync should improve from dance 1 to dance 2.
"""
import argparse
import json
import os
import socket
import socketserver
import statistics
import threading
import time

try:
    import legoeducation as le
except ImportError:          # lets the file be imported for tests without the package
    le = None

PORT = 8770
DEADMAN_S = 0.5
MOTION_DEG = 3               # position change that counts as "motion started"
MOTION_WAIT_S = 0.4
B_SIGN = int(os.environ.get("GC_LEGO_B_SIGN", "-1"))   # B arm mounted mirrored
ZERO_A = int(os.environ.get("GC_LEGO_ZERO_A", "0"))    # absolute position of "arm down"
ZERO_B = int(os.environ.get("GC_LEGO_ZERO_B", "0"))
ALLOWED_PEERS = tuple(os.environ.get("GC_LEGO_PEERS", "127.,172.").split(","))  # localhost + Docker bridges


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Lego:
    def __init__(self):
        self.dev = None
        self.lock = threading.Lock()
        self.rtt_ms = 40.0
        self.offset_ms = 0.0          # learned extra lead (motor spin-up etc.)
        self.error = None

    @property
    def ble(self):
        return bool(self.dev and self.dev.connected)

    def connect(self, card="0998", color="purple"):
        if le is None:
            self.error = "legoeducation not installed"
            return False
        colors = {"red": le.LEGO_COLOR_RED, "yellow": le.LEGO_COLOR_YELLOW, "blue": le.LEGO_COLOR_BLUE,
                  "green": le.LEGO_COLOR_GREEN, "purple": le.LEGO_COLOR_PURPLE,
                  "orange": le.LEGO_COLOR_ORANGE, "teal": le.LEGO_COLOR_TEAL, "white": le.LEGO_COLOR_WHITE}
        dev = le.DoubleMotor()
        for attempt in range(5):
            try:
                dev.connect(card_serial=card, card_color=colors.get(color, le.LEGO_COLOR_PURPLE),
                            device_notification_delay=20)
                break
            except Exception as e:  # noqa: BLE001
                self.error = repr(e)
                if "not ready" in str(e).lower() and attempt < 4:
                    time.sleep(1)
                    continue
                break
        if dev.connected:
            self.dev, self.error = dev, None
            log(f"BLE connected: Double Motor card {card} {color}")
            return True
        self.error = self.error or f"no Double Motor with card {card} found (on? card tapped? not paired elsewhere?)"
        log("BLE connect failed:", self.error)
        return False

    def calibrate(self, n=15):
        rtts = []
        with self.lock:
            for _ in range(n):
                t = time.perf_counter()
                self.dev.motor_set_speed(50, motor=le.MOTOR_BOTH, blocking=True)
                rtts.append((time.perf_counter() - t) * 1000)
        self.rtt_ms = statistics.median(rtts)
        log(f"calibrated: BLE rtt median {self.rtt_ms:.1f} ms (min {min(rtts):.0f}, max {max(rtts):.0f})")
        return self.rtt_ms

    def positions(self):
        m = self.dev.motor
        return [m[0].absolutePosition, m[1].absolutePosition]

    def execute(self, kf):
        """Issue one keyframe without waiting for the motion to finish."""
        p, a = kf["params"], kf["action"]
        d = self.dev
        with self.lock:
            if a == "pose":
                pa = (ZERO_A + int(p["a"])) % 360
                pb = (ZERO_B + B_SIGN * int(p["b"])) % 360
                sp = int(p.get("speed", 60))
                d.begin_batch()
                d.motor_run_to_absolute_position(pa, motor=le.MOTOR_LEFT, speed=sp, blocking=False)
                d.motor_run_to_absolute_position(pb, motor=le.MOTOR_RIGHT, speed=sp, blocking=False)
                d.end_batch(blocking=False)
            elif a == "run":
                d.begin_batch()
                for motor, v in ((le.MOTOR_LEFT, int(p["a"])), (le.MOTOR_RIGHT, B_SIGN * int(p["b"]))):
                    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if v >= 0
                                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
                    d.motor_run(direction=direction, motor=motor, speed=abs(v), blocking=False)
                d.end_batch(blocking=False)
            elif a == "tank":
                d.movement_move_tank(int(p["l"]), int(p["r"]), blocking=False)
            elif a == "stop":
                d.motor_stop(motor=le.MOTOR_BOTH, blocking=False)

    def stop(self):
        if self.ble:
            with self.lock:
                try:
                    self.dev.motor_stop(motor=le.MOTOR_BOTH, blocking=False)
                except Exception as e:  # noqa: BLE001
                    log("stop failed:", repr(e))

    def state(self):
        if not self.ble:
            return {"ble": False, "error": self.error}
        m, imu = self.dev.motor, self.dev.imu_device
        return {"ble": True, "offset_ms": round(self.offset_ms, 1), "rtt_ms": round(self.rtt_ms, 1),
                "motors": [{"abs": x.absolutePosition, "speed": x.speed, "state": x.motorState} for x in m],
                "imu": {"yaw": imu.yaw, "pitch": imu.pitch, "roll": imu.roll}}


class Player:
    """Plays one timeline at a time; deadman-protected."""

    def __init__(self, lego, send):
        self.lego, self.send = lego, send
        self.cancel = threading.Event()
        self.last_alive = time.time()
        self.thread = None

    def keepalive(self):
        self.last_alive = time.time()

    def start(self, dance_id, t_start, kfs):
        self.stop_play()
        self.cancel = threading.Event()
        self.last_alive = time.time()
        self.thread = threading.Thread(target=self._run, args=(dance_id, t_start, kfs, self.cancel),
                                       daemon=True)
        self.thread.start()

    def stop_play(self):
        self.cancel.set()
        self.lego.stop()

    def _wait_until(self, t, cancel):
        while True:
            now = time.time()
            if now - self.last_alive > DEADMAN_S:
                log("deadman: no keepalive, stopping")
                self.lego.stop()
                return False
            if t - now <= 0:
                return True
            if cancel.wait(min(t - now, 0.01)):
                return False

    def _motion_time(self, before, t_sent):
        """First time either output has moved >= MOTION_DEG, from 20 ms notifications."""
        end = t_sent + MOTION_WAIT_S
        while time.time() < end:
            now = self.lego.positions()
            if any(abs(((n - b) + 180) % 360 - 180) >= MOTION_DEG for n, b in zip(now, before)):
                return time.time()
            time.sleep(0.004)
        return None

    def _run(self, dance_id, t_start, kfs, cancel):
        lead = (self.lego.rtt_ms / 2 + self.lego.offset_ms) / 1000
        reps, motion_delays = [], []
        log(f"dance {dance_id}: {len(kfs)} keyframes, lead {lead * 1000:.0f} ms")
        for kf in kfs:
            t_cmd = t_start + kf["t"]
            if not self._wait_until(t_cmd - lead, cancel):
                break
            before = self.lego.positions()
            t_sent = time.time()
            try:
                self.lego.execute(kf)
            except Exception as e:  # noqa: BLE001
                log("execute failed:", repr(e))
                continue
            t_act = None
            if kf["action"] in ("pose", "run", "tank"):
                t_act = self._motion_time(before, t_sent)
                if t_act is not None:
                    motion_delays.append((t_act - t_sent) * 1000)
            if t_act is None:
                t_act = t_sent + self.lego.rtt_ms / 2000
            reps.append({"i": kf["i"], "beat": kf["beat"], "action": kf["action"],
                         "t_cmd": t_cmd, "t_sent": t_sent, "t_actual": t_act})
        if motion_delays:
            # Learn: next time, lead by the median observed command->motion delay.
            target = statistics.median(motion_delays) - self.lego.rtt_ms / 2
            self.lego.offset_ms = 0.5 * self.lego.offset_ms + 0.5 * max(0.0, target)
        if cancel.is_set():
            self.lego.stop()
        self.send({"op": "report", "dance_id": dance_id, "keyframes": reps,
                   "offset_ms": round(self.lego.offset_ms, 1)})
        log(f"dance {dance_id} done: {len(reps)}/{len(kfs)} keyframes, new offset {self.lego.offset_ms:.0f} ms")


LEGO = Lego()


def _sh(cmd, timeout=5):
    import subprocess
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


APPS_DIR = "/home/arduino/ArduinoApps"


def apps():
    """Installed App Lab apps: folder, name, description, README first paragraph, running?"""
    import glob
    import re
    running = {n[: -len("-main-1")] for n in _sh("docker ps --format '{{.Names}}'").splitlines()
               if n.endswith("-main-1")}
    out = []
    for y in sorted(glob.glob(f"{APPS_DIR}/*/app.yaml")):
        folder = os.path.basename(os.path.dirname(y))
        text = open(y).read()
        name = re.search(r"^name:\s*(.+)$", text, re.M)
        desc = re.search(r"^description:\s*(.+)$", text, re.M)
        readme = ""
        for r in ("README.md", "readme.md"):
            p = os.path.join(os.path.dirname(y), r)
            if os.path.exists(p):
                paras = [b.strip() for b in open(p).read().split("\n\n")
                         if b.strip() and not b.strip().startswith(("#", "```", "|", "!["))]
                readme = paras[0][:400] if paras else ""
                break
        out.append({"folder": folder,
                    "name": name.group(1).strip().strip('"') if name else folder,
                    "description": desc.group(1).strip().strip('"') if desc else "",
                    "readme": readme, "running": folder in running})
    return out


MODELS_DIR = "/var/lib/arduino-app-cli/models/llamacpp"
HASH_CACHE = os.path.expanduser("~/.gc-model-hashes.json")


def model_hash(model):
    """sha256 of a model's GGUF, for the planner's saved prompt cache key (a restored KV
    cache from different weights would give silently wrong plans). Hashing 1 GB takes
    ~15 s, so results are cached by (size, mtime)."""
    import hashlib
    import json as _json
    rel = os.path.normpath(str(model)).lstrip("/")
    if rel.startswith("..") or not rel:
        return {"error": "bad model"}
    path = os.path.join(MODELS_DIR, rel + ".gguf")
    if not os.path.isfile(path):
        return {"error": f"no model file for {model}"}
    st = os.stat(path)
    stamp = f"{st.st_size}:{int(st.st_mtime)}"
    try:
        cache = _json.load(open(HASH_CACHE))
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(path)
    if hit and hit.get("stamp") == stamp:
        return {"sha256": hit["sha256"], "cached": True}
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    cache[path] = {"stamp": stamp, "sha256": h.hexdigest()}
    try:
        _json.dump(cache, open(HASH_CACHE, "w"))
    except OSError:
        pass
    return {"sha256": h.hexdigest(), "cached": False}


def app_config(folder):
    """Top-level UPPER_CASE constants of an app's python/main.py, read with ast (never
    executed). f-strings come back as their literal text with {...} for the
    expressions, which is enough to read URL query parameters."""
    import ast
    folder = os.path.basename(str(folder))            # no path tricks
    path = os.path.join(APPS_DIR, folder, "python", "main.py")
    if not os.path.exists(path):
        return {"error": f"no app '{folder}'"}
    consts = {}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if not key.isupper():
                continue
            v = node.value
            try:
                consts[key] = ast.literal_eval(v)
                continue
            except (ValueError, SyntaxError):
                pass
            if isinstance(v, ast.JoinedStr):
                consts[key] = "".join(p.value if isinstance(p, ast.Constant) else "{...}" for p in v.values)
    return {"folder": folder, "constants": consts}


def sysinfo():
    """Host-level facts the app container can't see (it has its own OS image and
    only its own app folder). Used by the planner's self_report skill."""
    import glob
    import re
    info = {}
    info["hostname"] = _sh("hostname")
    m = re.search(r'PRETTY_NAME="([^"]+)"', open("/etc/os-release").read())
    info["os"] = m.group(1) if m else ""
    info["kernel"] = _sh("uname -r")
    info["model"] = open("/proc/device-tree/model").read().strip("\x00 \n") if os.path.exists("/proc/device-tree/model") else ""
    info["soc"] = [c for c in open("/proc/device-tree/compatible").read().split("\x00") if c][1:] \
        if os.path.exists("/proc/device-tree/compatible") else []
    info["cpus"] = os.cpu_count()
    mem = dict(re.findall(r"(\w+):\s+(\d+) kB", open("/proc/meminfo").read()))
    info["mem_total_mb"] = int(mem.get("MemTotal", 0)) // 1024
    info["mem_avail_mb"] = int(mem.get("MemAvailable", 0)) // 1024
    st = os.statvfs("/")
    info["disk_free_mb"] = st.f_bavail * st.f_frsize // 2**20
    info["disk_total_mb"] = st.f_blocks * st.f_frsize // 2**20
    info["uptime_s"] = int(float(open("/proc/uptime").read().split()[0]))
    info["load"] = open("/proc/loadavg").read().split()[:3]
    temps = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            temps.append(int(open(z).read()) / 1000)
        except (OSError, ValueError):
            pass
    info["temp_c"] = round(max(temps), 1) if temps else None
    info["ips"] = [l.split()[3].split("/")[0] for l in _sh("ip -4 -o addr show scope global").splitlines()
                   if not l.split()[1].startswith(("docker", "br-", "veth"))]
    apps = []
    for y in sorted(glob.glob("/home/arduino/ArduinoApps/*/app.yaml")):
        m = re.search(r"^name:\s*(.+)$", open(y).read(), re.M)
        apps.append((m.group(1).strip().strip('"') if m else os.path.basename(os.path.dirname(y))))
    info["apps_installed"] = apps
    info["containers"] = _sh("docker ps --format '{{.Names}}'").splitlines()
    info["models"] = re.findall(r"^\[(.+)\]$", open("/var/lib/arduino-app-cli/models/llamacpp/models.ini").read(), re.M) \
        if os.path.exists("/var/lib/arduino-app-cli/models/llamacpp/models.ini") else []
    usb = []
    for d in glob.glob("/sys/bus/usb/devices/*"):
        p = os.path.join(d, "product")
        if os.path.exists(p) and not os.path.basename(d).startswith("usb"):
            man = open(os.path.join(d, "manufacturer")).read().strip() if os.path.exists(os.path.join(d, "manufacturer")) else ""
            usb.append(f"{man} {open(p).read().strip()}".strip())
    info["usb"] = usb
    info["video"] = [open(n).read().strip() for n in glob.glob("/sys/class/video4linux/*/name")]
    info["sound_cards"] = [l.split(":", 1)[1].strip() for l in open("/proc/asound/cards").read().splitlines()
                           if re.match(r"\s*\d+ \[", l)] if os.path.exists("/proc/asound/cards") else []
    info["bluetooth"] = sorted(os.listdir("/sys/class/bluetooth")) if os.path.exists("/sys/class/bluetooth") else []
    info["app_cli"] = _sh("arduino-app-cli version | head -1")
    info["lego_ble"] = LEGO.ble
    return info


class Handler(socketserver.StreamRequestHandler):
    def setup(self):
        super().setup()
        self.wlock = threading.Lock()

    def send(self, msg):
        with self.wlock:
            try:
                self.wfile.write((json.dumps(msg) + "\n").encode())
                self.wfile.flush()
            except OSError:
                pass

    def handle(self):
        peer = self.client_address[0]
        if not peer.startswith(ALLOWED_PEERS):
            log("refused peer", peer)
            return
        log("client", peer)
        player = Player(LEGO, self.send)
        for line in self.rfile:
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            op, rid = req.get("op"), req.get("rid")
            try:
                if op == "keepalive":
                    player.keepalive()
                    continue
                if op == "hello":
                    out = {"ble": LEGO.ble, "device": "DoubleMotor"}
                elif op == "connect_ble":
                    ok = LEGO.ble or LEGO.connect(req.get("card", "0998"), req.get("color", "purple"))
                    out = {"ble": ok, "error": LEGO.error}
                elif op == "calibrate":
                    out = {"rtt_ms": LEGO.calibrate() if LEGO.ble else None}
                elif op == "timeline":
                    if not LEGO.ble:
                        out = {"ok": False, "error": "BLE not connected"}
                    else:
                        player.start(req["dance_id"], float(req["t_start"]), req["keyframes"])
                        out = {"ok": True}
                elif op == "stop":
                    player.stop_play()
                    out = {"ok": True}
                elif op == "state":
                    out = LEGO.state()
                elif op == "sysinfo":
                    out = sysinfo()
                elif op == "apps":
                    out = {"apps": apps()}
                elif op == "app_config":
                    out = app_config(req.get("folder", ""))
                elif op == "restart_runner":
                    # Ground Control's supervisor: the model runner stopped answering.
                    _sh("docker restart ground-control-llamacpp-models-runner-1", timeout=60)
                    out = {"ok": True}
                elif op == "prune_slots":
                    # Keep the planner's current saved prompt cache, delete stale ones (100 MB each).
                    keep = os.path.basename(str(req.get("keep", "")))
                    d = os.path.join(MODELS_DIR, "slots")
                    gone = [f for f in os.listdir(d) if f.startswith("gc_prefix_") and f != keep] if keep else []
                    for f in gone:
                        os.remove(os.path.join(d, f))
                    out = {"removed": gone}
                elif op == "model_hash":
                    out = model_hash(req.get("model", ""))
                else:
                    out = {"error": f"unknown op {op}"}
            except Exception as e:  # noqa: BLE001
                out = {"error": repr(e)}
            out["rid"] = rid
            self.send(out)
        player.stop_play()
        log("client gone; motors stopped")


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--card", default="0998")
    ap.add_argument("--color", default="purple")
    ap.add_argument("--no-autoconnect", action="store_true")
    a = ap.parse_args()
    if not a.no_autoconnect:
        threading.Thread(target=LEGO.connect, args=(a.card, a.color), daemon=True).start()
    with Server(("0.0.0.0", a.port), Handler) as srv:
        log(f"LEGO service on :{a.port} (peers {ALLOWED_PEERS})")
        srv.serve_forever()


if __name__ == "__main__":
    main()
