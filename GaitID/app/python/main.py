# UNO Q - Linux side
# Drains IMU samples from the sketch, then, depending on the settings below:
#   * streams them to the laptop (tools/record.py, stepcount/tools/*) for data collection
#   * logs them to logs/ on the board, so a walk needs no laptop (stepcount/tools/pull_logs.py
#     downloads them later)
#   * counts steps on the board with the numpy StepNet (MPU only) and shows the count
#   * classifies the activity every HOP_S seconds and shows S / W / R / B
#
# Run tools/sync_app.py on the laptop first so gaitid/, stepcount_board.py and models/ sit
# next to this file.

import json
import os
import socket
import time
from collections import deque

import numpy as np
from arduino.app_utils import App, Bridge

from gaitid import config as C
from gaitid.model import Model

# ---------------- settings ----------------
STREAM_TO = "192.168.1.195"       # laptop IP -> stream to the laptop tools; None = off
LOG = True                        # save every sample to logs/ on the board
COUNT_STEPS = True                # count steps on the board (needs models/stepnet_board.npz)
DISPLAY = "steps"                 # LED matrix: "steps" (running count) or "activity" (S/W/R/B)
MODEL = "model2_full.json"        # activity classifier, or "model1_peak.json"
SMOOTH = 3                        # majority vote over the last N activity predictions
# ------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(HERE, "models", MODEL)
model = Model.load(model_path) if os.path.exists(model_path) else None
print("Activity model:", MODEL if model else "none")

counter = None
steps_path = os.path.join(HERE, "models", "stepnet_board.npz")
if COUNT_STEPS and os.path.exists(steps_path):
    from stepcount_board import BoardStepNet, StepCounter
    counter = StepCounter(BoardStepNet(steps_path))
print("Step counter:", "on (MPU only)" if counter else "off")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if STREAM_TO else None
buf = deque(maxlen=C.WINDOW)
votes = deque(maxlen=SMOOTH)
since_pred = 0
last_shown = None


class SessionLog:
    """One CSV per power-on (logs/session_NNNN_<date>.csv) plus a small summary JSON that is
    rewritten every 10 s, so pulling the plug loses at most a few seconds."""

    def __init__(self):
        folder = os.path.join(HERE, "logs")
        os.makedirs(folder, exist_ok=True)
        n = len([f for f in os.listdir(folder) if f.endswith(".csv")]) + 1
        # session number first: the board clock can be off without Wi-Fi, the number can't
        self.name = f"session_{n:04d}_{time.strftime('%Y%m%d-%H%M')}"
        self.csv = open(os.path.join(folder, self.name + ".csv"), "w")
        self.csv.write("t,ax,ay,az,gx,gy,gz\n")
        self.summary_path = os.path.join(folder, self.name + ".json")
        self.started = time.strftime("%Y-%m-%d %H:%M:%S")
        self.samples, self.t0, self.last_save = 0, None, time.monotonic()
        print("Logging to", self.name + ".csv")

    def write(self, t, vals):
        if self.t0 is None:
            self.t0 = t
        self.csv.write(f"{t:.3f}," + ",".join(vals) + "\n")
        self.samples += 1

    def save(self, steps, force=False):
        if not force and time.monotonic() - self.last_save < 10:
            return
        self.last_save = time.monotonic()
        self.csv.flush()
        os.fsync(self.csv.fileno())
        with open(self.summary_path, "w") as f:
            json.dump({"session": self.name, "started": self.started, "samples": self.samples,
                       "seconds": round(self.samples / C.FS, 1),
                       "steps_on_board": None if steps is None else round(steps, 1)}, f)


log = SessionLog() if LOG else None


def display(fn, text):
    """LED matrix update that can never stop logging or counting."""
    try:
        Bridge.call(fn, text)
    except Exception as e:
        print(f"{fn} failed:", e)


def count_text(n):
    """Fits 3 characters of the 4x6 font: 0-999, then 1k2 = 1200 steps, up to 99k."""
    n = int(round(n))
    return str(n) if n < 1000 else f"{n // 1000}k{(n % 1000) // 100}" if n < 10000 else f"{n // 1000}k"


def loop():
    global since_pred, last_shown
    if not getattr(loop, "scanned", False):   # DEBUG: one-time I2C scan
        loop.scanned = True
        try:
            print("I2C scan:", Bridge.call("i2c_scan"))
        except Exception as e:
            print("I2C scan failed:", e)
        if counter and DISPLAY == "steps":
            display("show_count", "0")
    try:
        chunk = Bridge.call("get_imu")
    except Exception as e:
        print("get_imu failed:", e)
        time.sleep(0.5)
        return
    rows = [r.split(",") for r in (chunk or "").split(";") if r]

    if sock and rows:
        text = "\n".join(",".join([f"{float(r[0]) / 1000:.3f}"] + r[1:]) for r in rows)
        try:
            sock.sendto(text.encode(), (STREAM_TO, C.UDP_PORT))
        except OSError:
            pass                      # no network (out on a walk): logging carries on

    for r in rows:
        vals = [float(v) for v in r[1:]]
        if log:
            log.write(float(r[0]) / 1000, r[1:])
        if counter:
            counter.add(vals)
        buf.append(vals)
        since_pred += 1
        if model and len(buf) == C.WINDOW and since_pred >= C.HOP:
            since_pred = 0
            votes.append(model.predict(np.array(buf)))
            label = max(set(votes), key=votes.count)
            if label != last_shown:
                last_shown = label
                print("Activity:", label)
                if DISPLAY == "activity":
                    display("show", C.LETTERS[label])

    if counter and rows and counter.n - counter.committed >= int(1.5 * C.FS) + C.FS:
        before = int(round(counter.total))
        counter.update()                                  # about once a second
        print(f"PROBE update n={counter.n} total={counter.total:.2f}", flush=True)
        if DISPLAY == "steps" and int(round(counter.total)) != before:
            t0 = time.monotonic()
            print(f"PROBE show_count {count_text(counter.total)!r} ...", flush=True)
            display("show_count", count_text(counter.total))
            print(f"PROBE show_count returned in {time.monotonic() - t0:.2f}s", flush=True)
    if log:
        log.save(counter.total if counter else None)
    time.sleep(0.2)


App.run(user_loop=loop)
