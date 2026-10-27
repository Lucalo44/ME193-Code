# UNO Q - Linux side
# Drains IMU samples from the sketch, then
#   * optionally streams them to the laptop (tools/record.py) for data collection
#   * optionally classifies every HOP_S seconds with the trained model and shows
#     the result on the LED matrix (S / W / R / B).
#
# Run tools/sync_app.py on the laptop first so gaitid/ and models/ sit next to this file.

import os
import socket
import time
from collections import deque

import numpy as np
from arduino.app_utils import App, Bridge

from gaitid import config as C
from gaitid.model import Model

# ---------------- settings ----------------
STREAM_TO = "10.0.0.115"                  # laptop IP, e.g. "192.168.1.50" -> record mode; None = off
MODEL = "model2_full.json"        # or "model1_peak.json" to run the one-axis model
SMOOTH = 3                        # majority vote over the last N predictions
# ------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(HERE, "models", MODEL)
model = Model.load(model_path) if os.path.exists(model_path) else None
print("Model:", MODEL if model else "none (stream only)")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if STREAM_TO else None
buf = deque(maxlen=C.WINDOW)
votes = deque(maxlen=SMOOTH)
since_pred = 0
last_shown = None


def loop():
    global since_pred, last_shown
    if not getattr(loop, "scanned", False):   # DEBUG: one-time I2C scan
        loop.scanned = True
        print("I2C scan:", Bridge.call("i2c_scan"))
    try:
        chunk = Bridge.call("get_imu")
    except Exception as e:
        print("get_imu failed:", e)
        time.sleep(0.5)
        return
    rows = [r for r in (chunk or "").split(";") if r]

    if sock and rows:
        text = "\n".join(
            ",".join([f"{float(r.split(',')[0]) / 1000:.3f}"] + r.split(",")[1:]) for r in rows)
        sock.sendto(text.encode(), (STREAM_TO, C.UDP_PORT))

    for r in rows:
        buf.append([float(v) for v in r.split(",")[1:]])
        since_pred += 1
        if model and len(buf) == C.WINDOW and since_pred >= C.HOP:
            since_pred = 0
            votes.append(model.predict(np.array(buf)))
            label = max(set(votes), key=votes.count)
            if label != last_shown:
                last_shown = label
                print("Activity:", label)
                Bridge.call("show", C.LETTERS[label])
    time.sleep(0.2)


App.run(user_loop=loop)
