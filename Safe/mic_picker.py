"""Mic picker window: every input device with a live level meter, click one to use it.

    python3 mic_picker.py         # pick and save without collecting anything

Talk and watch which bar moves. The choice is saved by name to mic.json (names survive
devices being plugged/unplugged, indices don't) and reused by collect_voice.py and
live_voice.py, so training and use hear the same mic.
"""
import json

import cv2
import numpy as np

import config as C
from audio import draw_meter

ROW_H, W = 46, 640
FONT = cv2.FONT_HERSHEY_SIMPLEX


def input_devices():
    import sounddevice as sd
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def saved_mic():
    """Index of the saved mic in the current device list, or None."""
    try:
        name = json.loads(C.MIC_FILE.read_text())["name"]
    except (OSError, ValueError, KeyError):
        return None
    return next((i for i, n in input_devices() if n == name), None)


def auto_mic():
    """No window (UNO Q): the saved mic if its name still matches (Linux adds suffixes like
    ': USB Audio (hw:1,0)'), else the first USB/camera mic, else the default input."""
    devices = input_devices()
    if not devices:
        return None
    try:
        saved = json.loads(C.MIC_FILE.read_text())["name"].lower()
    except (OSError, ValueError, KeyError):
        saved = ""
    for i, n in devices:
        if saved and (saved in n.lower() or n.lower().split(":")[0].strip() in saved):
            return i
    for i, n in devices:
        if any(k in n.lower() for k in ("usb", "camera", "webcam")):
            return i
    import sounddevice as sd
    default = sd.default.device[0]
    return default if default is not None and default >= 0 else devices[0][0]


def save_mic(index, name):
    C.MIC_FILE.write_text(json.dumps({"index": index, "name": name}))


def _open_all():
    from audio import MicStream
    streams = {}
    for i, name in input_devices():
        try:
            streams[i] = (name, MicStream(i, seconds=0.5))
        except Exception as e:                      # busy or virtual devices can refuse
            print(f"  skipping [{i}] {name}: {e}")
    return streams


def pick_mic():
    """Shows the picker and returns the chosen device index (also saved), or None."""
    win = "Select microphone"
    current = saved_mic()
    print("Opening microphones... (macOS may ask for microphone access the first time)")
    streams = _open_all()
    choice, hover = None, None

    def on_mouse(ev, x, y, *_):
        nonlocal choice, hover
        k = (y - 50) // ROW_H if y >= 50 else -1
        idx = list(streams)[k] if 0 <= k < len(streams) else None
        hover = idx
        if ev == cv2.EVENT_LBUTTONDOWN and idx is not None:
            choice = idx

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while choice is None:
        canvas = np.full((60 + max(1, len(streams)) * ROW_H, W, 3), 20, np.uint8)
        msg = ("Talk, then click the mic whose bar moves (or press its number).  r = rescan, q = cancel"
               if streams else "No microphones found. Plug one in, then press r.  q = cancel")
        cv2.putText(canvas, msg, (10, 30), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for n, (i, (name, mic)) in enumerate(streams.items()):
            y = 50 + n * ROW_H
            if hover == i:
                cv2.rectangle(canvas, (2, y), (W - 3, y + ROW_H - 4), (0, 200, 255), 2)
            label = f"[{i}] {name}" + ("  (saved)" if i == current else "")
            cv2.putText(canvas, label[:48], (10, y + 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            level = mic.level_db(0.05)
            draw_meter(canvas, 10, y + 26, W - 90, 10, level, C.TRIGGER_DB)
            cv2.putText(canvas, f"{level:5.0f} dB", (W - 75, y + 36), FONT, 0.4, (200, 200, 200), 1)
        cv2.imshow(win, canvas)

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            for _, mic in streams.values():
                mic.close()
            streams = _open_all()
        elif ord("0") <= k <= ord("9") and k - ord("0") in streams:
            choice = k - ord("0")

    for _, mic in streams.values():
        mic.close()
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    if choice is not None:
        name = dict(input_devices())[choice]
        save_mic(choice, name)
        print(f"Using mic [{choice}] {name} (saved to {C.MIC_FILE.name})")
    return choice


def resolve_mic(index=None, force_pick=False):
    """--mic N wins; else the saved mic if it's still plugged in; else open the picker."""
    if index is not None:
        return index
    if not force_pick:
        saved = saved_mic()
        if saved is not None:
            return saved
    choice = pick_mic()
    if choice is None:
        raise SystemExit("No microphone selected.")
    return choice


if __name__ == "__main__":
    pick_mic()
