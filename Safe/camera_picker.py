"""Camera picker window: a live preview of every camera found, click one to use it.

    python3 camera_picker.py      # pick and save without collecting anything

OpenCV can't read camera names on macOS, only indices, and the order shifts when a USB camera
or an iPhone (Continuity Camera) is plugged in. Seeing each feed is the only reliable way to
know which index is the USB webcam. The choice is saved to camera.json and reused by
collect_data.py and run_safe.py, so training and the safe always use the same camera.
"""
import json

import cv2
import numpy as np

import config as C

TILE_W, TILE_H, COLS = 320, 240, 3
FONT = cv2.FONT_HERSHEY_SIMPLEX

try:                                            # hide OpenCV's "out of bound" spam while probing
    cv2.setLogLevel(0)
except AttributeError:
    pass


def saved_camera():
    try:
        return int(json.loads(C.CAMERA_FILE.read_text())["index"])
    except (OSError, ValueError, KeyError):
        return None


def save_camera(index):
    C.CAMERA_FILE.write_text(json.dumps({"index": index}))


def probe(max_index):
    """Open every index that actually returns a frame."""
    caps = {}
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
            caps[i] = cap
        else:
            cap.release()
    return caps


def first_working_camera(max_index=10):
    """Lowest index that actually returns a frame (for the headless UNO Q, where there is no
    picker window and video device numbers don't match the laptop's)."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return i
    return None


def _tile(frame, index, hover, current):
    t = np.zeros((TILE_H, TILE_W, 3), np.uint8)
    if frame is not None:
        t[:] = cv2.resize(frame, (TILE_W, TILE_H))
    cv2.rectangle(t, (0, 0), (TILE_W - 1, TILE_H - 1), (0, 200, 255) if hover else (80, 80, 80), 4 if hover else 1)
    label = f"[{index}] camera {index}" + ("  (saved)" if index == current else "")
    cv2.rectangle(t, (0, 0), (TILE_W, 30), (0, 0, 0), -1)
    cv2.putText(t, label, (8, 21), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def pick_camera(max_index=8):
    """Shows the picker and returns the chosen index (also saved). Returns None if cancelled."""
    win = "Select camera"
    current = saved_camera()
    print("Looking for cameras...")
    caps = probe(max_index)
    choice, hover = None, None

    def on_mouse(ev, x, y, *_):
        nonlocal choice, hover
        k = (y - 50) // TILE_H * COLS + x // TILE_W if y >= 50 else -1
        idx = list(caps)[k] if 0 <= k < len(caps) else None
        hover = idx
        if ev == cv2.EVENT_LBUTTONDOWN and idx is not None:
            choice = idx

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while choice is None:
        indices = list(caps)
        rows = max(1, -(-len(indices) // COLS))
        canvas = np.full((50 + rows * TILE_H, COLS * TILE_W, 3), 20, np.uint8)
        msg = ("Click the USB webcam (or press its number).  r = rescan, q = cancel" if indices
               else "No cameras found. Plug in the webcam, then press r.  q = cancel")
        cv2.putText(canvas, msg, (10, 32), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        for n, i in enumerate(indices):
            ok, frame = caps[i].read()
            r, c = divmod(n, COLS)
            canvas[50 + r * TILE_H:50 + (r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = \
                _tile(frame if ok else None, i, hover == i, current)
        cv2.imshow(win, canvas)

        k = cv2.waitKey(30) & 0xFF
        if k == ord("q") or k == 27:
            break
        if k == ord("r"):
            for cap in caps.values():
                cap.release()
            caps = probe(max_index)
        elif ord("0") <= k <= ord("9") and k - ord("0") in caps:
            choice = k - ord("0")

    for cap in caps.values():
        cap.release()
    cv2.destroyWindow(win)
    cv2.waitKey(1)
    if choice is not None:
        save_camera(choice)
        print(f"Using camera {choice} (saved to {C.CAMERA_FILE.name})")
    return choice


if __name__ == "__main__":
    pick_camera()
