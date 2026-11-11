"""Camera -> crop -> network input. Shared by collect_data.py, train.py and run_safe.py so the
nets see exactly the same pixels at training time and at the safe."""
import sys
import threading
import time

import cv2
import numpy as np

import config as C

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (C.CREASE_KERNEL, C.CREASE_KERNEL))


def open_camera(index=None):
    """Opens the given index, else the one saved by camera_picker.py, else CAM_INDEX."""
    if index is None:
        from camera_picker import saved_camera
        index = saved_camera()
        index = C.CAM_INDEX if index is None else index
    if sys.platform.startswith("linux"):
        # UNO Q: talk V4L2 directly (OpenCV otherwise picks GStreamer, which queues frames),
        # ask for MJPG so USB bandwidth allows 30 fps at 640x480, and keep only one buffer.
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.FRAME_H)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}. Run camera_picker.py to choose another.")
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.resize(frame, (C.FRAME_W, C.FRAME_H))
    return cv2.flip(frame, 1) if C.MIRROR else frame


class CameraGrabber:
    """Reads the camera on its own thread and keeps only the newest frame.

    Without this, the camera's buffer fills while the nets are busy, and every read hands back
    a frame from a few hundred ms ago, then jumps. With it, the nets and the live view always
    get the current frame, and the camera never waits on anything.
    """

    def __init__(self, cap):
        self.cap, self.frame, self.seq, self.alive = cap, None, 0, True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        fails = 0
        while self.alive:
            frame = read_frame(self.cap)
            if frame is None:
                fails += 1
                if fails > 50:                     # ~ camera unplugged
                    break
                time.sleep(0.02)
                continue
            fails = 0
            with self.cond:
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify_all()
        self.alive = False
        with self.cond:
            self.cond.notify_all()

    def read(self, last_seq, timeout=2.0):
        """Waits for a frame newer than last_seq. Returns (seq, frame), or (seq, None) if the
        camera died."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.alive, timeout=timeout)
            if self.seq == last_seq:
                return last_seq, None
            return self.seq, self.frame

    def latest(self):
        return self.seq, self.frame

    def release(self):
        self.alive = False
        self.thread.join(timeout=1)
        self.cap.release()


def roi_box(h, w):
    side = int(min(h, w) * C.ROI_SIZE)
    cx, cy = int(w * C.ROI_CENTER[0]), int(h * C.ROI_CENTER[1])
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    return x0, y0, side


def crop_roi(frame):
    x0, y0, s = roi_box(*frame.shape[:2])
    return cv2.resize(frame[y0:y0 + s, x0:x0 + s], (C.SAVE_SIZE, C.SAVE_SIZE),
                      interpolation=cv2.INTER_AREA)


def gesture_array(crop):
    """BGR crop -> (3, GESTURE_SIZE, GESTURE_SIZE) float32."""
    img = cv2.resize(crop, (C.GESTURE_SIZE, C.GESTURE_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((img - 0.5) / 0.25).transpose(2, 0, 1)


def crease_maps(crop):
    """BGR crop -> (equalized gray, crease map), both uint8 at PALM_SIZE.

    Creases are thin dark valleys in otherwise smooth skin. CLAHE flattens the lighting, then a
    black-hat (closing minus image) keeps only dark features narrower than the kernel, so the
    broad shading of the hand drops out and the lines are left.
    """
    gray = cv2.cvtColor(cv2.resize(crop, (C.PALM_SIZE, C.PALM_SIZE),
                                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    eq = _clahe.apply(gray)
    lines = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, _kernel)
    lines = cv2.normalize(lines, None, 0, 255, cv2.NORM_MINMAX)
    return eq, lines


def palm_array(crop):
    """BGR crop -> (2, PALM_SIZE, PALM_SIZE) float32: [equalized gray, crease map]."""
    eq, lines = crease_maps(crop)
    x = np.stack([eq, lines]).astype(np.float32) / 255.0
    return (x - 0.5) / 0.25


def crease_preview(crop, size=120):
    """Small BGR image of the crease map, for on-screen debugging."""
    _, lines = crease_maps(crop)
    return cv2.cvtColor(cv2.resize(lines, (size, size)), cv2.COLOR_GRAY2BGR)


TO_ARRAY = {"gesture": gesture_array, "owner": palm_array}
