"""
vision/camera.py -- one capture thread for the single webcam that serves both
AprilTag detection and pose detection. Publishes the latest raw (unflipped)
frame with its time.monotonic() arrival timestamp and a frame counter.

The VideoCapture is opened on the calling (main) thread; only read() runs in
the background thread. Nothing here ever calls cv2.imshow.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 960, height: int = 540):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.status = "ok"
        else:
            self.status = f"error: could not open camera {index}"
        self._frame: Optional[np.ndarray] = None
        self._t = 0.0
        self.frame_id = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fps = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def start(self) -> None:
        if not self.ok or self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                self.status = "error: frame read failed"
                time.sleep(0.05)
                continue
            self.status = "ok"
            with self._cond:
                self._frame, self._t = frame, t
                self.frame_id += 1
                self._cond.notify_all()
            dt = t - last
            last = t
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

    def wait_frame(self, after_id: int, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float, int]:
        """Block until a frame newer than `after_id` arrives."""
        with self._cond:
            self._cond.wait_for(lambda: self.frame_id > after_id or self._stop.is_set(), timeout=timeout)
            return self._frame, self._t, self.frame_id

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.cap.release()
