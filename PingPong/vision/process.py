"""
vision/process.py -- run the vision pipeline (camera, AprilTags, MediaPipe pose,
stroke classifier, arm tracking) in its own process.

The game sees a VisionProcess with the same interface as the in-process
VisionWorker (camera.status, tags_active, tag_lead/tag_progress, confirmed_tags,
pose_detected, prediction, judge, stroke_check_enabled, fps), plus `arm`, an
ArmTracker fed with every frame. The child process does all the camera and
MediaPipe work on its own CPU core, so it can never stall the game loop or the
paddle's Bluetooth handling, whatever machine it runs on.

time.monotonic() is the same clock in both processes on macOS, so frame
timestamps line up with the paddle's swing times.

    parent                                      child (vision_child)
    ------                                      --------------------
    control queue  -- "tags:1" / "tags:0" / "stop" -->
                   <-- {"type": "hello" | "frame" | "tag"} -- results queue
                   <-- jpeg_b64 --                    previews queue (latest wins)
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import threading
import time
from types import SimpleNamespace
from typing import Callable, Optional

import config as C
from vision.arm import ArmTracker
from vision.pose import StrokeJudge


def _put_latest(q, item) -> None:
    """Non-blocking put that drops the item if the consumer is behind."""
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


# --------------------------------------------------------------------------
# Child process
# --------------------------------------------------------------------------

def _orphaned(parent_pid: Optional[int]) -> bool:
    """True once the game process is gone (killed without a clean stop), so the
    child never lingers in the background holding the camera."""
    return parent_pid is not None and os.getppid() != parent_pid


def vision_child(camera_index: int, pose_data: Optional[str], results, previews, control,
                 source: str = "camera", parent_pid: Optional[int] = None) -> None:
    if source == "synthetic":
        return _synthetic_child(results, previews, control, parent_pid)
    from vision.camera import Camera
    from vision.pose import PoseClassifier
    from vision.worker import VisionWorker

    camera = Camera(camera_index)
    if not camera.ok:
        results.put({"type": "hello", "camera": camera.status, "stroke_check": False})
        return
    classifier = PoseClassifier()
    if pose_data:
        classifier.load(pose_data)
    worker = VisionWorker(camera, classifier,
                          on_preview=lambda b64: _put_latest(previews, b64),
                          on_frame=lambda d: _put_latest(results, d))
    results.put({"type": "hello", "camera": camera.status, "stroke_check": worker.stroke_check_enabled})
    worker.start()
    try:
        while True:
            try:
                cmd = control.get(timeout=0.05)
            except queue.Empty:
                cmd = None
            if cmd == "stop" or _orphaned(parent_pid):
                break
            if cmd in ("tags:1", "tags:0"):
                worker.tags_active = cmd == "tags:1"
            while not worker.confirmed_tags.empty():
                results.put({"type": "tag", "id": worker.confirmed_tags.get_nowait()})
    finally:
        worker.stop()


def _synthetic_child(results, previews, control, parent_pid: Optional[int] = None) -> None:
    """Test source: a scripted 'player' at 30 fps -- no camera, no MediaPipe.
    The right-hand side arm (on screen) swings its wrist out and back."""
    results.put({"type": "hello", "camera": "ok", "stroke_check": True})
    t0 = time.monotonic()
    tags_on = False
    n = 0
    while True:
        try:
            cmd = control.get(timeout=1 / 30)
        except queue.Empty:
            cmd = None
        if cmd == "stop" or _orphaned(parent_pid):
            return
        if cmd in ("tags:1", "tags:0"):
            tags_on = cmd == "tags:1"
        t = time.monotonic()
        swing = 0.5 * math.sin((t - t0) * 2 * math.pi)                  # wrist out and back, 1 Hz
        arms = [{"shoulder": [-0.4, 1.0], "elbow": [-0.5, 0.6], "wrist": [-0.5, 0.3], "vis": 0.9},
                {"shoulder": [0.4, 1.0], "elbow": [0.6, 0.7], "wrist": [0.6 + swing, 0.5], "vis": 0.9}]
        _put_latest(results, {"type": "frame", "t": t, "label": "forehand", "conf": 1.0, "pose": True,
                              "arms": {"arms": arms}, "tag_lead": 1 if tags_on else None,
                              "tag_progress": min(1.0, n / 8) if tags_on else 0.0, "fps": 30.0})
        if tags_on:
            n += 1
            if n == 8:
                results.put({"type": "tag", "id": 1})
        if n % 3 == 0:
            _put_latest(previews, "c3ludGhldGlj")


# --------------------------------------------------------------------------
# Parent side
# --------------------------------------------------------------------------

class VisionProcess:
    def __init__(self, camera_index: int, pose_data: Optional[str],
                 on_preview: Optional[Callable[[str], None]] = None, source: str = "camera"):
        ctx = mp.get_context("spawn")
        self._results = ctx.Queue(maxsize=256)
        self._previews = ctx.Queue(maxsize=2)
        self._control = ctx.Queue()
        self._proc = ctx.Process(target=vision_child, name="vision",
                                 args=(camera_index, pose_data, self._results, self._previews, self._control, source,
                                       os.getpid()),
                                 daemon=True)
        self.on_preview = on_preview
        self.camera = SimpleNamespace(status="starting")
        self._t_start = 0.0
        self._ready = False
        self.judge = StrokeJudge()
        self.arm = ArmTracker(C.HANDEDNESS)
        self.confirmed_tags: "queue.Queue[int]" = queue.Queue()
        self.tag_lead: Optional[int] = None
        self.tag_progress = 0.0
        self.pose_detected = False
        self.prediction: Optional[str] = None
        self.confidence = 0.0
        self.fps = 0.0
        self._stroke_check = False
        self._tags_active = False
        self._stop = threading.Event()

    # Same interface as VisionWorker ------------------------------------------
    @property
    def stroke_check_enabled(self) -> bool:
        return self._stroke_check

    @property
    def tags_active(self) -> bool:
        return self._tags_active

    @tags_active.setter
    def tags_active(self, on: bool) -> None:
        if on != self._tags_active:
            self._tags_active = on
            self._control.put("tags:1" if on else "tags:0")

    def start(self) -> None:
        self._t_start = time.monotonic()
        self._proc.start()
        threading.Thread(target=self._read_results, daemon=True, name="vision-results").start()
        threading.Thread(target=self._read_previews, daemon=True, name="vision-previews").start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._control.put("stop")
        except Exception:
            pass
        self._proc.join(timeout=3)
        if self._proc.is_alive():
            self._proc.terminate()

    # Plumbing -------------------------------------------------------------
    def _read_results(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._results.get(timeout=0.2)
            except queue.Empty:
                if not self._proc.is_alive() and (self.camera.status == "ok" or not self._ready):
                    self.camera.status = "error: vision process stopped"
                elif not self._ready:
                    self._loading_progress()
                continue
            kind = msg.get("type")
            if kind == "hello":
                self._ready = True
                self.camera.status = msg["camera"]
                print(f"Camera {'ready' if msg['camera'] == 'ok' else msg['camera']} "
                      f"after {time.monotonic() - self._t_start:.0f} s.", flush=True)
                self._stroke_check = bool(msg["stroke_check"])
            elif kind == "tag":
                self.confirmed_tags.put(msg["id"])
            elif kind == "frame":
                self.pose_detected = msg["pose"]
                self.prediction, self.confidence = msg["label"], msg["conf"]
                self.tag_lead, self.tag_progress = msg["tag_lead"], msg["tag_progress"]
                self.fps = msg["fps"]
                self.judge.add(msg["t"], msg["label"], msg["conf"])
                self.arm.update(msg["t"], msg["arms"])

    def _loading_progress(self) -> None:
        """Show that the vision process is still loading (MediaPipe + OpenCV). Usually
        a few seconds; much longer when macOS has to fetch the library files first,
        e.g. from iCloud Drive."""
        waited = int(time.monotonic() - self._t_start)
        self.camera.status = f"loading {waited}s"
        if waited >= 15 and not getattr(self, "_warned", False):
            self._warned = True
            print("Camera still loading after 15 s -- the vision libraries are slow to load "
                  "(see README: 'Slow first start'). The game works meanwhile; the camera "
                  "joins when it's ready.", flush=True)

    def _read_previews(self) -> None:
        while not self._stop.is_set():
            try:
                b64 = self._previews.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.on_preview:
                self.on_preview(b64)
