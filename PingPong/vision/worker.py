"""
vision/worker.py -- the vision thread: camera frame -> AprilTags (unflipped)
-> mirror -> pose landmarks -> KNN prediction -> StrokeJudge, plus a small
annotated JPEG preview for the browser's picture-in-picture.

Results are published as plain attributes (read by the game loop) and a
queue of confirmed tag ids.
"""

from __future__ import annotations

import base64
import queue
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

import config as C
from vision.apriltags import TagDebouncer, TagDetector
from vision.camera import Camera
from vision.pose import PoseClassifier, StrokeJudge, create_landmarker, extract_features

# Upper-body skeleton for the preview drawing (MediaPipe landmark indices).
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (0, 11), (0, 12)]


class VisionWorker:
    def __init__(self, camera: Camera, classifier: Optional[PoseClassifier],
                 on_preview: Optional[Callable[[str], None]] = None):
        self.camera = camera
        self.classifier = classifier
        self.judge = StrokeJudge()
        self.on_preview = on_preview
        self.tags_active = False
        self.tag_lead: Optional[int] = None
        self.tag_progress = 0.0
        self.confirmed_tags: "queue.Queue[int]" = queue.Queue()
        self.pose_detected = False
        self.prediction: Optional[str] = None
        self.confidence = 0.0
        self.fps = 0.0
        self._detector = TagDetector()
        self._debounce = TagDebouncer()
        self._landmarker = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_preview = 0.0
        self._last_ts_ms = -1
        try:
            self._landmarker = create_landmarker()
        except Exception as exc:  # noqa: BLE001 -- keep tags working even without pose
            print(f"Pose landmarker unavailable ({exc}); stroke check disabled.")

    @property
    def stroke_check_enabled(self) -> bool:
        return self._landmarker is not None and self.classifier is not None and self.classifier.trained

    def start(self) -> None:
        self.camera.start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="vision")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.camera.stop()
        if self._landmarker is not None:
            self._landmarker.close()

    def _run(self) -> None:
        import mediapipe as mp

        frame_id = 0
        last = time.monotonic()
        while not self._stop.is_set():
            frame, t_frame, new_id = self.camera.wait_frame(frame_id)
            if frame is None or new_id == frame_id:
                continue
            frame_id = new_id

            # --- AprilTags on the UNFLIPPED frame ---
            tags = []
            if self.tags_active:
                tags = self._detector.detect(frame)
                lead, progress, confirmed = self._debounce.update([i for i, _ in tags])
                self.tag_lead, self.tag_progress = lead, progress
                if confirmed is not None:
                    self.confirmed_tags.put(confirmed)
            elif self.tag_progress:
                self._debounce.reset()
                self.tag_lead, self.tag_progress = None, 0.0

            # --- pose on the mirrored frame (same as training) ---
            view = cv2.flip(frame, 1)
            landmarks = None
            if self._landmarker is not None:
                rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                ts_ms = max(self._last_ts_ms + 1, int(t_frame * 1000))
                self._last_ts_ms = ts_ms
                result = self._landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
                landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            self.pose_detected = landmarks is not None
            label, conf = None, 0.0
            if landmarks is not None and self.classifier is not None:
                feat = extract_features(landmarks)
                if feat is not None:
                    label, conf = self.classifier.predict(feat)
            self.prediction, self.confidence = label, conf
            self.judge.add(t_frame - C.POSE_FRAME_LATENCY_S, label, conf)

            now = time.monotonic()
            dt, last = now - last, now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 / dt
            if self.on_preview and now - self._last_preview >= 1.0 / C.CAMERA_PREVIEW_FPS:
                self._last_preview = now
                self.on_preview(self._preview(view, landmarks, tags, label, conf))

    def _preview(self, view: np.ndarray, landmarks, tags, label, conf) -> str:
        h, w = view.shape[:2]
        if landmarks is not None:
            pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
            for a, b in SKELETON:
                cv2.line(view, pts[a], pts[b], (80, 255, 200), 2, cv2.LINE_AA)
            for i in {i for pair in SKELETON for i in pair}:
                cv2.circle(view, pts[i], 4, (255, 255, 255), -1, cv2.LINE_AA)
        for tag_id, corners in tags:
            # Tags were found on the unflipped frame: mirror x to draw them.
            pts = np.array([[w - 1 - x, y] for x, y in corners], dtype=np.int32)
            cv2.polylines(view, [pts], True, (0, 200, 255), 3, cv2.LINE_AA)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(view, f"{tag_id}: {C.TAG_IDS.get(tag_id, '?')}", (int(cx) - 40, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        if label:
            cv2.putText(view, f"{label} {conf * 100:.0f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        small = cv2.resize(view, (320, int(320 * h / w)))
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else ""
