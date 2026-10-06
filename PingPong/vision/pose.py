"""
vision/pose.py -- forehand / backhand / ready pose classification.

Pipeline copied from Pose Racecar/gesture_car_control.py: MediaPipe Pose
Landmarker (lite) in VIDEO mode, upper-body landmarks normalized to the hips
and torso size, and a live-trained KNN classifier saved as .npz.

Frames are mirrored (selfie view) before pose detection, exactly like the
reference, so MediaPipe's LEFT_* and RIGHT_* landmarks are swapped relative to
the person. That's fine for the KNN as long as training (tools/train_pose.py)
and gameplay both use flipped frames -- they do. The centerline rule uses the
ball's position only, never landmark names.

StrokeJudge keeps ~1 s of (timestamp, label, confidence) predictions and
decides forehand vs backhand at the moment of a swing.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from collections import Counter, deque
from typing import List, Optional, Tuple

import numpy as np

import config as C

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Upper-body landmark indices (MediaPipe PoseLandmark values): nose, shoulders,
# elbows, wrists, hips -- same set as the reference project.
FEATURE_LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 11, 12, 23, 24
FEATURE_DIM = len(FEATURE_LANDMARKS) * 3


def ensure_model(path: str = C.POSE_MODEL_PATH) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading MediaPipe pose model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Download complete.")
    return path


def create_landmarker(model_path: str = C.POSE_MODEL_PATH):
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _xyz(landmarks, index: int) -> np.ndarray:
    p = landmarks[index]
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def extract_features(pose_landmarks) -> Optional[np.ndarray]:
    """Hip-centered, torso-size-normalized feature vector, or None if the
    torso is unusable. Identical to the reference's extract_features()."""
    if not pose_landmarks:
        return None
    mid_hip = (_xyz(pose_landmarks, LEFT_HIP) + _xyz(pose_landmarks, RIGHT_HIP)) / 2.0
    mid_shoulder = (_xyz(pose_landmarks, LEFT_SHOULDER) + _xyz(pose_landmarks, RIGHT_SHOULDER)) / 2.0
    torso_size = float(np.linalg.norm(mid_shoulder - mid_hip))
    if torso_size < 1e-6:
        return None
    feats: List[float] = []
    for idx in FEATURE_LANDMARKS:
        feats.extend(((_xyz(pose_landmarks, idx) - mid_hip) / torso_size).tolist())
    return np.asarray(feats, dtype=np.float64)


class PoseClassifier:
    """Minimal KNN over pose feature vectors (the reference GestureClassifier
    with this project's classes)."""

    def __init__(self, k: int = C.POSE_K, classes: List[str] = C.POSE_CLASSES):
        self.k = k
        self.classes = classes
        self.X = np.zeros((0, FEATURE_DIM), dtype=np.float64)
        self.y: List[str] = []

    def add_sample(self, feat: np.ndarray, label: str) -> None:
        self.X = np.vstack([self.X, feat[None, :]])
        self.y.append(label)

    def counts(self) -> dict:
        c = Counter(self.y)
        return {cls: c.get(cls, 0) for cls in self.classes}

    @property
    def trained(self) -> bool:
        counts = self.counts()
        return counts.get("forehand", 0) > 0 and counts.get("backhand", 0) > 0

    def clear(self) -> None:
        self.X = np.zeros((0, FEATURE_DIM), dtype=np.float64)
        self.y = []

    def predict(self, feat: np.ndarray) -> Tuple[Optional[str], float]:
        n = len(self.y)
        if n == 0:
            return None, 0.0
        k = min(self.k, n)
        dists = np.linalg.norm(self.X - feat[None, :], axis=1)
        nearest = [self.y[i] for i in np.argsort(dists)[:k]]
        label, votes = Counter(nearest).most_common(1)[0]
        return label, votes / k

    def save(self, path: str) -> None:
        np.savez(path, X=self.X, y=np.array(self.y, dtype="<U16"), k=np.array([self.k]))

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=False)
        self.X = data["X"]
        self.y = [str(v) for v in data["y"]]
        if "k" in data:
            self.k = int(data["k"][0])
        return True


class StrokeJudge:
    """Confidence-weighted vote over the pose predictions around a swing."""

    def __init__(self, history_s: float = C.POSE_HISTORY_S):
        self.history_s = history_s
        self._buf: deque = deque()
        self._lock = threading.Lock()

    def add(self, t: float, label: Optional[str], confidence: float) -> None:
        with self._lock:
            self._buf.append((t, label, confidence))
            while self._buf and t - self._buf[0][0] > self.history_s:
                self._buf.popleft()

    def judge(self, t_peak: float, window: float = C.STROKE_WINDOW_S) -> Tuple[Optional[str], float]:
        """Returns (stroke, confidence). stroke is None when only "ready" (or
        no pose at all) was seen near the swing -- that counts as a wrong
        stroke."""
        with self._lock:
            near = [(lab, conf) for t, lab, conf in self._buf if abs(t - t_peak) <= window]
        return judge_predictions(near)


def judge_predictions(preds) -> Tuple[Optional[str], float]:
    score = {"forehand": 0.0, "backhand": 0.0}
    for label, conf in preds:
        if label in score:
            score[label] += conf
    total = score["forehand"] + score["backhand"]
    if total <= 0:
        return None, 0.0
    best = max(score, key=score.get)
    return best, score[best] / total
