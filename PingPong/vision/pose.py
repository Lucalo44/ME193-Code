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


# Mirroring a pose left <-> right: negate x and swap each left landmark with its
# right partner (nose stays). Used to share training data between left- and
# right-handed players (POSE_HAND_MODE = "mirror").
_MIRROR_PAIRS = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23}
_MIRROR_ORDER = np.array([FEATURE_LANDMARKS.index(_MIRROR_PAIRS.get(i, i)) for i in FEATURE_LANDMARKS])
_MIRROR_SIGN = np.array([-1.0, 1.0, 1.0] * len(FEATURE_LANDMARKS))


def mirror_features(feat: np.ndarray) -> np.ndarray:
    """The same pose seen in a mirror (works on one vector or a stack of them)."""
    f = np.asarray(feat, dtype=np.float64)
    shaped = f.reshape(f.shape[:-1] + (len(FEATURE_LANDMARKS), 3))[..., _MIRROR_ORDER, :]
    return shaped.reshape(f.shape) * _MIRROR_SIGN


def saved_handedness() -> str:
    """The hand picked in the game's lobby (player_settings.json), else config."""
    import json
    try:
        with open(os.path.join(C.HERE, C.PLAYER_SETTINGS_FILE)) as f:
            hand = json.load(f).get("handedness")
    except (OSError, ValueError):
        hand = None
    return hand if hand in ("left", "right") else C.HANDEDNESS


class PoseClassifier:
    """Minimal KNN over pose feature vectors (the reference GestureClassifier
    with this project's classes).

    Every sample remembers who recorded it and with which hand. Samples are
    stored as recorded; `hand` is the current player's hand, and `mode` decides
    how the other hand's samples are used:
      "mirror"   -- everything is compared as if right-handed (left-handed
                    samples and live poses are mirrored), so all players share
                    one model;
      "separate" -- only samples recorded with the current player's hand count.
    """

    def __init__(self, k: int = C.POSE_K, classes: List[str] = C.POSE_CLASSES,
                 hand: Optional[str] = None, mode: str = C.POSE_HAND_MODE):
        self.k = k
        self.classes = classes
        self.hand = hand or saved_handedness()
        self.mode = mode
        self.X = np.zeros((0, FEATURE_DIM), dtype=np.float64)
        self.y: List[str] = []
        self.hands: List[str] = []
        self.players: List[str] = []
        self._view = None                       # (X, y) actually compared against

    def set_hand(self, hand: str) -> None:
        if hand in ("left", "right") and hand != self.hand:
            self.hand = hand
            self._view = None

    def add_sample(self, feat: np.ndarray, label: str, hand: Optional[str] = None,
                   player: str = "") -> None:
        self.X = np.vstack([self.X, feat[None, :]])
        self.y.append(label)
        self.hands.append(hand or self.hand)
        self.players.append(player)
        self._view = None

    def _active(self):
        if self._view is None:
            hands = np.array(self.hands, dtype="<U8")
            y = np.array(self.y, dtype="<U16")
            if self.mode == "separate":
                keep = hands == self.hand
                self._view = (self.X[keep], y[keep].tolist())
            else:
                X = self.X.copy()
                left = hands == "left"
                if left.any():
                    X[left] = mirror_features(X[left])
                self._view = (X, y.tolist())
        return self._view

    def counts(self) -> dict:
        """Samples per class that the current player is matched against."""
        c = Counter(self._active()[1])
        return {cls: c.get(cls, 0) for cls in self.classes}

    def counts_for(self, player: str) -> dict:
        c = Counter(lab for lab, who in zip(self.y, self.players) if who == player)
        return {cls: c.get(cls, 0) for cls in self.classes}

    @property
    def trained(self) -> bool:
        counts = self.counts()
        return counts.get("forehand", 0) > 0 and counts.get("backhand", 0) > 0

    def remove_player(self, player: str) -> int:
        """Drop one player's samples; returns how many were removed."""
        keep = [i for i, who in enumerate(self.players) if who != player]
        removed = len(self.y) - len(keep)
        self.X = self.X[keep] if keep else np.zeros((0, FEATURE_DIM), dtype=np.float64)
        self.y = [self.y[i] for i in keep]
        self.hands = [self.hands[i] for i in keep]
        self.players = [self.players[i] for i in keep]
        self._view = None
        return removed

    def clear(self) -> None:
        self.X = np.zeros((0, FEATURE_DIM), dtype=np.float64)
        self.y, self.hands, self.players = [], [], []
        self._view = None

    def predict(self, feat: np.ndarray) -> Tuple[Optional[str], float]:
        X, y = self._active()
        n = len(y)
        if n == 0:
            return None, 0.0
        if self.mode != "separate" and self.hand == "left":
            feat = mirror_features(feat)
        k = min(self.k, n)
        dists = np.linalg.norm(X - feat[None, :], axis=1)
        nearest = [y[i] for i in np.argsort(dists)[:k]]
        label, votes = Counter(nearest).most_common(1)[0]
        return label, votes / k

    def save(self, path: str) -> None:
        np.savez(path, X=self.X, y=np.array(self.y, dtype="<U16"), k=np.array([self.k]),
                 hand=np.array(self.hands, dtype="<U8"), player=np.array(self.players, dtype="<U32"))

    def load(self, path: str, legacy_hand: Optional[str] = None) -> bool:
        """Older files have no hand/player per sample: they're taken as recorded
        with `legacy_hand` (default: the hand picked in the game's lobby)."""
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=False)
        self.X = data["X"]
        self.y = [str(v) for v in data["y"]]
        n = len(self.y)
        self.hands = ([str(v) for v in data["hand"]] if "hand" in data
                      else [legacy_hand or saved_handedness()] * n)
        self.players = [str(v) for v in data["player"]] if "player" in data else ["original"] * n
        if "k" in data:
            self.k = int(data["k"][0])
        self._view = None
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
