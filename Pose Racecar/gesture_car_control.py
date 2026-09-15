#!/usr/bin/env python3
"""
gesture_car_control.py
=======================

Controls a LEGO Education Double Motor using body-pose gestures captured from a live
webcam feed.

How it works
------------
A MediaPipe Pose Landmarker tracks your upper-body pose from the webcam.
A lightweight k-nearest-neighbors classifier -- trained live, right in this
script, with no external ML framework required -- maps your pose to one of
eight classes:

    idle   stop   go   left   right   back_left   back_right   reverse

"idle" is a deliberate catch-all: it's your natural resting posture (arms
relaxed, not making any of the other gestures), and it exists so the
classifier has somewhere to put "no command" instead of being forced to
misread an incidental pose as one of the real commands. Predicting idle
(like predicting stop) simply keeps the motor stopped.

The Double Motor is tank-drive (independent left/right wheels), so on top
of the basic forward/turn commands there are three reverse maneuvers:

    reverse      both wheels backward together (straight reverse)
    back_left    only the LEFT wheel reverses, right wheel holds still --
                 pivots the car backward around its stationary right wheel
    back_right   only the RIGHT wheel reverses, left wheel holds still --
                 the mirror image of back_left

Two modes
---------
TRAIN mode (the mode the script starts in):
    Strike a pose and press the matching number key to record a labeled
    training sample from the current frame:

        0 = idle    1 = stop    2 = go       3 = left     4 = right
        5 = back_left           6 = back_right           7 = reverse

    Collect a few dozen samples per class, ideally from a few different
    distances/angles, for a more robust classifier. Don't skip idle --
    record it while standing naturally, shifting your weight, adjusting
    your posture, etc., so the classifier learns what "not gesturing"
    looks like. The motor is always held stopped while in TRAIN mode.

RUN mode (press 'm' to toggle into it):
    The classifier continuously predicts your pose from the live feed and,
    if a Double Motor is connected, drives it accordingly. Predictions are
    smoothed in two ways so brief noise doesn't jerk the motor around:
    a rolling majority vote over the last --smoothing frames, and a short
    grace period (--idle-grace) that lets a momentary dip in confidence
    -- the kind that naturally happens for a frame or two while you're
    physically moving from one gesture to another -- coast on the last
    confident command instead of instantly falling back to idle. Only
    once low confidence (or a lost pose) persists past that grace period
    does the motor actually stop.

Keyboard controls
------------------
    0 / 1 / 2 / 3 / 4   Record a training sample for idle / stop / go / left / right
    5 / 6 / 7       Record a training sample for back_left / back_right / reverse
    m               Toggle TRAIN <-> RUN mode
    x               Connect to a Double Motor over Bluetooth (background)
    v               Disconnect from the Double Motor
    s               Save the training dataset to disk
    l               Reload the training dataset from disk
    c               Clear all in-memory training samples (does not touch
                    what is already saved on disk until you press 's')
    + / -           Increase / decrease motor speed
    [ / ]           Decrease / increase k (neighbors used for classification)
    q / ESC         Quit (stops the motor and disconnects first)

Usage
-----
    source venv/bin/activate
    python gesture_car_control.py

    # Try the gesture pipeline without any LEGO hardware attached:
    python gesture_car_control.py --no-motor

On first run this script downloads a small (~6 MB) MediaPipe pose model to
./models/pose_landmarker_lite.task. An internet connection is required for
that one-time download; after that everything runs offline.

Requires the packages in requirements.txt: legoeducation, mediapipe (pinned
to 0.10.35 -- see note in requirements.txt), and opencv (installed
automatically as a mediapipe dependency).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
from collections import Counter, deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

try:
    import legoeducation as le
except ImportError:
    le = None  # --no-motor still works without the package installed


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CLASSES = ["idle", "stop", "go", "left", "right", "back_left", "back_right", "reverse"]
KEY_TO_CLASS = {
    ord("0"): "idle",
    ord("1"): "stop",
    ord("2"): "go",
    ord("3"): "left",
    ord("4"): "right",
    ord("5"): "back_left",
    ord("6"): "back_right",
    ord("7"): "reverse",
}

# "idle" isn't a motor command -- it's the classifier's catch-all for "no
# gesture is being made right now". Both it and an unconfident/lost-pose
# prediction should simply keep the motor stopped. Every other label maps
# to itself; MotorController.send() knows how to drive each one.
MOTOR_COMMAND_FOR_LABEL = {
    "idle": "stop",
    "stop": "stop",
    "go": "go",
    "left": "left",
    "right": "right",
    "back_left": "back_left",
    "back_right": "back_right",
    "reverse": "reverse",
}

# If the pose is lost (no person detected) for this many consecutive
# frames, drop any buffered predictions rather than keep acting on a
# stale, possibly no-longer-true gesture.
POSE_LOST_FRAMES_BEFORE_RESET = 5

# While physically moving from one real gesture to another, the pose
# briefly looks ambiguous and the windowed vote can dip below --confidence
# for a frame or two even though nothing is actually wrong. Requiring that
# dip to persist for this many consecutive frames before falling back to
# idle keeps that momentary transition from instantly overriding whatever
# gesture was last confidently recognized.
UNCERTAIN_FRAMES_BEFORE_IDLE = 4

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# --------------------------------------------------------------------------
# LEGO Connection Card filter
# --------------------------------------------------------------------------
# By default this script connects to the first Double Motor it finds over
# Bluetooth. If you have more than one LEGO device nearby, edit the values
# below to target one specific Double Motor by the Connection Card plugged
# into it.
#
#   CARD_COLOR: one of legoeducation's le.LEGO_COLOR_* constants, e.g.:
#       le.LEGO_COLOR_RED     le.LEGO_COLOR_BLUE   le.LEGO_COLOR_GREEN
#       le.LEGO_COLOR_YELLOW  le.LEGO_COLOR_TEAL   le.LEGO_COLOR_PURPLE
#       le.LEGO_COLOR_WHITE   le.LEGO_COLOR_MAGENTA le.LEGO_COLOR_ORANGE
#       le.LEGO_COLOR_AZURE
#   CARD_SERIAL: the card's printed serial number as a string, e.g. "0049".
#                Useful to disambiguate two cards of the same color.
#
# Leave either one as None to not filter on it. Example -- to only connect
# to the Double Motor with a red Connection Card:
#
#     CARD_COLOR = le.LEGO_COLOR_RED
#
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "0994"

# Upper-body landmarks used as classifier features. Legs are frequently out
# of frame at a desk/table setup and add nothing for these four motions.
FEATURE_LANDMARKS = [
    vision.PoseLandmark.NOSE,
    vision.PoseLandmark.LEFT_SHOULDER,
    vision.PoseLandmark.RIGHT_SHOULDER,
    vision.PoseLandmark.LEFT_ELBOW,
    vision.PoseLandmark.RIGHT_ELBOW,
    vision.PoseLandmark.LEFT_WRIST,
    vision.PoseLandmark.RIGHT_WRIST,
    vision.PoseLandmark.LEFT_HIP,
    vision.PoseLandmark.RIGHT_HIP,
]
FEATURE_DIM = len(FEATURE_LANDMARKS) * 3  # x, y, z per landmark


# --------------------------------------------------------------------------
# Pose feature extraction
# --------------------------------------------------------------------------

def _landmark_xyz(landmarks, index: int) -> np.ndarray:
    p = landmarks[index]
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def extract_features(pose_landmarks) -> Optional[np.ndarray]:
    """Convert one detected pose's landmarks into a translation/scale
    invariant feature vector, or None if the pose is unusable (missing a
    torso to normalize against).
    """
    if not pose_landmarks:
        return None

    left_hip = _landmark_xyz(pose_landmarks, vision.PoseLandmark.LEFT_HIP.value)
    right_hip = _landmark_xyz(pose_landmarks, vision.PoseLandmark.RIGHT_HIP.value)
    left_shoulder = _landmark_xyz(pose_landmarks, vision.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = _landmark_xyz(pose_landmarks, vision.PoseLandmark.RIGHT_SHOULDER.value)

    mid_hip = (left_hip + right_hip) / 2.0
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    torso_size = float(np.linalg.norm(mid_shoulder - mid_hip))
    if torso_size < 1e-6:
        return None

    features: List[float] = []
    for landmark_enum in FEATURE_LANDMARKS:
        p = _landmark_xyz(pose_landmarks, landmark_enum.value)
        normalized = (p - mid_hip) / torso_size
        features.extend(normalized.tolist())
    return np.asarray(features, dtype=np.float64)


# --------------------------------------------------------------------------
# Live-trained k-nearest-neighbors gesture classifier
# --------------------------------------------------------------------------

class GestureClassifier:
    """A minimal KNN classifier over pose feature vectors. Samples are
    added live while the user poses in front of the camera; no separate
    offline training step is needed.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.X = np.zeros((0, FEATURE_DIM), dtype=np.float64)
        self.y: List[str] = []

    def add_sample(self, feat: np.ndarray, label: str) -> None:
        self.X = np.vstack([self.X, feat[None, :]])
        self.y.append(label)

    def counts(self) -> dict:
        c = Counter(self.y)
        return {cls: c.get(cls, 0) for cls in CLASSES}

    def clear(self) -> None:
        self.X = np.zeros((0, FEATURE_DIM), dtype=np.float64)
        self.y = []

    def predict(self, feat: np.ndarray) -> Tuple[Optional[str], float]:
        n = len(self.y)
        if n == 0:
            return None, 0.0
        k = min(self.k, n)
        dists = np.linalg.norm(self.X - feat[None, :], axis=1)
        nearest_idx = np.argsort(dists)[:k]
        nearest_labels = [self.y[i] for i in nearest_idx]
        label, votes = Counter(nearest_labels).most_common(1)[0]
        confidence = votes / k
        return label, confidence

    def save(self, path: str) -> None:
        np.savez(
            path,
            X=self.X,
            y=np.array(self.y, dtype="<U16"),
            k=np.array([self.k]),
        )

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=False)
        self.X = data["X"]
        self.y = list(data["y"])
        if "k" in data:
            self.k = int(data["k"][0])
        return True


# --------------------------------------------------------------------------
# Model download helper
# --------------------------------------------------------------------------

def ensure_model(path: str) -> str:
    if os.path.exists(path):
        return path
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    print(f"Downloading MediaPipe pose model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Download complete.")
    return path


# --------------------------------------------------------------------------
# Double Motor controller (thin wrapper with background connect/disconnect
# so Bluetooth I/O never freezes the camera preview)
# --------------------------------------------------------------------------

class MotorController:
    def __init__(self, speed: int = 50, enabled: bool = True):
        self.enabled = enabled and le is not None
        self.speed = speed
        self.motor = None
        self.status = "disabled" if not self.enabled else "disconnected"
        self._last_command: Optional[str] = None

    def connect_async(self) -> None:
        if not self.enabled or self.status in ("connecting", "connected"):
            return
        self.status = "connecting"
        threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self) -> None:
        try:
            self.motor = le.DoubleMotor()
            self.motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
            if self.motor.connected:
                self.status = "connected"
                self._last_command = None
            else:
                self.status = "not found"
        except Exception as exc:  # noqa: BLE001 - surface any BLE error to the overlay
            self.status = f"error: {exc}"

    def disconnect_async(self) -> None:
        if not self.enabled or self.motor is None:
            return
        threading.Thread(target=self._disconnect, daemon=True).start()

    def _disconnect(self) -> None:
        try:
            if self.motor is not None:
                self.motor.movement_stop(blocking=False)
                self.motor.disconnect()
        except Exception:
            pass
        finally:
            self.status = "disconnected"
            self._last_command = None

    def _apply(self, label: str) -> None:
        """Issue the underlying motor call for a resolved motor command
        (i.e. one already passed through MOTOR_COMMAND_FOR_LABEL). Split out
        from send() so set_speed() can re-issue the current command at a
        new speed without going through the repeat-command dedupe below.
        """
        if label == "stop":
            self.motor.movement_stop(blocking=False)
        elif label == "go":
            self.motor.movement_move(direction=le.MOVEMENT_DIRECTION_FORWARD, speed=self.speed, blocking=False)
        elif label == "left":
            self.motor.movement_move(direction=le.MOVEMENT_DIRECTION_LEFT, speed=self.speed, blocking=False)
        elif label == "right":
            self.motor.movement_move(direction=le.MOVEMENT_DIRECTION_RIGHT, speed=self.speed, blocking=False)
        elif label == "reverse":
            # Full reverse: both wheels backward together.
            self.motor.movement_move_tank(-self.speed, -self.speed, blocking=False)
        elif label == "back_left":
            # Only the left wheel reverses; the right wheel holds still, so
            # the car pivots backward around its stationary right wheel.
            self.motor.movement_move_tank(-self.speed, 0, blocking=False)
        elif label == "back_right":
            # Mirror of back_left: only the right wheel reverses, pivoting
            # backward around the stationary left wheel.
            self.motor.movement_move_tank(0, -self.speed, blocking=False)

    def send(self, label: str) -> None:
        """Issue a motor command for the given gesture label. Repeated
        identical labels are ignored so continuous predictions don't flood
        the Bluetooth link.
        """
        if not self.enabled or self.status != "connected" or self.motor is None:
            return
        if label == self._last_command:
            return
        self._last_command = label
        try:
            self._apply(label)
        except Exception as exc:  # noqa: BLE001
            self.status = f"error: {exc}"

    def set_speed(self, speed: int) -> None:
        self.speed = max(0, min(100, speed))
        if self.status == "connected" and self.motor is not None and self._last_command not in (None, "stop"):
            try:
                # Re-issue the currently active command at the new speed.
                # (movement_set_speed() only affects movement_move-style
                # commands, not an in-progress movement_move_tank(), so we
                # resend explicitly to cover every command type.)
                self._apply(self._last_command)
            except Exception:
                pass

    def shutdown(self) -> None:
        if self.enabled and self.motor is not None:
            try:
                self.motor.movement_stop(blocking=False)
                self.motor.disconnect()
            except Exception:
                pass


# --------------------------------------------------------------------------
# On-screen overlay
# --------------------------------------------------------------------------

def draw_overlay(frame, *, mode, classifier: GestureClassifier, smoothed_label, raw_confidence, motor: MotorController) -> None:
    y = 24
    line_height = 22

    def put(text: str, color=(255, 255, 255)) -> None:
        nonlocal y
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += line_height

    mode_color = (0, 200, 255) if mode == "TRAIN" else (0, 255, 0)
    put(f"Mode: {mode}   (m: toggle)", mode_color)

    counts = classifier.counts()
    put("Samples  " + "  ".join(f"{cls}:{counts[cls]}" for cls in CLASSES[:5]))
    put("         " + "  ".join(f"{cls}:{counts[cls]}" for cls in CLASSES[5:]))

    if smoothed_label is not None:
        put(f"Prediction: {smoothed_label}  (last frame vote {raw_confidence * 100:.0f}%, k={classifier.k})", (0, 255, 255))
    else:
        put("Prediction: -- (no pose / no training data yet)", (0, 0, 255))

    put(f"Motor: {motor.status}   speed={motor.speed}%")
    put("-" * 46, (120, 120, 120))
    put("0/1/2/3/4: idle/stop/go/left/right   5/6/7: back-left/back-right/reverse")
    put("s: save   l: load   c: clear   x: connect   v: disconnect")
    put("+/-: speed   [ / ]: k   q: quit")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control a LEGO Education Double Motor with live-trained pose gestures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument(
        "--model",
        default=os.path.join("models", "pose_landmarker_lite.task"),
        help="Path to the MediaPipe pose landmarker .task model (auto-downloaded if missing)",
    )
    parser.add_argument("--data", default="gesture_data.npz", help="Path to save/load the training dataset")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for the KNN gesture classifier")
    parser.add_argument("--speed", type=int, default=50, help="Motor speed percentage (0-100) used in RUN mode")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Minimum smoothed vote fraction required to trust a prediction; below this the motor stops",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=7,
        help="Number of recent frames to majority-vote over before sending a motor command",
    )
    parser.add_argument(
        "--idle-grace",
        dest="idle_grace",
        type=int,
        default=UNCERTAIN_FRAMES_BEFORE_IDLE,
        help=(
            "Consecutive low-confidence frames required before falling back to idle "
            "(higher = smoother through gesture transitions, but slower to actually stop)"
        ),
    )
    parser.add_argument(
        "--min-samples",
        dest="min_samples",
        type=int,
        default=15,
        help="Warn if fewer than this many samples exist per class when entering RUN mode",
    )
    parser.add_argument(
        "--no-motor",
        dest="no_motor",
        action="store_true",
        help="Run the camera + classifier only; never attempt to connect to hardware",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    model_path = ensure_model(args.model)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
    landmark_style = vision.drawing_styles.get_default_pose_landmarks_style()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera index {args.camera}", file=sys.stderr)
        landmarker.close()
        return 1

    classifier = GestureClassifier(k=args.k)
    if classifier.load(args.data):
        print(f"Loaded {len(classifier.y)} training samples from {args.data}")

    motor = MotorController(speed=args.speed, enabled=not args.no_motor)
    if not args.no_motor and le is None:
        print("legoeducation is not installed or failed to import; running as if --no-motor was passed.")

    print(__doc__)

    mode = "TRAIN"
    smoothing: deque = deque(maxlen=max(1, args.smoothing))
    missing_pose_frames = 0
    uncertain_frames = 0
    last_confident_label = "idle"
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read a frame from the camera; stopping.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)  # mirror image for natural interaction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            feat = extract_features(pose_landmarks) if pose_landmarks else None

            if pose_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame, pose_landmarks, connections, landmark_drawing_spec=landmark_style
                )

            raw_prediction, raw_confidence = (None, 0.0)
            if feat is not None:
                missing_pose_frames = 0
                raw_prediction, raw_confidence = classifier.predict(feat)
                if raw_prediction is not None:
                    smoothing.append(raw_prediction)
            else:
                # No pose detected this frame (person out of view, tracking
                # briefly lost, etc). Don't let old buffered votes keep
                # driving the motor indefinitely -- drop them after a short
                # grace period.
                missing_pose_frames += 1
                if missing_pose_frames >= POSE_LOST_FRAMES_BEFORE_RESET:
                    smoothing.clear()

            if smoothing:
                label, votes = Counter(smoothing).most_common(1)[0]
                if votes / len(smoothing) >= args.confidence:
                    # Confident majority: adopt it immediately and reset
                    # the ambiguity counter.
                    last_confident_label = label
                    uncertain_frames = 0
                else:
                    # Ambiguous window -- likely just a brief transition
                    # between two real gestures. Keep coasting on the last
                    # confident label until the ambiguity has persisted for
                    # --idle-grace frames in a row, rather than instantly
                    # snapping to idle.
                    uncertain_frames += 1
                    if uncertain_frames >= args.idle_grace:
                        last_confident_label = "idle"
            elif missing_pose_frames >= POSE_LOST_FRAMES_BEFORE_RESET:
                last_confident_label = "idle"
                uncertain_frames = 0

            smoothed_label = last_confident_label if (smoothing or missing_pose_frames >= POSE_LOST_FRAMES_BEFORE_RESET) else None

            if mode == "RUN":
                if smoothed_label is not None:
                    motor.send(MOTOR_COMMAND_FOR_LABEL.get(smoothed_label, "stop"))
            else:
                motor.send("stop")  # fail-safe: never drive while training

            draw_overlay(
                frame,
                mode=mode,
                classifier=classifier,
                smoothed_label=smoothed_label,
                raw_confidence=raw_confidence,
                motor=motor,
            )
            cv2.imshow("LEGO Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            elif key in KEY_TO_CLASS:
                if feat is not None:
                    label = KEY_TO_CLASS[key]
                    classifier.add_sample(feat, label)
                    print(f"Recorded sample #{len(classifier.y)} for '{label}'")
                else:
                    print("No pose detected in frame -- sample not recorded.")
            elif key == ord("m"):
                mode = "RUN" if mode == "TRAIN" else "TRAIN"
                smoothing.clear()
                uncertain_frames = 0
                last_confident_label = "idle"
                if mode == "TRAIN":
                    motor.send("stop")
                else:
                    counts = classifier.counts()
                    missing = [c for c in CLASSES if counts[c] < args.min_samples]
                    if missing:
                        print(
                            f"Warning: fewer than {args.min_samples} samples for: {', '.join(missing)}. "
                            "Classifier predictions may be unreliable."
                        )
                print(f"Switched to {mode} mode.")
            elif key == ord("s"):
                classifier.save(args.data)
                print(f"Saved {len(classifier.y)} samples to {args.data}")
            elif key == ord("l"):
                if classifier.load(args.data):
                    print(f"Loaded {len(classifier.y)} samples from {args.data}")
                else:
                    print(f"No dataset found at {args.data}")
            elif key == ord("c"):
                classifier.clear()
                print("Cleared in-memory training samples (dataset on disk is untouched until you press 's').")
            elif key == ord("x"):
                if CARD_COLOR is not None or CARD_SERIAL is not None:
                    print(f"Connecting to Double Motor (card_color={CARD_COLOR}, card_serial={CARD_SERIAL})...")
                else:
                    print("Connecting to Double Motor (no Connection Card filter set -- first one found)...")
                motor.connect_async()
            elif key == ord("v"):
                print("Disconnecting from Double Motor...")
                motor.disconnect_async()
            elif key in (ord("+"), ord("=")):
                motor.set_speed(motor.speed + 5)
            elif key in (ord("-"), ord("_")):
                motor.set_speed(motor.speed - 5)
            elif key == ord("["):
                classifier.k = max(1, classifier.k - 1)
            elif key == ord("]"):
                classifier.k = classifier.k + 1
    finally:
        motor.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
