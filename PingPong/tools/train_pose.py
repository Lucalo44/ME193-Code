#!/usr/bin/env python3
"""
tools/train_pose.py -- record forehand / backhand / ready poses for the stroke
check. Same UX as TRAIN mode in Pose Racecar/gesture_car_control.py.

    python tools/train_pose.py
    python tools/train_pose.py --with-paddle      # also connect the Double Motor for swing-synced recording
    python tools/train_pose.py --player sam --hand right   # a friend adding their own samples

Stand where you'll play, in view of the webcam, and record samples:

    0 / 1 / 2     record one sample: ready / forehand / backhand
    b             toggle BURST mode: then 0/1/2 starts recording that class every
                  ~100 ms; press the same key (or space) again to stop
    y             toggle SWING-SYNC mode (needs --with-paddle or x): 0/1/2 now
                  *arms* a class, and every real paddle swing labels the frames
                  within +/- STROKE_WINDOW_S of its peak with the armed class --
                  training data with exactly the timing the game uses
    x             connect the Double Motor (for swing-sync)
    h             switch the hand being recorded (left / right)
    s / l / c     save / load samples, clear THIS player's samples   (pose_data.npz)
    [ / ]         decrease / increase k
    q / ESC       quit

Tips: record "ready" generously (standing, walking, waiting, holding the
paddle). For forehand and backhand, record both the wind-up and the
follow-through under the same label. Aim for at least MIN_SAMPLES_PER_CLASS
(20) per class; swing-sync recording gives the most game-like data.

Several players can share one data file: every sample remembers --player and
--hand. With POSE_HAND_MODE = "mirror" (config.py) left-handed samples are
mirrored, so left- and right-handed players train one model; "separate" uses
only the samples recorded with the current player's hand.

Frames are mirrored before pose detection exactly as in the game -- don't
change that, or the classifier won't match gameplay.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

import config as C  # noqa: E402
from vision.pose import PoseClassifier, create_landmarker, extract_features, saved_handedness  # noqa: E402

KEY_TO_CLASS = {ord("0"): "ready", ord("1"): "forehand", ord("2"): "backhand"}


def draw_text_panel(frame, lines, x=8, y=8):
    """Readable text: one solid stroke per line on a translucent dark panel
    (an outline under thin text smears into a double image when the window is scaled)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = []
    width, height = 0, 10
    for text, color, scale in lines:
        thick = 2 if scale >= 0.6 else 1
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        rows.append((text, color, scale, thick, th, base))
        width = max(width, tw)
        height += th + base + 8
    x2, y2 = min(frame.shape[1] - 1, x + width + 20), min(frame.shape[0] - 1, y + height)
    roi = frame[y:y2, x:x2]
    roi[:] = (roi * 0.35).astype(roi.dtype)
    ty = y + 8
    for text, color, scale, thick, th, base in rows:
        ty += th
        cv2.putText(frame, text, (x + 10, ty), font, scale, color, thick, cv2.LINE_AA)
        ty += base + 8


def draw_overlay(frame, *, classifier, player, label, conf, burst, burst_label, sync, armed, paddle_status, last_msg):
    lines = []

    def put(text, color=(255, 255, 255), scale=0.6):
        lines.append((text, color, scale))

    put(f"Player: {player}   {classifier.hand.upper()}-handed (h to switch)   data: {classifier.mode}",
        (255, 255, 255))
    mine, used = classifier.counts_for(player), classifier.counts()
    parts = []
    for cls in C.POSE_CLASSES:
        n = mine[cls]
        parts.append(f"{cls}:{n}" + ("!" if n < C.MIN_SAMPLES_PER_CLASS else ""))
    put("Your samples  " + "   ".join(parts) + "     model uses "
        + " / ".join(str(used[c]) for c in C.POSE_CLASSES), (255, 255, 255))
    if label:
        color = {"forehand": (77, 184, 255), "backhand": (255, 157, 125), "ready": (180, 255, 180)}.get(label, (255, 255, 255))
        put(f"Prediction: {label}  {conf * 100:.0f}%   (k={classifier.k})", color, 0.8)
    else:
        put("Prediction: -- (no pose / no samples)", (80, 80, 255))
    mode = "BURST" if burst else "single"
    if sync:
        mode = f"SWING-SYNC (armed: {armed or 'press 0/1/2'})  paddle: {paddle_status}"
    put(f"Mode: {mode}" + (f"   >>> recording {burst_label} <<<" if burst_label else ""),
        (0, 220, 255) if (burst_label or sync) else (200, 200, 200))
    put("0/1/2 ready/forehand/backhand   b burst   y swing-sync   x paddle   h hand", (200, 200, 200), 0.5)
    put("s save   l load   c clear your samples   [ ] k   q quit", (200, 200, 200), 0.5)
    if last_msg and time.time() - last_msg[1] < 2.5:
        put(last_msg[0], (0, 255, 255))
    draw_text_panel(frame, lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--data", default=os.path.join(HERE, C.POSE_DATA_FILE))
    ap.add_argument("--k", type=int, default=C.POSE_K)
    ap.add_argument("--with-paddle", action="store_true", help="connect the Double Motor at startup")
    ap.add_argument("--player", default="original",
                    help="who is recording (samples are tagged with it; default 'original')")
    ap.add_argument("--hand", choices=("left", "right"), default=None,
                    help="the hand this player plays with (default: the hand picked in the game's lobby)")
    args = ap.parse_args()

    landmarker = create_landmarker()
    connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
    style = vision.drawing_styles.get_default_pose_landmarks_style()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}", file=sys.stderr)
        return 1

    player = args.player
    classifier = PoseClassifier(k=args.k, hand=args.hand or saved_handedness())
    if classifier.load(args.data):
        print(f"Loaded {len(classifier.y)} samples from {args.data} "
              f"(players: {', '.join(sorted(set(classifier.players)))}): {classifier.counts()}")
    print(f"Recording as '{player}', {classifier.hand}-handed. Data mode: {classifier.mode}.")

    paddle = None

    def connect_paddle():
        nonlocal paddle
        if paddle is None:
            from hardware.paddle import Paddle
            from hardware.swing import load_calibration
            paddle = Paddle(load_calibration(os.path.join(HERE, C.SWING_CALIBRATION_FILE)))
            paddle.start()
            print("Connecting to the Double Motor...")

    if args.with_paddle:
        connect_paddle()

    print(__doc__)
    burst = False
    burst_label = None
    last_burst = 0.0
    sync = False
    armed = None
    recent = deque()                      # (t_frame, feat) for swing-sync labeling
    last_msg = None
    t_start = time.monotonic()
    last_ts = -1
    try:
        while True:
            ok, frame = cap.read()
            t_frame = time.monotonic()
            if not ok:
                print("Camera read failed.", file=sys.stderr)
                break
            frame = cv2.flip(frame, 1)   # mirrored, exactly like the game
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(last_ts + 1, int((t_frame - t_start) * 1000))
            last_ts = ts
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            feat = extract_features(landmarks) if landmarks else None
            if landmarks:
                vision.drawing_utils.draw_landmarks(frame, landmarks, connections, landmark_drawing_spec=style)
            label, conf = classifier.predict(feat) if feat is not None else (None, 0.0)

            t_cam = t_frame - C.POSE_FRAME_LATENCY_S
            if feat is not None:
                recent.append((t_cam, feat))
            while recent and t_cam - recent[0][0] > 2.0:
                recent.popleft()

            # Burst recording.
            if burst_label and feat is not None and t_frame - last_burst >= C.BURST_RECORD_INTERVAL_S:
                classifier.add_sample(feat, burst_label, player=player)
                last_burst = t_frame

            # Swing-synced recording.
            if paddle is not None:
                while not paddle.swing_events.empty():
                    ev = paddle.swing_events.get_nowait()
                    if sync and armed:
                        frames = [f for t, f in recent if abs(t - ev.t_peak) <= C.STROKE_WINDOW_S]
                        for f in frames:
                            classifier.add_sample(f, armed, player=player)
                        last_msg = (f"swing -> {len(frames)} '{armed}' samples (strength {ev.strength:.2f})", time.time())
                        print(last_msg[0])

            draw_overlay(frame, classifier=classifier, player=player, label=label, conf=conf, burst=burst,
                         burst_label=burst_label, sync=sync, armed=armed,
                         paddle_status=paddle.status if paddle else "not connected", last_msg=last_msg)
            cv2.imshow("Ping Pong pose training", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            elif key in KEY_TO_CLASS:
                cls = KEY_TO_CLASS[key]
                if sync:
                    armed = cls
                elif burst:
                    burst_label = None if burst_label == cls else cls
                elif feat is not None:
                    classifier.add_sample(feat, cls, player=player)
                    print(f"Recorded sample #{len(classifier.y)} for '{cls}'")
                else:
                    print("No pose detected -- sample not recorded.")
            elif key == ord(" "):
                burst_label = None
            elif key == ord("b"):
                burst, burst_label = not burst, None
                sync = False
            elif key == ord("y"):
                sync, burst, burst_label = not sync, False, None
                if sync and paddle is None:
                    print("Swing-sync needs the paddle: press x to connect.")
            elif key == ord("x"):
                connect_paddle()
            elif key == ord("s"):
                classifier.save(args.data)
                low = [c for c, n in classifier.counts().items() if n < C.MIN_SAMPLES_PER_CLASS]
                last_msg = (f"Saved {len(classifier.y)} samples", time.time())
                print(f"Saved {len(classifier.y)} samples to {args.data}: {classifier.counts()}")
                if low:
                    print(f"  Warning: fewer than {C.MIN_SAMPLES_PER_CLASS} samples for {', '.join(low)}")
            elif key == ord("l"):
                ok = classifier.load(args.data)
                last_msg = (f"Loaded {len(classifier.y)} samples" if ok else "No saved data", time.time())
            elif key == ord("c"):
                n = classifier.remove_player(player)
                last_msg = (f"Cleared {n} of {player}'s samples (disk untouched until you press s)", time.time())
            elif key == ord("h"):
                classifier.set_hand("left" if classifier.hand == "right" else "right")
                last_msg = (f"Recording {classifier.hand}-handed", time.time())
            elif key == ord("["):
                classifier.k = max(1, classifier.k - 1)
            elif key == ord("]"):
                classifier.k += 1
    finally:
        if paddle is not None:
            paddle.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
