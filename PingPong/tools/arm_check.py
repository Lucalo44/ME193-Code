#!/usr/bin/env python3
"""
tools/arm_check.py -- see the live arm tracking before playing.

    python tools/arm_check.py                 # camera only
    python tools/arm_check.py --with-paddle   # also connect the Double Motor: real swings
                                              # confirm which arm holds the paddle

Stand where you'll play, holding the paddle. The window shows:
  * your skeleton, with the PADDLE ARM highlighted (orange) -- it should be the arm
    holding the paddle. At first it's the arm on your handed side; with
    --with-paddle, every real swing votes for the arm that moved fastest.
  * a small "screen paddle" panel: where your hand moves the on-screen paddle,
    after pressing z at your ready position.

Keys:  z  capture your ready position     h  switch handedness (left/right)
       q  quit

Same mirrored view, landmarks and body-relative measurements as the game.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402

import config as C  # noqa: E402
from vision.arm import ARMS, ArmTracker, arm_points  # noqa: E402
from vision.pose import create_landmarker  # noqa: E402

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--with-paddle", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1
    landmarker = create_landmarker()
    tracker = ArmTracker(C.HANDEDNESS)
    paddle = None
    if args.with_paddle:
        from hardware.paddle import Paddle
        from hardware.swing import load_calibration
        paddle = Paddle(load_calibration(os.path.join(HERE, C.SWING_CALIBRATION_FILE)))
        paddle.start()
    last_ts = -1
    print(__doc__)
    try:
        while True:
            ok, frame = cap.read()
            t = time.monotonic()
            if not ok:
                break
            view = cv2.flip(frame, 1)                       # mirrored, exactly like the game
            h, w = view.shape[:2]
            ts = max(last_ts + 1, int(t * 1000))
            last_ts = ts
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(view, cv2.COLOR_BGR2RGB)), ts)
            lms = res.pose_landmarks[0] if res.pose_landmarks else None
            tracker.update(t - C.POSE_FRAME_LATENCY_S, arm_points(lms) if lms else None)
            if paddle is not None:
                while not paddle.swing_events.empty():
                    tracker.on_swing(paddle.swing_events.get_nowait().t_peak)

            st = tracker.state(t)
            idx = tracker.paddle_arm() if lms else None
            if lms:
                pts = [(int(p.x * w), int(p.y * h)) for p in lms]
                for a, b in SKELETON:
                    cv2.line(view, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)
                if idx is not None:
                    s, e, wr = ARMS[idx]
                    cv2.line(view, pts[s], pts[e], (0, 140, 255), 6, cv2.LINE_AA)
                    cv2.line(view, pts[e], pts[wr], (0, 140, 255), 6, cv2.LINE_AA)
                    cv2.circle(view, pts[wr], 10, (0, 140, 255), -1, cv2.LINE_AA)

            # Screen-paddle panel: where the hand moves the on-screen paddle.
            px, py, size = w - 230, 20, 210
            cv2.rectangle(view, (px, py), (px + size, py + size), (30, 30, 30), -1)
            cv2.putText(view, "screen paddle", (px + 8, py + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cx, cy = px + size // 2, py + size // 2 + 10
            cv2.circle(view, (cx, cy), 4, (120, 120, 120), -1)
            if st.get("ok"):
                k = C.POSE_ARM_SCALE_M * C.POSE_ARM_WEIGHT / C.POSE_ARM_MAX_M * (size / 2 - 15)
                dx = max(-1.0, min(1.0, st["hand"][0])) * k
                dy = max(-1.0, min(1.0, st["hand"][1])) * k
                cv2.circle(view, (int(cx + dx), int(cy - dy)), 14, (60, 60, 220), -1, cv2.LINE_AA)

            lines = [
                f"handedness: {tracker.handedness}   (h to switch)",
                ("paddle arm: " + ("screen-RIGHT" if idx is not None and tracker.history[-1][1][idx]["shoulder"][0] > 0
                                   else "screen-LEFT") + f"  [{tracker.chosen_by}]") if idx is not None
                else "paddle arm: -- (no pose)",
                f"swing votes (left/right of screen): {tracker.votes}" if paddle else "run with --with-paddle to confirm by swinging",
                (f"hand offset from ready: x {st['hand'][0]:+.2f}  y {st['hand'][1]:+.2f}  (torso lengths)"
                 if st.get("ok") and st.get("zeroed") else "press z at your ready position"),
            ]
            if paddle is not None:
                lines.append(f"paddle: {paddle.status}")
            for i, text in enumerate(lines):
                cv2.putText(view, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(view, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Arm tracking check", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("z"):
                print("ready position captured" if tracker.zero() else "no pose -- step into view")
            if key == ord("h"):
                tracker.handedness = "left" if tracker.handedness == "right" else "right"
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if paddle is not None:
            paddle.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
