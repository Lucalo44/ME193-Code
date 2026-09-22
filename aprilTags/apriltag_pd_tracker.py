"""Center an AprilTag in the camera frame using a PD controller on a LEGO
Double Motor.

The car carries an AprilTag and drives back and forth along a straight
track parallel to the camera (see apriltag_centroid.py for how the tag
itself is detected). This closes the loop: the tag's horizontal pixel
position is the measurement, the frame's horizontal center is the fixed
setpoint, and PD control turns "how far off-center" into a motor speed --
the same target/actual -> speed shape as pd_tracker.py, just with a camera
as the sensor instead of a second motor's shaft position.

Update CARD_COLOR/CARD_SERIAL below to match your Double Motor's Connection
Card (run find_devices.py if you don't know it). Tune Kp/Kd live with the
trackbars while watching the car settle -- too much Kp alone overshoots and
oscillates around center; adding Kd damps that out, same as in
pd_tracker.py. If the car drives away from center instead of toward it,
flip DIRECTION_SIGN to -1.

Every commanded speed outside the deadzone is floored at MIN_SPEED, since
Kp*error alone shrinks toward zero near the target and real motors have
enough static friction to just stall there instead of creeping the rest of
the way in. If it oscillates right around the centerline instead of
settling, that floor is overpowering the tiny remaining error -- raise
DEADZONE_PIXELS slightly, lower MIN_SPEED, or raise Kd to damp it out.

Press 'q' to quit.
"""

import time

import cv2
import legoeducation as le

from lelib import doubleMotor

# --- Hardware ---------------------------------------------------------------
CARD_COLOR = le.LEGO_COLOR_PURPLE  # placeholder - replace with your Double Motor's actual card
CARD_SERIAL = "5164"               # placeholder - replace with your Double Motor's actual card

# --- Vision -------------------------------------------------------------
FAMILY = cv2.aruco.DICT_APRILTAG_36h11
CAMERA_INDEX = 0

# --- Control ------------------------------------------------------------
MAX_SPEED = 100          # speed cap sent to the motors, in percent
MIN_SPEED = 10          # smallest speed that reliably overcomes the car's own
                         # static friction -- below this it just stalls
                         # instead of creeping closer. Raise if it still
                         # stalls short of center; lower if it overshoots.
DEADZONE_PIXELS = 7      # error smaller than this counts as "centered" -> stop
DIRECTION_SIGN = 1      # flip to -1 if the car drives the wrong way on your track
RIGHT_MOTOR_SIGN = -1   # the two motors are mirror-mounted, so equal signed
                         # speeds spin the car in place instead of driving it
                         # straight -- this inverts the right side to cancel
                         # that out. Flip to +1 if it makes things worse.
MAX_SPEED_STEP = 10       # max change in commanded speed per frame, in percent --
                         # caps how fast speed can ramp so it glides instead of
                         # jumping (e.g. straight to MIN_SPEED at the deadzone
                         # edge). Lower = smoother but slower to react.
D_SMOOTHING = 0.15        # 0-1 weight on each new derivative sample -- a raw
                         # frame-to-frame derivative amplifies ordinary pixel
                         # jitter in the detected centroid into speed spikes;
                         # this low-pass-filters it. Lower = smoother but laggier.

# The tag's real size is fixed, so its apparent width in pixels is a proxy
# for distance from the camera -- bigger on screen means closer. This scales
# the PD output by (reference width / apparent width), so a tag that looks
# small (far away) drives faster and one that looks big (close) drives
# slower, on top of the existing centering behavior.
TAG_SIZE_REF_PIXELS = 120        # apparent tag width, in pixels, considered
                                 # "neutral" distance (factor = 1x) -- measure
                                 # this at your track's typical distance
DISTANCE_FACTOR_MIN = 0.5       # clamp so an extreme distance can't overwhelm Kp/Kd
DISTANCE_FACTOR_MAX = 2.0

KP_INIT, KP_MAX = 0.20, 10   # speed percent per pixel of error
KD_INIT, KD_MAX = 0.05, 0.5   # speed percent per (pixel/second) of error's rate of change


def find_centroid(corners) -> tuple[int, int]:
    """corners is one tag's 4x2 array of (x, y) pixel coordinates."""
    cx = int(corners[:, 0].mean())
    cy = int(corners[:, 1].mean())
    return cx, cy


def try_connect(car: doubleMotor) -> bool:
    try:
        car.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    except Exception as exc:
        print(f"Could not connect to the Double Motor: {exc}")
        return False
    return True


def main():
    tag_dict = cv2.aruco.getPredefinedDictionary(FAMILY)
    detector = cv2.aruco.ArucoDetector(tag_dict, cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    car = doubleMotor()
    print("Connecting to Double Motor...")
    connected = try_connect(car)
    if not connected:
        print("Running in camera preview-only mode (no motor commands will be sent).")

    window = "AprilTag PD Centering (q to quit)"
    cv2.namedWindow(window)
    cv2.createTrackbar("Kp x100", window, int(KP_INIT * 100), int(KP_MAX * 100), lambda _: None)
    cv2.createTrackbar("Kd x100", window, int(KD_INIT * 100), int(KD_MAX * 100), lambda _: None)

    prev_error = None
    prev_time = time.monotonic()
    smoothed_d_error = 0.0
    last_speed = 0.0

    try:
        while True:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break  # window was closed via its titlebar, not 'q'

            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            dt = now - prev_time
            prev_time = now

            kp = cv2.getTrackbarPos("Kp x100", window) / 100.0
            kd = cv2.getTrackbarPos("Kd x100", window) / 100.0

            frame_center_x = frame.shape[1] // 2
            cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame.shape[0]), (255, 255, 0), 1)

            corners_list, ids, _rejected = detector.detectMarkers(frame)

            desired_speed = 0.0
            if ids is not None and len(ids) > 0:
                pts = corners_list[0].reshape(4, 2).astype(int)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                cx, cy = find_centroid(pts)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                error = frame_center_x - cx
                # Target (frame center) never moves, so unlike pd_tracker.py
                # there's no "derivative kick" risk in differentiating the
                # error directly -- d(error)/dt is just -d(actual)/dt here.
                raw_d_error = 0.0 if prev_error is None or dt <= 0 else (error - prev_error) / dt
                smoothed_d_error += D_SMOOTHING * (raw_d_error - smoothed_d_error)
                prev_error = error

                tag_width = max(1, int(pts[:, 0].max() - pts[:, 0].min()))
                distance_factor = TAG_SIZE_REF_PIXELS / tag_width
                distance_factor = max(DISTANCE_FACTOR_MIN, min(DISTANCE_FACTOR_MAX, distance_factor))

                if abs(error) > DEADZONE_PIXELS:
                    raw = (kp * error + kd * smoothed_d_error) * distance_factor
                    # Faster the farther off-center it is, slower as it
                    # nears the line -- but floored at MIN_SPEED so it
                    # doesn't stall out before actually getting there.
                    magnitude = max(MIN_SPEED, min(MAX_SPEED, abs(raw)))
                    desired_speed = DIRECTION_SIGN * (magnitude if raw >= 0 else -magnitude)

                cv2.putText(
                    frame, f"error: {error:+d}px  speed: {last_speed:+.0f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
                cv2.putText(
                    frame, f"tag width: {tag_width}px  distance factor: x{distance_factor:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
            else:
                prev_error = None
                smoothed_d_error = 0.0
                cv2.putText(
                    frame, "no tag found - stopped", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            # Slew-limit the actual command so it glides toward desired_speed
            # instead of jumping straight there -- this is what actually
            # makes the motion continuous, since desired_speed itself can
            # still jump (e.g. snapping to MIN_SPEED right at the deadzone
            # edge, or to 0 the instant the tag drops out of frame).
            step = max(-MAX_SPEED_STEP, min(MAX_SPEED_STEP, desired_speed - last_speed))
            last_speed += step

            if connected:
                car.motor_run(motor=le.MOTOR_LEFT, speed=int(round(last_speed)), blocking=False)
                car.motor_run(motor=le.MOTOR_RIGHT, speed=int(round(RIGHT_MOTOR_SIGN * last_speed)), blocking=False)

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if connected:
            car.motor_stop(motor=le.MOTOR_BOTH)
            car.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
