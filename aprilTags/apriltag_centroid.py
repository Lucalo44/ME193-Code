"""Detect an AprilTag in the webcam feed and show its centroid.

Usage:
    python apriltag_centroid.py

For each detected tag: draws a red outline around its four corners, marks
the centroid, and prints/overlays its pixel coordinates. Press 'q' to quit.

This is a starting point, not the finished car-tracking project - it only
detects and displays. Driving the car so it stops centered under the tag
(and speeding up/slowing down based on how big the tag looks) is the next
step: look at how pd_tracker.py's control_loop() turns an "error" into a
motor speed command, then write a similar loop here where the error is
(frame_center_x - centroid_x) instead of (target_position - actual_position).
"""

import cv2

FAMILY = cv2.aruco.DICT_APRILTAG_36h11
CAMERA_INDEX = 0


def find_centroid(corners: "cv2.typing.MatLike") -> tuple[int, int]:
    """corners is one tag's 4x2 array of (x, y) pixel coordinates."""
    cx = int(corners[:, 0].mean())
    cy = int(corners[:, 1].mean())
    return cx, cy


def main():
    tag_dict = cv2.aruco.getPredefinedDictionary(FAMILY)
    detector = cv2.aruco.ArucoDetector(tag_dict, cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            corners_list, ids, _rejected = detector.detectMarkers(frame)

            if ids is not None:
                for corners, tag_id in zip(corners_list, ids.flatten()):
                    pts = corners.reshape(4, 2).astype(int)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

                    cx, cy = find_centroid(pts)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame, f"id {tag_id}: ({cx}, {cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )
                    print(f"tag {tag_id} centroid: ({cx}, {cy})")

            cv2.imshow("AprilTag Centroid (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
