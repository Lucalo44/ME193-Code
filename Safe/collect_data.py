"""Record labeled hand crops from the USB camera.

    python3 collect_data.py --task gesture     # finger counts: keys 0-5 pick the label (0 = none)
    python3 collect_data.py --task owner       # palm ID: m = me, n = not_me (friends' hands)

A camera picker window opens first: click the USB webcam. (--camera N skips it.)
SPACE starts/stops recording (each start is a new session), q quits.
Saves every --every'th frame to data/<task>/<label>/.
Tips:
  gesture: move, tilt and rotate your hand while recording; record "none" with an empty box,
           a hand half in/out, and a fist.
  owner:   palm facing the camera, cycling through 1-5 fingers. Get as many different
           not_me hands as you can; that is what makes the check strict.
"""
import argparse
import time

import cv2

import config as C
from camera_picker import pick_camera
from preprocess import crease_preview, crop_roi, open_camera, read_frame, roi_box

KEYS = {
    "gesture": {ord(str(i)): ("none" if i == 0 else str(i)) for i in range(6)},
    "owner": {ord("m"): "me", ord("n"): "not_me"},
}


def count(task):
    return {c: len(list((C.DATA_DIR / task / c).glob("*.jpg"))) for c in C.CLASSES[task]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gesture", "owner"], default="gesture")
    ap.add_argument("--every", type=int, default=3, help="save every Nth frame (default 3, ~10 fps)")
    ap.add_argument("--max", type=int, default=400, help="stop a session after this many images")
    ap.add_argument("--camera", type=int, help="camera index; skips the picker window")
    a = ap.parse_args()

    cam = a.camera if a.camera is not None else pick_camera()
    if cam is None:
        raise SystemExit("No camera selected.")

    for c in C.CLASSES[a.task]:
        (C.DATA_DIR / a.task / c).mkdir(parents=True, exist_ok=True)
    keys = KEYS[a.task]
    label = C.CLASSES[a.task][-1]
    counts = count(a.task)
    recording, session, n_session, frame_i = False, "", 0, 0

    cap = open_camera(cam)
    print(__doc__)
    while True:
        frame = read_frame(cap)
        if frame is None:
            break
        crop = crop_roi(frame)
        frame_i += 1

        if recording and frame_i % a.every == 0:
            path = C.DATA_DIR / a.task / label / f"{session}_{n_session:05d}.jpg"
            cv2.imwrite(str(path), crop)
            n_session += 1
            counts[label] += 1
            if n_session >= a.max:
                recording = False
                print(f"session {session}: hit --max ({a.max})")

        view = frame.copy()
        x0, y0, s = roi_box(*view.shape[:2])
        cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), (0, 0, 255) if recording else (0, 200, 255), 2)
        cv2.putText(view, f"label: {label}  {'REC' if recording else 'paused'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if recording else (255, 255, 255), 2)
        summary = "  ".join(f"{k}:{v}" for k, v in counts.items())
        cv2.putText(view, summary, (10, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if a.task == "owner":
            p = crease_preview(crop)
            view[40:40 + p.shape[0], view.shape[1] - p.shape[1] - 10:view.shape[1] - 10] = p
        cv2.imshow(f"collect: {a.task}", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            recording = not recording
            if recording:
                session, n_session = str(int(time.time() * 1000)), 0
            print(f"{'recording' if recording else 'paused'} [{label}]  {counts}")
        elif k in keys:
            label = keys[k]
            print("label ->", label)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
