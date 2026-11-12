"""Record labeled hand crops from the USB camera.

    python3 collect_data.py --task gesture     # finger counts: keys 0-5 pick the label (0 = none)
    python3 collect_data.py --task owner       # palm ID: m = me, n = not_me (friends' hands)

A camera picker window opens first: click the USB webcam. (--camera N skips it.)
--board records through the UNO Q's own webcam instead, while the safe app runs on the board:
the best data, because it is the exact view the safe sees in its mount.
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
import urllib.request

import cv2
import numpy as np

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
    ap.add_argument("--board", nargs="?", const="auto", metavar="HOST",
                    help="record from the UNO Q safe's webcam (needs the safe app running); optional board address")
    a = ap.parse_args()

    if a.board:
        from view_safe import PORT, Reader, board_hosts, find_safe
        hosts = board_hosts() + ["localhost"] if a.board == "auto" else [a.board]
        url = find_safe(hosts, PORT)
        if url is None:
            raise SystemExit("Couldn't reach the safe app on the board. Deploy it first (sync_to_unoq.py --deploy).")
        board = Reader(urllib.request.urlopen(url + "/crop", timeout=5))
        board.start()
        print(f"Recording from the board's webcam via {url}")
        if a.every == 3:
            a.every = 1                            # the board serves ~10 fps; keep every frame
    else:
        cam = a.camera if a.camera is not None else pick_camera()
        if cam is None:
            raise SystemExit("No camera selected.")

    for c in C.CLASSES[a.task]:
        (C.DATA_DIR / a.task / c).mkdir(parents=True, exist_ok=True)
    keys = KEYS[a.task]
    label = C.CLASSES[a.task][-1]
    counts = count(a.task)
    recording, session, n_session, frame_i = False, "", 0, 0

    cap = None if a.board else open_camera(cam)
    print(__doc__)
    last_t = None
    x0, y0, s_box = roi_box(C.FRAME_H, C.FRAME_W)
    frame = crop = None
    while True:
        new_frame = True
        if a.board:
            # Never block waiting for the board: keys are read only by the waitKey at the bottom.
            # (An inner waitKey here used to swallow SPACE, so recording never started.)
            if not board.alive:
                print("board stream ended")
                break
            if board.frame is None or board.t_frame == last_t:
                new_frame = False
            else:
                last_t, crop = board.t_frame, board.frame.copy()
                frame = np.zeros((C.FRAME_H, C.FRAME_W, 3), np.uint8)   # draw the crop where the box is
                frame[y0:y0 + s_box, x0:x0 + s_box] = cv2.resize(crop, (s_box, s_box))
        else:
            frame = read_frame(cap)
            if frame is None:
                break
            crop = crop_roi(frame)
        if frame is None:                          # board hasn't sent its first frame yet
            frame_wait = np.zeros((C.FRAME_H, C.FRAME_W, 3), np.uint8)
            cv2.putText(frame_wait, "waiting for the board's camera...", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(f"collect: {a.task}", frame_wait)
            if (cv2.waitKey(30) & 0xFF) == ord("q"):
                break
            continue
        frame_i += new_frame

        if recording and new_frame and frame_i % a.every == 0:
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

        k = cv2.waitKey(1 if not a.board else 15) & 0xFF
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

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
