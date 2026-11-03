"""Run the hand-code safe.

    python3 run_safe.py                    # virtual safe, gesture code only
    python3 run_safe.py --owner-check      # also require that the hand is mine (palm creases)
    python3 run_safe.py --hardware unoq    # on the UNO Q: button/LEDs/bolt over the Bridge
    python3 run_safe.py --headless         # no window (UNO Q without a display)

Flow: press the button -> LEDs go amber -> show three finger counts (1-5) in the box, pulling
your hand out of the box between each one -> the safe opens or stays locked.
Keys: SPACE = button, o = toggle owner check, r = relock now, q = quit.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

import config as C
import models
import set_password
import ui
from preprocess import crease_preview, crop_roi, gesture_array, open_camera, palm_array, read_frame, roi_box
from safe_io import UnoQSafe, VirtualSafe

GESTURES = C.CLASSES["gesture"]
ME = C.CLASSES["owner"].index("me")


class CodeReader:
    """Turns a noisy per-frame prediction stream into discrete digits.

    A digit counts once it is predicted confidently for STABLE_FRAMES frames in a row. After
    that, the box has to read "none" for RELEASE_FRAMES before the next digit is taken, which
    is what lets a code like 3-3-1 work and stops one long hold from entering 3-3-3.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.label, self.run, self.owner = None, 0, []
        self.need_release, self.none_run = False, 0
        self.last_owner = None

    @property
    def progress(self):
        return 0.0 if self.need_release else self.run / C.STABLE_FRAMES

    def update(self, label, conf, owner_p):
        confident = conf >= C.CONF_THRESH
        if self.need_release:
            self.none_run = self.none_run + 1 if (label == "none" and confident) else 0
            if self.none_run >= C.RELEASE_FRAMES:
                self.need_release, self.label, self.run = False, None, 0
            return None
        if label == "none" or not confident:
            self.label, self.run, self.owner = None, 0, []
            return None
        if label != self.label:
            self.label, self.run, self.owner = label, 0, []
        self.run += 1
        if owner_p is not None:
            self.owner.append(owner_p)
        if self.run >= C.STABLE_FRAMES:
            self.need_release, self.none_run = True, 0
            self.last_owner = float(np.mean(self.owner)) if self.owner else None
            return int(label)
        return None


@torch.no_grad()
def predict(model, array):
    return model(torch.from_numpy(array)[None]).softmax(1)[0].numpy()


def draw_camera(frame, crop, label, conf, owner_p, reader, state, owner_mode):
    view = frame.copy()
    x0, y0, s = roi_box(*view.shape[:2])
    color = {"ARMED": (0, 170, 255), "UNLOCKED": (60, 210, 60),
             "DENIED": (40, 40, 230), "LOCKOUT": (40, 40, 230)}.get(state, (150, 150, 150))
    cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), color, 2)
    if state == "ARMED":
        cv2.rectangle(view, (x0, y0 + s + 4), (x0 + int(s * min(1, reader.progress)), y0 + s + 10), color, -1)
        if reader.need_release:
            cv2.putText(view, "remove hand", (x0 + 8, y0 + s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    txt = f"{label} ({conf:.2f})"
    if owner_p is not None:
        txt += f"  me {owner_p:.2f}"
    cv2.putText(view, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if owner_mode:
        pv = crease_preview(crop)
        view[40:40 + pv.shape[0], 10:10 + pv.shape[1]] = pv
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner-check", action="store_true", help="require the palm-crease owner net to say 'me'")
    ap.add_argument("--hardware", choices=["virtual", "unoq"], default="virtual")
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()

    gesture_net = models.load("gesture")
    if gesture_net is None:
        raise SystemExit("No models/gesture.pt yet: collect_data.py --task gesture, then train.py --task gesture.")
    owner_net = models.load("owner")
    owner_mode = a.owner_check and owner_net is not None
    if a.owner_check and owner_net is None:
        print("No models/owner.pt yet, so owner check is off. Train it with train.py --task owner.")
    if not C.PASSWORD_FILE.exists():
        raise SystemExit("No password.json yet. Run set_password.py first.")

    safe = UnoQSafe() if a.hardware == "unoq" else VirtualSafe()
    safe.lock()
    safe.set_leds("off")
    reader = CodeReader()
    state, t_state, entered, owner_scores, fails = "IDLE", time.time(), [], [], 0
    status = "Press the button"
    fps = deque(maxlen=30)

    def go(new_state, new_status, leds, digits=0):
        nonlocal state, t_state, status
        state, t_state, status = new_state, time.time(), new_status
        safe.set_leds(leds, digits)
        print(f"[{time.strftime('%H:%M:%S')}] {new_state}: {new_status}")

    win = "Hand Safe"
    if not a.headless:
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, lambda ev, x, y, *_: safe.press()
                             if ev == cv2.EVENT_LBUTTONDOWN and ui.in_button(x, y, C.FRAME_W) else None)

    cap = open_camera()
    print(__doc__)
    while True:
        t0 = time.time()
        frame = read_frame(cap)
        if frame is None:
            print("camera read failed")
            break
        crop = crop_roi(frame)
        probs = predict(gesture_net, gesture_array(crop))
        label, conf = GESTURES[int(probs.argmax())], float(probs.max())
        owner_p = None
        if owner_mode and label != "none":
            owner_p = float(predict(owner_net, palm_array(crop))[ME])

        now = time.time()
        pressed = safe.button_pressed()
        if state == "IDLE" and pressed:
            entered, owner_scores = [], []
            reader.reset()
            go("ARMED", f"Show {C.CODE_LENGTH} numbers", "armed")
        elif state == "ARMED":
            digit = reader.update(label, conf, owner_p)
            if digit is not None:
                entered.append(digit)
                owner_scores.append(reader.last_owner)
                safe.set_leds("progress", len(entered))
                status = f"{len(entered)} of {C.CODE_LENGTH}"
                print(f"  digit {len(entered)} read" + (f" (me {reader.last_owner:.2f})" if owner_mode else ""))
            if len(entered) == C.CODE_LENGTH:
                code_ok = set_password.check(entered)
                owner_ok = not owner_mode or all(s is not None and s >= C.OWNER_THRESH for s in owner_scores)
                if code_ok and owner_ok:
                    fails = 0
                    safe.unlock()
                    go("UNLOCKED", "UNLOCKED", "granted", C.CODE_LENGTH)
                else:
                    fails += 1
                    why = "wrong code" if not code_ok else "hand not recognized"
                    print(f"  denied: {why} ({fails}/{C.MAX_FAILS})")
                    go("DENIED", "ACCESS DENIED", "denied", C.CODE_LENGTH)
            elif now - t_state > C.ENTRY_TIMEOUT_S:
                go("IDLE", "Timed out. Press the button", "off")
        elif state == "UNLOCKED" and now - t_state > C.UNLOCK_HOLD_S:
            safe.lock()
            go("IDLE", "Locked. Press the button", "off")
        elif state == "DENIED" and now - t_state > C.DENIED_SHOW_S:
            if fails >= C.MAX_FAILS:
                fails = 0
                go("LOCKOUT", f"Locked out for {C.LOCKOUT_S}s", "lockout")
            else:
                go("IDLE", "Press the button", "off")
        elif state == "LOCKOUT":
            status = f"Locked out: {C.LOCKOUT_S - (now - t_state):.0f}s"
            if now - t_state > C.LOCKOUT_S:
                go("IDLE", "Press the button", "off")

        fps.append(time.time() - t0)
        if a.headless:
            continue
        cam = draw_camera(frame, crop, label, conf, owner_p, reader, state, owner_mode)
        cv2.putText(cam, f"{len(fps) / max(1e-6, sum(fps)):.0f} fps", (C.FRAME_W - 80, C.FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        panel = ui.draw_panel(safe, status, owner_mode, owner_net is not None)
        cv2.imshow(win, np.hstack([cam, panel]))

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            safe.press()
        elif k == ord("o") and owner_net is not None:
            owner_mode = not owner_mode
            safe.lock()
            go("IDLE", f"Owner check {'on' if owner_mode else 'off'}", "off")   # restart any entry
        elif k == ord("r"):
            safe.lock()
            go("IDLE", "Locked. Press the button", "off")

    safe.lock()
    safe.set_leds("off")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
