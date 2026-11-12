"""Run the hand-code safe.

    python3 run_safe.py                    # virtual safe, gesture code only
    python3 run_safe.py --owner-check      # also require that the hand is mine (palm creases)
    python3 run_safe.py --hardware unoq    # on the UNO Q: button/LEDs/bolt over the Bridge
    python3 run_safe.py --headless         # no window (UNO Q without a display)
    python3 run_safe.py --pick-camera      # choose the camera again (otherwise uses the saved one)
    python3 run_safe.py --pick-mic         # choose the mic again (voice steps)
    python3 run_safe.py --no-voice         # camera only
    python3 run_safe.py --stream-port 8765 # serve the view for view_safe.py (the UNO Q does this)

Flow (UNO Q): the safe boots open; a press locks it (servo 90 deg clockwise); the next press
arms it -> LEDs show three boxes -> enter three numbers (1-5). Each step is either
held up by hand for 3 s (the ring on the dial fills; pull your hand out after) or said out
loud ("two"). Which steps are spoken is part of the code (set_password.py 1h 3v 5h), so every
step accepts both and only the full sequence is checked -> the safe opens (and stays open until
the next press) or stays locked.
Keys: SPACE = button, o = toggle owner check, r = relock now, q = quit.
"""
import argparse
import queue
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch

import config as C
import models
import set_password
import ui
from audio import ClipTrigger, MicStream, draw_meter, voice_array
from camera_picker import first_working_camera, pick_camera, saved_camera
from mic_picker import auto_mic, resolve_mic, saved_mic
from preprocess import CameraGrabber, crease_preview, crop_roi, gesture_array, open_camera, palm_array, roi_box
from safe_io import UnoQSafe, VirtualSafe

GESTURES = C.CLASSES["gesture"]
WORDS = C.CLASSES["voice"]
ME = C.CLASSES["owner"].index("me")
SPEAKER_ME = C.CLASSES["speaker"].index("me")
SPOKEN = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}


class CodeReader:
    """Turns a noisy per-frame prediction stream into discrete digits.

    A digit is confirmed once it has been held in the box for HOLD_S seconds' worth of votes.

    Frames vote instead of each having to be sure: a frame votes for its top label when that
    label's probability is at least VOTE_CONF. Each vote adds the frame's time to that digit's
    credit (an unsure frame right after a vote counts for the digit in the lead), and drains every other digit's credit at HOLD_DECAY times that
    rate. The first digit to reach HOLD_S of credit is confirmed. A clean hold therefore takes
    exactly HOLD_S (the same 3 s at 30 fps on the laptop or 10 fps on the UNO Q); a hand that
    flickers to other digits or to "unsure" just takes a little longer instead of starting over.
    If no digit gets a vote for GRACE_S (hand gone), all credit is dropped.
    After a confirm, the box has to go RELEASE_S without any digit votes before the next digit
    is taken, which is what lets a code like 3-3-1 work and stops one long hold entering 3-3-3.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.credit, self.owner = {}, {}
        self.t_prev, self.t_last = None, 0.0
        self.need_release, self.t_free = False, None
        self.last_owner = None
        self.confirmed, self.t_confirmed = None, -1e9

    @property
    def label(self):
        """The digit currently in the lead (what the UI shows), or None."""
        if self.need_release or not self.credit:
            return None
        lead = max(self.credit, key=self.credit.get)
        return lead if self.credit[lead] > 0 else None

    def progress(self, now=None):
        lead = self.label
        return 0.0 if lead is None else min(1.0, self.credit[lead] / C.HOLD_S)

    def remaining(self, now=None):
        lead = self.label
        return C.HOLD_S if lead is None else max(0.0, C.HOLD_S - self.credit[lead])

    def hold_info(self, now):
        """What the UI needs to draw the hold ring."""
        flash = self.confirmed if now - self.t_confirmed < C.CONFIRM_FLASH_S else None
        return {"label": self.label, "progress": self.progress(), "remaining": self.remaining(),
                "release": self.need_release, "flash": flash}

    def update(self, label, conf, owner_p, now):
        dt = 0.0 if self.t_prev is None else min(now - self.t_prev, 0.5)   # cap a stalled frame
        self.t_prev = now
        vote = label if conf >= C.VOTE_CONF else None
        digit = vote if vote not in (None, "none") else None

        if self.need_release:                                   # waiting for the hand to leave
            if digit is not None:
                self.t_free = None
            else:
                self.t_free = self.t_free or now
                if now - self.t_free >= C.RELEASE_S:
                    self.need_release = False
            return None

        if digit is not None:
            self.t_last = now
            self.credit[digit] = self.credit.get(digit, 0.0) + dt
            if owner_p is not None:
                self.owner.setdefault(digit, []).append(owner_p)
        elif vote is None and self.label is not None and now - self.t_last <= C.GRACE_S:
            self.credit[self.label] += dt                       # unsure frame mid-hold: keep the clock running
        if vote is not None:                                    # any confident vote drains the rest
            for d in self.credit:
                if d != digit:
                    self.credit[d] = max(0.0, self.credit[d] - C.HOLD_DECAY * dt)
        if self.credit and now - self.t_last > C.GRACE_S:       # hand gone
            self.credit, self.owner = {}, {}
            return None

        lead = self.label
        if lead is not None and self.credit[lead] >= C.HOLD_S:
            scores = self.owner.get(lead, [])
            self.last_owner = float(np.mean(scores)) if scores else None
            self.confirmed, self.t_confirmed = int(lead), now
            self.credit, self.owner = {}, {}
            self.need_release, self.t_free = True, None
            return int(lead)
        return None


@torch.no_grad()
def predict(model, array):
    return model(torch.from_numpy(array)[None]).softmax(1)[0].numpy()


class PresenceGate:
    """Is anything in the box, compared with the empty background?

    Works on a 32x32 blurred grayscale copy of the crop with its mean removed (so auto-exposure
    shifts don't count). The background is learned from the first frames and keeps blending in
    while the box reads empty. If the box reads occupied for PRESENCE_RESET_S while staying
    still (camera bumped, a box left in view), that becomes the new background.
    """

    def __init__(self):
        self.bg, self.prev, self.t_occupied, self.frac = None, None, None, 0.0
        self.t_start = None

    @staticmethod
    def _small(crop):
        g = cv2.cvtColor(cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
        return g - g.mean()

    def update(self, crop, now):
        g = self._small(crop)
        if self.t_start is None:
            self.t_start = now
        if self.bg is None or now - self.t_start < C.PRESENCE_WARMUP_S:
            self.bg, self.prev = g, g              # camera exposure is still settling: keep the newest
            return False
        self.frac = float((np.abs(g - self.bg) > C.PRESENCE_PIX).mean())
        # "moving" = a real change between frames (a hand), not dim-light sensor noise
        moving = float((np.abs(g - self.prev) > C.PRESENCE_PIX).mean()) > C.PRESENCE_MOVE_FRAC
        self.prev = g
        present = self.frac > C.PRESENCE_FRAC
        if not present:
            self.bg += C.PRESENCE_LEARN * (g - self.bg)
            self.t_occupied = None
        else:
            if moving or self.t_occupied is None:
                self.t_occupied = now
            elif now - self.t_occupied > C.PRESENCE_RESET_S:
                self.bg, self.t_occupied = g.copy(), None     # the scene itself changed
                print("[presence] background re-learned")
                return False
        return present


class VoiceWorker:
    """Classifies voice clips on its own thread, so a ~160 ms clip on the UNO Q never stalls
    the camera loop (which is what made hand holds stutter). Drops clips if it falls behind."""

    def __init__(self, voice_net, speaker_net):
        self.voice_net, self.speaker_net = voice_net, speaker_net
        self.inbox, self.outbox = queue.Queue(maxsize=2), queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, clip, want_speaker, t):
        try:
            self.inbox.put_nowait((clip, want_speaker, t))
        except queue.Full:
            pass

    def _run(self):
        while True:
            clip, want_speaker, t = self.inbox.get()
            arr = voice_array(clip)
            vp = predict(self.voice_net, arr)
            me = float(predict(self.speaker_net, arr)[SPEAKER_ME]) if (want_speaker and self.speaker_net) else None
            self.outbox.put((t, vp, me))

    def results(self):
        out = []
        while True:
            try:
                out.append(self.outbox.get_nowait())
            except queue.Empty:
                return out


def draw_camera(frame, crop, label, conf, owner_p, hold, state, owner_mode):
    view = frame.copy()
    x0, y0, s = roi_box(*view.shape[:2])
    color = {"ARMED": (0, 170, 255), "OPEN": (60, 210, 60),
             "DENIED": (40, 40, 230), "LOCKOUT": (40, 40, 230)}.get(state, (150, 150, 150))
    cv2.rectangle(view, (x0, y0), (x0 + s, y0 + s), color, 2)
    if state == "ARMED" and hold is not None:
        cv2.rectangle(view, (x0, y0 + s + 4), (x0 + s, y0 + s + 10), (60, 60, 60), -1)
        cv2.rectangle(view, (x0, y0 + s + 4), (x0 + int(s * hold["progress"]), y0 + s + 10), color, -1)
        if hold["release"]:
            msg = "remove hand"
        elif hold["label"]:
            msg = f"holding {hold['label']}: {hold['remaining']:.1f}s"
        else:
            msg = f"hold a number for {C.HOLD_S:.0f}s"
        cv2.putText(view, msg, (x0 + 8, y0 + s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    txt = f"{label} ({conf:.2f})"
    if owner_p is not None:
        txt += f"  me {owner_p:.2f}"
    cv2.putText(view, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if owner_mode and crop is not None:
        pv = crease_preview(crop)
        view[40:40 + pv.shape[0], 10:10 + pv.shape[1]] = pv
    return view


def _main(cleanup):
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner-check", action="store_true", help="require the palm-crease owner net to say 'me'")
    ap.add_argument("--hardware", choices=["virtual", "unoq"], default="virtual")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--pick-camera", action="store_true", help="open the camera picker window first")
    ap.add_argument("--camera", help="camera index, or 'auto' = first camera that returns frames (UNO Q)")
    ap.add_argument("--no-voice", action="store_true", help="don't open the mic; hand steps only")
    ap.add_argument("--pick-mic", action="store_true", help="open the mic picker window first")
    ap.add_argument("--stream-port", type=int, help="serve the annotated view as MJPEG on this port (view_safe.py)")
    ap.add_argument("--mic", help="mic device index, or 'auto' = saved name / USB mic / default (UNO Q)")
    ap.add_argument("--start-open", action="store_true",
                    help="boot with the bolt open; the first button press locks it (UNO Q, first start only)")
    ap.add_argument("--perf", type=float, default=10.0, help="print a [perf] line every N seconds (0 = off)")
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

    voice_net = None if a.no_voice else models.load("voice")
    speaker_net = models.load("speaker") if voice_net is not None else None
    mic = trigger = voice = None
    if not a.no_voice and voice_net is None:
        print("No models/voice.pt yet, so spoken steps can't be entered. "
              "Train it with collect_voice.py + train.py --task voice.")
    if voice_net is not None:
        try:
            if a.mic == "auto":
                mic_index = auto_mic()
            elif a.headless and a.mic is None and saved_mic() is None:
                mic_index = None
            else:
                mic_index = resolve_mic(None if a.mic is None else int(a.mic), a.pick_mic)
            if mic_index is None:
                raise RuntimeError("no usable microphone (run mic_picker.py, or pass --mic)")
            mic = MicStream(mic_index)
            cleanup.append(mic.close)
            trigger = ClipTrigger(mic)
            voice = VoiceWorker(voice_net, speaker_net)
            print(f"Voice on: {mic.name}" + ("" if speaker_net else "  (no speaker.pt, so owner check skips voice steps)"))
        except Exception as e:                    # no PortAudio, no mic, mic busy: run camera-only
            print(f"Voice is off: {e}")
            voice_net = speaker_net = mic = trigger = voice = None
    heard, t_heard = None, -1e9                   # last spoken step, for the UI
    t_step = -1e9                                 # when the last step (either kind) was confirmed
    t_active = -1e9                               # last sign of someone entering (hold building, voice clip)

    safe = UnoQSafe() if a.hardware == "unoq" else VirtualSafe()
    reader = CodeReader()
    gate = PresenceGate() if C.PRESENCE_GATE else None
    presence_n = loops_n = 0
    entered, owner_scores, fails = [], [], 0
    if a.start_open:                              # OPEN -press-> IDLE (locked) -press-> ARMED
        safe.unlock()
        safe.set_leds("open")
        state, status = "OPEN", "Open. Press the button to lock"
    else:
        safe.lock()
        safe.set_leds("off")
        state, status = "IDLE", "Locked. Press the button"
    t_state = time.time()
    loop_times = deque(maxlen=30)

    def go(new_state, new_status, leds, digits=0):
        nonlocal state, t_state, status
        state, t_state, status = new_state, time.time(), new_status
        safe.set_leds(leds, digits)
        print(f"[{time.strftime('%H:%M:%S')}] {new_state}: {new_status}")

    cam = a.camera
    if cam == "auto":
        cam = first_working_camera()
        if cam is None:
            raise SystemExit("No camera returned a frame. Is the USB webcam plugged in?")
        print(f"Camera: index {cam}")
    elif cam is not None:
        cam = int(cam)
    if cam is None and not a.headless and (a.pick_camera or saved_camera() is None):
        cam = pick_camera()
        if cam is None:
            raise SystemExit("No camera selected.")
    grabber = CameraGrabber(open_camera(cam))
    cleanup.append(grabber.release)

    # The latest results of the loop, for whoever draws the view (local window or stream thread).
    snap = {"label": "none", "conf": 0.0, "owner_p": None, "hold": None, "state": state,
            "status": status, "loop_fps": 0.0}

    def render():
        """Draws the full view from the NEWEST camera frame plus the latest results."""
        _, frame = grabber.latest()
        if frame is None:
            return None
        sp, now = snap, time.time()
        crop = crop_roi(frame) if owner_mode else None
        view = draw_camera(frame, crop, sp["label"], sp["conf"], sp["owner_p"], sp["hold"], sp["state"], owner_mode)
        cv2.putText(view, f"net {sp['loop_fps']:.0f} fps", (C.FRAME_W - 105, C.FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if mic is not None:                       # mic meter + last word, bottom-left of the camera
            draw_meter(view, 10, C.FRAME_H - 30, 200, 10, mic.level_db(), trigger.threshold)
            if heard and now - t_heard < 2.0:
                cv2.putText(view, f'heard "{heard[0]}" ({heard[1]:.2f})', (10, C.FRAME_H - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 210, 60), 2)
        panel = ui.draw_panel(safe, sp["status"], owner_mode, owner_net is not None, sp["hold"])
        return np.hstack([view, panel])

    last_clip = [0, None]
    streamer = None
    if a.stream_port:
        from stream import FrameStreamer
        streamer = FrameStreamer(a.stream_port, render=render, on_press=safe.press,
                                 state=lambda: {"state": state, "status": status, "steps": len(entered)},
                                 raw=lambda: (lambda sf: (sf[0], crop_roi(sf[1]) if sf[1] is not None else None))(grabber.latest()),
                                 audio=None if mic is None else {
                                     "clip": lambda: (last_clip[0], last_clip[1]),
                                     "noise": lambda: mic.latest(C.CLIP_S),
                                     "level": lambda: (mic.level_db(), trigger.threshold)})
        cleanup.append(streamer.close)

    win = "Hand Safe"
    if not a.headless:
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, lambda ev, x, y, *_: safe.press()
                             if ev == cv2.EVENT_LBUTTONDOWN and ui.in_button(x, y, C.FRAME_W) else None)

    print(__doc__)
    seq, t_perf, cam_seq0, worst_gap, t_prev, votes = 0, time.time(), 0, 0.0, None, Counter()
    while True:
        seq, frame = grabber.read(seq)            # newest frame; never a stale buffered one
        if frame is None:
            print("camera stopped returning frames")
            break
        t0 = time.time()
        if t_prev is not None:
            worst_gap = max(worst_gap, t0 - t_prev)
        t_prev = t0
        crop = crop_roi(frame)
        present = gate.update(crop, t0) if gate else True
        if present:
            probs = predict(gesture_net, gesture_array(crop))
            label, conf = GESTURES[int(probs.argmax())], float(probs.max())
        else:
            label, conf = "none", 1.0              # box matches the empty background
        presence_n += present
        loops_n += 1
        owner_p = None
        if owner_mode and label != "none":
            owner_p = float(predict(owner_net, palm_array(crop))[ME])

        now = time.time()
        clip = trigger.poll(now) if trigger else None
        if clip is not None:
            last_clip[0], last_clip[1] = last_clip[0] + 1, clip   # served to collect_voice.py --board
        if clip is not None and state == "ARMED":
            voice.submit(clip, owner_mode, now)   # classified on the voice thread
        pressed = safe.button_pressed()
        if state == "OPEN" and pressed:
            safe.lock()                           # servo turns 90 degrees clockwise
            go("IDLE", "Locked. Press again to enter the code", "off")
        elif state == "IDLE" and pressed:
            entered, owner_scores = [], []
            reader.reset()
            go("ARMED", f"Show or say {C.CODE_LENGTH} numbers", "armed")
        elif state == "ARMED":
            votes[label if conf >= C.VOTE_CONF else "unsure"] += 1
            step = None
            digit = reader.update(label, conf, owner_p, now) if len(entered) < C.CODE_LENGTH else None
            if digit is not None:
                step = ("h", digit, reader.last_owner)
            if reader.hold_info(now)["label"]:
                t_active = now                    # a hold is building: the user is mid-entry
            for t_clip, vp, me_p in (voice.results() if voice else []):
                if t_clip >= t_state:
                    t_active = now                # they spoke (even if it wasn't understood)
                if step is not None or t_clip < t_state or len(entered) >= C.CODE_LENGTH:
                    continue                      # one step per loop; ignore clips from before arming
                word, vconf = WORDS[int(vp.argmax())], float(vp.max())
                if word != "none" and vconf >= C.VOICE_CONF:
                    step = ("v", int(word), me_p)
                    heard, t_heard = (SPOKEN[word], vconf, word), now
                else:
                    print(f"  (heard {word} at {vconf:.2f}, ignored)")
            if step is not None:
                mode, d, score = step
                entered.append((mode, d))
                owner_scores.append((mode, score))
                t_step = now
                safe.set_leds("progress", len(entered))
                status = f"{len(entered)} of {C.CODE_LENGTH}" if len(entered) < C.CODE_LENGTH else "Checking..."
                print(f"  step {len(entered)}: {'hand' if mode == 'h' else 'voice'}"
                      + (f" (me {score:.2f})" if score is not None else ""))
            if len(entered) == C.CODE_LENGTH and now - t_step >= C.CONFIRM_FLASH_S:
                code_ok = set_password.check(entered)
                owner_ok = not owner_mode or all(
                    (s is not None and s >= C.OWNER_THRESH) if m == "h"
                    else (speaker_net is None or (s is not None and s >= C.SPEAKER_THRESH))
                    for m, s in owner_scores)
                if code_ok and owner_ok:
                    fails = 0
                    safe.unlock()
                    go("OPEN", "UNLOCKED. Press the button to lock", "granted", C.CODE_LENGTH)
                else:
                    fails += 1
                    why = "wrong code" if not code_ok else "hand not recognized"
                    print(f"  denied: {why} ({fails}/{C.MAX_FAILS})")
                    go("DENIED", "ACCESS DENIED", "denied", C.CODE_LENGTH)
            elif len(entered) < C.CODE_LENGTH and now - max(t_state, t_step, t_active) > C.ENTRY_TIMEOUT_S:
                go("IDLE", "Timed out. Press the button", "off")
        else:
            if voice:
                voice.results()                   # not armed: discard
        if state == "OPEN" and safe.leds == "granted" and now - t_state > C.GRANTED_SHOW_S:
            safe.set_leds("open")                 # check mark, then the open padlock until locked
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

        loop_times.append(time.time() - t0)
        hold = reader.hold_info(now) if state == "ARMED" else None
        if hold is not None and heard and now - t_heard < C.CONFIRM_FLASH_S:
            hold["flash"], hold["flash_text"] = heard[2], "HEARD"
        safe.tick(now, hold)                      # LED matrix on the UNO Q (hold bar, digit flash)
        snap = {"label": label, "conf": conf, "owner_p": owner_p, "hold": hold, "state": state,
                "status": status, "loop_fps": len(loop_times) / max(1e-6, sum(loop_times))}

        if a.perf and now - t_perf >= a.perf:     # one line of health numbers for the app log
            dt = now - t_perf
            line = (f"[perf] camera {(seq - cam_seq0) / dt:.0f} fps, net {snap['loop_fps']:.1f} fps, "
                    f"worst gap {worst_gap * 1000:.0f} ms")
            if trigger:
                line += (f", mic floor {trigger.floor if trigger.floor is not None else float('nan'):.0f} dB "
                         f"-> trigger {trigger.threshold:.0f} dB, {trigger.clips} clips")
                trigger.clips = 0
            if getattr(safe, "_t_fail", None) is not None:
                line += f", BRIDGE DOWN {now - safe._t_fail:.0f}s"
            elif getattr(safe, "errors", 0):
                line += f", bridge errors {safe.errors}"
            if getattr(safe, "errors", 0):
                safe.errors = 0
            if gate:
                line += f", box occupied {presence_n / max(1, loops_n):.0%} (last {gate.frac:.0%} changed)"
            if votes:
                line += ", armed votes " + " ".join(f"{k}:{v}" for k, v in votes.most_common(4))
            print(line)
            t_perf, cam_seq0, worst_gap, votes, presence_n, loops_n = now, seq, 0.0, Counter(), 0, 0

        if a.headless:
            continue
        view = render()
        if view is not None:
            cv2.imshow(win, view)
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


def main():
    """Runs the safe; whatever stops it (q, an error, a lost camera), the camera, mic and stream
    port are released, so the UNO Q's automatic restart can open them again."""
    cleanup = []
    try:
        _main(cleanup)
    finally:
        for close in reversed(cleanup):
            try:
                close()
            except Exception as e:
                print("cleanup:", e)
        try:
            cv2.destroyAllWindows()
        except cv2.error:                         # headless OpenCV on the UNO Q has no windows
            pass


if __name__ == "__main__":
    main()
