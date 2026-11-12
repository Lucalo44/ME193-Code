"""Record labeled 1 s voice clips from the microphone.

    python3 collect_voice.py --task voice      # words: keys 0-5 pick the label (0 = none)
    python3 collect_voice.py --task speaker    # who:   m = me, n = not_me (other people)

A mic picker window opens first: talk, and click the mic whose bar moves. (--mic N skips it.)
SPACE starts/stops a session (each start is a new session), +/- move the trigger level, q quits.

With a word label selected, every time your voice crosses the trigger line (orange) a 1 s
clip is cut, starting 0.2 s before you spoke, and saved to data/<task>/<label>/.
Say the word, pause ~half a second, say it again. Vary loudness, speed, and distance.
The "none" label instead saves one clip every second whether or not anything is said:
record silence, room noise, typing, other words, coughs, music. The more junk, the fewer
false triggers later. Its clips are also mixed into the others as background noise in training.

--board records through the UNO Q's own webcam mic instead, while the safe app runs on the
board. The board's trigger cuts the clips, exactly as it does during a real unlock:
    python3 collect_voice.py --board            # or --board 192.168.1.191
(+/- do nothing in this mode; the board sets its own trigger level.)

Like the hand data: record every label in every place/mic you care about, in rounds.
"""
import argparse
import io
import time
import urllib.request

import cv2
import numpy as np

import config as C
from audio import ClipTrigger, MicStream, draw_meter, read_wav, spec_image, write_wav
from mic_picker import pick_mic

KEYS = {
    "voice": {ord(str(i)): ("none" if i == 0 else str(i)) for i in range(6)},
    "speaker": {ord("m"): "me", ord("n"): "not_me"},
}
SAY = {"none": "(background - don't say a digit)", "1": 'say "one"', "2": 'say "two"',
       "3": 'say "three"', "4": 'say "four"', "5": 'say "five"',
       "me": "you: say anything (the digit words are best)", "not_me": "someone else talking"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class BoardMic:
    """Stands in for MicStream + ClipTrigger: asks the safe app on the board for its clips."""

    def __init__(self, url):
        self.url, self.seq, self.level, self.threshold = url, None, -90.0, -35.0
        self.name, self.sr, self.armed, self.trigger_db, self.t_poll = f"board mic ({url})", C.SAMPLE_RATE, True, 0, 0.0

    def _get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=3) as r:
            self.level = float(r.headers.get("X-Level", self.level))
            self.threshold = float(r.headers.get("X-Trigger", self.threshold))
            return int(r.headers.get("X-Seq", -1)), r.read()

    def poll(self, now):                          # newest triggered clip, or None
        if now - self.t_poll < 0.08:
            return None
        self.t_poll = now
        seq, body = self._get(f"/clip?after={self.seq if self.seq is not None else 10**9}")
        if self.seq is None:                      # first call: skip whatever was cut before we joined
            self.seq = seq
            return None
        if not body:
            return None
        self.seq = seq
        return read_wav(io.BytesIO(body))

    def latest(self, seconds=C.CLIP_S):
        return read_wav(io.BytesIO(self._get("/noise")[1]))

    def level_db(self):
        return self.level

    def close(self):
        pass


def count(task):
    return {c: len(list((C.DATA_DIR / task / c).glob("*.wav"))) for c in C.CLASSES[task]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(C.AUDIO_TASKS), default="voice")
    ap.add_argument("--mic", type=int, help="input device index; skips the picker window")
    ap.add_argument("--board", nargs="?", const="auto", metavar="HOST",
                    help="record from the UNO Q safe's mic (needs the safe app running); optional board address")
    ap.add_argument("--max", type=int, default=150, help="stop a session after this many clips")
    a = ap.parse_args()

    if a.board:
        from view_safe import PORT, board_hosts, find_safe
        url = find_safe(board_hosts() + ["localhost"] if a.board == "auto" else [a.board], PORT)
        if url is None:
            raise SystemExit("Couldn't reach the safe app on the board. Is it running?")
        mic = trigger = BoardMic(url)
    else:
        mic_index = a.mic if a.mic is not None else pick_mic()
        if mic_index is None:
            raise SystemExit("No microphone selected.")
        mic = MicStream(mic_index)
        trigger = ClipTrigger(mic)
    for c in C.CLASSES[a.task]:
        (C.DATA_DIR / a.task / c).mkdir(parents=True, exist_ok=True)
    keys = KEYS[a.task]
    label = C.CLASSES[a.task][-1]
    counts = count(a.task)
    recording, session, n_session = False, "", 0
    next_bg, last_clip, last_saved_at = 0.0, None, -1e9
    print(__doc__)
    print(f"mic: {mic.name} @ {mic.sr} Hz")

    while True:
        now = time.time()
        clip = None
        if recording and label == "none":
            if now >= next_bg:                      # background: one clip per second, no trigger
                clip, next_bg = mic.latest(C.CLIP_S), now + C.CLIP_S
        else:
            clip = trigger.poll(now)                # always runs so you can test the trigger
            if clip is not None and not recording:
                last_clip, clip = clip, None        # paused: show it, don't save it
        if clip is not None:
            write_wav(C.DATA_DIR / a.task / label / f"{session}_{n_session:05d}.wav", clip)
            n_session += 1
            counts[label] += 1
            last_clip, last_saved_at = clip, now
            if n_session >= a.max:
                recording = False
                print(f"session {session}: hit --max ({a.max})")

        # ---- draw ----
        img = np.full((420, 640, 3), 20, np.uint8)
        rec_col = (0, 0, 255) if recording else (255, 255, 255)
        cv2.putText(img, f"label: {label}   {'REC' if recording else 'paused (SPACE)'}", (10, 30),
                    FONT, 0.7, rec_col, 2, cv2.LINE_AA)
        cv2.putText(img, SAY[label], (10, 58), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        level = mic.level_db()
        draw_meter(img, 10, 80, 520, 18, level, trigger.threshold)
        cv2.putText(img, f"{level:4.0f} dB  trig {trigger.threshold:.0f} (min {trigger.trigger_db}, +/-)", (10, 120), FONT, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        if not trigger.armed:
            cv2.putText(img, "capturing...", (540, 95), FONT, 0.5, (60, 210, 60), 1, cv2.LINE_AA)
        if last_clip is not None:
            img[140:340, 10:630] = spec_image(last_clip, 620, 200)
            cv2.putText(img, "last clip" + (" (SAVED)" if now - last_saved_at < 0.6 else ""), (14, 160),
                        FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        summary = "  ".join(f"{k}:{v}" for k, v in counts.items())
        cv2.putText(img, summary, (10, 370), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"mic: {mic.name}", (10, 400), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.imshow(f"collect: {a.task}", img)

        k = cv2.waitKey(10) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            recording = not recording
            if recording:
                session, n_session, next_bg = str(int(time.time() * 1000)), 0, now
            print(f"{'recording' if recording else 'paused'} [{label}]  {counts}")
        elif k in (ord("+"), ord("=")):
            trigger.trigger_db = min(-5, trigger.trigger_db + 2)
        elif k in (ord("-"), ord("_")):
            trigger.trigger_db = max(-75, trigger.trigger_db - 2)
        elif k in keys:
            label = keys[k]
            print("label ->", label)

    mic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
