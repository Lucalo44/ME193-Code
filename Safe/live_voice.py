"""Try the trained voice nets live.

    python3 live_voice.py                # uses the mic saved by the picker
    python3 live_voice.py --pick-mic     # choose the mic again

Speak a digit; each time the trigger fires, the 1 s clip is classified and shown with the
probability of every class (and P(me) if models/speaker.pt exists). +/- move the trigger,
q quits. This is the test bench; wiring voice into run_safe.py comes next.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

import config as C
import models
from audio import ClipTrigger, MicStream, draw_meter, spec_image, voice_array
from mic_picker import resolve_mic

FONT = cv2.FONT_HERSHEY_SIMPLEX
WORDS = {"none": "none", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five"}


@torch.no_grad()
def predict(model, clip):
    return model(torch.from_numpy(voice_array(clip))[None]).softmax(1)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", type=int, help="input device index")
    ap.add_argument("--pick-mic", action="store_true", help="open the mic picker window first")
    a = ap.parse_args()

    voice_net = models.load("voice")
    if voice_net is None:
        raise SystemExit("No models/voice.pt yet: collect_voice.py --task voice, then train.py --task voice.")
    speaker_net = models.load("speaker")
    classes = C.CLASSES["voice"]
    me = C.CLASSES["speaker"].index("me")

    mic = MicStream(resolve_mic(a.mic, a.pick_mic))
    trigger = ClipTrigger(mic)
    print(__doc__)
    print(f"mic: {mic.name} @ {mic.sr} Hz   speaker check: {'on' if speaker_net else 'no model'}")
    last_clip, probs, p_me, t_last = None, None, None, -1e9
    history = deque(maxlen=8)

    while True:
        now = time.time()
        clip = trigger.poll(now)
        if clip is not None:
            t0 = time.time()
            probs = predict(voice_net, clip)
            p_me = float(predict(speaker_net, clip)[me]) if speaker_net else None
            last_clip, t_last = clip, now
            k = int(probs.argmax())
            word = WORDS[classes[k]] if probs[k] >= C.VOICE_CONF else "?"
            if word != "none":
                history.appendleft(word + (f" (me {p_me:.2f})" if p_me is not None else ""))
            print(f"heard: {word:>5}  p={probs[k]:.2f}" + (f"  me={p_me:.2f}" if p_me is not None else "")
                  + f"  ({(time.time() - t0) * 1000:.0f} ms)")

        img = np.full((460, 640, 3), 20, np.uint8)
        level = mic.level_db()
        draw_meter(img, 10, 20, 520, 16, level, trigger.threshold)
        cv2.putText(img, f"{level:4.0f} dB  trig {trigger.threshold:.0f}", (535, 33), FONT, 0.4, (200, 200, 200), 1)
        if last_clip is not None:
            img[50:210, 10:630] = spec_image(last_clip, 620, 160)
        if probs is not None:
            fresh = now - t_last < 1.5
            for i, c in enumerate(classes):                       # probability bars
                y = 235 + i * 26
                col = (60, 210, 60) if i == probs.argmax() and probs[i] >= C.VOICE_CONF else (120, 120, 120)
                cv2.putText(img, WORDS[c], (10, y + 14), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(img, (80, y), (80 + int(300 * probs[i]), y + 18), col if fresh else (70, 70, 70), -1)
                cv2.putText(img, f"{probs[i]:.2f}", (390, y + 14), FONT, 0.45, (200, 200, 200), 1)
            if p_me is not None:
                col = (60, 210, 60) if p_me >= C.SPEAKER_THRESH else (40, 40, 230)
                cv2.putText(img, f"speaker: P(me) = {p_me:.2f}", (10, 410), FONT, 0.6, col, 2, cv2.LINE_AA)
        cv2.putText(img, "heard:", (460, 250), FONT, 0.5, (200, 200, 200), 1)
        for i, h in enumerate(history):
            cv2.putText(img, h, (460, 275 + i * 20), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"mic: {mic.name}", (10, 445), FONT, 0.45, (150, 150, 150), 1)
        cv2.imshow("Voice test", img)

        k = cv2.waitKey(10) & 0xFF
        if k == ord("q"):
            break
        if k in (ord("+"), ord("=")):
            trigger.trigger_db = min(-5, trigger.trigger_db + 2)
        elif k in (ord("-"), ord("_")):
            trigger.trigger_db = max(-75, trigger.trigger_db - 2)

    mic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
