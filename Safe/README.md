# Safe

The goal here is to build a properly fun lock around the Arduino UNO Q. You walk up to the safe and push a button, which lights the LEDs. Then you hold three finger counts (each 1–5) under a USB camera, and the safe either opens or stays shut. The physical safe is still in progress. Until it exists, the safe is a virtual window on my laptop, drawn by [ui.py](ui.py).

## How it works

In order to keep the whole stack to PyTorch and OpenCV (both of which install cleanly on the UNO Q's Debian side), I skipped hand-landmark libraries. Instead I crop a fixed square "place your hand here" box out of each frame and classify it directly. All of the crop and input logic lives in [preprocess.py](preprocess.py). The collector, the trainer and the safe all import it, so the nets see identical pixels at training time and at the lock.

There are two small CNNs, both defined in [models.py](models.py):

1. The gesture net takes a 96px RGB crop and outputs `none / 1 / 2 / 3 / 4 / 5`. The `none` class matters more than it looks. It is how the safe knows your hand has left the box between digits, which in turn is what lets a code like 3-3-1 work.
2. The owner net decides `me / not_me` from the palm's crease lines. A raw webcam image is mostly lighting and skin tone, which says nothing useful about whose hand it is. To get around that, the crop is converted to gray and equalized with CLAHE. A morphological black-hat then keeps only the dark features narrower than a 9px kernel, and that is roughly what a crease is. The net sees the equalized gray and this crease map as two channels. Of note, the owner data is never mirrored during augmentation, since a flipped palm is effectively someone else's palm.

[run_safe.py](run_safe.py) turns the noisy per-frame predictions into digits. A digit only counts once it has been held in the box for 3s. I timed this by the clock rather than by frame count, since the laptop and the UNO Q won't run at the same frame rate. Over 3s the classifier is bound to flicker, so misreads shorter than 0.35s don't reset the hold. To show the hold, a ring around the safe's dial fills as you hold, with the digit and a countdown in the middle, and it flashes green on confirm. After that, the box has to read `none` for 0.3s before the next digit is accepted. With owner check on, P(me) is averaged over each digit's frames, and all three digits have to clear 0.85. Three wrong attempts in a row lock the keypad for 30s. Every one of these numbers lives in [config.py](config.py).

## Running it

```bash
pip install -r requirements.txt
```

Of note, OpenCV on macOS only knows cameras by index, and that order shuffles when a USB webcam or an iPhone is connected. To deal with this, [camera_picker.py](camera_picker.py) opens a window with a live tile from every camera it finds. You click the USB webcam, and the choice is saved to `camera.json` so the safe reuses the camera the nets were trained on.

1. Set the code with [set_password.py](set_password.py), e.g. `python3 set_password.py 4 2 5`. Only a salted hash is stored, in a git-ignored `password.json`.
2. Gesture-only safe:
   1. Record data with `python3 collect_data.py --task gesture`. Keys 0–5 pick the label (0 = none), and SPACE starts or stops a session. Move and tilt your hand while recording. For `none`, record an empty box, a hand half in or out, and a fist.
   2. Train with `python3 train.py --task gesture`. Train/val is split by recording session, so the val accuracy isn't just neighboring frames of the training set.
   3. Run `python3 run_safe.py`. Click PRESS or hit SPACE, then show your three numbers.
3. Owner-only safe (palm creases):
   1. Record with `python3 collect_data.py --task owner`. Use `m` for my hand and `n` for anyone else's. Keep the palm facing the camera while cycling through 1–5. The more different `not_me` hands, the stricter the check gets.
   2. Train with `python3 train.py --task owner`. Read the printed false-accept rate, since that is the number that matters for a lock.
   3. Run `python3 run_safe.py --owner-check`, or press `o` in the window to toggle the check on and off live.
4. On the UNO Q, the same code runs headless as the `safe` app in [ArduinoUNOq/safe](../ArduinoUNOq/safe/README.md). That README has the button and servo wiring. [sync_to_unoq.py](sync_to_unoq.py) copies this folder's code, nets and code file into the app, and `tools/deploy.py safe` puts it on the board. The start button, servo bolt and LED matrix go through the Bridge via [safe_io.py](safe_io.py).

## Voice

With the hand pipeline working, I added a voice side built the same way. [mic_picker.py](mic_picker.py) does for microphones what the camera picker does, showing a live level meter per input so I can talk and click whichever bar moves. Unlike cameras, mics report their names, so the choice is saved by name and survives the USB webcam being replugged.

[collect_voice.py](collect_voice.py) cuts a 1s clip every time my voice crosses a trigger level, starting 0.2s before the word. The `none` label instead grabs a clip every second regardless, which is how silence, typing and other words get recorded. [audio.py](audio.py) turns each clip into a 40-band log-mel spectrogram. I wrote the STFT and mel filterbank in numpy rather than pulling in librosa, in keeping with building this from the ground up. Each band's mean is subtracted per clip, which strips out the fixed coloring of whichever mic recorded it. The spectrogram (40 × 98) then goes into the same small CNN as the hand.

There are two tasks here, mirroring the hand:

1. `voice` recognizes the spoken words "one" through "five" plus `none`.
2. `speaker` decides `me / not_me`, the audio twin of the palm-crease check. Training never speeds up these clips, since that shifts pitch, and pitch is part of who is talking (the same reason palms are never mirrored).

Training reuses [train.py](train.py) and the per-session split. Augmentation shifts the clip in time, changes the gain, and mixes in real background from the `none` recordings at 3–20dB SNR. It then blanks a random frequency band and time stretch (SpecAugment). [live_voice.py](live_voice.py) is the test bench: speak a digit and it shows the spectrogram, every class's probability, and P(me). With both halves working, the code became a mix of the two: hand 1, say "two", hand 5 (`set_password.py h1 v2 h5`). Specifically, every step takes either a hand or a voice, and only the full sequence is checked at the end. That way the screen never hints which step is meant to be spoken, so the hand/voice pattern is part of the secret. Holding up 1-2-5 by hand gets you the same "ACCESS DENIED" as a wrong number.

`data/` is git-ignored, since it is a pile of pictures of my hand.
